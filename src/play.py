"""뇌 <-> 게임 폐루프 + 대조군. 게이트 2 의 측정 도구.

정책
  brain   전체 뇌 166,700 뉴런. 매 결정마다 운석 -> LC4 자극 -> DN -> 3채널 -> 액션
  greedy  같은 기하학을 규칙으로 (뇌 없음). "그냥 규칙이랑 뭐가 달라"의 그 규칙
  random  무작위 액션
  noop    가만히
  lesion:* 뇌에서 특정 세포군을 끈 상태
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
from ocatari.core import OCAtari
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS
from vision import Vision, LC4Map, ship_heading_deg, ASPECT
from decode import Decoder

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
ACT_EVERY = 4
STEPS_PER_DECISION = 333          # dt 0.2 -> 66.6 ms, 링 길이 9 의 배수

# 발사는 뇌가 정하지 않는다. 모든 정책에 동일하게 거는 고정 요소다.
# 안 걸면 운석이 안 줄어 웨이브가 영원히 안 끝나고 '가만히 있기'가 최선이 된다.
# (부과값으로 회계에 적는다. 정책 간 비교는 동일 조건이라 공정하다)
FIRE_MAP = {"NOOP": "FIRE", "LEFT": "LEFTFIRE", "RIGHT": "RIGHTFIRE", "UP": "UPFIRE"}


def with_fire(a, actions):
    n = actions[a]
    return actions.index(FIRE_MAP.get(n, n))


# 🔴 [실측] 발사 버튼은 **계속 누르고 있으면 한 발도 안 나간다.**
#   frameskip=1 에서 1프레임만 누르면 ROM 이 못 읽고, 계속 누르면 뗄 때까지 재발사가 안 된다.
#   probe_fire4.py: 2400프레임 동안
#     계속 누름 / 1프레임 누름   -> 점수 20, 운석 개수 최대 5 (= 아무것도 못 부숨)
#     2프레임 누르고 6 뗌        -> 점수 130, 운석 개수 최대 8 (쪼개진다)
#     4프레임 누르고 4 뗌        -> 점수 130
#   우리는 결정당 4프레임 내내 같은 액션을 유지했으므로 **FIRE 를 계속 누르고 있었고,
#   따라서 지금까지 한 발도 안 쏘고 있었다.**
#   -> 결정 안에서 앞 2프레임만 FIRE, 뒤 2프레임은 떼서 회전/추진만 유지한다.
FIRE_PRESS = 2                      # ACT_EVERY 중 앞 몇 프레임을 누를지


def frame_action(base, fire, k, act_every=ACT_EVERY):
    """결정 안의 k번째 프레임에 실제로 넣을 액션. 회전·추진은 4프레임 내내 유지된다."""
    return fire if k % act_every < FIRE_PRESS else base


class BrainPolicy:
    def __init__(self, lesion=None, th50=7.2, cap=150.0, k=16, min_intensity=0.0,
                 align_slop=1, hold=0, inertia=True):
        import pandas as pd
        self.C = dict(np.load(G/"circuit_idx.npz"))
        P = np.load(G/"lc4_position.npz", allow_pickle=True)
        nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
        self.nside = nodes["somaSide"].astype("string").fillna("").to_numpy()
        self._P = P
        self.map = LC4Map(P["idx"], P["pos"], P["side"], P["valid"], float(P["theta_L"]), k=k)
        self.readout = dict(np.load(G/"vnc_readout.npz", allow_pickle=True))
        self.dec = Decoder(self.C, self.nside, self.readout,
                           align_slop=align_slop, min_intensity=min_intensity)
        # 관성 보정 (이슈 #36). 끄면 예전 동작 — A/B 비교용으로 남긴다.
        self.inertia = inertia
        self.hold = hold          # 정렬되면 몇 결정 동안 추진을 유지할지
        self._hold_left = 0
        self.b = BrainRT(params=PARAMS)
        self.b.reset(); self.b.set_poisson(self.C["LC4"], 1.0)     # has_poi 켜기
        self.b.capture(STEPS_PER_DECISION, tally=True)
        self.b.reset()
        self.gain, self.cap = th50, cap
        self.sec = STEPS_PER_DECISION*PARAMS["dt"]/1000.0
        self.lesion_name = lesion
        self.frame = 0
        if lesion:
            for name in lesion.split("+"):
                self.b.lesion(torch.as_tensor(self.C[name], device="cuda"))

    def rate_of(self, idx):
        """등급 판독: 창 전체의 평균 시냅스 구동(mV).
        스파이크 수를 세면 DN 한 종류가 2세포뿐이라 창당 1~2개로 양자화된다.
        Jang & von Reyn 2023 이 DNp02 를 subthreshold 로 기록한 것과도 맞는 판독이다."""
        if len(idx) == 0: return 0.0
        return float(self._gac[idx].mean())

    def __call__(self, looms, orientation, actions, vel=None):
        idx, rates = self.map.rates(looms, self.b.N, self.gain, self.cap)
        self.b.set_poisson_rates(idx, rates)
        self.b.tally.zero_(); self.b.gacc.zero_()
        self.b.seed.fill_(self.frame); self.frame += 1
        self.b.graph.replay()
        torch.cuda.synchronize()
        self._tal = self.b.tally.cpu().numpy()
        self._gac = (self.b.gacc/STEPS_PER_DECISION).cpu().numpy()
        ch = self.dec.channels(self.rate_of)
        a, tgt, st = self.dec.action(ch, orientation, actions,
                                     vel=vel if self.inertia else None)
        # 이력: 한 번 정렬되면 hold 결정 동안 추진을 유지한다.
        # 회전이 결정당 22.5도라 정렬 창을 지나쳐 버리는 문제(이슈 #3)에 대한 대응.
        if st == "추진":
            self._hold_left = self.hold
        elif self._hold_left > 0:
            self._hold_left -= 1
            a = actions.index("UP")
        return a, ch


    def set_controller(self, k=None, align_slop=None, th50=None, cap=None, hold=None):
        """뇌(그래프 캡처)는 그대로 두고 컨트롤러만 갈아끼운다."""
        if k is not None:
            P = self._P
            self.map = LC4Map(P["idx"], P["pos"], P["side"], P["valid"],
                              float(P["theta_L"]), k=k)
        if align_slop is not None: self.dec.align_slop = align_slop
        if th50 is not None: self.gain = th50
        if cap is not None: self.cap = cap
        if hold is not None: self.hold = hold
        self._hold_left = 0
        self.dec.reset()


class GreedyPolicy:
    """같은 기하학, 뇌 없음. 팽창률 가중 위협 벡터의 반대로 간다."""
    def __init__(self, min_dtheta=0.0, inertia=True):
        self.min_dtheta = min_dtheta
        self.inertia = inertia
    def __call__(self, looms, orientation, actions, vel=None):
        lat = fore = w = 0.0
        for L in looms:
            if L["dtheta"] <= self.min_dtheta: continue
            psi = np.radians(L["phi_rel"])
            lat += L["dtheta"]*np.sin(psi); fore += L["dtheta"]*np.cos(psi)
            w += L["dtheta"]
        ch = dict(lateral=lat, fore=fore, intensity=w, norm=float(np.hypot(lat, fore)))
        if w <= 0 or ch["norm"] < 1e-9:
            return actions.index("NOOP"), ch
        psi_e = np.degrees(np.arctan2(lat, fore)) + 180.0
        head = ship_heading_deg(orientation)
        world = (head - psi_e) % 360.0
        if self.inertia and vel is not None:                 # 뇌 쪽과 같은 보정
            vx, vy = float(vel[0]), float(vel[1]); spd = float(np.hypot(vx, vy))
            if spd > 1e-6:
                wx, wy = np.cos(np.radians(world)), np.sin(np.radians(world))
                tx, ty = wx*spd - vx, wy*spd - vy
                if np.hypot(tx, ty) > 1e-6:
                    world = np.degrees(np.arctan2(ty, tx)) % 360.0
        tgt = int(round((world - 90.0)/22.5)) % 16
        diff = (tgt - orientation + 8) % 16 - 8
        if abs(diff) <= 1: return actions.index("UP"), ch
        return (actions.index("LEFT") if diff > 0 else actions.index("RIGHT")), ch


def run_episode(env, policy, V, actions, max_frames=9000, rng=None, log=None, fire=True,
                noop_start=30, threat_r=30.0):
    """threat_r 은 스칼라 또는 반경 목록. 반경은 **채점에만** 쓰이고 정책에는 전혀
    들어가지 않으므로, 여러 반경을 한 번의 주행에서 동시에 집계한다 (3배 절약 +
    반경 간 비교가 '같은 주행'이라 잡음이 안 섞인다)."""
    """noop_start: 시작 시 무작위 no-op 프레임 수의 상한 (ALE 표준 평가 규약).

    [실측] ALE Asteroids 는 env.reset(seed=...) 를 줘도 **게임이 완전히 동일하다.**
    운석 초기 배치가 고정이라 시드로는 변동이 안 생긴다 (에피소드 간 표준편차 0.0,
    튜닝 시드와 보고 시드의 결과가 소수점까지 일치). 시드 분리로 과적합을 막으려면
    다른 변동원이 필요하다. 표준 방식대로 시작 시 무작위 no-op 을 넣는다.
    """
    obs = env.reset(seed=int(rng.integers(0, 2**31)) if rng else 0)
    if noop_start and rng is not None:
        for _ in range(int(rng.integers(1, noop_start + 1))):
            env.step(actions.index("NOOP"))
    V.reset()
    if hasattr(policy, "dec"): policy.dec.reset()
    if hasattr(policy, "b"): policy.b.reset(); policy.frame = 0
    action = (actions.index("NOOP"), actions.index("FIRE"))
    score = 0.0; frames = 0; alive_runs = []; cur = 0; had_ship = False
    n_up = 0; n_dec = 0; n_left = 0; n_right = 0
    # --- 사건 기반 지표 (이슈 #6): 접근 중인 운석이 위험 반경 안에 든 '사건' 단위로 센다
    radii = [float(threat_r)] if np.isscalar(threat_r) else [float(x) for x in threat_r]
    ev_open = {R: {} for R in radii}      # 반경 -> {운석 id: 사건 시작 프레임}
    ev_total = {R: 0 for R in radii}
    ev_hit = {R: 0 for R in radii}
    lives_prev = None
    for f in range(max_frames):
        obs, rew, trunc, term, info = env.step(
            frame_action(action[0], action[1], f))         # OCAtari 순서
        score += float(rew); frames += 1
        if term or trunc: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            if had_ship and cur > 0: alive_runs.append(cur); cur = 0
            a0 = actions.index("NOOP")
            action = (a0, actions.index("FIRE") if fire else a0); continue
        had_ship = True; cur += ACT_EVERY
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        # --- 사건 추적: 피격은 목숨 감소로 판정한다 (Player 소실은 하이퍼스페이스와 구분 불가)
        lives = info.get("lives", None) if isinstance(info, dict) else None
        hit_now = (lives_prev is not None and lives is not None and lives < lives_prev)
        lives_prev = lives
        for R in radii:
            op = ev_open[R]; near = set()
            for L in looms:
                if L["dtheta"] > 0 and L["dist"] <= R:
                    near.add(L["id"])
                    if L["id"] not in op:
                        op[L["id"]] = f; ev_total[R] += 1
            if hit_now and op:
                ev_hit[R] += 1               # 위협 사건이 열려 있는 동안의 피격
                op.clear()
            else:
                for i in list(op):           # 반경 밖으로 나갔거나 사라진 사건은 종료
                    if i not in near: del op[i]

        try:
            action, ch = policy(looms, ori, actions, vel=V.ship_v)
        except TypeError:            # vel 을 안 받는 옛 정책 (가만히 있기/무작위)
            action, ch = policy(looms, ori, actions)
        n_dec += 1
        if actions[action] == "UP": n_up += 1
        elif actions[action] == "LEFT": n_left += 1
        elif actions[action] == "RIGHT": n_right += 1
        base_a = action
        action = (base_a, with_fire(base_a, actions)) if fire else (base_a, base_a)
        if log is not None:
            log.append(dict(f=f, n_loom=len(looms), action=int(action),
                            **{k: float(v) for k, v in ch.items()
                               if isinstance(v, (int, float))}))
    if cur > 0: alive_runs.append(cur)
    R0 = radii[0]
    return dict(score=score, frames=frames, up_frac=(n_up/max(n_dec,1)),
                n_dec=n_dec, n_left=n_left, n_right=n_right,
                turn_bias=((n_left-n_right)/max(n_left+n_right, 1)),
                events=ev_total[R0], hits=ev_hit[R0],
                events_by_r={R: ev_total[R] for R in radii},
                hits_by_r={R: ev_hit[R] for R in radii},
                escape_rate=(1.0 - ev_hit[R0]/ev_total[R0]) if ev_total[R0] else float("nan"),
                exposure=ev_total[R0]/max(frames,1)*1000.0,
                mean_life=float(np.mean(alive_runs)) if alive_runs else 0.0,
                max_life=float(max(alive_runs)) if alive_runs else 0.0,
                n_lives=len(alive_runs))


def make_env():
    return OCAtari("ALE/Asteroids-v5", mode="ram", hud=False, frameskip=1,
                   repeat_action_probability=0.0)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "brain"
    n_ep  = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mf    = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    env = make_env(); actions = env.unwrapped.get_action_meanings(); V = Vision()
    rng = np.random.default_rng(12345)
    if which == "brain":      pol = BrainPolicy()
    elif which.startswith("lesion:"): pol = BrainPolicy(lesion=which.split(":",1)[1])
    elif which == "greedy":   pol = GreedyPolicy()
    elif which == "random":
        r2 = np.random.default_rng(7)
        pol = lambda looms, ori, acts: (int(r2.integers(0, len(acts))), {"norm": 0.0})
    else:
        pol = lambda looms, ori, acts: (acts.index("NOOP"), {"norm": 0.0})
    res = []
    t0 = time.perf_counter()
    for e in range(n_ep):
        r = run_episode(env, pol, V, actions, max_frames=mf, rng=rng)
        res.append(r)
        print(f"  ep{e}: score {r['score']:6.0f}  frames {r['frames']:5d}  "
              f"평균 생존 {r['mean_life']:6.1f}f  최장 {r['max_life']:6.1f}f  "
              f"목숨 {r['n_lives']}", flush=True)
    el = time.perf_counter()-t0
    agg = dict(policy=which, n_ep=n_ep,
               score=float(np.mean([r["score"] for r in res])),
               mean_life=float(np.mean([r["mean_life"] for r in res])),
               max_life=float(np.mean([r["max_life"] for r in res])),
               frames=float(np.mean([r["frames"] for r in res])),
               wall_s=el)
    print(f"[{which}] 점수 {agg['score']:.0f}  평균생존 {agg['mean_life']:.1f}f  "
          f"최장 {agg['max_life']:.1f}f  ({el:.0f}s)")
    (ROOT/"out"/f"play_{which.replace(':','_')}.json").write_text(
        json.dumps(dict(agg=agg, eps=res), indent=2), encoding="utf-8")
