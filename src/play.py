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


class BrainPolicy:
    def __init__(self, lesion=None, th50=7.2, cap=150.0, k=16, min_intensity=0.0):
        import pandas as pd
        self.C = dict(np.load(G/"circuit_idx.npz"))
        P = np.load(G/"lc4_position.npz", allow_pickle=True)
        nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
        self.nside = nodes["somaSide"].astype("string").fillna("").to_numpy()
        self.map = LC4Map(P["idx"], P["pos"], P["side"], P["valid"], float(P["theta_L"]), k=k)
        self.dec = Decoder(self.C, self.nside, min_intensity=min_intensity)
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

    def __call__(self, looms, orientation, actions):
        idx, rates = self.map.rates(looms, self.b.N, self.gain, self.cap)
        self.b.set_poisson_rates(idx, rates)
        self.b.tally.zero_(); self.b.gacc.zero_()
        self.b.seed.fill_(self.frame); self.frame += 1
        self.b.graph.replay()
        torch.cuda.synchronize()
        self._tal = self.b.tally.cpu().numpy()
        self._gac = (self.b.gacc/STEPS_PER_DECISION).cpu().numpy()
        ch = self.dec.channels(self.rate_of)
        a, tgt, st = self.dec.action(ch, orientation, actions)
        return a, ch


class GreedyPolicy:
    """같은 기하학, 뇌 없음. 팽창률 가중 위협 벡터의 반대로 간다."""
    def __init__(self, min_dtheta=0.0):
        self.min_dtheta = min_dtheta
    def __call__(self, looms, orientation, actions):
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
        tgt = int(round((((head - psi_e) % 360.0) - 90.0)/22.5)) % 16
        diff = (tgt - orientation + 8) % 16 - 8
        if abs(diff) <= 1: return actions.index("UP"), ch
        return (actions.index("LEFT") if diff > 0 else actions.index("RIGHT")), ch


def run_episode(env, policy, V, actions, max_frames=9000, rng=None, log=None, fire=True):
    obs = env.reset(seed=int(rng.integers(0, 2**31)) if rng else 0)
    V.reset()
    if hasattr(policy, "dec"): policy.dec.reset()
    if hasattr(policy, "b"): policy.b.reset(); policy.frame = 0
    action = actions.index("NOOP")
    score = 0.0; frames = 0; alive_runs = []; cur = 0; had_ship = False
    for f in range(max_frames):
        obs, rew, trunc, term, info = env.step(action)     # OCAtari 순서
        score += float(rew); frames += 1
        if term or trunc: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            if had_ship and cur > 0: alive_runs.append(cur); cur = 0
            action = actions.index("FIRE" if fire else "NOOP"); continue
        had_ship = True; cur += ACT_EVERY
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        action, ch = policy(looms, ori, actions)
        if fire: action = with_fire(action, actions)
        if log is not None:
            log.append(dict(f=f, n_loom=len(looms), action=int(action),
                            **{k: float(v) for k, v in ch.items()
                               if isinstance(v, (int, float))}))
    if cur > 0: alive_runs.append(cur)
    return dict(score=score, frames=frames,
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
