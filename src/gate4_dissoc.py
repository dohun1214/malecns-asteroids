"""게이트 4 (b) — **이중 해리**. 게이트 4 의 1차 산출물이다. 점수가 아니다.

  LC10a 를 끄면  -> 조준만 무너지고 회피는 멀쩡해야 한다
  DNp11 을 끄면  -> 회피만 무너지고 조준은 멀쩡해야 한다

이게 나오면 "아무 뉴런이나 꺼도 망가지는 것 아니냐"가 막힌다. 한 방향 해리로는 못 막는다.

같은 뉴런을 껐는데 **모드에 따라 결과가 달라야** 한다는 점이 핵심이다.
그래서 (모드 2) x (병변 3) = 6 조건을 같은 시드로 짝지어 돌린다.

지표
  조준  각폭 안 비율 · 발사당 명중률 · 가장 가까운 운석의 각거리(중앙)
  회피  목숨당 생존 프레임 · 위협 사건 탈출률
  ⚠️ 발사 규칙은 6 조건 전부 동일하다. 그래서 명중률 차이를 '조준 차이'라고 말할 수 있다.
"""
import sys, json, gc
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, PursuitPolicy, make_env, Vision, with_fire,
                  frame_action, ACT_EVERY, _takes_vel)

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEEDS = [11, 33, 44]
def say(*a): print(*a, flush=True)

ENV = make_env(); A = ENV.unwrapped.get_action_meanings(); V = Vision()


def n_mis(env):
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def run(policy, seed, frames=FRAMES):
    env = ENV
    rng = np.random.default_rng(seed)
    takes = _takes_vel(policy)
    env.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset()
    if hasattr(policy, "dec"): policy.dec.reset()
    if hasattr(policy, "b"): policy.b.reset(); policy.frame = 0
    act = (A.index("NOOP"), A.index("FIRE"))
    prev = 0; shots = hits = 0; score = 0.0
    aimed = ndec = 0; nearest = []
    lives_prev = None; alive_runs = []; cur = 0
    n_deaths = 0        # 🔴 이슈 #56: '배 보이는 구간' 지표는 화면 판정에 흔들린다.
                        #    정책/병변 비교에는 lives 감소로 센 쪽을 쓴다.
    n_frames = 0
    for f in range(frames):
        _, rew, tr, te, info = env.step(frame_action(act[0], act[1], f))
        score += float(rew)
        if rew > 0: hits += 1
        m = n_mis(env)
        if m > prev: shots += (m - prev)
        prev = m
        if tr or te: break
        if f % ACT_EVERY: continue
        lv = info.get("lives", None) if isinstance(info, dict) else None
        if lives_prev is not None and lv is not None and lv < lives_prev:
            if cur > 0: alive_runs.append(cur)
            cur = 0; n_deaths += 1
        lives_prev = lv
        xy, head, looms = V.looming(ENV.objects)
        if xy is None:
            a0 = A.index("NOOP"); act = (a0, A.index("FIRE")); continue
        cur += ACT_EVERY
        ori = 0
        for o in ENV.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        ndec += 1
        if looms:
            phi = np.array([L["phi_rel"] for L in looms])
            th = np.array([L["theta"] for L in looms])
            d = np.array([L["dist"] for L in looms])
            if (np.abs(phi) <= th/2.0).any(): aimed += 1
            nearest.append(abs(float(phi[int(np.argmin(d))])))
        a, _c = (policy(looms, ori, A, vel=V.ship_v) if takes
                 else policy(looms, ori, A))
        act = (a, with_fire(a, A))
    if cur > 0: alive_runs.append(cur)
    return dict(shots=shots, hits=hits, score=score,
                aimed=aimed/max(ndec, 1),
                near=float(np.median(nearest)) if nearest else float("nan"),
                life=float(np.mean(alive_runs)) if alive_runs else 0.0,
                n_deaths=n_deaths,
                life_per_death=float(f + 1)/float(n_deaths + 1))


CONDS = [("도망", None), ("도망", "LC10a"), ("도망", "DNp11"),
         ("쫓기", None), ("쫓기", "LC10a"), ("쫓기", "DNp11")]
R = {}
say(f"{'모드':<5} {'병변':<7} | {'각폭안':>7} {'발사당명중':>10} {'최근접각':>8} "
    f"| {'죽음당생존':>10} {'점수':>7}")
say("-"*78)
for mode, les in CONDS:
    if mode == "도망":
        pol = BrainPolicy(lesion=les, inertia=True) if les != "LC10a" else None
        if pol is None:
            # BrainPolicy 는 circuit_idx 이름만 받는다. LC10a 는 인덱스로 직접 끈다.
            pol = BrainPolicy(inertia=True)
            import pandas as pd
            nd = pd.read_feather(Path(__file__).resolve().parent.parent.parent
                                 /"malecns-song"/"graph"/"nodes.feather")
            t = nd["type"].astype("string").fillna("").to_numpy()
            pol.b.lesion(torch.as_tensor(np.where(t == "LC10a")[0], device="cuda"))
            pol.lesion_name = "LC10a"
    else:
        pol = PursuitPolicy(lesion=les)
    rs = [run(pol, s) for s in SEEDS]
    g = lambda k: float(np.mean([r[k] for r in rs]))
    R[f"{mode}/{les or '온전'}"] = {k: g(k) for k in rs[0]}
    say(f"{mode:<5} {les or '온전':<7} | {g('aimed')*100:6.1f}% "
        f"{g('hits')/max(g('shots'),1)*100:9.1f}% {g('near'):7.1f}도 "
        f"| {g('life_per_death'):10.0f} {g('score'):7.0f}")
    del pol; gc.collect(); torch.cuda.empty_cache()

Path("out").mkdir(exist_ok=True)
json.dump(R, open("out/gate4_dissoc.json", "w"), ensure_ascii=False, indent=1)


def d(a, b, k):
    x, y = R[a][k], R[b][k]
    return (y - x)/abs(x)*100 if abs(x) > 1e-9 else float("nan")


say("\n=== 이중 해리 ===")
say(f"{'':22s} {'조준 (각폭 안)':>16s} {'회피 (죽음당 생존)':>20s}")
for mode in ("도망", "쫓기"):
    for les in ("LC10a", "DNp11"):
        say(f"{mode} 모드에서 {les:>6s} 끄기  {d(f'{mode}/온전', f'{mode}/{les}', 'aimed'):+14.1f}% "
            f"{d(f'{mode}/온전', f'{mode}/{les}', 'life_per_death'):+19.1f}%")
say("\n저장: out/gate4_dissoc.json")
