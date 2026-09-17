"""게이트 4 (c) — 음성 대조. **이게 없으면 (b)의 −100% 는 아무 말도 아니다.**

쫓기 모드에서 LC10a 275세포를 끄면 조준이 0% 가 됐다. 그런데 그게
  ① LC10a **여서** 인지
  ② 그냥 **275개를 껐기** 때문인지
는 (b)만으로 못 가른다. 게이트 2 에서 쓴 것과 같은 논리다 (13문서).

그래서 **같은 수(275)의 무작위 세포**를 끄고 같은 걸 잰다. 시드 5개.
조준이 안 무너지면 해리가 선다. 무너지면 (b)는 철회다.

추가로 **LC10a 를 끄면 뇌가 아무 신호도 안 낸다**는 점을 확인한다 — 그건 '방향이
틀어진다'가 아니라 '입력이 끊긴다'이므로, 도피 쪽 병변(신호는 나오는데 방향이 틀린다)과
성격이 다르다. 정직하게 구분해서 적어야 한다.
"""
import sys, json, gc
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (PursuitPolicy, make_env, Vision, with_fire, frame_action, ACT_EVERY)

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEEDS = [11, 33, 44]
LES_SEEDS = [0, 1, 2, 3, 4]
def say(*a): print(*a, flush=True)
ENV = make_env(); A = ENV.unwrapped.get_action_meanings(); V = Vision()
ROOT = Path(__file__).resolve().parent.parent
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
LC10A = np.where(typ == "LC10a")[0]
N = len(nodes)


def n_mis(env):
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def run(policy, seed, frames=FRAMES):
    rng = np.random.default_rng(seed)
    ENV.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): ENV.step(A.index("NOOP"))
    V.reset(); policy.b.reset(); policy.frame = 0
    act = (A.index("NOOP"), A.index("FIRE"))
    prev = 0; shots = hits = 0; score = 0.0; aimed = ndec = 0
    nearest = []; nz = 0
    for f in range(frames):
        _, rew, tr, te, _ = ENV.step(frame_action(act[0], act[1], f))
        score += float(rew)
        if rew > 0: hits += 1
        m = n_mis(ENV)
        if m > prev: shots += (m - prev)
        prev = m
        if tr or te: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(ENV.objects)
        if xy is None:
            a0 = A.index("NOOP"); act = (a0, A.index("FIRE")); continue
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
        a, ch = policy(looms, ori, A)
        if abs(ch.get("steer", 0.0)) > 1e-12: nz += 1
        act = (a, with_fire(a, A))
    return dict(aimed=aimed/max(ndec, 1), hits=hits, shots=shots, score=score,
                near=float(np.median(nearest)) if nearest else float("nan"),
                signal=nz/max(ndec, 1))


R = {}
say(f"{'조건':<20} {'각폭안':>7} {'발사당명중':>10} {'최근접각':>8} {'신호있는결정':>12} {'점수':>7}")
say("-"*72)


def report(name, pol):
    rs = [run(pol, s) for s in SEEDS]
    g = lambda k: float(np.mean([r[k] for r in rs]))
    R[name] = {k: g(k) for k in rs[0]}
    say(f"{name:<20} {g('aimed')*100:6.1f}% {g('hits')/max(g('shots'),1)*100:9.1f}% "
        f"{g('near'):7.1f}도 {g('signal')*100:11.1f}% {g('score'):7.0f}")


pol = PursuitPolicy()
report("온전", pol)
del pol; gc.collect(); torch.cuda.empty_cache()

pol = PursuitPolicy(lesion="LC10a")
report("LC10a 275세포", pol)
del pol; gc.collect(); torch.cuda.empty_cache()

for s in LES_SEEDS:
    rng = np.random.default_rng(100 + s)
    pick = rng.choice(N, size=len(LC10A), replace=False)
    pol = PursuitPolicy()
    pol.b.lesion(torch.as_tensor(pick, device="cuda"))
    report(f"무작위 275 (시드{s})", pol)
    del pol; gc.collect(); torch.cuda.empty_cache()

rnd = [R[k] for k in R if k.startswith("무작위")]
say("\n=== 판정 ===")
a0 = R["온전"]["aimed"]; aL = R["LC10a 275세포"]["aimed"]
am = float(np.mean([r["aimed"] for r in rnd])); asd = float(np.std([r["aimed"] for r in rnd]))
say(f"  온전 각폭안 {a0*100:.1f}%")
say(f"  LC10a 끄기  {aL*100:.1f}%   ({(aL-a0)/a0*100:+.1f}%)")
say(f"  무작위 275  {am*100:.1f} ± {asd*100:.1f}%   ({(am-a0)/a0*100:+.1f}%)")
if a0 > 0:
    per_cell = abs((aL-a0)/a0) / max(abs((am-a0)/a0), 1e-9)
    say(f"  -> LC10a 가 같은 수의 무작위보다 {per_cell:.1f}배")
json.dump(R, open("out/gate4_control.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/gate4_control.json")
