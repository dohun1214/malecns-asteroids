"""게이트 4 (e) — rewire 를 **실제 게임**에서. (d)는 합성 자극 스윕이었다.

🔴 **시드 3개로 처음 쟀다가 과대평가했다.** 합성 스윕을 시드 10개로 다시 재보니
   방향 소실이 85.8% -> 60.1% 로 내려갔고, 용량 반응도 '급락'이 아니라 **단조 증가**였다
   (gate4_rewire2.py). 여기도 시드를 6개로 늘린다.

쫓기 모드에서 `LC10a -> AOTU019/025` 배선만 섞고 조준 지표를 잰다.
세포는 하나도 안 꺼져 있으므로 **신호는 나온다.** 방향만 뒤섞인다.
"""
import sys, json, gc
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import PursuitPolicy, make_env, Vision, with_fire, frame_action, ACT_EVERY
import rewire as RW

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEEDS = [11, 33, 44]
RW_SEEDS = [0, 1, 2, 3, 4, 5]   # 🔴 시드 3개로는 과대평가가 났다 (gate4_rewire2.py)
def say(*a): print(*a, flush=True)
ENV = make_env(); A = ENV.unwrapped.get_action_meanings(); V = Vision()
ROOT = Path(__file__).resolve().parent.parent
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
LC10A = np.where(typ == "LC10a")[0]
AOTU = np.concatenate([np.where(typ == t)[0] for t in ("AOTU019", "AOTU025")])
crow, packed0, C, pre = RW.load()
N = crow.size - 1
post0, w0 = RW.unpack(packed0)
sel = RW.select(N, pre, packed0, LC10A, AOTU)
lm = np.zeros(N, bool); lm[LC10A] = True
am = np.zeros(N, bool); am[AOTU] = True
other_all = np.flatnonzero(lm[pre] & ~am[post0])
rng0 = np.random.default_rng(99)
sel_other = other_all[rng0.choice(len(other_all), size=len(sel), replace=False)]
say(f"섞는 엣지 {len(sel)}개 (전체의 {len(sel)/len(packed0)*100:.4f}%)")


def n_mis(env):
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def run(policy, seed, frames=FRAMES):
    rng = np.random.default_rng(seed)
    ENV.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): ENV.step(A.index("NOOP"))
    V.reset(); policy.b.reset(); policy.frame = 0
    act = (A.index("NOOP"), A.index("FIRE"))
    prev = shots = hits = 0; score = 0.0; aimed = ndec = nz = 0; near = []
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
            near.append(abs(float(phi[int(np.argmin(d))])))
        a, ch = policy(looms, ori, A)
        if abs(ch.get("steer", 0.0)) > 1e-12: nz += 1
        act = (a, with_fire(a, A))
    return dict(aimed=aimed/max(ndec, 1), hits=hits, shots=shots, score=score,
                near=float(np.median(near)) if near else float("nan"),
                signal=nz/max(ndec, 1))


pol = PursuitPolicy()
R = {}
say(f"\n{'조건':<24} {'각폭안':>7} {'발사당명중':>10} {'최근접각':>8} {'신호':>7} {'점수':>7}")
say("-"*68)


def report(name, packed=None):
    if packed is None:
        pol.b.packed.copy_(torch.from_numpy(packed0.astype(np.int32)).to(pol.b.dev))
    else:
        pol.b.packed.copy_(torch.from_numpy(packed.astype(np.int32)).to(pol.b.dev))
    rs = [run(pol, s) for s in SEEDS]
    g = lambda k: float(np.mean([r[k] for r in rs]))
    R[name] = {k: g(k) for k in rs[0]}
    say(f"{name:<24} {g('aimed')*100:6.1f}% {g('hits')/max(g('shots'),1)*100:9.1f}% "
        f"{g('near'):7.1f}도 {g('signal')*100:6.1f}% {g('score'):7.0f}")


report("온전")
for sd in RW_SEEDS:
    pk, _ = RW.rewire(packed0, sel, seed=sd, frac=1.0)
    report(f"lc10a->AOTU 섞기 (s{sd})", pk)
for sd in RW_SEEDS[:4]:
    pk, _ = RW.rewire(packed0, sel_other, seed=sd, frac=1.0)
    report(f"LC10a 다른출력 (s{sd})", pk)

a0 = R["온전"]["aimed"]
rw = [v["aimed"] for k, v in R.items() if k.startswith("lc10a->")]
ot = [v["aimed"] for k, v in R.items() if k.startswith("LC10a 다른")]
sg = [v["signal"] for k, v in R.items() if k.startswith("lc10a->")]
say("\n=== 판정 ===")
say(f"  온전 각폭안 {a0*100:.1f}%  신호 {R['온전']['signal']*100:.1f}%")
say(f"  배선 섞기   {np.mean(rw)*100:.1f} ± {np.std(rw)*100:.1f}%"
    f"  ({(np.mean(rw)-a0)/a0*100:+.1f}%)   **신호 {np.mean(sg)*100:.1f}% 유지**")
say(f"  다른 출력   {np.mean(ot)*100:.1f} ± {np.std(ot)*100:.1f}%"
    f"  ({(np.mean(ot)-a0)/a0*100:+.1f}%)")
json.dump(R, open("out/gate4_rewire_game.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/gate4_rewire_game.json")
