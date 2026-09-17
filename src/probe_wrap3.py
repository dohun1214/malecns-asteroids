"""좌표 범위를 **종류별로** 다시 본다.

probe_wrap 에서 y 가 18~528 로 나왔는데, 정지 상태 측정(probe_wrap2)에서는
운석 y 가 24~192 였다. 둘 중 하나가 이상하다 — 섞어서 재면 안 된다.
Player 와 Asteroid 를 따로, 분위수까지 본다. 화면 밖 값이 있으면 그게 쓰레기다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, with_fire, frame_action, ACT_EVERY

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEEDS = [11, 33, 44]
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
pol = BrainPolicy(inertia=True)

P = {"Player": [[], []], "Asteroid": [[], []], "PlayerMissile": [[], []]}
raw = {"Player": 0, "Asteroid": 0}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    env.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); pol.b.reset(); pol.frame = 0; pol.dec.reset()
    act = (A.index("NOOP"), A.index("FIRE"))
    for f in range(FRAMES):
        _, _, tr, te, _ = env.step(frame_action(act[0], act[1], f))
        if tr or te: break
        for o in env.objects:
            if o is None: continue
            n = type(o).__name__
            if n not in P: continue
            w, h = o.wh
            if n in raw: raw[n] += 1
            if w <= 0 or h <= 0: continue          # 기존 필터
            P[n][0].append(float(o.xy[0])); P[n][1].append(float(o.xy[1]))
        if f % ACT_EVERY: continue
        ship, asts = V.parse(env.objects)
        if ship is None:
            a0 = A.index("NOOP"); act = (a0, A.index("FIRE")); continue
        ori = int(getattr(ship, "orientation", 0))
        xy, head, looms = V.looming(env.objects)
        a_, _c = pol(looms, ori, A, vel=V.ship_v)
        act = (a_, with_fire(a_, A))

for n, (x, y) in P.items():
    if not x: say(f"{n:14s} 없음"); continue
    x = np.array(x); y = np.array(y)
    q = lambda v, p: np.percentile(v, p)
    say(f"{n:14s} n={len(x):7,}")
    say(f"   x  min {x.min():6.0f}  1% {q(x,1):6.0f}  50% {q(x,50):6.0f}"
        f"  99% {q(x,99):6.0f}  max {x.max():6.0f}")
    say(f"   y  min {y.min():6.0f}  1% {q(y,1):6.0f}  50% {q(y,50):6.0f}"
        f"  99% {q(y,99):6.0f}  max {y.max():6.0f}")
    out = int(np.sum((x < 0) | (x > 160) | (y < 0) | (y > 210)))
    say(f"   화면(0~160 x 0~210) 밖 {out:,} ({out/len(x)*100:.2f}%)")
    if out:
        m = (x < 0) | (x > 160) | (y < 0) | (y > 210)
        say(f"      밖 표본 x {x[m][:8].round(0)}  y {y[m][:8].round(0)}")
json.dump({n: dict(n=len(v[0]), xmin=float(min(v[0])), xmax=float(max(v[0])),
                   ymin=float(min(v[1])), ymax=float(max(v[1])))
           for n, v in P.items() if v[0]},
          open("out/probe_wrap3.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/probe_wrap3.json")
