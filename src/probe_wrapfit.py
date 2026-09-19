"""★ 랩 주기를 **행동으로** 확정한다. 이게 다섯 번의 실패 뒤 찾은 답이다.

착상: 주기가 틀리면 이음매를 건너는 운석의 방위각이 **한쪽으로 치우쳐** 틀린다.
그러면 **좌/우 회전이 비대칭**해진다. 그러니 **좌회전 편향이 최소가 되는 주기**가 맞는 값이다.
회피 품질(가장 가까운 운석의 각거리, 클수록 잘 피함)도 같이 본다 — 최대가 되어야 한다.

민감도 예비 측정(시드 3개, probe_wrapsens)에서 둘 다 **177 에서 최적**이 나왔다:
    편향  168:+26.1  172:+15.6  **177:+8.2**  182:+13.9  210:+27.9
    각거리 168:117.9  172:122.1  **177:124.7**  182:115.0  210:112.7
여기서는 시드를 늘리고 격자를 촘촘히 한다.

★ 자가시험: **같은 방법으로 WRAP_X 를 훑으면 160 에서 최적이 나와야 한다.**
  (배를 옆으로 밀어 x=4 -> 164, d=160 으로 직접 관측한 값)
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision as VS
from play import BrainPolicy, make_env, Vision, with_fire, frame_action, ACT_EVERY

FRAMES = 3000
SEEDS = [11, 33, 44, 55, 77, 91, 103, 127]
def say(*a): print(*a, flush=True)
ENV = make_env(); A = ENV.unwrapped.get_action_meanings(); V = Vision()
pol = BrainPolicy(inertia=True)


def run(seed):
    rng = np.random.default_rng(seed)
    ENV.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): ENV.step(A.index("NOOP"))
    V.reset(); pol.b.reset(); pol.frame = 0; pol.dec.reset()
    act = (A.index("NOOP"), A.index("FIRE"))
    near = []; nL = nR = 0
    for f in range(FRAMES):
        _, _, tr, te, _ = ENV.step(frame_action(act[0], act[1], f))
        if tr or te: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(ENV.objects)
        if xy is None:
            a0 = A.index("NOOP"); act = (a0, A.index("FIRE")); continue
        ori = 0
        for o in ENV.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        if looms:
            phi = np.array([L["phi_rel"] for L in looms])
            d = np.array([L["dist"] for L in looms])
            near.append(abs(float(phi[int(np.argmin(d))])))
        a, _c = pol(looms, ori, A, vel=V.ship_v)
        nm = A[a]; nL += nm == "LEFT"; nR += nm == "RIGHT"
        act = (a, with_fire(a, A))
    return (abs((nL-nR)/max(nL+nR, 1)),
            float(np.median(near)) if near else float("nan"))


def scan(axis, cands, fixed):
    say(f"\n[{axis}] (다른 축은 {fixed} 고정, 시드 {len(SEEDS)}개)")
    say(f"   {'주기':>5} {'|좌회전 편향|':>13} {'최근접 각거리':>14}")
    out = {}
    for P in cands:
        if axis == "y": VS.WRAP_X, VS.WRAP_Y = float(fixed), float(P)
        else:           VS.WRAP_X, VS.WRAP_Y = float(P), float(fixed)
        rs = [run(s) for s in SEEDS]
        b = float(np.mean([r[0] for r in rs])); n = float(np.mean([r[1] for r in rs]))
        bs = float(np.std([r[0] for r in rs])/np.sqrt(len(rs)))
        out[P] = (b, n)
        say(f"   {P:5d} {b*100:11.1f}% ±{bs*100:.1f} {n:13.1f}도")
    bb = min(out, key=lambda k: out[k][0]); nb = max(out, key=lambda k: out[k][1])
    say(f"   -> 편향 최소 **{bb}**   각거리 최대 **{nb}**")
    return out, bb, nb


oy, by, ny = scan("y", [166, 170, 174, 176, 177, 178, 180, 186, 196, 210], 160)
ox, bx, nx = scan("x", [150, 156, 158, 160, 162, 166, 176], 177)
VS.WRAP_X, VS.WRAP_Y = 160.0, 177.0
say(f"\n★ 자가시험: x 에서 편향 최소 {bx} / 각거리 최대 {nx}  (정답 160) "
    f"-> {'통과' if abs(bx-160) <= 2 or abs(nx-160) <= 2 else '실패'}")
json.dump(dict(y={str(k): v for k, v in oy.items()},
               x={str(k): v for k, v in ox.items()},
               best_y=by, best_x=bx),
          open("out/probe_wrapfit.json", "w"), ensure_ascii=False, indent=1)
