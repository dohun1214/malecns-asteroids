"""y 랩 주기 확정 — 이번엔 **관성 주행**으로. (probe_wrap7 은 자가시험에서 떨어졌다)

probe_wrap7 이 왜 틀렸나: 계속 추진하면 **배가 가속한다.** 등속을 가정한 v x T 가 깨진다.
자가시험(x 축에서 160 이 나와야 함)이 10.6 을 내놔서 바로 걸렸다.
*자가시험을 안 붙였으면 89.3 을 y 주기로 적을 뻔했다.*

이번 방법: 30프레임만 추진하고 **그 뒤로는 NOOP**. 17문서에서 이 게임은 마찰이 사실상
없다고 실측했으므로 배는 등속으로 미끄러진다. 그러면 랩에서 랩까지 = 정확히 한 바퀴다.
자가시험은 그대로 x 축(=160).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()


def coast(turns, push=150, frames=9000):
    env.reset(seed=0)
    for _ in range(13): env.step(A.index("NOOP"))
    for _ in range(turns):
        for k in range(4): env.step(A.index("LEFT") if k < 2 else A.index("NOOP"))
    for _ in range(push): env.step(A.index("UP"))
    out = []
    for f in range(frames):
        env.step(A.index("NOOP"))
        ship, _ = V.parse(env.objects)
        out.append(None if ship is None else (float(ship.xy[0]), float(ship.xy[1])))
    return out


def analyse(seq, axis, lo_band, hi_band, bound=210.0):
    """랩 = 값이 lo_band 아래에서 hi_band 위로 (또는 반대로) 건너뛰는 사건."""
    vals = []
    for i, p in enumerate(seq):
        v = None if p is None else p[axis]
        if v is not None and v > bound: v = None      # 전이 구간
        vals.append(v)
    ok = [(i, v) for i, v in enumerate(vals) if v is not None]
    # 등속 확인: 작은 연속 변위만 모은다
    d = [b[1]-a[1] for a, b in zip(ok, ok[1:]) if b[0]-a[0] == 1 and abs(b[1]-a[1]) <= 6]
    if not d: return None
    v = float(np.mean(d))
    laps = []
    for a, b in zip(ok, ok[1:]):
        if a[1] < lo_band and b[1] > hi_band and v < 0: laps.append(a[0])
        if a[1] > hi_band and b[1] < lo_band and v > 0: laps.append(a[0])
    if len(laps) < 3: return dict(v=v, n=len(laps), P=None, laps=laps)
    gaps = np.diff(laps)
    return dict(v=v, n=len(laps), gaps=gaps.tolist(), P=(abs(v)*gaps).tolist())


for turns, axis, name, lo, hi in ((0, 1, "y", 50.0, 140.0), (4, 0, "x", 50.0, 110.0)):
    seq = coast(turns)
    r = analyse(seq, axis, lo, hi)
    if r is None or r["P"] is None:
        say(f"[{name}] 랩 {0 if r is None else r['n']}회 — 부족"); continue
    P = np.array(r["P"])
    say(f"[{name}] 관성 속도 {r['v']:+.4f} px/frame,  랩 {r['n']}회")
    say(f"   랩 간격(프레임) {r['gaps'][:8]}")
    say(f"   주기 {np.round(P,1)[:8]}")
    say(f"   **중앙 {np.median(P):.1f}   평균 {P.mean():.1f} ± {P.std():.1f}**")
json.dump({}, open("out/probe_wrap8.json", "w"))
