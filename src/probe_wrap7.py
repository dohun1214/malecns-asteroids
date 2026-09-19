"""y 랩 주기 확정 — **랩 사이의 주기**로 잰다.

앞선 방법들이 왜 실패했나
  - 전이 구간(화면 밖 520~528) 길이가 2~6프레임으로 들쭉날쭉해 앞뒤 값 역산이 흩어졌다
  - 운석 등속성 스캔은 '크기가 같은 운석이 1개뿐'인 표본에 y 랩이 한 번도 안 걸렸다

이번 방법: 배를 한 방향으로만 밀면 **등속**이다. 화면 안 구간에서 속도 v 를 정확히 재고,
연속한 두 랩 사이의 프레임 수 T 를 세면  **주기 P = v x T** 다.
전이 구간의 길이와 무관하다 — 랩에서 랩까지가 정확히 한 바퀴이기 때문이다.

자가시험: 같은 방법을 **x 축**에 적용하면 160 이 나와야 한다 (직접 관측으로 확정된 값).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()


def trace(turns, frames):
    env.reset(seed=0)
    for _ in range(13): env.step(A.index("NOOP"))
    for _ in range(turns):
        for k in range(4): env.step(A.index("LEFT") if k < 2 else A.index("NOOP"))
    out = []
    for f in range(frames):
        env.step(A.index("UP"))
        ship, _ = V.parse(env.objects)
        out.append(None if ship is None else (float(ship.xy[0]), float(ship.xy[1])))
    return out


def period(seq, axis, bound):
    """seq: [(x,y) or None].  axis 0=x 1=y.  bound: 정상값 상한."""
    v_all = []
    laps = []          # 랩이 일어난 프레임 인덱스
    prev = None
    for i, p in enumerate(seq):
        cur = None if p is None else p[axis]
        if cur is not None and cur > bound: cur = None      # 전이 구간 버림
        if prev is not None and cur is not None:
            d = cur - prev[1]
            if abs(d) <= 6 and i - prev[0] == 1:
                v_all.append(d)
            elif abs(d) > 20:
                laps.append((prev[0], i, prev[1], cur))
        if cur is not None: prev = (i, cur)
    nz = [x for x in v_all if x != 0]
    if not nz or len(laps) < 2: return None
    v = float(np.mean(v_all))                  # 프레임당 평균 변위 (0 포함)
    # 랩 시작 프레임 사이 간격
    gaps = [laps[k+1][0] - laps[k][0] for k in range(len(laps)-1)]
    return dict(v=v, gaps=gaps, n_lap=len(laps),
                P=[abs(v)*g for g in gaps])


for turns, axis, name, bound in ((0, 1, "y (위로 추진)", 210.0),
                                 (4, 0, "x (옆으로 추진)", 200.0)):
    seq = trace(turns, 3000)
    r = period(seq, axis, bound)
    if r is None: say(f"[{name}] 랩을 2회 이상 못 잡았다"); continue
    P = np.array(r["P"])
    say(f"[{name}] 프레임당 변위 {r['v']:+.4f} px,  랩 {r['n_lap']}회")
    say(f"   랩 간격(프레임) {r['gaps'][:8]}")
    say(f"   주기 추정 {np.round(P,1)[:8]}")
    say(f"   **중앙 {np.median(P):.1f}   평균 {P.mean():.1f} ± {P.std():.1f}**")
json.dump({}, open("out/probe_wrap7.json", "w"))
