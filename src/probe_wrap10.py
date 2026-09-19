"""y 랩 주기를 **좌표 집합**으로 확정한다. 지금 코드의 177 은 ±2px 불확실로 남아 있다.

관찰: 배와 운석의 y 가 **전부 짝수**다 (18, 20, ..., 192, 194). x 도 보자.
좌표가 2 간격으로 양자화돼 있고 놀이터가 [18, 194] 라면,
위로 갈 때 18 다음은 16 인데 그게 194 로 감긴다  ->  16 ≡ 194 (mod P)  ->  **P = 178**.
(픽셀로 잰 '놀이터 높이 177' 은 **칠해진 행 수**지 좌표 주기가 아니다. 1 차이가 난다.)

재는 것
  1 실제로 관측되는 y 값의 **집합** — 최소/최대/간격
  2 x 도 같은 방식으로 (정답 160 을 알고 있으니 **자가시험**이 된다)
  3 놀이터 밖 값(520~528)은 제외
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, with_fire, frame_action, ACT_EVERY
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()

ship_x, ship_y, ast_x, ast_y = set(), set(), set(), set()
rng = np.random.default_rng(11)
env.reset(seed=0)
for _ in range(17): env.step(A.index("NOOP"))
act = (A.index("NOOP"), A.index("FIRE"))
for f in range(30000):
    _, _, tr, te, _ = env.step(frame_action(act[0], act[1], f))
    if tr or te:
        env.reset(seed=0)
        for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
        continue
    for o in env.objects:
        if o is None: continue
        n = type(o).__name__
        if n not in ("Player", "Asteroid"): continue
        w, h = o.wh
        if w <= 0 or h <= 0: continue
        x, y = float(o.xy[0]), float(o.xy[1])
        if y > 210: continue                       # 전이 구간 제외
        (ship_x if n == "Player" else ast_x).add(x)
        (ship_y if n == "Player" else ast_y).add(y)
    if f % ACT_EVERY: continue
    b = int(rng.integers(0, 4))
    base = [A.index(x) for x in ("NOOP", "LEFT", "RIGHT", "UP")][b]
    act = (base, with_fire(base, A))


def report(name, s, known=None):
    v = np.array(sorted(s))
    d = np.diff(v)
    step = int(np.min(d)) if len(d) else 0
    say(f"{name:12s} n={len(v):4d}  범위 {v.min():.0f} ~ {v.max():.0f}  "
        f"최소 간격 {step}  간격 분포 {dict(zip(*np.unique(d, return_counts=True)))}")
    P = v.max() - v.min() + step
    say(f"             -> 주기 추정 = (max - min) + 간격 = **{P:.0f}**"
        + (f"   (정답 {known} 과 {'일치' if abs(P-known) < 1e-9 else '불일치'})" if known else ""))
    return float(P)


say(f"관측 프레임 30,000\n")
px = report("배 x", ship_x, known=160)
py = report("배 y", ship_y)
ax = report("운석 x", ast_x)
ay = report("운석 y", ast_y)
say(f"\n합친 범위  x {min(min(ship_x), min(ast_x)):.0f}~{max(max(ship_x), max(ast_x)):.0f}"
    f"   y {min(min(ship_y), min(ast_y)):.0f}~{max(max(ship_y), max(ast_y)):.0f}")
json.dump(dict(ship_x=[min(ship_x), max(ship_x)], ship_y=[min(ship_y), max(ship_y)],
               ast_x=[min(ast_x), max(ast_x)], ast_y=[min(ast_y), max(ast_y)],
               P_x=px, P_y=py), open("out/probe_wrap10.json", "w"),
          ensure_ascii=False, indent=1)
say("저장: out/probe_wrap10.json")
