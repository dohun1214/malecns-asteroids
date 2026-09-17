"""랩 주기를 배 궤적으로 직접 잰다. 한 방향으로만 추진해서 화면을 가로지르게 한다.

지금까지 알아낸 것
  - 운석 x 4~159 / y 24~192 (화면 밖 0%)
  - Player y 가 520~528 로 튀는 프레임이 있다 (무작위 정책 1.8%, 뇌 정책 18.9%)
    -> `Vision.parse` 가 그걸 배로 받아들인다. 그 프레임의 방위각은 전부 틀린다.

여기서는 배를 한 방향으로 밀어 **좌표가 한 바퀴 도는 것**을 직접 본다.
점프 직전/직후 값에서 랩 주기를 읽는다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, frame_action, ACT_EVERY, with_fire

def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()


def trace(turns, frames=1400):
    """turns 번 좌회전한 뒤 계속 추진하며 배 좌표를 기록한다."""
    env.reset(seed=0)
    for _ in range(13): env.step(A.index("NOOP"))
    for _ in range(turns):
        for k in range(4): env.step(A.index("LEFT") if k < 2 else A.index("NOOP"))
    out = []
    for f in range(frames):
        env.step(A.index("UP"))
        ship, _ = V.parse(env.objects)
        out.append((None, None) if ship is None
                   else (float(ship.xy[0]), float(ship.xy[1])))
    return out


for turns, name in ((0, "위쪽(orientation 0)"), (4, "90도 회전"),
                    (2, "45도 회전")):
    tr = trace(turns)
    xs = [p[0] for p in tr if p[0] is not None]
    ys = [p[1] for p in tr if p[1] is not None]
    if not xs: say(f"{name}: 배가 안 보임"); continue
    say(f"\n[{name}]  x {min(xs):.0f}~{max(xs):.0f}   y {min(ys):.0f}~{max(ys):.0f}"
        f"   (배 있는 프레임 {len(xs)}/{len(tr)})")
    # 점프 찾기
    jumps = []
    prev = None
    for p in tr:
        if p[0] is None: prev = None; continue
        if prev is not None:
            dx, dy = p[0]-prev[0], p[1]-prev[1]
            if abs(dx) > 20 or abs(dy) > 20:
                jumps.append((prev, p, round(dx, 1), round(dy, 1)))
        prev = p
    say(f"   점프 {len(jumps)}회")
    for j in jumps[:6]:
        say(f"      {j[0]} -> {j[1]}   d=({j[2]}, {j[3]})")
    off = [p for p in tr if p[1] is not None and p[1] > 210]
    if off:
        say(f"   🔴 화면 밖(y>210) 프레임 {len(off)}개, y 고유값 "
            f"{sorted(set(int(p[1]) for p in off))[:10]}")
say("\n(참고) 운석이 도는 범위는 x 4~159 / y 24~192 로 이미 쟀다.")
