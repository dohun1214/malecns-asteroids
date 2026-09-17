"""y 랩 주기를 정확히 잰다. x 는 160 으로 확정됐다 (4 -> 164, d=160).

배를 위로만 밀면서 좌표를 프레임 단위로 찍는다. 화면 밖 값(520~528)은 전이 구간이므로
**그 구간을 건너뛰고** 직전/직후의 정상 좌표와 프레임 수·속도로 주기를 역산한다.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
env.reset(seed=0)
for _ in range(13): env.step(A.index("NOOP"))
seq = []
for f in range(700):
    env.step(A.index("UP"))
    ship, _ = V.parse(env.objects)
    seq.append(None if ship is None else float(ship.xy[1]))

say("프레임별 배 y (위로만 추진, 처음 40)")
say("  " + " ".join("--" if v is None else f"{v:.0f}" for v in seq[:40]))

# 정상 구간의 프레임당 속도
ok = [(i, v) for i, v in enumerate(seq) if v is not None and v <= 210]
spd = []
for (i0, v0), (i1, v1) in zip(ok, ok[1:]):
    if i1 == i0+1 and abs(v1-v0) < 20: spd.append(v0-v1)   # 위로 = y 감소
v = float(np.median([s for s in spd if s > 0])) if any(s > 0 for s in spd) else 0.0
say(f"\n정상 구간 프레임당 상승 {v:.3f} px (중앙값)")

say("\n전이 구간 (화면 밖 값 앞뒤)")
i = 1
shown = 0
while i < len(seq)-1 and shown < 3:
    if seq[i] is not None and seq[i] > 210 and (seq[i-1] is None or seq[i-1] <= 210):
        j = i
        while j < len(seq) and seq[j] is not None and seq[j] > 210: j += 1
        before = seq[i-1]; after = seq[j] if j < len(seq) else None
        n_out = j - i
        say(f"   직전 y={before}  화면 밖 {n_out}프레임 "
            f"({[int(x) for x in seq[i:j]]})  직후 y={after}")
        if before is not None and after is not None and v > 0:
            travel = v*(n_out+1)
            say(f"      그 사이 이동량 ~{travel:.1f}px  ->  "
                f"주기 = (직후 + 이동량) - 직전 = {after + travel - before:.1f}")
        shown += 1
        i = j
    i += 1
