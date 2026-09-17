"""게임 화면에서 운석이 안 보이고 빈 사각형만 보이는 이유 — 실측.

가설: 아타리 2600 은 한 주사선에 여러 스프라이트를 못 그려서 **프레임마다 번갈아 그린다**(깜빡임).
      대시보드는 4프레임에 한 장만 보내므로 **위상이 고정**되고, 하필 운석을 안 그리는 위상에
      잠기면 운석이 영원히 안 보인다.
      (운석 dx/dy 가 절반의 프레임에서 0이던 것, 배가 20% 안 보이던 것과 같은 계열)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, ACT_EVERY

env = make_env(); A = env.unwrapped.get_action_meanings()
ale = env._env.env.env.ale
env.reset(seed=3)
for _ in range(120): env.step(A.index("FIRE"))


def boxes():
    out = []
    for o in env.objects:
        if o and type(o).__name__ == "Asteroid":
            w, h = o.wh
            if w > 0 and h > 0:
                out.append((int(o.xy[0]), int(o.xy[1]), int(w), int(h)))
    return out


N = 60
frames, bx = [], []
for f in range(N):
    env.step(A.index("FIRE"))
    frames.append(np.asarray(ale.getScreenRGB(), dtype=np.uint8))
    bx.append(boxes())

print(f"프레임 {N}. OCAtari 가 보고하는 운석 수: "
      f"{[len(b) for b in bx[:12]]} ...", flush=True)

fill = []
for f in range(N):
    tot = hit = 0
    for (x, y, w, h) in bx[f]:
        roi = frames[f][max(0,y):y+h, max(0,x):x+w]
        if roi.size == 0: continue
        tot += roi.shape[0]*roi.shape[1]
        hit += int((roi.sum(axis=2) > 40).sum())
    fill.append(hit/tot if tot else np.nan)
fill = np.asarray(fill)
print("\n운석 박스 안이 실제로 칠해진 비율 (프레임별)", flush=True)
for p in range(ACT_EVERY):
    m = fill[p::ACT_EVERY]
    print(f"  4프레임 위상 {p}: 평균 {np.nanmean(m)*100:5.1f}%   "
          + " ".join(f"{v*100:4.0f}" for v in m[:8]), flush=True)
print(f"  전체 평균 {np.nanmean(fill)*100:.1f}%   "
      f"0%인 프레임 {int((fill < 0.02).sum())}/{N}", flush=True)

# 최댓값 합성(연속 k프레임)을 하면 얼마나 복구되나
print("\n연속 k프레임 최댓값 합성 후 같은 지표", flush=True)
for k in (1, 2, 3, 4):
    v = []
    for f in range(k-1, N):
        comp = np.maximum.reduce(frames[f-k+1:f+1])
        tot = hit = 0
        for (x, y, w, h) in bx[f]:
            roi = comp[max(0,y):y+h, max(0,x):x+w]
            if roi.size == 0: continue
            tot += roi.shape[0]*roi.shape[1]
            hit += int((roi.sum(axis=2) > 40).sum())
        v.append(hit/tot if tot else np.nan)
    print(f"  k={k}: 평균 {np.nanmean(v)*100:5.1f}%   "
          f"0%인 프레임 {int((np.asarray(v) < 0.02).sum())}/{len(v)}", flush=True)

# 화면 전체에서 '켜진 픽셀' 수가 프레임마다 어떻게 변하는가
on = np.array([(fr.sum(axis=2) > 40).sum() for fr in frames])
print(f"\n켜진 픽셀 수: 평균 {on.mean():.0f}, 최소 {on.min()}, 최대 {on.max()}", flush=True)
print("  처음 12프레임:", list(on[:12]), flush=True)
