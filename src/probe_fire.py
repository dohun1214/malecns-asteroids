"""① OCAtari 의 x 좌표가 화면 픽셀과 맞는가 (운석이 세로로만 움직이는 게 진짜인가)
   ② 발사가 실제로 되는가 — 아타리는 버튼을 떼야 다시 쏘는 게임이 많다
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env

env = make_env(); A = env.unwrapped.get_action_meanings()
ale = env._env.env.env.ale
def scr(): return np.asarray(ale.getScreenRGB(), dtype=np.uint8)
def blend():
    a = scr(); env.step(A.index("NOOP")); return np.maximum(a, scr())


# ---------------------------------------------------------- ① 좌표 대조
env.reset(seed=11)
for _ in range(80): env.step(A.index("NOOP"))
print("① OCAtari 좌표 vs 화면 픽셀 (운석 밝은 픽셀의 무게중심)", flush=True)
print(f"  {'프레임':>5} {'OCAtari x,y':>16} {'픽셀 무게중심 x,y':>20}", flush=True)
for f in range(6):
    for _ in range(12): env.step(A.index("NOOP"))
    im = np.maximum(scr(), (env.step(A.index("NOOP")), scr())[1])
    m = im.sum(axis=2) > 60
    m[:20] = False                       # 점수 영역 제외
    ys, xs = np.nonzero(m)
    oc = [(round(float(o.xy[0]),1), round(float(o.xy[1]),1))
          for o in env.objects if o and type(o).__name__ == "Asteroid" and o.wh[0] > 0]
    print(f"  {f:>5} {str(oc[:2]):>16} "
          f"  픽셀 x {xs.mean():6.1f} y {ys.mean():6.1f}  (켜진 픽셀 {m.sum()})", flush=True)

# 운석 하나를 픽셀로 직접 추적해서 dx, dy 를 잰다
env.reset(seed=11)
for _ in range(80): env.step(A.index("NOOP"))
cent = []
for f in range(80):
    im = scr(); env.step(A.index("NOOP")); im = np.maximum(im, scr())
    m = im.sum(axis=2) > 60; m[:20] = False
    lab = np.zeros_like(m, dtype=np.int32)
    ys, xs = np.nonzero(m)
    if len(xs): cent.append((xs.mean(), ys.mean()))
c = np.asarray(cent)
d = np.diff(c, axis=0)
d = d[np.abs(d).sum(axis=1) < 20]
print(f"\n  전체 무게중심 이동 dx 평균 {d[:,0].mean():+.3f}  dy 평균 {d[:,1].mean():+.3f}"
      f"   |dx|>0 인 비율 {np.mean(np.abs(d[:,0])>0.05)*100:.0f}%", flush=True)

# ---------------------------------------------------------- ② 발사
def trial(mode, n=3600, seed=11):
    env.reset(seed=seed)
    for _ in range(40): env.step(A.index("NOOP"))
    rng = np.random.default_rng(0)
    score = 0.0; last = A.index("NOOP")
    for f in range(n):
        rot = rng.choice(["NOOP", "LEFT", "RIGHT", "UP"], p=[.55, .15, .15, .15])
        if mode == "hold":
            a = {"NOOP":"FIRE","LEFT":"LEFTFIRE","RIGHT":"RIGHTFIRE","UP":"UPFIRE"}[rot]
        elif mode == "pulse":
            a = ({"NOOP":"FIRE","LEFT":"LEFTFIRE","RIGHT":"RIGHTFIRE","UP":"UPFIRE"}[rot]
                 if f % 8 == 0 else rot)
        elif mode == "pulse4":
            a = ({"NOOP":"FIRE","LEFT":"LEFTFIRE","RIGHT":"RIGHTFIRE","UP":"UPFIRE"}[rot]
                 if f % 4 == 0 else rot)
        else:
            a = rot
        o, r, tr, te, info = env.step(A.index(a))
        score += float(r)
        if tr or te: break
    return score

print(f"\n② 발사 방식별 3600프레임(60초) 점수 — 배는 무작위로 회전/추진", flush=True)
for m in ("norfire", "hold", "pulse4", "pulse"):
    ss = [trial(m, seed=s) for s in (11, 12, 13)]
    print(f"  {m:<9} {np.mean(ss):7.1f}   시드별 {ss}", flush=True)
