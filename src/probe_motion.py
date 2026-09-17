"""두 가지 실측.
  A. 운석이 정말 세로로만 움직이는가 (이동 방향 분포)
  B. 발사가 실제로 되고 있는가 — 아타리는 버튼을 **떼야** 다시 쏘는 게임이 많다.
     우리는 매 프레임 FIRE 를 누르고 있어서 연사가 안 될 수 있다.
"""
import sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, ACT_EVERY

env = make_env(); A = env.unwrapped.get_action_meanings()
print("액션 목록:", A, flush=True)


def objs_by(name):
    out = []
    for o in env.objects:
        if o and type(o).__name__ == name:
            w, h = o.wh
            if w > 0 and h > 0:
                out.append((float(o.xy[0]), float(o.xy[1]), int(w), int(h)))
    return out


def class_names():
    return collections.Counter(type(o).__name__ for o in env.objects if o)


# ---------------------------------------------------------------- A. 이동 방향
env.reset(seed=11)
for _ in range(60): env.step(A.index("FIRE"))
print("\n화면의 객체 종류:", dict(class_names()), flush=True)

prev = objs_by("Asteroid")
angs, spds = [], []
STEP = 2                     # OCAtari 위치는 2프레임 주기로 갱신된다
for f in range(1200):
    for _ in range(STEP): env.step(A.index("FIRE"))
    cur = objs_by("Asteroid")
    used = set()
    for (x, y, w, h) in cur:
        best, bd, bi = None, 1e9, None
        for j, (px, py, pw, ph) in enumerate(prev):
            if pw != w or ph != h or j in used: continue
            dx = x-px; dy = y-py
            dx -= 160.0*round(dx/160.0); dy -= 210.0*round(dy/210.0)
            d = dx*dx+dy*dy
            if d < bd: bd, best, bi = d, (dx, dy), j
        if best is not None and bd <= 12**2 and bd > 1e-9:
            used.add(bi)
            dx, dy = best
            angs.append(np.degrees(np.arctan2(-dy, dx)) % 180.0)   # 0=가로, 90=세로
            spds.append(np.hypot(dx, dy)/STEP)
    prev = cur

angs = np.asarray(angs); spds = np.asarray(spds)
print(f"\nA. 운석 이동 표본 {len(angs):,}개  (STEP={STEP}프레임)", flush=True)
hist, edges = np.histogram(angs, bins=12, range=(0, 180))
for i in range(12):
    lo, hi = edges[i], edges[i+1]
    lab = "가로" if lo < 15 or hi > 165 else ("세로" if 75 <= lo < 105 else "")
    print(f"  {lo:5.0f}~{hi:5.0f}도 {hist[i]/len(angs)*100:5.1f}%  "
          f"{'#'*int(hist[i]/max(hist)*40)} {lab}", flush=True)
near_v = ((angs > 60) & (angs < 120)).mean()
near_h = ((angs < 30) | (angs > 150)).mean()
print(f"  세로에 가까움(60~120도) {near_v*100:.1f}%   가로에 가까움 {near_h*100:.1f}%", flush=True)
print(f"  속도 중앙값 {np.median(spds):.2f} px/프레임, "
      f"고유 속도값 {sorted(set(np.round(spds,2)))[:10]}", flush=True)

# ---------------------------------------------------------------- B. 발사
def trial(mode, n=1800):
    env.reset(seed=11)
    for _ in range(40): env.step(A.index("FIRE"))
    score = 0.0; bullets = 0; seen = 0; fired_frames = 0
    for f in range(n):
        if mode == "hold":      a = A.index("FIRE")
        elif mode == "alt2":    a = A.index("FIRE") if (f % 2 == 0) else A.index("NOOP")
        elif mode == "alt4":    a = A.index("FIRE") if (f % 4 == 0) else A.index("NOOP")
        elif mode == "alt8":    a = A.index("FIRE") if (f % 8 == 0) else A.index("NOOP")
        else:                   a = A.index("NOOP")
        if A[a].endswith("FIRE"): fired_frames += 1
        o, r, tr, te, info = env.step(a)
        score += float(r)
        if tr or te: break
        b = [x for x in env.objects if x and "Bullet" in type(x).__name__
             or (x and "Missile" in type(x).__name__)]
        bullets += len(b); seen += 1
    return score, bullets/max(seen,1), fired_frames

print(f"\nB. 발사 방식별 1800프레임 (30초)", flush=True)
print(f"  {'방식':<8}{'점수':>8}{'화면 평균 탄환':>14}{'FIRE 누른 프레임':>16}", flush=True)
for m in ("noop", "hold", "alt2", "alt4", "alt8"):
    s, b, ff = trial(m)
    print(f"  {m:<8}{s:>8.0f}{b:>14.2f}{ff:>16}", flush=True)
