"""배가 결정 프레임에서 왜 안 보이는가 — 실측."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, ACT_EVERY

env = make_env(); A = env.unwrapped.get_action_meanings()
env.reset(seed=0)
for _ in range(20): env.step(A.index("NOOP"))
n = 4000
have_obj = np.zeros(n, bool); have_ok = np.zeros(n, bool); lives = np.zeros(n, int)
wh = []
for f in range(n):
    o_, r, tr, te, info = env.step(A.index("FIRE"))
    if tr or te:
        env.reset(seed=int(f)); continue
    lives[f] = info.get("lives", -1)
    for o in env.objects:
        if o and type(o).__name__ == "Player":
            have_obj[f] = True
            w, h = o.wh
            wh.append((int(w), int(h)))
            if w > 0 and h > 0: have_ok[f] = True
            break

def runs(mask):
    out = []; c = 0
    for v in mask:
        if v: c += 1
        elif c: out.append(c); c = 0
    if c: out.append(c)
    return np.array(out) if out else np.array([0])

print(f"프레임 {n}")
print(f"  Player 객체 존재      {have_obj.mean()*100:5.1f}%")
print(f"  그중 w>0,h>0 (통과)   {have_ok.mean()*100:5.1f}%   <- Vision.parse 가 실제로 받는 것")
print(f"  객체는 있는데 버려짐   {(have_obj & ~have_ok).mean()*100:5.1f}%")
u, c = np.unique(np.array(wh), axis=0, return_counts=True)
print(f"  Player wh 분포: " + "  ".join(f"{tuple(a)}×{b}" for a, b in
                                        sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1])[:6]))
ab = runs(~have_ok)
print(f"  '안 보임' 연속 구간: {len(ab)}개, 평균 {ab.mean():.1f}프레임, 최대 {ab.max()}프레임")
print(f"    길이 분포 " + "  ".join(f"{q}%:{np.percentile(ab,q):.0f}" for q in (50,75,90,99)))
# 4프레임 위상별로 보이는 비율
for ph in range(ACT_EVERY):
    m = have_ok[ph::ACT_EVERY]
    print(f"  위상 {ph}: 보임 {m.mean()*100:5.1f}%")
