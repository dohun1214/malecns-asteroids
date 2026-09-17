"""랩 주기를 **실측**한다. 160 x 210 은 가정이었고, y 범위가 18~528 로 나왔다.

`vision.py` 는 배 속도와 운석 추적에서 `dx -= 160*round(dx/160)`,
`dy -= 210*round(dy/210)` 를 쓴다. **주기가 틀리면 그 보정도 틀린 것이다.**

방법: 운석을 프레임 단위로 추적해서 **위치가 갑자기 크게 튀는 순간**(감싸기)을 잡고,
그 점프 크기를 모은다. 점프 크기의 최빈값이 곧 랩 주기다.
운석은 (w,h) 로 종류를 구분할 수 있고 등속이므로 매칭이 쉽다.
"""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, frame_action, ACT_EVERY

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()

env.reset(seed=0)
for _ in range(11): env.step(A.index("NOOP"))
prev = {}
jx, jy = [], []
allx, ally = [], []
shipx, shipy = [], []
act = A.index("NOOP")
for f in range(FRAMES):
    _, _, tr, te, _ = env.step(act)
    if tr or te:
        env.reset(seed=0); prev = {}; continue
    if f % 2: continue                     # 스프라이트 깜빡임 위상 고정
    ship, asts = V.parse(env.objects)
    if ship is not None:
        shipx.append(float(ship.xy[0])); shipy.append(float(ship.xy[1]))
    cur = {}
    for a in asts:
        x, y = float(a.xy[0]), float(a.xy[1]); w, h = a.wh
        allx.append(x); ally.append(y)
        cur.setdefault((w, h), []).append((x, y))
    for k, v in cur.items():
        if k not in prev: continue
        for (x, y) in v:
            # 직전 프레임의 같은 크기 운석 중 가장 가까운 것과 잇는다
            best = min(prev[k], key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
            dx, dy = x - best[0], y - best[1]
            if abs(dx) > 40: jx.append(abs(dx))
            if abs(dy) > 40: jy.append(abs(dy))
    prev = cur

allx = np.array(allx); ally = np.array(ally)
say(f"운석 좌표 범위   x {allx.min():.0f} ~ {allx.max():.0f}"
    f"   y {ally.min():.0f} ~ {ally.max():.0f}")
say(f"배 좌표 범위     x {min(shipx):.0f} ~ {max(shipx):.0f}"
    f"   y {min(shipy):.0f} ~ {max(shipy):.0f}")
for nm, j in (("x", jx), ("y", jy)):
    if not j:
        say(f"{nm} 축 점프 없음"); continue
    c = collections.Counter(int(round(v)) for v in j)
    say(f"{nm} 축 점프 {len(j)}회.  상위: {c.most_common(6)}")
    say(f"      중앙 {np.median(j):.1f}  평균 {np.mean(j):.1f}")
json.dump(dict(xmin=float(allx.min()), xmax=float(allx.max()),
               ymin=float(ally.min()), ymax=float(ally.max()),
               jx=[float(v) for v in jx[:200]], jy=[float(v) for v in jy[:200]]),
          open("out/probe_wrap2.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/probe_wrap2.json")
