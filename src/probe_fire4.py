"""발사가 아예 안 먹힌다. 왜인가 — 누르는 길이/떼는 길이, 게임 시작 여부를 훑는다.

가설
  H1 frameskip=1 이라 1프레임 누름이 너무 짧아서 ROM 이 못 읽는다
  H2 아타리는 버튼을 떼야 다시 쏜다 (계속 누르면 1발 후 멈춤)
  H3 게임이 시작 안 된 상태(데모/어트랙트)라 조작은 되는데 사격만 막혀 있다
지표: 운석 개수가 한 번이라도 변하는가 (부수면 큰 운석이 쪼개져 개수가 는다) + 점수
"""
import sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env

env = make_env(); A = env.unwrapped.get_action_meanings()
ale = env._env.env.env.ale


def n_ast():
    return sum(1 for o in env.objects
               if o and type(o).__name__ == "Asteroid" and o.wh[0] > 0)


def trial(press, release, n=2400, seed=11, start_fire=0):
    env.reset(seed=seed)
    for _ in range(start_fire): env.step(A.index("FIRE"))
    for _ in range(40): env.step(A.index("NOOP"))
    score = 0.0; counts = set(); first = None; per = max(press+release, 1)
    for f in range(n):
        a = A.index("FIRE") if (press > 0 and f % per < press) else A.index("NOOP")
        o, r, tr, te, info = env.step(a)
        score += float(r)
        c = n_ast(); counts.add(c)
        if first is None and len(counts) > 1: first = f
        if tr or te: break
    return score, sorted(counts), first


print(f"{'누름':>4}{'뗌':>4}{'시작FIRE':>8}{'점수':>7}{'운석 개수 변화':>16}{'첫 변화':>9}", flush=True)
rows = []
for press, release, sf in [(0,0,0), (1,7,0), (2,6,0), (4,4,0), (8,8,0), (1,1,0),
                           (60,0,0), (1,7,30), (1,7,120), (2,30,0), (4,28,0)]:
    s, c, fi = trial(press, release, start_fire=sf)
    rows.append((press, release, sf, s, c, fi))
    print(f"{press:>4}{release:>4}{sf:>8}{s:>7.0f}{str(c):>16}{str(fi):>9}", flush=True)

# 액션 공간 확인 — full_action_space 로 하면 달라지나
from ocatari.core import OCAtari
e2 = OCAtari("ALE/Asteroids-v5", mode="ram", hud=False, frameskip=1,
             repeat_action_probability=0.0, full_action_space=True)
print(f"\nfull_action_space 액션 수: {len(e2.unwrapped.get_action_meanings())}", flush=True)
A2 = e2.unwrapped.get_action_meanings()
e2.reset(seed=11)
for _ in range(40): e2.step(A2.index("NOOP"))
def n2():
    return sum(1 for o in e2.objects if o and type(o).__name__ == "Asteroid" and o.wh[0] > 0)
cs = set(); sc = 0.0
for f in range(2400):
    a = A2.index("FIRE") if f % 8 < 2 else A2.index("NOOP")
    o, r, tr, te, info = e2.step(a); sc += float(r); cs.add(n2())
    if tr or te: break
print(f"  full_action_space 2프레임 누름/6 뗌: 점수 {sc:.0f}, 운석 개수 {sorted(cs)}", flush=True)
