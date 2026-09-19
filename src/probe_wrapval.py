"""WRAP_Y 를 **목적에 맞는 기준**으로 검증한다.

픽셀로 잰 값은 177 이다. 그게 맞다면, 운석 추적이 가장 잘 이어져야 한다 —
`Vision.looming` 은 직전 결정의 같은 크기 운석 중 가장 가까운 것과 잇고, 못 이으면
**새 ID** 를 준다. 주기가 틀리면 감싸는 운석에서 연결이 끊긴다.

후보 주기를 훑으며 **결정당 새 ID 수**를 센다. 최소가 되는 값이 맞는 주기다.
자가시험: x 축을 같은 방법으로 훑으면 160 이 나와야 한다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision as VS
from play import make_env, Vision, with_fire, frame_action, ACT_EVERY
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings()
FRAMES = 4000


def run(px, py):
    VS.WRAP_X, VS.WRAP_Y = float(px), float(py)
    V = Vision()
    rng = np.random.default_rng(7)
    env.reset(seed=0)
    for _ in range(19): env.step(A.index("NOOP"))
    V.reset()
    act = (A.index("NOOP"), A.index("FIRE"))
    new_ids = 0; n_obj = 0; n_dec = 0
    last = -1
    for f in range(FRAMES):
        _, _, tr, te, _ = env.step(frame_action(act[0], act[1], f))
        if tr or te:
            env.reset(seed=0); V.reset(); continue
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        n_dec += 1
        for L in looms:
            n_obj += 1
            if L["id"] > last: new_ids += 1; last = L["id"]
        b = int(rng.integers(0, 4))
        base = [A.index(x) for x in ("NOOP", "LEFT", "RIGHT", "UP")][b]
        act = (base, with_fire(base, A))
    return new_ids, n_obj, n_dec


say("[y] WRAP_X 는 160 고정, WRAP_Y 만 바꾼다")
best = None
for py in (160, 168, 172, 175, 176, 177, 178, 180, 184, 190, 200, 210):
    ni, no, nd = run(160.0, py)
    r = ni/max(no, 1)
    say(f"   WRAP_Y={py:3d}   새 ID {ni:5d} / 관측 {no:6d} = {r*100:5.2f}%")
    if best is None or r < best[1]: best = (py, r)
say(f"   -> 최소는 **WRAP_Y = {best[0]}** ({best[1]*100:.2f}%)")

say("\n[x] 자가시험 — WRAP_Y 는 177 고정, WRAP_X 를 훑으면 160 이 나와야 한다")
bx = None
for px in (140, 150, 156, 158, 160, 162, 166, 176, 200):
    ni, no, nd = run(px, 177.0)
    r = ni/max(no, 1)
    say(f"   WRAP_X={px:3d}   새 ID {ni:5d} / 관측 {no:6d} = {r*100:5.2f}%")
    if bx is None or r < bx[1]: bx = (px, r)
say(f"   -> 최소는 **WRAP_X = {bx[0]}** ({bx[1]*100:.2f}%)")
json.dump(dict(best_y=best[0], best_x=bx[0]),
          open("out/probe_wrapval.json", "w"), ensure_ascii=False, indent=1)
