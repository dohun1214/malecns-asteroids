"""Player.orientation 값이 화면에서 어느 방향인가 — 추진해서 측정한다.

문서에 없다. 액션 매핑이 여기 걸려 있으므로 추측하면 안 된다.
방법: orientation 을 k 로 맞춘 뒤 UP(추진)을 주고 배가 실제로 움직인 방향을 잰다.
"""
import json
from pathlib import Path
import numpy as np
from ocatari.core import OCAtari

ROOT = Path(__file__).resolve().parent.parent
env = OCAtari("ALE/Asteroids-v5", mode="ram", hud=False, frameskip=1,
              repeat_action_probability=0.0)
AM = env.unwrapped.get_action_meanings()
NOOP, UP, LEFT, RIGHT = AM.index("NOOP"), AM.index("UP"), AM.index("LEFT"), AM.index("RIGHT")

def player(e):
    for o in e.objects:
        if o and type(o).__name__ == "Player": return o
    return None

res = {}
print(f"{'orient':>7} {'추진 전 xy':>16} {'추진 후 xy':>16} {'변위':>16} "
      f"{'화면각(도)':>11} {'수학각(도)':>11}")
for target in range(16):
    env.reset(seed=1)
    for _ in range(120): env.step(NOOP)
    # orientation 을 target 으로 회전
    guard = 0
    while True:
        p = player(env)
        if p is None:
            env.step(NOOP); guard += 1
            if guard > 400: break
            continue
        if int(p.orientation) == target: break
        env.step(LEFT); guard += 1
        if guard > 400: break
    p = player(env)
    if p is None or int(p.orientation) != target:
        print(f"{target:>7}  도달 실패"); continue
    xy0 = np.array(p.xy, float)
    # 화면 랩어라운드를 넘지 않도록 프레임마다 변위를 누적한다
    prev = xy0.copy(); d = np.zeros(2); lost = False
    W, H = 160.0, 210.0
    for k in range(60):
        env.step(UP if k < 40 else NOOP)
        q = player(env)
        if q is None: lost = True; break
        cur = np.array(q.xy, float)
        step = cur - prev
        step[0] -= W*round(step[0]/W); step[1] -= H*round(step[1]/H)   # 랩 보정
        d += step; prev = cur
    if lost:
        print(f"{target:>7}  추진 중 Player 소실"); continue
    xy1 = prev
    # 화면 좌표는 y 가 아래로 증가한다. 수학각은 y 를 뒤집은 것
    scr = np.degrees(np.arctan2(d[1], d[0])) % 360
    mth = np.degrees(np.arctan2(-d[1], d[0])) % 360
    print(f"{target:>7} {str(tuple(xy0)):>16} {str(tuple(xy1)):>16} "
          f"{str((round(d[0],1), round(d[1],1))):>16} {scr:>11.1f} {mth:>11.1f}")
    res[target] = dict(dx=float(d[0]), dy=float(d[1]), screen_deg=float(scr),
                       math_deg=float(mth), moved=float(np.hypot(*d)))

good = {k: v for k, v in res.items() if v["moved"] > 6}
if len(good) >= 8:
    ks = np.array(sorted(good)); ang = np.unwrap(np.radians([good[k]["math_deg"] for k in ks]))
    A = np.c_[ks, np.ones(len(ks))]
    slope, icpt = np.linalg.lstsq(A, np.degrees(ang), rcond=None)[0]
    print(f"\n측정된 관계: 수학각 = {slope:.2f} x orientation + {icpt:.1f}")
    print(f"  -> 한 칸 {slope:.2f}도 (기대 ±22.5), 회전 방향 "
          f"{'반시계(수학 양)' if slope > 0 else '시계'}")
    pred = (slope*ks + icpt)
    err = np.degrees(np.angle(np.exp(1j*np.radians(np.degrees(ang) - pred))))
    print(f"  잔차 최대 {np.abs(err).max():.1f}도 (작으면 선형 관계 확정)")
    res["fit"] = dict(slope=float(slope), intercept=float(icpt),
                      max_resid=float(np.abs(err).max()))
(ROOT/"out"/"probe_heading.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"\n-> out/probe_heading.json")
