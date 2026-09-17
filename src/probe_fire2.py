"""발사가 화면에 **아무 영향도 없는가**를 결정적으로 본다.
같은 회전 시퀀스에서 FIRE 비트만 켜고/끄고 돌려서 화면을 바이트 단위로 비교한다.
전부 같으면 FIRE 가 무시되고 있는 것이다.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env

env = make_env(); A = env.unwrapped.get_action_meanings()
ale = env._env.env.env.ale
FIREOF = {"NOOP":"FIRE","LEFT":"LEFTFIRE","RIGHT":"RIGHTFIRE","UP":"UPFIRE"}


def run(fire_mode, n=900, seed=11):
    env.reset(seed=seed)
    for _ in range(40): env.step(A.index("NOOP"))
    rng = np.random.default_rng(0)
    frames = []; score = 0.0; rews = []
    for f in range(n):
        rot = rng.choice(["NOOP", "LEFT", "RIGHT", "UP"], p=[.55, .15, .15, .15])
        if fire_mode == "none": a = rot
        elif fire_mode == "hold": a = FIREOF[rot]
        elif fire_mode == "pulse8": a = FIREOF[rot] if f % 8 == 0 else rot
        o, r, tr, te, info = env.step(A.index(a))
        score += float(r)
        if r: rews.append((f, float(r)))
        if f % 30 == 0: frames.append(np.asarray(ale.getScreenRGB(), dtype=np.uint8).copy())
        if tr or te: break
    return frames, score, rews


base, s0, r0 = run("none")
for m in ("hold", "pulse8"):
    fr, s, rr = run(m)
    n = min(len(base), len(fr))
    diff = [i for i in range(n) if not np.array_equal(base[i], fr[i])]
    print(f"{m:<8} 점수 {s:6.0f} (발사없음 {s0:.0f})  "
          f"화면 스냅샷 {n}장 중 다른 것 {len(diff)}장  첫 차이 {diff[0] if diff else '없음'}",
          flush=True)
print(f"\n발사없음 보상 이벤트 {len(r0)}개: {r0[:8]}", flush=True)

# 진짜로 시작된 게임인가 — lives / 게임오버가 도는지
env.reset(seed=11)
lv = []
for f in range(2400):
    o, r, tr, te, info = env.step(A.index("FIRE"))
    lv.append(info.get("lives"))
    if tr or te:
        print(f"\n에피소드 종료 @ {f}프레임, lives 시퀀스 앞 {lv[:5]} 끝 {lv[-5:]}", flush=True)
        break
else:
    print(f"\n2400프레임 안 끝남. lives 고유값 {sorted(set(lv))}, "
          f"처음 {lv[0]} -> 마지막 {lv[-1]}", flush=True)

# 액션별로 화면이 바뀌는지 (조작이 먹긴 하는가)
env.reset(seed=11)
for _ in range(60): env.step(A.index("NOOP"))
st = ale.cloneSystemState()
outs = {}
for a in ("NOOP", "LEFT", "RIGHT", "UP", "FIRE"):
    ale.restoreSystemState(st)
    for _ in range(20): env.step(A.index(a))
    outs[a] = np.asarray(ale.getScreenRGB(), dtype=np.uint8).copy()
print("\n같은 지점에서 액션만 바꿔 20프레임 후 화면 비교 (NOOP 기준 차이 픽셀 수)", flush=True)
for a, im in outs.items():
    print(f"  {a:<6} {int((im != outs['NOOP']).any(axis=2).sum()):>6}", flush=True)
