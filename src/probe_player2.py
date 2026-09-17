"""정지 상태에서는 Player 가 항상 1개·화면 안이었다. 그런데 실제 플레이에서는
y 가 520~528 인 표본이 18.86% 나왔다. **죽음/부활 동안 스프라이트를 화면 밖에
주차해 두는 것**으로 보인다. 그렇다면 `Vision.parse` 는 그걸 배로 받아들인다.

실제 플레이(발사 O, 죽음 O)에서 `parse` 가 고른 배의 y 를 직접 센다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, frame_action, ACT_EVERY, with_fire

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
rng = np.random.default_rng(3)
env.reset(seed=0)
for _ in range(17): env.step(A.index("NOOP"))

tot = bad = 0; ys = []
lives_prev = None; bad_near_death = 0; since_death = 10**9
act = (A.index("NOOP"), A.index("FIRE"))
for f in range(FRAMES):
    _, _, tr, te, info = env.step(frame_action(act[0], act[1], f))
    if tr or te:
        env.reset(seed=0); V.reset(); lives_prev = None; continue
    lv = info.get("lives") if isinstance(info, dict) else None
    if lives_prev is not None and lv is not None and lv < lives_prev: since_death = 0
    else: since_death += 1
    lives_prev = lv
    ship, _ = V.parse(env.objects)
    if ship is not None:
        tot += 1; y = float(ship.xy[1]); ys.append(y)
        if y > 210:
            bad += 1
            if since_death < 200: bad_near_death += 1
    if f % ACT_EVERY: continue
    a = int(rng.integers(0, 5))
    base = [A.index(x) for x in ("NOOP", "LEFT", "RIGHT", "UP", "NOOP")][a]
    act = (base, with_fire(base, A))

ys = np.array(ys)
say(f"parse 가 배를 반환한 프레임 {tot:,}")
say(f"  그중 y > 210 (화면 밖) : {bad:,}  (**{bad/max(tot,1)*100:.1f}%**)")
say(f"     그중 죽은 직후 200프레임 안 : {bad_near_death:,} "
    f"({bad_near_death/max(bad,1)*100:.1f}%)")
say(f"  y 분위수  1% {np.percentile(ys,1):.0f}  50% {np.percentile(ys,50):.0f}"
    f"  90% {np.percentile(ys,90):.0f}  99% {np.percentile(ys,99):.0f}  max {ys.max():.0f}")
say(f"  화면 밖 y 의 고유값: {sorted(set(int(v) for v in ys[ys>210]))[:12]}")
json.dump(dict(tot=tot, bad=bad, frac=bad/max(tot,1)),
          open("out/probe_player2.json", "w"), ensure_ascii=False, indent=1)
say("저장: out/probe_player2.json")
