"""🔴 Player 의 y 가 18.86% 의 프레임에서 520~528 이다. 화면 높이는 210 이다.

`Vision.parse` 는 `for o in objs: ... if name == "Player": ship = o` 로 **마지막** Player 를
쓴다. 프레임에 Player 가 여러 개면 뒤엣것이 이긴다. 목숨 표시 아이콘 같은 게
Player 로 잡히고 있으면 **배 위치가 통째로 틀린다.**

프레임당 Player 개수와 각각의 좌표·크기를 센다.
"""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, frame_action, ACT_EVERY, with_fire

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
env.reset(seed=0)
rng = np.random.default_rng(11)
for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))

cnt = collections.Counter()
groups = collections.Counter()
multi = []
act = (A.index("NOOP"), A.index("FIRE"))
for f in range(FRAMES):
    _, _, tr, te, info = env.step(frame_action(act[0], act[1], f))
    if tr or te: break
    ps = [o for o in env.objects
          if o is not None and type(o).__name__ == "Player"
          and o.wh[0] > 0 and o.wh[1] > 0]
    cnt[len(ps)] += 1
    for o in ps:
        y = float(o.xy[1])
        groups[("화면 안" if y <= 210 else "화면 밖", tuple(o.wh))] += 1
    if len(ps) > 1 and len(multi) < 6:
        multi.append([(float(o.xy[0]), float(o.xy[1]), tuple(o.wh)) for o in ps]
                     + [info.get("lives")])
    if f % ACT_EVERY: continue
    act = (A.index("NOOP"), A.index("FIRE"))

say(f"프레임당 Player 개수 분포: {dict(sorted(cnt.items()))}")
say("\n(위치, 크기)별 개수")
for k, v in groups.most_common(10):
    say(f"   {k[0]:6s} wh={k[1]}  {v:,}")
say("\nPlayer 가 2개 이상인 프레임 표본 (마지막 항목은 lives)")
for m in multi: say(f"   {m}")

# parse 가 실제로 무엇을 고르는가
env.reset(seed=0)
for _ in range(11): env.step(A.index("NOOP"))
bad = tot = 0
for f in range(FRAMES):
    env.step(A.index("NOOP"))
    ship, _ = V.parse(env.objects)
    if ship is None: continue
    tot += 1
    if float(ship.xy[1]) > 210: bad += 1
say(f"\n`Vision.parse` 가 고른 배의 y 가 210 을 넘는 비율: {bad}/{tot} "
    f"({bad/max(tot,1)*100:.1f}%)")
json.dump(dict(counts={str(k): v for k, v in cnt.items()},
               bad=bad, tot=tot), open("out/probe_player.json", "w"),
          ensure_ascii=False, indent=1)
say("저장: out/probe_player.json")
