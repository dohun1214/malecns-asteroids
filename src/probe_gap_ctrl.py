"""❓ [이슈 #65] '가려는 방향 vs 실제 가는 방향' 오차가 90도를 넘는다 (뇌 103도, 규칙 125도).
이 지표가 제어를 재는지, 아니면 기하가 만드는 값인지 대조군으로 본다.

'가려는 방향'(회피 방위)은 팽창률 가중이라 **배가 향해 가는 쪽의 운석**이 가장 크게 잡힌다
(상대 속도가 크니까). 그러면 배가 움직이기만 해도 회피 방위는 이동 방향의 반대로 쏠린다.
→ 조종과 무관하게 **무작위로 움직이는 배**에서도 같은 지표를 잰다.
   무작위도 90도를 넘으면 지표가 기하에 오염된 것이다.
'가려는 방향'은 규칙 정책의 채널(뇌 없음)로 잰다. 액션만 바꾼다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, GreedyPolicy, frame_action, ACT_EVERY, with_fire
from vision import player_xy, ship_heading_deg, ASPECT, WRAP_Y, WRAP_X

N_EP, MAXF = 8, 4000
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings()
def w(d, p): return d - p*round(d/p)

def run(mode, seed0=7):
    rng = np.random.default_rng(seed0); arng = np.random.default_rng(99)
    pol = GreedyPolicy(); V = Vision(); gaps = []; ups = 0; n = 0
    for _ in range(N_EP):
        env.reset(seed=int(rng.integers(0, 2**31)))
        for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
        V.reset(); action = (A.index("NOOP"), A.index("FIRE")); prev = None
        for f in range(MAXF):
            _, _, tr, te, _ = env.step(frame_action(action[0], action[1], f))
            if tr or te: break
            if f % ACT_EVERY: continue
            xy, head, looms = V.looming(env.objects)
            if xy is None:
                prev = None; action = (A.index("NOOP"), A.index("FIRE")); continue
            ori = 0
            for o in env.objects:
                if o and type(o).__name__ == "Player" and o.wh[0] > 0:
                    ori = int(getattr(o, "orientation", 0)); break
            a, ch = pol(looms, ori, A, vel=V.ship_v)
            if mode == "random":
                a = A.index(arng.choice(["UP", "LEFT", "RIGHT", "NOOP"], p=[0.25, 0.3, 0.3, 0.15]))
            if ch.get("norm", 0) > 1e-9 and prev is not None:
                psi = np.degrees(np.arctan2(ch["lateral"], ch["fore"])) + 180.0
                want = (ship_heading_deg(ori) - psi) % 360.0
                dx = w(xy[0]-prev[0], WRAP_X); dy = -w(xy[1]-prev[1], WRAP_Y)/ASPECT
                if np.hypot(dx, dy) > 0.3:
                    go = np.degrees(np.arctan2(dy, dx)) % 360.0
                    gaps.append(abs((want - go + 180.0) % 360.0 - 180.0))
            prev = xy; n += 1; ups += A[a] == "UP"
            action = (a, with_fire(a, A))
    g = np.asarray(gaps)
    return dict(n=len(g), median=float(np.median(g)), mean=float(g.mean()),
                over90=float((g > 90).mean()), up=ups/max(n, 1))

out = {m: run(m) for m in ["rule", "random"]}
for k, v in out.items():
    say(f"{k:7s} n={v['n']:5d}  중앙 {v['median']:6.1f}  평균 {v['mean']:6.1f}  >90 {v['over90']*100:5.1f}%  추진 {v['up']*100:.1f}%")
json.dump(out, open("out/probe_gap_ctrl.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
