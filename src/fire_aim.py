"""총이 운석을 겨누고 있는가, 아니면 그냥 정면으로 나가는가.

발사 규칙 자체에는 조준이 없다 (매 결정 앞 2프레임 무조건 FIRE). 그래도 총알은
**배가 향한 방향**으로 나가고, 배 방향은 뇌가 정한다. 그러므로 물어야 할 것은
'무작위인가'가 아니라 **'배 방향이 운석 쪽으로 쏠려 있는가'** 다.

측정 (발사가 걸리는 매 결정마다):
  1 총구선에 가장 가까운 운석의 각거리 min|phi_rel|
  2 총구선이 운석의 각폭 안에 들어가는 비율  (|phi| <= theta/2)  = '지금 쏘면 맞는다'
  3 ±15도 안에 운석이 있는 비율
  4 가장 가까운(거리) 운석의 각거리        -> 회피 중이면 180도 근처여야 한다
대조: **같은 기하에서 배 방향만 균등 무작위로 돌린 것** (운석 배치가 균등하지 않으므로
      이 대조가 없으면 '겨눈다'를 말할 수 없다). 결정마다 32회 재추출.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, GreedyPolicy, make_env, Vision, with_fire,
                  frame_action, ACT_EVERY, _takes_vel)

NRAND = 32
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEEDS = [11, 33, 44]


def n_missile(env):
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def measure(env, policy, V, A, seed, frames=FRAMES):
    rng = np.random.default_rng(seed)
    takes_vel = _takes_vel(policy) if policy is not None else False
    env.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))):
        env.step(A.index("NOOP"))
    V.reset()
    if hasattr(policy, "dec"): policy.dec.reset()
    if hasattr(policy, "b"): policy.b.reset(); policy.frame = 0
    action = (A.index("NOOP"), A.index("FIRE"))
    prev_mis = 0; shots = 0; hits = 0; score = 0.0
    real, ctrl = [], []          # (min|phi|, 명중가능, ±15도, 최근접운석 각거리)
    for f in range(frames):
        _, rew, tr, te, _ = env.step(frame_action(action[0], action[1], f))
        score += float(rew)
        if rew > 0: hits += 1
        m = n_missile(env)
        if m > prev_mis: shots += (m - prev_mis)
        prev_mis = m
        if tr or te: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            a0 = A.index("NOOP"); action = (a0, A.index("FIRE")); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        if policy is None:
            a = A.index("NOOP"); ch = {}
        elif takes_vel:
            a, ch = policy(looms, ori, A, vel=V.ship_v)
        else:
            a, ch = policy(looms, ori, A)
        action = (a, with_fire(a, A))
        if looms:
            phi = np.array([L["phi_rel"] for L in looms])
            th = np.array([L["theta"] for L in looms])
            d = np.array([L["dist"] for L in looms])
            near_i = int(np.argmin(d))

            def stat(off):
                p = (phi - off + 180.0) % 360.0 - 180.0
                ap = np.abs(p)
                return (float(ap.min()), float((ap <= th/2.0).any()),
                        float((ap <= 15.0).any()), float(abs(p[near_i])))
            real.append(stat(0.0))
            for off in rng.uniform(-180, 180, NRAND):
                ctrl.append(stat(off))
    real = np.array(real); ctrl = np.array(ctrl)
    return dict(seed=seed, n=len(real), shots=shots, hits=hits, score=score,
                real_min=float(np.median(real[:, 0])),
                ctrl_min=float(np.median(ctrl[:, 0])),
                real_hit=float(real[:, 1].mean()), ctrl_hit=float(ctrl[:, 1].mean()),
                real_15=float(real[:, 2].mean()), ctrl_15=float(ctrl[:, 2].mean()),
                real_near=float(np.median(real[:, 3])),
                ctrl_near=float(np.median(ctrl[:, 3])))


def agg(rows, k):
    return float(np.mean([r[k] for r in rows]))


env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
out = {}
brain = BrainPolicy(inertia=True)
greedy = GreedyPolicy(inertia=True)
for name, pol in (("전체 뇌", brain), ("규칙 기반", greedy), ("가만히 있기", None)):
    rows = [measure(env, pol, V, A, s) for s in SEEDS]
    out[name] = rows
    print(f"\n[{name}]  결정 {agg(rows,'n'):.0f} · 발사 {agg(rows,'shots'):.0f} · "
          f"명중 {agg(rows,'hits'):.0f} · 점수 {agg(rows,'score'):.0f}", flush=True)
    print(f"  총구선에 가장 가까운 운석 각거리(중앙)  실제 {agg(rows,'real_min'):6.1f}도"
          f"   |  무작위 방향 {agg(rows,'ctrl_min'):6.1f}도")
    print(f"  '지금 쏘면 맞는다'(각폭 안)            실제 {agg(rows,'real_hit')*100:6.1f}%"
          f"   |  무작위 방향 {agg(rows,'ctrl_hit')*100:6.1f}%")
    print(f"  ±15도 안에 운석                        실제 {agg(rows,'real_15')*100:6.1f}%"
          f"   |  무작위 방향 {agg(rows,'ctrl_15')*100:6.1f}%")
    print(f"  가장 가까운 운석의 각거리(중앙)        실제 {agg(rows,'real_near'):6.1f}도"
          f"   |  무작위 방향 {agg(rows,'ctrl_near'):6.1f}도")
    print(f"  발사당 명중률  {agg(rows,'hits')/max(agg(rows,'shots'),1)*100:.1f}%")

Path(__file__).resolve().parent.parent.joinpath("out").mkdir(exist_ok=True)
json.dump(out, open(Path(__file__).resolve().parent.parent/"out"/"fire_aim.json", "w"),
          ensure_ascii=False, indent=1)
print("\n저장: out/fire_aim.json", flush=True)
