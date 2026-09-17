"""발사 쪽에 남은 것 — 천장을 먼저 잰다.

조준을 고치기 전에 **고쳐봐야 얼마나 좋아지는지**를 알아야 한다. 안 그러면
게이트 4 를 다 만들고 나서 "그래서 얼마나 나아졌는데?" 에 답을 못 한다.

  A 누름 -> 발사 전환율. 우리는 매 결정(4프레임) 누르는데 실제로 나가는 건 몇 발인가.
    아타리는 화면에 떠 있는 총알 수에 상한이 있어서, 자주 누른다고 더 나가지 않는다.
  B **완벽 조준 상한.** 가장 가까운 운석 쪽으로 무조건 돌리는 '신탁' 정책을 만들어
    같은 발사 규칙으로 돌린다. 이건 **자 대신 쓰는 도구지 후보 정책이 아니다** —
    부과 규칙이므로 절대 출시하지 않는다.
  C TTMn 이 이 모델에서 실제로 발화하는가 (데모용 격발이 가능한지).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, GreedyPolicy, make_env, Vision, with_fire,
                  frame_action, ACT_EVERY, _takes_vel, ship_heading_deg)

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEEDS = [11, 33, 44]
def say(*a): print(*a, flush=True)
R = {}


class OracleAim:
    """가장 가까운 운석 쪽으로 돌린다. **측정 도구다. 출시 금지.**"""
    def __call__(self, looms, orientation, actions, vel=None):
        ch = {}
        if not looms:
            return actions.index("NOOP"), ch
        L = min(looms, key=lambda d: d["dist"])
        phi = L["phi_rel"]          # 정면 0, 오른쪽 양수
        if abs(phi) <= 11.25:
            return actions.index("NOOP"), ch
        return actions.index("RIGHT" if phi > 0 else "LEFT"), ch


def n_mis(env):
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def run(policy, seed, frames=FRAMES):
    env = ENV; A = ACTS; V = VIS
    rng = np.random.default_rng(seed)
    takes = _takes_vel(policy) if policy is not None else False
    env.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset()
    if hasattr(policy, "dec"): policy.dec.reset()
    if hasattr(policy, "b"): policy.b.reset(); policy.frame = 0
    action = (A.index("NOOP"), A.index("FIRE"))
    prev = 0; shots = 0; hits = 0; score = 0.0
    presses = 0; maxmis = 0; aimed = 0; ndec = 0
    for f in range(frames):
        _, rew, tr, te, _ = env.step(frame_action(action[0], action[1], f))
        score += float(rew)
        if rew > 0: hits += 1
        m = n_mis(env); maxmis = max(maxmis, m)
        if m > prev: shots += (m - prev)
        prev = m
        if tr or te: break
        if f % ACT_EVERY: continue
        presses += 1                      # 결정마다 한 번씩 누른다
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            a0 = A.index("NOOP"); action = (a0, A.index("FIRE")); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        ndec += 1
        if looms:
            ap = np.abs([L["phi_rel"] for L in looms])
            th = np.array([L["theta"] for L in looms])
            if (ap <= th/2.0).any(): aimed += 1
        if policy is None: a = A.index("NOOP")
        elif takes: a, _c = policy(looms, ori, A, vel=V.ship_v)
        else: a, _c = policy(looms, ori, A)
        action = (a, with_fire(a, A))
    return dict(shots=shots, hits=hits, score=score, presses=presses,
                maxmis=maxmis, aimed=aimed/max(ndec, 1))


ENV = make_env(); ACTS = ENV.unwrapped.get_action_meanings(); VIS = Vision()
brain = BrainPolicy(inertia=True); greedy = GreedyPolicy(inertia=True)

say("A/B  누름->발사 전환율과 완벽 조준 상한  (시드 3개 x %d프레임)\n" % FRAMES)
say(f"{'정책':12s} {'누름':>6s} {'발사':>6s} {'전환':>6s} {'명중':>6s} "
    f"{'발사당명중':>10s} {'조준률':>7s} {'점수':>7s} {'동시최대':>8s}")
rows = {}
for name, pol in (("가만히 있기", None), ("규칙 기반", greedy),
                  ("전체 뇌", brain), ("★신탁 조준", OracleAim())):
    rs = [run(pol, s) for s in SEEDS]
    g = lambda k: float(np.mean([r[k] for r in rs]))
    rows[name] = {k: g(k) for k in rs[0]}
    say(f"{name:12s} {g('presses'):6.0f} {g('shots'):6.0f} "
        f"{g('shots')/max(g('presses'),1)*100:5.1f}% {g('hits'):6.0f} "
        f"{g('hits')/max(g('shots'),1)*100:9.1f}% {g('aimed')*100:6.1f}% "
        f"{g('score'):7.0f} {g('maxmis'):8.0f}")
R["AB"] = rows

say("\nC  TTMn 이 이 모델에서 발화하는가 (전체 뇌, 시드 11, 900프레임)")
import torch
b = brain.b
idx = brain.C.get("TTMn", None)
if idx is None:
    keys = [k for k in brain.C.keys()]
    say(f"   circuit_idx 키: {keys}")
else:
    say(f"   TTMn 인덱스 {np.asarray(idx).tolist()}")
tot = {}
ENV.reset(seed=0)
for _ in range(15): ENV.step(ACTS.index("NOOP"))
VIS.reset(); brain.b.reset(); brain.frame = 0; brain.dec.reset()
action = (ACTS.index("NOOP"), ACTS.index("FIRE"))
watch = {k: np.asarray(brain.C[k]).ravel() for k in brain.C
         if k in ("TTMn", "DLMn", "PSI", "DNp01", "DNp02", "DNp11", "DNp04", "MDN")}
acc = {k: 0.0 for k in watch}; nd = 0
for f in range(900):
    _, _, tr, te, _ = ENV.step(frame_action(action[0], action[1], f))
    if tr or te: break
    if f % ACT_EVERY: continue
    xy, head, looms = VIS.looming(ENV.objects)
    if xy is None:
        a0 = ACTS.index("NOOP"); action = (a0, ACTS.index("FIRE")); continue
    ori = 0
    for o in ENV.objects:
        if o and type(o).__name__ == "Player":
            ori = int(getattr(o, "orientation", 0)); break
    a, _c = brain(looms, ori, ACTS, vel=VIS.ship_v)
    action = (a, with_fire(a, ACTS))
    nd += 1
    tal = brain._tal
    for k, ii in watch.items(): acc[k] += float(tal[ii].sum())
for k in watch:
    say(f"   {k:7s} 세포 {watch[k].size:3d}  결정당 평균 스파이크 {acc[k]/max(nd,1):8.2f}")
R["C"] = {k: acc[k]/max(nd, 1) for k in watch}

Path("out").mkdir(exist_ok=True)
json.dump(R, open("out/fire_ceiling.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/fire_ceiling.json")
