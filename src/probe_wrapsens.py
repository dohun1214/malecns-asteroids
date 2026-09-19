"""WRAP_Y 의 잔여 불확실성이 **결과를 바꾸는가.**

주기를 정확히 못 박았다. 시도 5개가 각각 다른 값을 냈다:
  #7 89.3 (자가시험 실패)  #8 18~34 (자가시험 실패)  #9 177 (픽셀 행, 자가시험 통과)
  #10 168~178 (좌표 범위, 자가시험 실패 — 스프라이트가 이음매를 걸치면 범위가 주기보다 크다)
  #11 158 (운석 감싸기 사건 10/11 일치. 단 이음매 근처에서 검출이 끊겨 **과소평가**다)

확정된 하한: 운석 좌표 범위 24~192 = **169 이상**. 상한: 배 좌표 범위 18~194 = **177 이하**
(배 x 가 164 인데 정답이 160 이었으므로 범위는 주기를 **과대**평가한다).
=> **169 ~ 177.** 지금 코드는 177.

정확한 값을 더 쫓는 대신 **그 범위 안에서 결과가 흔들리는지**를 잰다.
안 흔들리면 잔여 불확실성은 결론에 영향이 없다. 흔들리면 그때 더 판다.
비교용으로 **옛 값 210** 도 같이 돌린다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision as VS
from play import (BrainPolicy, GreedyPolicy, make_env, Vision, with_fire,
                  frame_action, ACT_EVERY, _takes_vel)

FRAMES = 3000
SEEDS = [11, 33, 44]
CANDS = [168, 172, 177, 182, 210]
def say(*a): print(*a, flush=True)
ENV = make_env(); A = ENV.unwrapped.get_action_meanings(); V = Vision()
pol = BrainPolicy(inertia=True)


def n_mis(env):
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def run(seed):
    rng = np.random.default_rng(seed)
    ENV.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): ENV.step(A.index("NOOP"))
    V.reset(); pol.b.reset(); pol.frame = 0; pol.dec.reset()
    act = (A.index("NOOP"), A.index("FIRE"))
    prev = shots = hits = 0; score = 0.0
    near = []; nL = nR = nU = ndec = 0
    for f in range(FRAMES):
        _, rew, tr, te, _ = ENV.step(frame_action(act[0], act[1], f))
        score += float(rew)
        if rew > 0: hits += 1
        m = n_mis(ENV)
        if m > prev: shots += (m - prev)
        prev = m
        if tr or te: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(ENV.objects)
        if xy is None:
            a0 = A.index("NOOP"); act = (a0, A.index("FIRE")); continue
        ori = 0
        for o in ENV.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        ndec += 1
        if looms:
            phi = np.array([L["phi_rel"] for L in looms])
            d = np.array([L["dist"] for L in looms])
            near.append(abs(float(phi[int(np.argmin(d))])))
        a, _c = pol(looms, ori, A, vel=V.ship_v)
        nm = A[a]
        nL += nm == "LEFT"; nR += nm == "RIGHT"; nU += nm == "UP"
        act = (a, with_fire(a, A))
    return dict(score=score, hits=hits, shots=shots,
                near=float(np.median(near)) if near else float("nan"),
                bias=(nL-nR)/max(nL+nR, 1), thrust=nU/max(ndec, 1))


R = {}
say(f"{'WRAP_Y':>7} {'점수':>7} {'발사당명중':>10} {'최근접각':>9} {'좌회전편향':>10} {'추진':>7}")
say("-"*58)
for P in CANDS:
    VS.WRAP_Y = float(P)
    rs = [run(s) for s in SEEDS]
    g = lambda k: float(np.mean([r[k] for r in rs]))
    R[P] = {k: g(k) for k in rs[0]}
    say(f"{P:7d} {g('score'):7.0f} {g('hits')/max(g('shots'),1)*100:9.1f}% "
        f"{g('near'):8.1f}도 {g('bias')*100:+9.1f}% {g('thrust')*100:6.1f}%")
VS.WRAP_Y = 177.0

lo = [R[p] for p in (168, 172, 177)]
say("\n=== 판정 ===")
for k, nm in (("score", "점수"), ("near", "최근접각"), ("bias", "좌회전 편향")):
    v = [x[k] for x in lo]
    say(f"  169~177 범위 안 {nm:10s} {min(v):8.1f} ~ {max(v):8.1f}"
        f"   (폭 {max(v)-min(v):.1f})   |  옛 값 210 에서는 {R[210][k]:.1f}")
json.dump({str(k): v for k, v in R.items()},
          open("out/probe_wrapsens.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/probe_wrapsens.json")
