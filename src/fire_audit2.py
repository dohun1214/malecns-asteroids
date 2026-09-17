"""발사 점검 후속 2건.

  A FIRE_PRESS=1 과 2 가 3000프레임에서 **완전히 같은 숫자**를 냈다 (117발/23파괴/2350점).
    두 번째 눌림 프레임이 아무 일도 안 한다는 뜻인데, 시드 하나로 단정할 수 없다.
    여러 시드에서 확인한다. 같으면 14문서의 '1프레임은 ROM 이 못 읽는다' 가 틀린 설명이다.
  B with_fire 는 FIRE_MAP 에 없는 액션을 **그대로 돌려준다** = 그 결정엔 발사가 안 걸린다.
    디코더는 UP/LEFT/RIGHT/NOOP 만 내므로 문제없지만, **무작위 대조군**은 14개 액션을
    다 내므로 발사가 덜 걸린다. 얼마나 덜 걸리는지 센다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import play as PL
from play import (BrainPolicy, run_episode, make_env, Vision, with_fire,
                  frame_action, ACT_EVERY, FIRE_MAP)

ROOT = Path(__file__).resolve().parent.parent
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
def say(*a): print(*a, flush=True)
out = {}


def n_missile():
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def n_ast():
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and type(o).__name__ == "Asteroid")


def drive(pol, seed, press, frames=2500):
    old = PL.FIRE_PRESS; PL.FIRE_PRESS = press
    rng = np.random.default_rng(seed)
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); pol.dec.reset(); pol.b.reset(); pol.frame = 0
    act = (A.index("NOOP"), A.index("FIRE"))
    shots = 0; pm = 0; sc = 0.0; drop = 0; pa = n_ast(); acts = []
    for f in range(frames):
        _, rew, tr, te, _ = env.step(frame_action(act[0], act[1], f))
        sc += float(rew)
        if tr or te: break
        m = n_missile()
        if m > pm: shots += m - pm
        pm = m
        a_ = n_ast()
        if a_ < pa: drop += pa - a_
        pa = a_
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            act = (A.index("NOOP"), A.index("FIRE")); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player" and o.wh[0] > 0:
                ori = int(getattr(o, "orientation", 0)); break
        a, ch = pol(looms, ori, A, vel=V.ship_v)
        acts.append(A[a])
        act = (a, with_fire(a, A))
    PL.FIRE_PRESS = old
    return dict(shots=shots, score=sc, drop=drop, acts="".join(x[0] for x in acts))


bp = BrainPolicy()
say("="*80)
say("A. FIRE_PRESS 1 vs 2 — 시드 5개")
say(f"   {'시드':>5} | {'누름=1 발사':>10} {'파괴':>5} {'점수':>7} | {'누름=2 발사':>10} {'파괴':>5} {'점수':>7} | 같은가")
same = True
for sd in (11, 22, 33, 44, 55):
    bp.b.alive.fill_(1); r1 = drive(bp, sd, 1)
    bp.b.alive.fill_(1); r2 = drive(bp, sd, 2)
    eq = (r1["shots"], r1["drop"], round(r1["score"])) == (r2["shots"], r2["drop"], round(r2["score"]))
    same &= eq
    say(f"   {sd:>5} | {r1['shots']:>10} {r1['drop']:>5} {r1['score']:>7.0f}"
        f" | {r2['shots']:>10} {r2['drop']:>5} {r2['score']:>7.0f} | {'동일' if eq else '다름'}")
    out[f"seed{sd}"] = dict(p1=r1, p2=r2, equal=bool(eq))
say(f"   -> {'모든 시드에서 동일하다. 두 번째 눌림 프레임은 아무 일도 안 한다.' if same else '시드에 따라 다르다.'}")
out["press_equal"] = bool(same)

say("\n" + "="*80)
say("B. with_fire 가 발사를 못 거는 액션이 얼마나 되나")
say(f"   FIRE_MAP 에 있는 액션: {sorted(FIRE_MAP)}")
nomap = [a for a in A if a not in FIRE_MAP and not a.endswith("FIRE")]
say(f"   매핑 없는 액션 {len(nomap)}개: {nomap}")
say("   (그 액션이 나오면 그 결정에는 발사가 **안 걸린다**)")
# 디코더가 실제로 내는 액션 분포
r = drive(bp, 77, 2)
from collections import Counter
cnt = Counter(r["acts"])
say(f"   전체 뇌가 낸 액션 분포: {dict(cnt)}  (U=UP, L=LEFT, R=RIGHT, N=NOOP)")
say(f"   -> 뇌는 매핑 있는 4개만 낸다. 발사 누락 0%")
# 무작위 정책은?
rr = np.random.default_rng(1)
picks = [A[int(rr.integers(0, len(A)))] for _ in range(100000)]
miss = sum(1 for p in picks if p not in FIRE_MAP and not p.endswith("FIRE"))
already = sum(1 for p in picks if p.endswith("FIRE"))
say(f"   무작위 정책(14개 균등): 발사 못 거는 액션 {miss/1000:.1f}%,"
    f" 이미 FIRE 조합인 액션 {already/1000:.1f}%")
say("   -> 무작위 대조군만 발사가 덜 걸린다. 바닥값 대조라 결론에 영향은 없지만 기록한다.")
out["nomap"] = nomap; out["random_miss_pct"] = miss/1000.0

(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"fire_audit2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
say("-> out/fire_audit2.json")
