"""발사가 제대로 되고 있는지 **정밀** 점검 (사용자 요청).

14문서에서 "한 발도 안 나가던" 버그를 고쳤지만, 고친 뒤의 발사가 **최적인지**,
그리고 **정책 간에 정말 동일하게** 걸리는지는 안 봤다. 여기서 본다.

  1 실제 액션 스트림이 의도한 눌림 패턴인가 (프레임 단위로 찍어서 확인)
  2 총알이 실제로 몇 발 나가는가 (객체 등장 이벤트로 센다)
  3 FIRE_PRESS 를 바꾸면 어떻게 되나 — 지금 값(2)이 맞는 선택인가
  4 회전/추진이 발사 때문에 끊기지 않는가 (FIRE_MAP 이 조합 액션으로 가는가)
  5 모든 정책에 **동일하게** 걸리는가
  6 총알이 뇌의 입력을 오염시키지 않는가 (Vision 이 총알을 운석으로 안 보는가)
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import play as PL
from play import (BrainPolicy, GreedyPolicy, run_episode, make_env, Vision,
                  with_fire, frame_action, ACT_EVERY, FIRE_MAP)

ROOT = Path(__file__).resolve().parent.parent
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
def say(*a): print(*a, flush=True)
out = {}

# ── 1. 액션 매핑 ───────────────────────────────────────────────────────────
say("="*86)
say("1. 발사 조합 액션이 실제로 존재하는가 (회전·추진이 끊기면 안 된다)")
say(f"   액션 공간 {len(A)}개: {A}")
bad = []
for base, fired in FIRE_MAP.items():
    okb, okf = base in A, fired in A
    if not (okb and okf): bad.append((base, fired))
    say(f"   {base:<5} -> {fired:<10} {'있음' if okf else '🔴 없음'}"
        f"   (with_fire 결과 = {A[with_fire(A.index(base), A)] if okb else '?'})")
say(f"   -> {'전부 존재' if not bad else '🔴 빠진 매핑: '+str(bad)}")
# 디코더가 낼 수 있는 액션이 전부 매핑에 있는가
dec_acts = {"UP", "LEFT", "RIGHT", "NOOP"}
miss = dec_acts - set(FIRE_MAP)
say(f"   디코더가 내는 액션 {sorted(dec_acts)} 중 매핑 없는 것: "
    f"{sorted(miss) if miss else '없음'}")
out["map_ok"] = (not bad) and (not miss)

# ── 2. 프레임 단위 눌림 패턴 ───────────────────────────────────────────────
say("\n" + "="*86)
say("2. 실제 프레임별 눌림 패턴 — 의도는 '2프레임 누르고 2프레임 뗀다'")
base_a, fire_a = A.index("LEFT"), A.index("LEFTFIRE")
pat = [A[frame_action(base_a, fire_a, f)] for f in range(16)]
say("   " + " ".join(f"{f:>2}" for f in range(16)))
say("   " + " ".join(("발" if p.endswith("FIRE") else " ·") for p in pat))
press = [p.endswith("FIRE") for p in pat]
runs, cur = [], 1
for i in range(1, len(press)):
    if press[i] == press[i-1]: cur += 1
    else: runs.append((press[i-1], cur)); cur = 1
runs.append((press[-1], cur))
say(f"   연속 구간: " + " ".join(f"{'누름' if b else '뗌'}{n}" for b, n in runs))
say(f"   -> 주기 {ACT_EVERY}프레임 중 {PL.FIRE_PRESS}프레임 누름."
    f" **떼는 구간이 있어야 재발사가 된다** (14문서)")
out["pattern"] = ["FIRE" if p else "-" for p in press]

# ── 3~5. 실제 주행 ─────────────────────────────────────────────────────────
def n_missile():
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and "Missile" in type(o).__name__)


def n_ast():
    return sum(1 for o in env.objects if o and o.wh[0] > 0
               and type(o).__name__ == "Asteroid")


def drive(policy, frames=3000, seed=31337, fire=True, press=None):
    """총알 '등장 이벤트' 를 센다. 화면에 있는 개수가 아니라 새로 나간 발수다."""
    old = PL.FIRE_PRESS
    if press is not None: PL.FIRE_PRESS = press
    rng = np.random.default_rng(seed)
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset()
    if hasattr(policy, "dec"): policy.dec.reset()
    if hasattr(policy, "b"): policy.b.reset(); policy.frame = 0
    act = (A.index("NOOP"), A.index("FIRE")) if fire else (A.index("NOOP"),)*2
    shots = 0; prev_m = 0; score = 0.0; ast_drop = 0; prev_a = n_ast()
    seen_ast_max = prev_a; waves = 0
    takes = PL._takes_vel(policy)
    for f in range(frames):
        _, rew, tr, te, info = env.step(frame_action(act[0], act[1], f))
        score += float(rew)
        if tr or te: break
        m = n_missile()
        if m > prev_m: shots += (m - prev_m)       # 새로 생긴 총알 = 발사
        prev_m = m
        a_ = n_ast()
        if a_ < prev_a: ast_drop += (prev_a - a_)
        if a_ - prev_a >= 2: waves += 1
        prev_a = a_
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            act = ((A.index("NOOP"), A.index("FIRE")) if fire
                   else (A.index("NOOP"),)*2); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player" and o.wh[0] > 0:
                ori = int(getattr(o, "orientation", 0)); break
        a, ch = (policy(looms, ori, A, vel=V.ship_v) if takes else policy(looms, ori, A))
        act = (a, with_fire(a, A)) if fire else (a, a)
    PL.FIRE_PRESS = old
    return dict(shots=shots, score=score, ast_drop=ast_drop, waves=waves, frames=f+1)


bp = BrainPolicy()
say("\n" + "="*86)
say("3. FIRE_PRESS 를 바꾸면 — 지금 값 2 가 맞는 선택인가 (전체 뇌, 3000프레임)")
say(f"   {'누름':>4} {'발사':>6} {'운석파괴':>9} {'웨이브':>7} {'점수':>7}")
best = None
for press in (0, 1, 2, 3, 4):
    r = drive(bp, press=press)
    say(f"   {press:>4} {r['shots']:>6} {r['ast_drop']:>9} {r['waves']:>7} {r['score']:>7.0f}")
    out[f"press{press}"] = r
    if best is None or r["score"] > best[1]: best = (press, r["score"])
say(f"   -> 점수 최고는 누름={best[0]} (지금 값 {PL.FIRE_PRESS})")
say("   누름=0 은 발사 없음, 누름=4 는 '계속 누름' = 14문서의 그 버그 상태다")

say("\n" + "="*86)
say("4. 모든 정책에 동일하게 걸리는가 (같은 시드, 발사 켬/끔)")
say(f"   {'정책':<14} {'발사':>6} {'운석파괴':>9} {'점수':>7} | {'발사끔 점수':>11}")
rr = np.random.default_rng(3)
POLS = [("가만히 있기", lambda l, o, a: (A.index("NOOP"), {"norm": 0.0})),
        ("무작위", lambda l, o, a: (int(rr.integers(0, len(A))), {"norm": 0.0})),
        ("규칙 기반", GreedyPolicy()), ("전체 뇌", bp)]
for name, pol in POLS:
    if hasattr(pol, "b"): pol.b.alive.fill_(1)
    on = drive(pol, frames=2000)
    off = drive(pol, frames=2000, fire=False)
    say(f"   {name:<14} {on['shots']:>6} {on['ast_drop']:>9} {on['score']:>7.0f}"
        f" | {off['score']:>11.0f}")
    out[f"pol_{name}"] = dict(on=on, off=off)

# ── 6. 총알이 뇌 입력을 오염시키는가 ───────────────────────────────────────
say("\n" + "="*86)
say("6. 총알이 뇌의 입력으로 새어 들어가는가 (Vision 이 운석으로 세면 안 된다)")
env.reset(seed=7)
for _ in range(60): env.step(A.index("FIRE") if _ % 4 < 2 else A.index("NOOP"))
objs = env.objects
kinds = {}
for o in objs:
    if o and o.wh[0] > 0: kinds[type(o).__name__] = kinds.get(type(o).__name__, 0)+1
ship, asts = V.parse(objs)
say(f"   화면 객체: {kinds}")
say(f"   Vision.parse 가 운석으로 센 것 {len(asts)}개"
    f"  (Asteroid {kinds.get('Asteroid', 0)}개와 일치해야 한다)")
leak = len(asts) != kinds.get("Asteroid", 0)
say(f"   -> {'🔴 총알이 새어 들어간다' if leak else '새지 않는다'}")
out["leak"] = bool(leak)

say("\n" + "="*86)
say("판정")
p2 = out["press2"]; p4 = out["press4"]; p0 = out["press0"]
say(f"  매핑 {'정상' if out['map_ok'] else '🔴 결함'}"
    f"  / 입력 오염 {'없음' if not out['leak'] else '🔴 있음'}")
say(f"  발사 {p2['shots']}발 (누름=0 일 때 {p0['shots']}발, 누름=4 일 때 {p4['shots']}발)")
say(f"  운석 파괴 {p2['ast_drop']} (누름=4 일 때 {p4['ast_drop']})")
if p2["shots"] > 20 and p2["ast_drop"] > p4["ast_drop"]:
    say("  -> 발사가 실제로 나가고, 계속 누르는 상태(옛 버그)보다 확실히 낫다.")
else:
    say("  -> 🔴 기대와 다르다. 숫자를 다시 볼 것.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"fire_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
say("-> out/fire_audit.json")
