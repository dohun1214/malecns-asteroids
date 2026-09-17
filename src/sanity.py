"""'아무 일도 안 일어나는 것'을 잡는 자동 점검 (이슈 #33).

같은 계열 함정에 네 번 걸렸다. 전부 "결정당 4프레임"과 아타리 프레임 단위가 어긋나서
생겼고, 전부 **에러 없이 그럴듯한 숫자**를 냈다. 마지막 하나(발사 0발)는 지표가 아니라
화면을 본 사람이 찾았다. 헤드리스 숫자는 "아무 일도 안 일어남"을 못 본다.

그래서 여기서는 결과가 좋은지가 아니라 **작동이 실제로 일어나는지**를 임계값으로 검사한다.
큰 변경 뒤에 한 번씩 돌린다. 하나라도 FAIL 이면 종료 코드 1.

  python src/sanity.py [프레임수]
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, make_env, Vision, with_fire, frame_action,
                  ACT_EVERY)
from vision import ship_heading_deg

ROOT = Path(__file__).resolve().parent.parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MAXF = int(ARGS[0]) if ARGS else 3000

# 자가시험 — 점검이 **실제로 그 버그를 잡는지** 확인한다.
#   --selftest 는 발사를 일부러 옛날 방식(결정 4프레임 내내 누름)으로 되돌린다.
#   그 상태에서 '발사가 실제로 나간다' 가 통과하면 이 점검은 쓸모없는 것이다.
SELFTEST = "--selftest" in sys.argv
if SELFTEST:
    _orig = frame_action
    def frame_action(base, fire, k, act_every=ACT_EVERY):   # noqa: F811
        return fire                                          # 계속 누르고 있기 = 옛 버그
def say(*a): print(*a, flush=True)

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
ale = env._env.env.env.ale
pol = BrainPolicy()

rng = np.random.default_rng(20260917)
env.reset(seed=int(rng.integers(0, 2**31)))
for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
V.reset(); pol.dec.reset(); pol.b.reset(); pol.frame = 0

def n_ast():
    return sum(1 for o in env.objects
               if o and type(o).__name__ == "Asteroid" and o.wh[0] > 0)

def bg_pixels(img):
    """배경이 아닌 픽셀 수. 아타리 배경은 단색이라 최빈색을 배경으로 본다."""
    q = img.reshape(-1, 3)
    col, cnt = np.unique(q, axis=0, return_counts=True)
    return int(len(q) - cnt.max())

action = (A.index("NOOP"), A.index("FIRE"))
names = {}                       # 관측된 객체 타입 이름
ast_series, ship_seen, lit, lit2 = [], [], [], []
v_nonzero, v_total = 0, 0
n_fire_obj = 0
acts = {"UP": 0, "LEFT": 0, "RIGHT": 0, "NOOP": 0, "기타": 0}
spk, chan, nloom = [], [], []
prev = None
score = 0.0; frames_done = 0
t0 = time.perf_counter()

for f in range(MAXF):
    obs, rew, tr, te, info = env.step(frame_action(action[0], action[1], f))
    score += float(rew); frames_done += 1
    if tr or te: break
    for o in env.objects:
        if o and getattr(o, "wh", (0, 0))[0] > 0:
            names[type(o).__name__] = names.get(type(o).__name__, 0) + 1
    ast_series.append(n_ast())
    # 총알이 화면에 있는 **프레임 수**를 센다 (객체 수가 아니라).
    if any(o and o.wh[0] > 0 and ("Missile" in type(o).__name__
                                  or "Bullet" in type(o).__name__)
           for o in env.objects):
        n_fire_obj += 1
    img = np.asarray(ale.getScreenRGB(), dtype=np.uint8)
    lit.append(bg_pixels(img))
    lit2.append(bg_pixels(np.maximum(img, prev)) if prev is not None else bg_pixels(img))
    prev = img

    if f % ACT_EVERY: continue
    xy, head, looms = V.looming(env.objects)
    ship_seen.append(xy is not None)
    for L in looms:
        v_total += 1
        if abs(L.get("dtheta", 0.0)) > 1e-12: v_nonzero += 1
    if xy is None:
        action = (A.index("NOOP"), A.index("FIRE")); continue
    ori = 0
    for o in env.objects:
        if o and type(o).__name__ == "Player" and o.wh[0] > 0:
            ori = int(getattr(o, "orientation", 0)); break
    a, ch = pol(looms, ori, A, vel=V.ship_v)
    spk.append(int(pol._tal.sum())); nloom.append(len(looms))
    chan.append([float(ch["lateral"]), float(ch["fore"]), float(ch["norm"])])
    acts[A[a] if A[a] in acts else "기타"] += 1
    action = (a, with_fire(a, A))

dt = time.perf_counter() - t0
ast = np.asarray(ast_series); chan = np.asarray(chan) if chan else np.zeros((1, 3))
dec = sum(acts.values())
drops = int((np.diff(ast) < 0).sum())
rises = int((np.diff(ast) >= 2).sum())
vis = float(np.mean(ship_seen)) if ship_seen else 0.0
turn = acts["LEFT"] + acts["RIGHT"]

say(f"{MAXF}프레임 / 결정 {dec}회 / {dt:.0f}s / 점수 {score:.0f}"
    + ("   ⚠️ 자가시험 모드: 발사를 일부러 망가뜨렸다" if SELFTEST else ""))
say(f"관측된 객체 타입: " + ", ".join(f"{k}×{v}" for k, v in sorted(names.items())))
say("")

FAIL = []
def chk(name, ok, got, want):
    (FAIL.append(name) if not ok else None)
    say(f"  {'통과' if ok else '🔴실패'}  {name:<34} {got:<30} 기준 {want}")

say("점검")
# 🔴 처음엔 '운석이 줄어든 순간' 으로 걸었는데 **자가시험에서 안 잡혔다.**
#   발사가 0발이어도 배가 죽으면 운석이 리셋되면서 개수가 줄어든다.
#   진짜 신호는 총알 객체다: 정상 3,444회 관측 vs 망가진 상태 22회.
fire_frac = n_fire_obj/max(frames_done, 1)
chk("발사가 실제로 나간다", fire_frac > 0.10,
    f"총알이 보이는 프레임 {fire_frac*100:.1f}% ({n_fire_obj}/{frames_done})", "> 10%")
chk("총이 운석을 맞힌다", drops >= 3, f"운석이 줄어든 순간 {drops}회", ">= 3")
# 웨이브 전환은 짧은 주행에서는 안 일어나는 게 정상이라 **기록만** 한다.
# (운석을 다 부숴야 넘어간다. 3,000프레임이면 대개 못 끝낸다.)
#   발사 버그의 서명은 "운석 개수가 아예 안 변한다" 이고 그건 위 검사가 잡는다.
chk("배가 보인다", 0.55 <= vis <= 0.98, f"결정의 {vis*100:.1f}%", "55~98%")
chk("운석 속도가 0으로 안 굳었다", v_total == 0 or v_nonzero/max(v_total, 1) > 0.5,
    f"{v_nonzero}/{v_total} ({v_nonzero/max(v_total,1)*100:.0f}%)", "> 50%")
chk("화면에 운석이 그려진다 (합성 후)", min(lit2) > 50,
    f"합성 후 최소 {min(lit2)}px (합성 전 {min(lit)}px)", "> 50px")
# 🔴 "발화 최소 > 0" 으로 걸면 안 된다. 위협이 하나도 없는 결정은 자극이 0 이고
#   이 모델은 기저 발화가 0 이라 **발화 0 이 정상**이다 (처음에 이걸로 헛 FAIL 을 냈다).
#   자극이 실제로 들어간 결정만 본다.
sp_stim = [v for v, n in zip(spk, nloom) if n > 0]
ok_stim = len(sp_stim) > 0 and (sum(1 for v in sp_stim if v > 0)/len(sp_stim)) > 0.99
chk("자극이 있으면 뇌가 발화한다", ok_stim,
    f"자극 있는 결정 {len(sp_stim)}회 중 발화 "
    f"{sum(1 for v in sp_stim if v>0)}회 (평균 {np.mean(sp_stim) if sp_stim else 0:.0f})", "> 99%")
chk("액션이 한쪽으로 안 죽었다", min(acts["UP"], acts["LEFT"], acts["RIGHT"]) > 0,
    f"추진 {acts['UP']} 좌 {acts['LEFT']} 우 {acts['RIGHT']}", "각각 > 0")
chk("판독이 상수로 안 굳었다", float(chan[:, 0].std()) > 1e-6 and float(chan[:, 1].std()) > 1e-6,
    f"좌우 sd {chan[:,0].std():.4f} 전후 sd {chan[:,1].std():.4f}", "> 0")

# ── 발사 위상 취약성 검사 (fire_audit2.py 에서 드러난 것) ──────────────────
#   무작위 no-op 시작이 결정 경계와 ROM 폴링 위상을 매번 어긋나게 한다.
#   FIRE_PRESS=1 이면 **전 시드에서 0발**, =4 면 1발이다 (실측).
#   위 검사는 시드 하나만 보므로 이 취약성을 못 잡는다.
#   🔴 '총알이 보이는 프레임 수' 로 세면 안 된다 — 한 발만 쏴도 오래 떠 있어서
#      press=4(=옛 버그)도 통과한다. **새로 생긴 총알 수(발사 수)** 로 센다.
def quick_shots(seed, base_name, frames=900):
    rng2 = np.random.default_rng(seed)
    env.reset(seed=int(rng2.integers(0, 2**31)))
    for _ in range(int(rng2.integers(1, 31))): env.step(A.index("NOOP"))
    b = A.index(base_name); act = (b, with_fire(b, A)); n = 0; prev = 0
    for f in range(frames):
        env.step(frame_action(act[0], act[1], f))
        m = sum(1 for o in env.objects if o and o.wh[0] > 0
                and "Missile" in type(o).__name__)
        if m > prev: n += m - prev
        prev = m
    return n


fire_seeds = [quick_shots(s_, b_) for s_ in (11, 33, 44) for b_ in ("NOOP", "LEFT")]
chk("발사가 시드에 안 흔들린다", min(fire_seeds) >= 10,
    f"시드3 x base2 발사 수 최소 {min(fire_seeds)} (전부 {fire_seeds})", ">= 10")



bias = (acts["LEFT"] - acts["RIGHT"])/max(turn, 1)
say("")
say("기록 (임계값 아님)")
say(f"     웨이브 전환(운석 재보충) {rises}회 — 짧은 주행에선 0 이 정상")
say(f"     좌/우 회전 편향 {bias*100:+.1f}%  (좌 {acts['LEFT']} 우 {acts['RIGHT']}) — 이슈 #32")
say(f"     추진 비율 {acts['UP']/max(dec,1)*100:.1f}%   운석 수 {ast.min()}~{ast.max()}")
say(f"     화면 켜진 픽셀 합성 전 {np.mean(lit):.0f} / 합성 후 {np.mean(lit2):.0f}")

say("")
say("="*70)
if SELFTEST:
    caught = "발사가 실제로 나간다" in FAIL
    say(f"자가시험: 망가진 발사를 {'잡았다' if caught else '못 잡았다'}"
        f"  -> 이 점검은 {'쓸모가 있다' if caught else '쓸모가 없다'}")
    sys.exit(0 if caught else 1)
if FAIL:
    say(f"🔴 {len(FAIL)}개 실패: " + ", ".join(FAIL))
else:
    say("전부 통과")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"sanity.json").write_text(json.dumps(dict(
    frames=MAXF, decisions=dec, score=score, drops=drops, rises=rises,
    ship_visible=vis, v_nonzero=v_nonzero, v_total=v_total,
    lit_min=min(lit), lit2_min=min(lit2), spk_min=int(min(spk)) if spk else 0,
    acts=acts, turn_bias=bias, objects=names, fail=FAIL), indent=2, ensure_ascii=False),
    encoding="utf-8")
say("-> out/sanity.json")
sys.exit(1 if FAIL else 0)
