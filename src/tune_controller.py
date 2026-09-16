"""이슈 #3: 컨트롤러 파라미터 조정.

⚠️ 과적합 방지 — 튜닝 시드와 보고 시드를 분리한다.
   TUNE 시드로 고르고, 한 번도 안 본 REPORT 시드에서 최종 수치를 낸다.

   ⚠️⚠️ 시드만으로는 안 된다. ALE Asteroids 는 env.reset(seed=...) 를 줘도
   게임이 완전히 동일하다(운석 초기 배치 고정). 실제로 처음 돌렸을 때
   튜닝 시드와 보고 시드 결과가 소수점까지 일치했다.
   -> ALE 표준 평가 규약인 '시작 시 무작위 no-op(1~30프레임)'으로 변동을 만든다.

⚠️ 회계 — 이건 '컨트롤러'를 생존 기준으로 고른 것이지 모델 피팅이 아니다.
   w_syn / dt / 연결성은 손대지 않는다. 모든 lesion 조건이 같은 컨트롤러를 쓰므로
   lesion 증거는 영향받지 않는다. 다만 '부과한 것'에 적는다.
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, GreedyPolicy, run_episode, make_env, Vision

ROOT = Path(__file__).resolve().parent.parent
TUNE_SEED, REPORT_SEED = 12345, 8675309
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
def bench(pol, seed, n=N_EP, mf=MAXF):
    rng = np.random.default_rng(seed)
    rs = [run_episode(env, pol, V, A, max_frames=mf, rng=rng) for _ in range(n)]
    life = np.array([r["mean_life"] for r in rs])
    return (float(life.mean()), float(np.mean([r["up_frac"] for r in rs])),
            float(np.mean([r["max_life"] for r in rs])), float(life.std()))

def say(*a): print(*a, flush=True)
bp = BrainPolicy()
say(f"튜닝 시드 {TUNE_SEED} / 보고 시드 {REPORT_SEED}  (에피소드 {N_EP} x {MAXF}프레임)\n")

# ---------------------------------------------------------------- 1차: align_slop
say(f"{'align_slop':>11} {'hold':>5} | {'생존':>8} {'추진%':>7} {'최장':>8}")
best = None
for slop in (1, 2, 3):
    for hold in (0, 2):
        bp.set_controller(align_slop=slop, hold=hold)
        life, up, mx, sd = bench(bp, TUNE_SEED)
        say(f"{slop:>11} {hold:>5} | {life:>8.1f} ±{sd:>5.0f} {up*100:>6.1f}% {mx:>8.1f}")
        if best is None or life > best[0]: best = (life, slop, hold)
say(f"  -> 최적 align_slop={best[1]} hold={best[2]} (생존 {best[0]:.1f})")
SLOP, HOLD = best[1], best[2]

# ---------------------------------------------------------------- 2차: k
say(f"\n{'k':>11} | {'생존':>8} {'추진%':>7} {'최장':>8}")
bestk = None
for k in (8, 16, 24, 32):
    bp.set_controller(k=k, align_slop=SLOP, hold=HOLD)
    life, up, mx, sd = bench(bp, TUNE_SEED)
    say(f"{k:>11} | {life:>8.1f} ±{sd:>5.0f} {up*100:>6.1f}% {mx:>8.1f}")
    if bestk is None or life > bestk[0]: bestk = (life, k)
K = bestk[1]
say(f"  -> 최적 k={K} (생존 {bestk[0]:.1f})")

# ---------------------------------------------------------------- 3차: th50 / cap 민감도
say(f"\n{'th50':>6} {'cap':>6} | {'생존':>8} {'추진%':>7}")
bestt = None
for th50 in (7.2, 14.4):
    for cap in (150.0, 220.0):
        bp.set_controller(k=K, align_slop=SLOP, hold=HOLD, th50=th50, cap=cap)
        life, up, mx, sd = bench(bp, TUNE_SEED)
        say(f"{th50:>6.1f} {cap:>6.0f} | {life:>8.1f} ±{sd:>5.0f} {up*100:>6.1f}%")
        if bestt is None or life > bestt[0]: bestt = (life, th50, cap)
TH50, CAP = bestt[1], bestt[2]
say(f"  -> 최적 th50={TH50} cap={CAP} (생존 {bestt[0]:.1f})")

# ---------------------------------------------------------------- 보고: 안 본 시드
say(f"\n{'='*66}\n보고용 시드 {REPORT_SEED} — 튜닝에 한 번도 안 쓴 시드")
cfg = dict(k=K, align_slop=SLOP, hold=HOLD, th50=TH50, cap=CAP)
say(f"선택된 설정: {cfg}")
say(f"\n{'정책':<22} {'생존':>8} {'추진%':>7} {'최장':>8}")
rows = {}
bp.set_controller(**{"k": 16, "align_slop": 1, "hold": 0, "th50": 7.2, "cap": 150.0})
rows["뇌 (조정 전)"] = bench(bp, REPORT_SEED)
bp.set_controller(**cfg)
rows["뇌 (조정 후)"] = bench(bp, REPORT_SEED)
rows["규칙 기반"] = bench(GreedyPolicy(), REPORT_SEED)
rows["가만히 있기"] = bench(lambda l,o,a:(a.index("NOOP"), {"norm":0.0}), REPORT_SEED)
r2 = np.random.default_rng(7)
rows["무작위"] = bench(lambda l,o,a:(int(r2.integers(0,len(a))), {"norm":0.0}), REPORT_SEED)
for k_, (life, up, mx, sd) in rows.items():
    say(f"{k_:<22} {life:>8.1f} ±{sd:>5.0f} {up*100:>6.1f}% {mx:>8.1f}")
(ROOT/"out"/"tune_controller.json").write_text(json.dumps(
    dict(cfg=cfg, tune_seed=TUNE_SEED, report_seed=REPORT_SEED,
         report={k_: dict(life=v[0], up=v[1], max=v[2], sd=v[3]) for k_, v in rows.items()}),
    indent=2), encoding="utf-8")
say(f"\n-> out/tune_controller.json")
