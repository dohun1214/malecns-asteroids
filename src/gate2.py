"""게이트 2: Asteroids 실전에서 회로가 무작위 조작보다 오래 사는가. (02문서 7장)

지표는 '점수'가 아니라 '생존'이다. 이 컨트롤러는 발사를 안 하므로 웨이브가 안 끝나고
난이도가 고정된다 — 도피 능력만 보는 깨끗한 조건이 된다.

대조군
  noop     가만히         (아무것도 안 해도 오래 살면 이 시험은 무의미하다)
  random   무작위 액션     (02문서 8장의 성공 기준: "무작위 조작보다 오래 산다")
  greedy   같은 기하학을 규칙으로. **"그냥 규칙이랑 뭐가 달라"의 그 규칙**
  brain    전체 뇌 166,700
  lesion   DNp02 / DNp11 각 2개, 그리고 음성 대조로 무작위 2개
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, GreedyPolicy, run_episode, make_env, Vision

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

env = make_env(); actions = env.unwrapped.get_action_meanings(); V = Vision()
FIRE = os.environ.get("NOFIRE", "") == ""
def bench(name, pol, seed=12345):
    rng = np.random.default_rng(seed)
    rs = [run_episode(env, pol, V, actions, max_frames=MAXF, rng=rng, fire=FIRE)
          for _ in range(N_EP)]
    a = dict(policy=name,
             mean_life=float(np.mean([r["mean_life"] for r in rs])),
             sd_life=float(np.std([r["mean_life"] for r in rs])),
             max_life=float(np.mean([r["max_life"] for r in rs])),
             lives=float(np.mean([r["n_lives"] for r in rs])),
             score=float(np.mean([r["score"] for r in rs])),
             frames=float(np.mean([r["frames"] for r in rs])))
    print(f"{name:<26} {a['mean_life']:8.1f} ±{a['sd_life']:5.1f} {a['max_life']:9.1f} "
          f"{a['lives']:7.1f} {a['score']:8.0f}", flush=True)
    return a

print(f"에피소드 {N_EP}개 x 최대 {MAXF} 프레임, 동일 시드. "
      f"발사 {'모든 정책에 고정' if FIRE else '없음'}\n")
print(f"{'정책':<26} {'평균 생존(f)':>9} {'':>6} {'최장(f)':>9} {'목숨':>7} {'점수':>8}")
R = []
r2 = np.random.default_rng(7)
R.append(bench("noop", lambda l, o, a: (a.index("NOOP"), {"norm": 0.0})))
R.append(bench("random", lambda l, o, a: (int(r2.integers(0, len(a))), {"norm": 0.0})))
R.append(bench("greedy (규칙)", GreedyPolicy()))

t0 = time.perf_counter()
bp = BrainPolicy()
print(f"  (뇌 준비 {time.perf_counter()-t0:.0f}s)", flush=True)
C = bp.C
rand2 = np.random.default_rng(11).choice(166700, size=2, replace=False)

def with_lesion(cells):
    bp.b.alive.fill_(1)
    if cells is not None and len(cells):
        bp.b.lesion(torch.as_tensor(np.asarray(cells), device="cuda"))
    bp.frame = 0
    return bp

R.append(bench("brain (온전)", with_lesion(None)))
R.append(bench("brain lesion DNp02(2)", with_lesion(C["DNp02"])))
R.append(bench("brain lesion DNp11(2)", with_lesion(C["DNp11"])))
R.append(bench("brain lesion 무작위2 (음성)", with_lesion(rand2)))
R.append(bench("brain lesion LC4 절반", with_lesion(C["LC4"][::2])))
bp.b.alive.fill_(1)

base = next(r for r in R if r["policy"] == "brain (온전)")
rnd  = next(r for r in R if r["policy"] == "random")
grd  = next(r for r in R if r["policy"] == "greedy (규칙)")
print(f"\n판정")
print(f"  뇌 vs 무작위 : {base['mean_life']/max(rnd['mean_life'],1e-9):.2f}배  "
      f"-> {'통과 (02문서 8장 성공 기준)' if base['mean_life'] > rnd['mean_life'] else '실패'}")
print(f"  뇌 vs 규칙   : {base['mean_life']/max(grd['mean_life'],1e-9):.2f}배  "
      f"(규칙보다 못해도 무방 — 요점은 '다르게 한다'다)")
for r in R:
    if r["policy"].startswith("brain lesion"):
        print(f"  {r['policy']:<26} 온전 대비 {r['mean_life']/max(base['mean_life'],1e-9):.2f}배")
(ROOT/"out"/"gate2.json").write_text(json.dumps(R, indent=2), encoding="utf-8")
print(f"\n-> out/gate2.json")
