"""게이트 2 재정의 (이슈 #6): 충돌 임박 상황에 조건부인 지표.

총 생존 시간은 '도피를 잘하는가'가 아니라 '덜 움직이는가'를 잰다.
배가 정지해 있으면 대부분의 운석이 그냥 빗나가서, 아무것도 안 하는 게 1등이 됐다.
손으로 짠 규칙 기반 컨트롤러도 똑같이 졌다 -> 회로가 아니라 채점법 문제다.

  탈출 성공률 = 1 - (피격으로 끝난 위협 사건) / (전체 위협 사건)
  노출 빈도   = 1,000프레임당 위협 사건 수   (성공률과 분리해서 본다)

위험 반경 R 은 눈대중이 아니라 실측으로 정했다 (probe_threat.py, 결정 4,818회/사망 18회).
  죽기 직전 접근 운석까지의 최소거리  중앙값 6,  최대 15.8
  R=15 는 사망의 83%,  R=20 은 100% 를 포함한다. R=40 은 전체 결정의 38% 가
  반경 안이라 '위협'이라는 말이 무의미해진다.
-> R = 15 / 20 / 25 세 값에서 확인한다. 한 값에만 성립하면 못 쓴다.
반경은 채점에만 쓰이고 정책에는 안 들어가므로 한 번의 주행에서 동시에 집계한다.
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, GreedyPolicy, run_episode, make_env, Vision

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
RADII = [15.0, 20.0, 25.0]
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
def say(*a): print(*a, flush=True)


def bench(pol, seed=12345):
    """한 정책을 N_EP 에피소드 주행. 반경별 사건 집계를 한 번에 얻는다."""
    rng = np.random.default_rng(seed)
    rs = [run_episode(env, pol, V, A, max_frames=MAXF, rng=rng, threat_r=RADII)
          for _ in range(N_EP)]
    out = {}
    for R in RADII:
        ev = sum(r["events_by_r"][R] for r in rs)
        hit = sum(r["hits_by_r"][R] for r in rs)
        p = (1.0 - hit/ev) if ev else float("nan")
        out[R] = dict(events=ev, hits=hit, escape=p,
                      se=float(np.sqrt(p*(1-p)/ev)) if ev else float("nan"),
                      exposure=float(np.mean([r["events_by_r"][R]/max(r["frames"],1)*1000.0
                                              for r in rs])))
    life = float(np.mean([r["mean_life"] for r in rs]))
    score = float(np.mean([r["score"] for r in rs]))
    up = float(np.mean([r["up_frac"] for r in rs]))
    return dict(by_r=out, life=life, score=score, up_frac=up)


r2 = np.random.default_rng(7)
POLS = [("가만히 있기", lambda l, o, a: (a.index("NOOP"), {"norm": 0.0})),
        ("무작위", lambda l, o, a: (int(r2.integers(0, len(a))), {"norm": 0.0})),
        ("규칙 기반", GreedyPolicy())]
bp = BrainPolicy(); C = bp.C
rand2 = np.random.default_rng(11).choice(166700, size=2, replace=False)


def with_lesion(cells):
    bp.b.alive.fill_(1)
    if cells is not None and len(cells):
        bp.b.lesion(torch.as_tensor(np.asarray(cells), device="cuda"))
    bp.frame = 0; bp.dec.reset()
    return bp


BRAINS = [("전체 뇌", None), ("뇌 lesion DNp02(2)", C["DNp02"]),
          ("뇌 lesion DNp11(2)", C["DNp11"]),
          ("뇌 lesion 무작위2 (음성)", rand2), ("뇌 lesion LC4 절반", C["LC4"][::2])]

res = {}
t0 = time.perf_counter()
for name, pol in POLS:
    res[name] = bench(pol); say(f"  [{time.perf_counter()-t0:6.0f}s] {name} 완료")
for name, les in BRAINS:
    res[name] = bench(with_lesion(les)); say(f"  [{time.perf_counter()-t0:6.0f}s] {name} 완료")
bp.b.alive.fill_(1)

NAMES = [n for n, _ in POLS] + [n for n, _ in BRAINS]
for R in RADII:
    say(f"\n{'='*82}\n위험 반경 R = {R:.0f}   (에피소드 {N_EP} x {MAXF}프레임, 무작위 no-op 시작)")
    say(f"{'정책':<26} {'사건':>7} {'피격':>6} {'탈출 성공률':>16} {'노출/1000f':>11} {'생존':>8}")
    for n in NAMES:
        d = res[n]["by_r"][R]
        say(f"{n:<26} {d['events']:>7} {d['hits']:>6} "
            f"{d['escape']*100:>10.1f}±{d['se']*100:<4.1f}% "
            f"{d['exposure']:>11.1f} {res[n]['life']:>8.0f}")
    b_, n_, g_ = (res["전체 뇌"]["by_r"][R], res["가만히 있기"]["by_r"][R],
                  res["규칙 기반"]["by_r"][R])
    d = b_["escape"] - n_["escape"]; sd = float(np.hypot(b_["se"], n_["se"]))
    say(f"  뇌 - 가만히 = {d*100:+.1f}%p (합성 SE {sd*100:.1f}%p, "
        f"{abs(d)/sd if sd else 0:.1f}σ) -> {'뇌가 낫다' if d > 0 else '뇌가 못하다'}")
    say(f"  뇌 - 규칙   = {(b_['escape']-g_['escape'])*100:+.1f}%p")

say(f"\n{'='*82}\nR 민감도: 결론이 세 값에서 같은가")
say(f"{'정책':<26} " + " ".join(f"{'R='+str(int(r)):>10}" for r in RADII))
for n in NAMES:
    say(f"{n:<26} " + " ".join(f"{res[n]['by_r'][r]['escape']*100:>9.1f}%" for r in RADII))

out = {n: dict(life=res[n]["life"], score=res[n]["score"], up_frac=res[n]["up_frac"],
               by_r={str(int(R)): res[n]["by_r"][R] for R in RADII}) for n in NAMES}
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"gate2_events.json").write_text(
    json.dumps(dict(n_ep=N_EP, max_frames=MAXF, radii=RADII, rows=out),
               indent=2, ensure_ascii=False), encoding="utf-8")
say(f"\n-> out/gate2_events.json  ({time.perf_counter()-t0:.0f}s)")
