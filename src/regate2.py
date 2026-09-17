"""🔴 11문서의 '게임 실력 주장 철회'를 **발사가 실제로 되는 상태에서** 다시 잰다.

철회 당시 우리는 "발사를 모든 정책에 고정했다"고 믿고 있었다. 실제로는 결정당 4프레임 내내
FIRE 를 누르고 있어서 **한 발도 안 나가고 있었다** (probe_fire4.py).
운석이 하나도 안 줄어드는 상태였으므로 '가만히 있기'가 유리한 게 당연했다.

발사가 되면 이야기가 달라진다: 조준하려면 돌아야 하고, 웨이브를 끝내야 다음으로 간다.
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, GreedyPolicy, run_episode, make_env, Vision

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
def say(*a): print(*a, flush=True)


def bench(pol, fire, seed=12345):
    rng = np.random.default_rng(seed)
    rs = [run_episode(env, pol, V, A, max_frames=MAXF, rng=rng, fire=fire)
          for _ in range(N_EP)]
    f = lambda k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
    return dict(score=f("score"), life=f("mean_life"), frames=f("frames"),
                up=f("up_frac"), lives=f("n_lives"))


r2 = np.random.default_rng(7)
POLS = [("가만히 있기", lambda l, o, a: (a.index("NOOP"), {"norm": 0.0})),
        ("무작위", lambda l, o, a: (int(r2.integers(0, len(a))), {"norm": 0.0})),
        ("규칙 기반", GreedyPolicy())]
bp = BrainPolicy()
POLS.append(("전체 뇌 166,700", bp))

t0 = time.perf_counter()
out = {}
say(f"에피소드 {N_EP} x 최대 {MAXF}프레임, 무작위 no-op 시작\n")
for fire in (True, False):
    say(f"{'='*78}\n발사 {'작동' if fire else '없음(예전 상태)'}")
    say(f"{'정책':<18}{'점수':>16}{'목숨당 생존':>16}{'프레임':>12}{'추진%':>8}")
    for name, pol in POLS:
        if isinstance(pol, BrainPolicy):
            pol.b.alive.fill_(1); pol.frame = 0; pol.dec.reset()
        d = bench(pol, fire); out[f"{name}|{fire}"] = d
        say(f"{name:<18}{d['score'][0]:>9.0f}±{d['score'][1]:<6.0f}"
            f"{d['life'][0]:>9.0f}±{d['life'][1]:<6.0f}"
            f"{d['frames'][0]:>12.0f}{d['up'][0]*100:>8.1f}")

b = out["전체 뇌 166,700|True"]; n = out["가만히 있기|True"]; g = out["규칙 기반|True"]
say(f"\n{'='*78}\n판정 (발사 작동 기준)")
for k, lab in (("score", "점수"), ("life", "목숨당 생존")):
    say(f"  {lab}: 뇌 {b[k][0]:.0f}  vs 가만히 {n[k][0]:.0f}  vs 규칙 {g[k][0]:.0f}"
        f"   -> 뇌가 가만히보다 {'낫다' if b[k][0] > n[k][0] else '못하다'}"
        f" ({b[k][0]/max(n[k][0],1e-9):.2f}배)")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"regate2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
say(f"\n-> out/regate2.json  ({time.perf_counter()-t0:.0f}s)")
