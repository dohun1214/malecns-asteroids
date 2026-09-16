"""B단계: LC4 를 때리면 신호가 하행뉴런까지 실제로 가는가.

이게 안 되면 게이트 1(a)(c)가 둘 다 무너진다. 01문서 7.1 이 반복해서 경고하는
"자극을 주지 않으면 아무것도 안 일어난다 / 생리적 크기로 넣으면 아무것도 안 일어난다"
의 실전 버전이다.

특이성 대조가 핵심: 무작위 126개를 같은 세기로 때렸을 때와 비교해야
"많이 때려서 켜진 것"이 아님을 보일 수 있다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
C = np.load(ROOT/"graph"/"circuit_idx.npz")
DNS = ["DNp01","DNp02","DNp03","DNp04","DNp06","DNp11"]
VNCD = ["TTMn","PSI","DLMn","GFC1","GFC2","GFC3","GFC4","MDN"]

T_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
W_SWEEP = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.15]
R = {"t_ms": T_MS}
def say(*a): print(*a, flush=True)

rng = np.random.default_rng(7)
RAND126 = rng.choice(166700, size=126, replace=False)
RAND185 = rng.choice(166700, size=185, replace=False)

CONDS = [
    ("자극 없음",            None),
    ("LC4 전체 126",         C["LC4"]),
    ("LC4 좌 71",            C["LC4_L"]),
    ("LC4 우 55",            C["LC4_R"]),
    ("무작위 126 (대조)",     RAND126),
    ("LPLC2 185",            C["LPLC2"]),
    ("무작위 185 (대조)",     RAND185),
]

for w_syn in W_SWEEP:
    p = dict(PARAMS); p["w_syn"] = w_syn
    b = BrainRT(params=p)
    steps = int(round(T_MS / p["dt"]))
    say(f"\n{'='*94}")
    say(f"w_syn = {w_syn}   자극 {p['w_syn']*p['f_poi']:.1f} mV (임계의 "
        f"{p['w_syn']*p['f_poi']/7:.1f}배)   {T_MS:.0f} ms = {steps} step")
    say(f"{'조건':<20} {'활성뉴런':>9} {'활성%':>7} {'평균Hz':>7} | " +
        " ".join(f"{d[-3:]:>6}" for d in DNS) + " | " + " ".join(f"{v:>5}" for v in VNCD))
    res = {}
    for name, tgt in CONDS:
        b.reset()
        if tgt is None: b.set_poisson(None, 0)
        else:           b.set_poisson(tgt, 150.0)
        b.run_tally(steps)
        torch.cuda.synchronize()
        tal = b.tally
        nact = int((tal > 0).sum().item())
        sec = steps * p["dt"] / 1000.0
        mean_hz = float(tal.sum().item()) / b.N / sec
        dn = {d: float(b.rates(C[d], steps).mean()) for d in DNS}
        vn = {v: float(b.rates(C[v], steps).mean()) if len(C[v]) else 0.0 for v in VNCD}
        say(f"{name:<20} {nact:>9,} {nact/b.N*100:>6.2f}% {mean_hz:>7.2f} | " +
            " ".join(f"{dn[d]:>6.1f}" for d in DNS) + " | " +
            " ".join(f"{vn[v]:>5.1f}" for v in VNCD))
        res[name] = dict(active=nact, active_pct=nact/b.N*100, mean_hz=mean_hz,
                         dn=dn, vnc=vn,
                         stim_hz=float(b.rates(tgt, steps).mean()) if tgt is not None else 0.0)
    R[f"w{w_syn}"] = res

    # 특이성: LC4 대 무작위 126
    a, c = res["LC4 전체 126"], res["무작위 126 (대조)"]
    say(f"\n  특이성 (LC4 126 vs 무작위 126, 같은 자극 세기):")
    for d in DNS:
        x, y = a["dn"][d], c["dn"][d]
        ratio = x/y if y > 0.01 else float("inf") if x > 0.01 else 1.0
        say(f"    {d}: LC4 {x:6.1f} Hz / 무작위 {y:6.1f} Hz  ->  "
            f"{'inf' if ratio==float('inf') else f'{ratio:.1f}x'}")
    say(f"    자극 뉴런 자체 발화율: LC4 {a['stim_hz']:.0f} Hz / 무작위 {c['stim_hz']:.0f} Hz "
        f"(같아야 정상 — 자극 세기가 동일하다는 확인)")
    # 편측성
    l, r = res["LC4 좌 71"], res["LC4 우 55"]
    say(f"  편측성: 좌 자극시 DNp02 {l['dn']['DNp02']:.1f} / DNp11 {l['dn']['DNp11']:.1f} Hz,"
        f"  우 자극시 DNp02 {r['dn']['DNp02']:.1f} / DNp11 {r['dn']['DNp11']:.1f} Hz")

(OUT/"gate_b.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
say(f"\n-> out/gate_b.json")
