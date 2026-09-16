"""B-2: 하행뉴런이 포화되지 않는 동작 구간을 찾는다.

150 Hz 로 LC4 126개를 전부 때리면 DN 이 320 Hz 까지 올라간다. 불응기 2.2ms 상한이
약 450 Hz 이므로 포화에 가깝고, 포화 상태에서는 DNp02-DNp11 차이가 뭉개진다.

⚠️ 동작점은 '방향 선택성이 제일 잘 나오는 값'이 아니라 '포화되지 않는 값'으로 고른다.
   전자로 고르면 그건 피팅이고 순환논증 회계의 '피팅한 것' 칸에 들어간다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
C = np.load(ROOT/"graph"/"circuit_idx.npz")
DNS = ["DNp01","DNp02","DNp03","DNp04","DNp06","DNp11"]
T_MS = 300.0
p = dict(PARAMS)
steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
rng = np.random.default_rng(7)
R = {}
def say(*a): print(*a, flush=True)

RFC_MAX = 1000.0 / p["t_rfc"]
say(f"불응기 {p['t_rfc']} ms -> 이론상 최대 발화율 {RFC_MAX:.0f} Hz")
say(f"\n{'='*92}\n(1) 자극 '세기' 스윕 — LC4 126개 전부, Poisson 비율만 변경")
say(f"{'rate':>6} {'자극뉴런Hz':>10} {'활성%':>7} | " + " ".join(f"{d[-3:]:>7}" for d in DNS)
    + " | " + f"{'DN최대/450':>10}")
for hz in (1, 2, 5, 10, 20, 40, 80, 150):
    b.reset(); b.set_poisson(C["LC4"], float(hz)); b.run_tally(steps)
    torch.cuda.synchronize()
    stim = float(b.rates(C["LC4"], steps).mean())
    act = int((b.tally > 0).sum().item())/b.N*100
    dn = {d: float(b.rates(C[d], steps).mean()) for d in DNS}
    mx = max(dn.values())
    say(f"{hz:>6} {stim:>10.1f} {act:>6.2f}% | " + " ".join(f"{dn[d]:>7.1f}" for d in DNS)
        + f" | {mx/RFC_MAX*100:>9.0f}%")
    R[f"rate_{hz}"] = dict(stim_hz=stim, active_pct=act, dn=dn, sat=mx/RFC_MAX)

say(f"\n{'='*92}\n(2) 자극 '범위' 스윕 — 150 Hz 고정, LC4 중 일부만 (실제 looming 은 시야 일부)")
say(f"{'n_LC4':>6} {'활성%':>7} | " + " ".join(f"{d[-3:]:>7}" for d in DNS) + " | " + f"{'포화':>6}")
for n in (2, 4, 8, 16, 32, 64, 126):
    sub = rng.choice(C["LC4"], size=n, replace=False)
    b.reset(); b.set_poisson(sub, 150.0); b.run_tally(steps)
    torch.cuda.synchronize()
    act = int((b.tally > 0).sum().item())/b.N*100
    dn = {d: float(b.rates(C[d], steps).mean()) for d in DNS}
    mx = max(dn.values())
    say(f"{n:>6} {act:>6.2f}% | " + " ".join(f"{dn[d]:>7.1f}" for d in DNS)
        + f" | {mx/RFC_MAX*100:>5.0f}%")
    R[f"n_{n}"] = dict(active_pct=act, dn=dn, sat=mx/RFC_MAX)

say(f"\n포화 기준: DN 최대 발화율이 이론 상한의 30% 미만 (약 135 Hz 이하) 인 구간을 쓴다.")
(ROOT/"out"/"drive_sweep.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
