"""같은 시작점에서 여러 번 돌리면 비트 단위로 같아야 한다.
다르면 경쟁 조건이고, 02문서 10.4 분기 비교 데모가 원리적으로 불가능해진다."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

p = dict(PARAMS)
rng = np.random.default_rng(0)
idx = rng.choice(166700, size=400, replace=False)
thr = p["v_th"] - p["v_0"]
b = BrainRT(params=p)
R = {}

def run(n, poisson=False):
    b.reset()
    if poisson:
        b.set_drive(None, 0); b.set_poisson(idx, 150.0)
    else:
        b.set_poisson(None, 0); b.set_drive(idx, thr*1.2)
    b.run_eager(n)
    return b.v.clone(), b.g.clone(), b.refr.clone().int(), int(b.over.item())

ok_all = True
for tag, poi in (("직접 주입", False), ("Poisson 자극", True)):
    for n in (10, 40, 200, 999):
        outs = [run(n, poi) for _ in range(3)]
        dv = max(float((outs[0][0]-o[0]).abs().max().item()) for o in outs[1:])
        dg = max(float((outs[0][1]-o[1]).abs().max().item()) for o in outs[1:])
        dr = max(int((outs[0][2]-o[2]).abs().max().item()) for o in outs[1:])
        ok = (dv == 0 and dg == 0 and dr == 0); ok_all &= ok
        print(f"  {tag:>10}  {n:4d} step x3:  |dv|max={dv:.3e}  |dg|max={dg:.3e}  "
              f"|drefr|max={dr}  -> {'DETERMINISTIC' if ok else 'NONDETERMINISTIC'}")
        R[f"{tag}_{n}"] = dict(dv=dv, dg=dg, drefr=dr, deterministic=bool(ok))
R["all_deterministic"] = bool(ok_all)
print(f"\n재현성: {'전 조건 비트 단위 동일' if ok_all else '깨짐'}")
(Path(__file__).resolve().parent.parent/"out"/"determinism.json").write_text(
    json.dumps(R, indent=2, default=float), encoding="utf-8")
