import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS
from verify import BrainRef

p = dict(PARAMS)
rng = np.random.default_rng(0)
idx = rng.choice(166700, size=400, replace=False)
thr = p["v_th"] - p["v_0"]
rt = BrainRT(params=p); rt.reset(); rt.set_drive(idx, thr*1.2)
ref = BrainRef(p)
drive = torch.zeros(rt.N, device="cuda"); drive[torch.from_numpy(idx).cuda().long()] = thr*1.2
drive_set = set(idx.tolist())

prev = {}
for t in range(10):
    prev["g_rt"] = rt.g.clone(); prev["g_ref"] = ref.g.clone()
    prev["v_rt"] = rt.v.clone(); prev["v_ref"] = ref.v.clone()
    prev["r_rt"] = rt.refr.clone().int(); prev["r_ref"] = ref.refr.clone()
    prev["inc"] = None
    ref.step(drive=drive); rt.run_eager(1)

dg = (rt.g - ref.g).abs()
n = int((dg > 1e-6).sum())
print(f"step 9 직후: g 불일치 뉴런 {n}개,  max {dg.max().item():.6f}")
top = torch.topk(dg, min(8, rt.N)).indices.cpu().numpy()
print(f"{'뉴런':>8} {'inD':>4} {'g_before':>12} {'g_rt':>12} {'g_ref':>12} {'dg':>10} "
      f"{'r_rt':>5} {'r_ref':>5} {'drive':>6}")
for j in top:
    j = int(j)
    if dg[j].item() <= 1e-6: continue
    print(f"{j:>8} {'':>4} {prev['g_ref'][j].item():12.5f} {rt.g[j].item():12.5f} "
          f"{ref.g[j].item():12.5f} {dg[j].item():10.5f} "
          f"{int(prev['r_rt'][j].item()):5d} {int(prev['r_ref'][j].item()):5d} "
          f"{'Y' if j in drive_set else '-':>6}")
    gb = prev['g_ref'][j].item()
    print(f"         g_before*B = {gb*ref.B:.6f} ;  g_rt-g_before = {rt.g[j].item()-gb:.6f} ;"
          f"  g_ref-g_before*B = {ref.g[j].item()-gb*ref.B:.6f}")
dr = (rt.refr.int() - ref.refr).abs()
print(f"\nrefr 불일치 뉴런 {int((dr>0).sum())}개")
dv = (rt.v - ref.v).abs()
print(f"v 불일치 뉴런 {int((dv>1e-6).sum())}개  max {dv.max().item():.6f}")
