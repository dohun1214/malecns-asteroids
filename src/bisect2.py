"""스텝별로 스파이크 목록/cnt/scatter 직후 inc 를 3회 비교해 첫 갈림을 찾는다."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS, _scatter, _membrane, POST_BITS, POST_MASK

p = dict(PARAMS)
rng = np.random.default_rng(0)
idx = rng.choice(166700, size=400, replace=False)
thr = p["v_th"] - p["v_0"]
b = BrainRT(params=p)
STEPS = 14

def trace():
    b.reset(); b.set_drive(idx, thr*1.2)
    rec = []
    for t in range(STEPS):
        sw = t % b.L; sr = (t - b.D) % b.L
        cnt_before = int(b.cnt[sr].item())
        sp_before = np.sort(b.sp[sr, :cnt_before].cpu().numpy()) if cnt_before else np.empty(0, np.int32)
        _scatter[(b.nprog, b.lanes)](b.sp[sr], b._cnt_view[sr], b.crow, b.packed, b.inc,
            POST_MASK=POST_MASK, PB=POST_BITS, NPROG=b.nprog, LANES=b.lanes,
            EBLOCK=b.eblock, num_warps=4)
        torch.cuda.synchronize()
        inc_after = b.inc.clone()
        b._cnt_view[sw].zero_()
        _membrane[(b.nblk,)](b.v, b.g, b.refr, b.inc, b.alive, b.rfc, b.drive, b.lam,
            b.sp[sw], b._cnt_view[sw], b.over, b.N, b.cap, b.seed, t, b.poi_amp,
            b.A, b.B, b.KBA, b.p["v_0"], b.p["v_th"], b.p["v_rst"],
            float(b.p["w_syn"]), HAS_POI=b.has_poi, BLOCK=b.block, num_warps=4)
        torch.cuda.synchronize()
        c2 = int(b.cnt[sw].item())
        rec.append(dict(t=t, sr=sr, cnt_in=cnt_before, sp_in=sp_before,
                        inc_sum=int(inc_after.sum().item()),
                        inc_abs=int(inc_after.abs().sum().item()),
                        inc_nz=int((inc_after != 0).sum().item()),
                        cnt_out=c2,
                        sp_out=np.sort(b.sp[sw, :c2].cpu().numpy()) if c2 else np.empty(0, np.int32),
                        v=b.v.clone(), g=b.g.clone()))
    return rec

runs = [trace() for _ in range(3)]
print(f"{'t':>3} {'sr':>3} {'cnt_in':>7} {'sp_in동일':>9} {'inc_nz':>7} {'inc동일':>8} "
      f"{'cnt_out':>8} {'sp_out동일':>10} {'v동일':>6}")
for t in range(STEPS):
    a = runs[0][t]
    sp_in_same = all(np.array_equal(a["sp_in"], r[t]["sp_in"]) for r in runs[1:])
    inc_same = all(a["inc_sum"] == r[t]["inc_sum"] and a["inc_abs"] == r[t]["inc_abs"]
                   and a["inc_nz"] == r[t]["inc_nz"] for r in runs[1:])
    cnt_out_same = all(a["cnt_out"] == r[t]["cnt_out"] for r in runs[1:])
    sp_out_same = all(np.array_equal(a["sp_out"], r[t]["sp_out"]) for r in runs[1:])
    v_same = all(float((a["v"]-r[t]["v"]).abs().max().item()) == 0.0 for r in runs[1:])
    print(f"{t:>3} {a['sr']:>3} {a['cnt_in']:>7} {str(sp_in_same):>9} {a['inc_nz']:>7} "
          f"{str(inc_same):>8} {a['cnt_out']:>8} {str(sp_out_same) if cnt_out_same else 'cnt다름':>10} "
          f"{str(v_same):>6}")
    if not inc_same:
        print(f"     inc_nz 3회: {[r[t]['inc_nz'] for r in runs]}  "
              f"inc_abs 3회: {[r[t]['inc_abs'] for r in runs]}")
        break
