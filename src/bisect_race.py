"""경쟁 조건이 scatter 커널인지 membrane 커널인지 이분한다."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lif_rt
from lif_rt import BrainRT, PARAMS, _scatter, _membrane, POST_BITS, POST_MASK

G = Path(__file__).resolve().parent.parent/"graph"
crow_np = np.load(G/"out_crow.npy").astype(np.int64)
post_np = np.load(G/"out_post.npy").astype(np.int64)
w_np    = np.load(G/"out_w.npy").astype(np.int32)
N = crow_np.size - 1
crow_c = torch.from_numpy(crow_np).cuda()
post_c = torch.from_numpy(post_np).cuda()
w_c    = torch.from_numpy(w_np).cuda()

def torch_scatter(b, sr):
    c = int(b.cnt[sr].item())
    if c == 0: return
    src = b.sp[sr, :c].long()
    lo, hi = crow_c[src], crow_c[src+1]
    deg = hi - lo
    tot = int(deg.sum().item())
    if tot == 0: return
    starts = torch.repeat_interleave(lo, deg)
    off = torch.arange(tot, device="cuda") - torch.repeat_interleave(
        torch.cumsum(deg, 0) - deg, deg)
    e = starts + off
    b.inc.index_add_(0, post_c[e], w_c[e])

def make_step(b, mode):
    def step(t):
        sw = t % b.L; sr = (t - b.D) % b.L
        if mode in ("full", "torch_scatter"):
            if mode == "full":
                _scatter[(b.nprog, b.lanes)](b.sp[sr], b._cnt_view[sr], b.crow, b.packed,
                    b.inc, POST_MASK=POST_MASK, PB=POST_BITS, NPROG=b.nprog,
                    LANES=b.lanes, EBLOCK=b.eblock, num_warps=4)
            else:
                torch_scatter(b, sr)
        b._cnt_view[sw].zero_()
        _membrane[(b.nblk,)](b.v, b.g, b.refr, b.inc, b.alive, b.rfc, b.drive, b.lam,
            b.sp[sw], b._cnt_view[sw], b.over, b.N, b.cap, b.seed, t, b.poi_amp,
            b.A, b.B, b.KBA, b.p["v_0"], b.p["v_th"], b.p["v_rst"],
            float(b.p["w_syn"]), HAS_POI=b.has_poi, BLOCK=b.block, num_warps=4)
    return step

p = dict(PARAMS)
rng = np.random.default_rng(0)
idx = rng.choice(166700, size=400, replace=False)
thr = p["v_th"] - p["v_0"]
b = BrainRT(params=p)

for mode in ("no_scatter", "torch_scatter", "full"):
    st = make_step(b, mode)
    res = []
    for _ in range(3):
        b.reset(); b.set_drive(idx, thr*1.2)
        for t in range(30): st(t)
        torch.cuda.synchronize()
        res.append((b.v.clone(), b.g.clone()))
    dv = max(float((res[0][0]-r[0]).abs().max().item()) for r in res[1:])
    dg = max(float((res[0][1]-r[1]).abs().max().item()) for r in res[1:])
    print(f"{mode:>14}: 30 step x3  |dv|max={dv:.3e}  |dg|max={dg:.3e}  "
          f"-> {'DETERMINISTIC' if dv==0 and dg==0 else 'NONDETERMINISTIC'}")
