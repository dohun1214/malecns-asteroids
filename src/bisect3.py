"""(1) 9스텝째 inc/g/v 를 원소 단위로 3회 비교
   (2) membrane 커널 안에서 INC 를 0으로 쓰는 대신 바깥에서 zero_() 하면 결정론이 돌아오는가"""
import sys
from pathlib import Path
import numpy as np, torch, triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS, _scatter, _membrane, POST_BITS, POST_MASK


@triton.jit
def _membrane_nozero(V, G, R, INC, ALIVE, RFC, DRIVE, SP, CNT,
                     n, cap, A, B, KBA, v_0, v_th, v_rst, w_syn, BLOCK: tl.constexpr):
    """INC 를 읽기만 한다 (0으로 되돌리는 건 호출자가 별도 커널로)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(V + off, mask=m, other=0.0)
    g = tl.load(G + off, mask=m, other=0.0)
    r = tl.load(R + off, mask=m, other=1).to(tl.int32)
    inc = tl.load(INC + off, mask=m, other=0)
    incf = inc.to(tl.float32) * w_syn
    active = r == 0
    g = tl.where(active, g * B + incf, g + incf)
    v = tl.where(active, v_0 + (v - v_0) * A + g * KBA, v)
    v = v + tl.load(DRIVE + off, mask=m, other=0.0)
    r = tl.maximum(r - 1, 0)
    s = (v > v_th) & active & m
    v = tl.where(s, v_rst, v)
    g = tl.where(s, 0.0, g)
    r = tl.where(s, tl.load(RFC + off, mask=m, other=0).to(tl.int32), r)
    tl.store(V + off, v, mask=m)
    tl.store(G + off, g, mask=m)
    tl.store(R + off, r.to(tl.int8), mask=m)
    alive = tl.load(ALIVE + off, mask=m, other=0)
    e = s & (alive != 0)
    ei = e.to(tl.int32)
    num = tl.sum(ei, axis=0)
    if num > 0:
        base = tl.atomic_add(CNT, num)
        pos = base + tl.cumsum(ei, axis=0) - ei
        tl.store(SP + pos, off.to(tl.int32), mask=e & (pos < cap))


p = dict(PARAMS)
rng = np.random.default_rng(0)
idx = rng.choice(166700, size=400, replace=False)
thr = p["v_th"] - p["v_0"]
b = BrainRT(params=p)

def step(t, variant):
    sw = t % b.L; sr = (t - b.D) % b.L
    _scatter[(b.nprog, b.lanes)](b.sp[sr], b._cnt_view[sr], b.crow, b.packed, b.inc,
        POST_MASK=POST_MASK, PB=POST_BITS, NPROG=b.nprog, LANES=b.lanes,
        EBLOCK=b.eblock, num_warps=4)
    b._cnt_view[sw].zero_()
    if variant == "inkernel_zero":
        _membrane[(b.nblk,)](b.v, b.g, b.refr, b.inc, b.alive, b.rfc, b.drive, b.lam,
            b.sp[sw], b._cnt_view[sw], b.over, b.N, b.cap, b.seed, t, b.poi_amp,
            b.A, b.B, b.KBA, b.p["v_0"], b.p["v_th"], b.p["v_rst"],
            float(b.p["w_syn"]), HAS_POI=False, BLOCK=b.block, num_warps=4)
    else:
        _membrane_nozero[(b.nblk,)](b.v, b.g, b.refr, b.inc, b.alive, b.rfc, b.drive,
            b.sp[sw], b._cnt_view[sw], b.N, b.cap, b.A, b.B, b.KBA,
            b.p["v_0"], b.p["v_th"], b.p["v_rst"], float(b.p["w_syn"]),
            BLOCK=b.block, num_warps=4)
        b.inc.zero_()

# --- (1) 9스텝째 원소 단위 비교 (커널 내부 0쓰기 버전)
snaps = []
for _ in range(3):
    b.reset(); b.set_drive(idx, thr*1.2)
    for t in range(9): step(t, "inkernel_zero")
    torch.cuda.synchronize()
    inc9 = b.inc.clone()          # scatter 전 (0이어야 함)
    sw, sr = 9 % b.L, 0
    _scatter[(b.nprog, b.lanes)](b.sp[sr], b._cnt_view[sr], b.crow, b.packed, b.inc,
        POST_MASK=POST_MASK, PB=POST_BITS, NPROG=b.nprog, LANES=b.lanes,
        EBLOCK=b.eblock, num_warps=4)
    torch.cuda.synchronize()
    inc_after = b.inc.clone(); v_in = b.v.clone(); g_in = b.g.clone()
    b._cnt_view[sw].zero_()
    _membrane[(b.nblk,)](b.v, b.g, b.refr, b.inc, b.alive, b.rfc, b.drive, b.lam,
        b.sp[sw], b._cnt_view[sw], b.over, b.N, b.cap, b.seed, 9, b.poi_amp,
        b.A, b.B, b.KBA, b.p["v_0"], b.p["v_th"], b.p["v_rst"],
        float(b.p["w_syn"]), HAS_POI=False, BLOCK=b.block, num_warps=4)
    torch.cuda.synchronize()
    snaps.append(dict(inc_pre=inc9, inc=inc_after, v_in=v_in, g_in=g_in,
                      v=b.v.clone(), g=b.g.clone(), inc_post=b.inc.clone()))

a = snaps[0]
for k in ("inc_pre", "inc", "v_in", "g_in", "v", "g", "inc_post"):
    d = max(float((a[k].float() - s[k].float()).abs().max().item()) for s in snaps[1:])
    nb = max(int(((a[k].float() - s[k].float()) != 0).sum().item()) for s in snaps[1:])
    print(f"  step9 {k:>9}: 3회 최대차 {d:12.5f}  다른 원소 {nb:6d}")
j = int((a["g"] - snaps[1]["g"]).abs().argmax().item())
print(f"  최대 차 뉴런 {j}: inc={[int(s['inc'][j].item()) for s in snaps]}  "
      f"g_in={[float(s['g_in'][j].item()) for s in snaps]}  "
      f"g_out={[float(s['g'][j].item()) for s in snaps]}  "
      f"inc_post={[int(s['inc_post'][j].item()) for s in snaps]}")

# --- (2) 커널 밖 zero_() 변형의 결정론
for variant in ("inkernel_zero", "outside_zero"):
    res = []
    for _ in range(3):
        b.reset(); b.set_drive(idx, thr*1.2)
        for t in range(30): step(t, variant)
        torch.cuda.synchronize()
        res.append((b.v.clone(), b.g.clone()))
    dv = max(float((res[0][0]-r[0]).abs().max().item()) for r in res[1:])
    dg = max(float((res[0][1]-r[1]).abs().max().item()) for r in res[1:])
    print(f"  {variant:>14}: 30step x3  |dv|={dv:.3e} |dg|={dg:.3e} -> "
          f"{'DETERMINISTIC' if dv==0 and dg==0 else 'NONDETERMINISTIC'}")
