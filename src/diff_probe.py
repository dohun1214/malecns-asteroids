"""스파이크는 같은데 v 가 갈라지는 첫 스텝을 찾아 원인을 특정한다."""
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

for t in range(14):
    # --- 이번 스텝에 '도착할' 전류를 양쪽에서 같은 위상으로 뽑는다
    slot = t % 9
    inc_ref = ref.ring[slot].clone()                 # ref 가 지금 읽을 것
    spikes_prev_ref = None
    # rt: scatter 만 먼저 돌려서 inc 를 채운다
    from lif_rt import _scatter, POST_BITS, POST_MASK
    _scatter[(rt.nprog, rt.lanes)](rt.sp[slot], rt._cnt_view[slot], rt.crow, rt.packed,
                                   rt.inc, POST_MASK=POST_MASK, PB=POST_BITS,
                                   NPROG=rt.nprog, LANES=rt.lanes, EBLOCK=rt.eblock, num_warps=4)
    torch.cuda.synchronize()
    inc_rt = rt.inc.float().clone()
    rt.inc.zero_()                                    # 원상 복구 (아래에서 정상 step 을 다시 돈다)
    dinc = (inc_rt - inc_ref).abs()
    nbad = int((dinc > 0).sum()); mx = float(dinc.max().item())
    cnt_ref_prev = None
    print(f"t={t:2d}  cnt[slot{slot}]={int(rt.cnt[slot].item()):5d}  "
          f"inc 불일치 뉴런 {nbad:6d}  max {mx:8.1f}   "
          f"|dv|max={float((rt.v-ref.v).abs().max().item()):.3e}  "
          f"|dg|max={float((rt.g-ref.g).abs().max().item()):.3e}", flush=True)
    if nbad and t <= 10:
        j = int(dinc.argmax().item())
        print(f"     뉴런 {j}: scatter {inc_rt[j].item():.0f} vs gather {inc_ref[j].item():.0f}")
        # 이 뉴런에 실제로 입력을 준 프리시냅틱은?
        crow = np.load(Path(__file__).resolve().parent.parent/"graph"/"out_crow.npy")
        post = np.load(Path(__file__).resolve().parent.parent/"graph"/"out_post.npy")
        w    = np.load(Path(__file__).resolve().parent.parent/"graph"/"out_w.npy")
        pre  = np.repeat(np.arange(crow.size-1, dtype=np.int64), np.diff(crow))
        e = np.flatnonzero(post == j)
        c = int(rt.cnt[slot].item())
        fired = set(rt.sp[slot, :c].cpu().numpy().tolist())
        contrib = [(int(pre[x]), float(w[x])) for x in e if int(pre[x]) in fired]
        print(f"     발화한 프리시냅틱 {len(contrib)}개, 합 {sum(c2 for _,c2 in contrib):.0f}")
        print(f"     상위: {sorted(contrib, key=lambda z:-abs(z[1]))[:6]}")
        break
    ref.step(drive=drive)
    rt.run_eager(1)
