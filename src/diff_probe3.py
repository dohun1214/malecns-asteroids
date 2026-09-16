import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS, _scatter, POST_BITS, POST_MASK
from verify import BrainRef

G = Path(__file__).resolve().parent.parent/"graph"
p = dict(PARAMS)
rng = np.random.default_rng(0)
idx = rng.choice(166700, size=400, replace=False)
thr = p["v_th"] - p["v_0"]
rt = BrainRT(params=p); rt.reset(); rt.set_drive(idx, thr*1.2)
ref = BrainRef(p)
drive = torch.zeros(rt.N, device="cuda"); drive[torch.from_numpy(idx).cuda().long()] = thr*1.2

spk_ref = ref.step(drive=drive)
rt.run_eager(1)
c = int(rt.cnt[0].item())
lst = rt.sp[0, :c].cpu().numpy()
ref_idx = np.flatnonzero(spk_ref.cpu().numpy())
print(f"step0: rt cnt={c}  ref={ref_idx.size}")
print(f"  집합 동일? {set(lst.tolist()) == set(ref_idx.tolist())}")
print(f"  rt 목록 중복 {c - len(set(lst.tolist()))}개,  정렬돼 있나 {bool((np.diff(lst)>0).all())}")
miss = set(ref_idx.tolist()) - set(lst.tolist())
extra = set(lst.tolist()) - set(ref_idx.tolist())
print(f"  ref 에만 {sorted(miss)[:10]}   rt 에만 {sorted(extra)[:10]}")

# 커널이 만든 그 목록을 그대로 scatter 에 넣어 gather 와 비교
crow = np.load(G/"out_crow.npy").astype(np.int64); post = np.load(G/"out_post.npy").astype(np.int64)
w = np.load(G/"out_w.npy").astype(np.float32); packed = np.load(G/"out_packed.npy").astype(np.int32)
N = crow.size-1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
order = np.argsort(post, kind="stable"); T = np.zeros(N+1, np.int64)
np.cumsum(np.bincount(post[order], minlength=N), out=T[1:])
W = torch.sparse_csr_tensor(torch.from_numpy(T).cuda(), torch.from_numpy(pre[order]).cuda(),
                            torch.from_numpy(w[order]).cuda(), size=(N,N), dtype=torch.float32, device="cuda")
crow_t = torch.from_numpy(crow.astype(np.int32)).cuda(); packed_t = torch.from_numpy(packed).cuda()

for label, arr in (("커널이 만든 순서", lst), ("같은 집합 정렬본", np.sort(lst))):
    s = torch.zeros(N, device="cuda"); s[torch.from_numpy(arr.astype(np.int64)).cuda()] = 1.0
    refv = torch.mv(W, s)
    sp = torch.zeros(rt.cap, dtype=torch.int32, device="cuda"); sp[:len(arr)] = torch.from_numpy(arr.astype(np.int32)).cuda()
    cnt = torch.tensor([len(arr)], dtype=torch.int32, device="cuda")
    inc = torch.zeros(N, dtype=torch.int32, device="cuda")
    _scatter[(rt.nprog, rt.lanes)](sp, cnt, crow_t, packed_t, inc, POST_MASK=POST_MASK,
        PB=POST_BITS, NPROG=rt.nprog, LANES=rt.lanes, EBLOCK=rt.eblock, num_warps=4)
    torch.cuda.synchronize()
    d = (inc.float()-refv).abs()
    nb = int((d>0).sum())
    print(f"  [{label}] 불일치 {nb}개 max {d.max().item():.1f}")
    if nb:
        j = int(d.argmax().item())
        e = np.flatnonzero(post == j)
        fired = set(arr.tolist())
        contrib = [(int(pre[x]), float(w[x])) for x in e if int(pre[x]) in fired]
        print(f"     뉴런 {j}: scatter {inc[j].item()} vs gather {refv[j].item():.0f}; "
              f"기여 프리 {contrib}")
        for pr,_ in contrib:
            deg = int(crow[pr+1]-crow[pr])
            print(f"       pre {pr}: 출차수 {deg}, 목록 위치 {int(np.flatnonzero(arr==pr)[0])}")
