"""커널 검증을 동역학에서 분리한다.

같은 스파이크 벡터 s 를 줬을 때
  pre-major scatter (Triton, int32 원자 누산)  vs  post-major gather (cuSPARSE fp32 mv)
가 '정확히' 같은 inc 를 내야 한다. 가중치가 정수라 fp32 부분합이 전부 정확 표현
범위(|합| <= 12만 << 2^24) 안이므로 반올림 여지가 없다. 조금이라도 다르면 커널 버그다.

이게 통과하면, 긴 시뮬에서 생기는 차이는 연결 연산이 아니라
막전위 갱신의 부동소수점 결합순서(FMA) 차이 -> 스파이크 타이밍 혼돈이다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, triton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import _scatter, POST_BITS, POST_MASK

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"

crow_np = np.load(G/"out_crow.npy").astype(np.int64)
post_np = np.load(G/"out_post.npy").astype(np.int64)
w_np    = np.load(G/"out_w.npy").astype(np.float32)
packed  = np.load(G/"out_packed.npy").astype(np.int32)
N = crow_np.size - 1

pre_np = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow_np))
order  = np.argsort(post_np, kind="stable")
Tind   = np.zeros(N+1, dtype=np.int64)
np.cumsum(np.bincount(post_np[order], minlength=N), out=Tind[1:])
W = torch.sparse_csr_tensor(torch.from_numpy(Tind).cuda(),
                            torch.from_numpy(pre_np[order]).cuda(),
                            torch.from_numpy(w_np[order]).cuda(),
                            size=(N, N), dtype=torch.float32, device="cuda")

crow_t   = torch.from_numpy(crow_np.astype(np.int32)).cuda()
packed_t = torch.from_numpy(packed).cuda()
inc      = torch.zeros(N, dtype=torch.int32, device="cuda")

NPROG, LANES, EBLOCK = 512, 16, 64
R = {"N": int(N), "nnz": int(packed.size)}
print(f"N={N} nnz={packed.size}")

ok_all = True
for frac in (0.0, 0.0001, 0.005, 0.02, 0.10):
    rng = np.random.default_rng(int(frac*1e6) + 7)
    k = int(N*frac)
    idx = np.sort(rng.choice(N, size=k, replace=False)) if k else np.empty(0, np.int64)
    s = torch.zeros(N, device="cuda"); 
    if k: s[torch.from_numpy(idx).cuda().long()] = 1.0
    ref = torch.mv(W, s)

    sp  = torch.zeros(max(k, 1), dtype=torch.int32, device="cuda")
    if k: sp[:k] = torch.from_numpy(idx.astype(np.int32)).cuda()
    cnt = torch.tensor([k], dtype=torch.int32, device="cuda")
    inc.zero_()
    _scatter[(NPROG, LANES)](sp, cnt, crow_t, packed_t, inc,
                             POST_MASK=POST_MASK, PB=POST_BITS, NPROG=NPROG,
                             LANES=LANES, EBLOCK=EBLOCK, num_warps=4)
    torch.cuda.synchronize()
    got = inc.float()
    d = (got - ref).abs().max().item()
    nz = int((ref != 0).sum())
    exact = (d == 0.0)
    ok_all &= exact
    print(f"  발화 {k:>6,} ({frac*100:5.2f}%)  영향 받은 뉴런 {nz:>6,}  "
          f"|diff|max = {d:.1f}  {'EXACT' if exact else 'MISMATCH'}")
    R[f"frac_{frac}"] = dict(k=k, affected=nz, maxdiff=d, exact=bool(exact))

# 허브 뉴런 단독 — 부하 불균형 경로가 정확한지
deg = np.diff(crow_np)
hub = int(np.argmax(deg))
s = torch.zeros(N, device="cuda"); s[hub] = 1.0
ref = torch.mv(W, s)
sp = torch.tensor([hub], dtype=torch.int32, device="cuda")
cnt = torch.tensor([1], dtype=torch.int32, device="cuda")
inc.zero_()
_scatter[(NPROG, LANES)](sp, cnt, crow_t, packed_t, inc, POST_MASK=POST_MASK,
                         PB=POST_BITS, NPROG=NPROG, LANES=LANES, EBLOCK=EBLOCK, num_warps=4)
torch.cuda.synchronize()
d = (inc.float() - ref).abs().max().item()
ok_all &= (d == 0.0)
print(f"  허브 뉴런 {hub} 단독 (출차수 {deg[hub]:,})  |diff|max = {d:.1f}  "
      f"{'EXACT' if d==0 else 'MISMATCH'}")
R["hub"] = dict(idx=hub, deg=int(deg[hub]), maxdiff=d)
R["all_exact"] = bool(ok_all)
print(f"\nscatter 커널: {'모든 조건 EXACT' if ok_all else '불일치 있음'}")
(ROOT/"out"/"test_scatter.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
