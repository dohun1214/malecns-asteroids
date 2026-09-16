"""E 의 graph/ 산출물 -> 이벤트 구동용 pre-major(출력 엣지) 패킹 배열.

post-major CSR (inc = W @ s) 는 발화 여부와 무관하게 매 스텝 전체 nnz 를 읽는다.
이벤트 구동은 '발화한 뉴런의 출력 엣지만' 읽어야 하므로 pre-major 가 필요하다.
E 의 csr_indptr/csr_indices/csr_syn 이 이미 pre-major 다 (lif.py _pre_edge_list 참조).

패킹: int32 = (signed_weight << 18) | (post & 0x3FFFF)
  post   하위 18 bit : 0..262143  >= N=166,700
  weight 상위 14 bit : -8192..8191 (시냅스 수 x 부호)

가중치를 '상위' 비트에 두는 게 중요하다. 커널에서 산술 우측 시프트(p >> 18)가
부호 확장을 공짜로 해준다. 반대로 post 를 상위에 두면 post >= 131072 인 엣지에서
최상위 비트가 서서 int32 가 음수가 되고, 산술 시프트가 post 를 망가뜨린다.
(torch 에 실용적인 uint32 텐서가 없어서 int32 로 다뤄야 한다)
"""
import os, json
from pathlib import Path
import numpy as np

SRC = Path(os.environ.get("MALECNS_GRAPH",
      Path(__file__).resolve().parent.parent.parent / "malecns-song" / "graph"))
DST = Path(__file__).resolve().parent.parent / "graph"
DST.mkdir(exist_ok=True)

indptr = np.load(SRC/"csr_indptr.npy")     # pre-major row pointer
post   = np.load(SRC/"csr_indices.npy")    # postsynaptic index
syn    = np.load(SRC/"csr_syn.npy")        # synapse count (unsigned)
sign   = np.load(SRC/"sign.npy")           # per-neuron sign, int8
N   = indptr.size - 1
nnz = post.size
print(f"N={N}  nnz={nnz}")
print(f"indptr={indptr.dtype} post={post.dtype} syn={syn.dtype} sign={sign.dtype}")

pre = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
w   = syn.astype(np.int64) * sign[pre].astype(np.int64)

print(f"\n엣지 시냅스 수  min={syn.min()} max={syn.max()} mean={syn.mean():.2f} median={np.median(syn):.0f}")
print(f"부호 있는 가중치 min={w.min()} max={w.max()}")
for b in (8, 9, 12, 14, 16):
    lo, hi = -(1 << (b-1)), (1 << (b-1)) - 1
    over = int(((w < lo) | (w > hi)).sum())
    print(f"  {b:2d}bit signed [{lo},{hi}] : 초과 엣지 {over:,} ({over/nnz*100:.4f}%)")

POST_BITS, W_BITS = 18, 14
assert N < (1 << POST_BITS)
WLO, WHI = -(1 << (W_BITS-1)), (1 << (W_BITS-1)) - 1
n_clip = int(((w < WLO) | (w > WHI)).sum())
w_c = np.clip(w, WLO, WHI).astype(np.int64)
packed = ((w_c << POST_BITS) | (post.astype(np.int64) & ((1 << POST_BITS) - 1))
          ).astype(np.int32)

# 왕복 검증 — 커널이 하는 것과 정확히 같은 연산으로
back_post = (packed.astype(np.int64) & ((1 << POST_BITS) - 1))
back_w = packed.astype(np.int64) >> POST_BITS          # numpy 는 산술 시프트
assert (back_post == post.astype(np.int64)).all(), "post 왕복 실패"
assert (back_w == w_c).all(), "weight 왕복 실패"
print(f"\n패킹 왕복 검증 OK.  clip 된 엣지 {n_clip:,} ({n_clip/nnz*100:.4f}%)")

out_deg = np.diff(indptr)
print(f"출차수  mean={out_deg.mean():.1f} median={np.median(out_deg):.0f} max={out_deg.max():,}")
print(f"  -> scatter 부하 불균형 {out_deg.max()/max(np.median(out_deg),1):.0f}배 (2D 그리드 필요)")

np.save(DST/"out_crow.npy",   indptr.astype(np.int32))
np.save(DST/"out_packed.npy", packed.astype(np.int32))
np.save(DST/"out_post.npy",   post.astype(np.int32))    # 레퍼런스 경로용
np.save(DST/"out_w.npy",      w_c.astype(np.float32))   # 레퍼런스 경로용 (시냅스 수, w_syn 곱하기 전)
np.save(DST/"sign.npy",       sign)
(DST/"prep_report.json").write_text(json.dumps(dict(
    N=int(N), nnz=int(nnz), post_bits=POST_BITS, w_bits=W_BITS,
    w_min=int(w.min()), w_max=int(w.max()), n_clipped=n_clip,
    out_deg_max=int(out_deg.max()), out_deg_median=int(np.median(out_deg)),
    bytes_packed=int(packed.nbytes)), indent=2), encoding="utf-8")
print(f"\nwrote {DST}  (packed {packed.nbytes/2**20:.1f} MB, crow {indptr.nbytes/2**20:.1f} MB)")
