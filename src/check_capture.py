"""
05-기술-구현-가이드.md 2.2 - 착수 전 확인.

(1) torch.sparse CSR mv 가 torch.cuda.graph 안에서 캡처되는가?
(2) torch.sparse_csr_tensor 가 values 버퍼를 앨리어싱하는가?
(3) (파생) 캡처된 그래프의 replay 가 그래프 밖에서 바뀐 입력/가중치를 반영하는가?
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

G = Path(os.environ.get("MALECNS_GRAPH",
                        Path(__file__).resolve().parent.parent.parent / "malecns-song" / "graph"))

R = {}
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {torch.cuda.get_device_name(0)}")
print(f"graph dir: {G}")

crow_np = np.load(G / "csrT_indptr.npy")
col_np  = np.load(G / "csrT_indices.npy")
val_np  = np.load(G / "csrT_val.npy")
N   = crow_np.size - 1
nnz = col_np.size
print(f"N={N}  nnz={nnz}  crow={crow_np.dtype} col={col_np.dtype} val={val_np.dtype}")
R["N"], R["nnz"] = int(N), int(nnz)

IDX = os.environ.get("IDX_DTYPE", "int64")
_np_idx = np.int64 if IDX == "int64" else np.int32
print(f"index dtype: {IDX}  (crow/col 는 반드시 같은 dtype 이어야 한다)")
R["idx_dtype"] = IDX
crow = torch.from_numpy(crow_np.astype(_np_idx)).cuda()
col  = torch.from_numpy(col_np.astype(_np_idx)).cuda()
val  = torch.from_numpy(val_np.astype(np.float32)).cuda()

W = torch.sparse_csr_tensor(crow, col, val, size=(N, N),
                            dtype=torch.float32, device="cuda")

# ---------------------------------------------------------------- (2) aliasing
alias_val  = (W.values().data_ptr()      == val.data_ptr())
alias_col  = (W.col_indices().data_ptr() == col.data_ptr())
alias_crow = (W.crow_indices().data_ptr()== crow.data_ptr())
print(f"\n[2] ALIASING  values={alias_val}  col={alias_col}  crow={alias_crow}")
R["alias_values"], R["alias_col"], R["alias_crow"] = alias_val, alias_col, alias_crow

# ---------------------------------------------------------------- (1) capture
x = torch.zeros(N, device="cuda")
y = torch.zeros(N, device="cuda")
# 희소 입력: 2% 발화 흉내
rng = np.random.default_rng(0)
spk = rng.choice(N, size=int(N * 0.02), replace=False)
spk_t = torch.from_numpy(spk).cuda().long()
x.index_fill_(0, spk_t, 1.0)

torch.cuda.synchronize()
for _ in range(5):
    y.copy_(torch.mv(W, x))
torch.cuda.synchronize()
ref = y.clone()
print(f"    warm-up mv ok, |y|_1 = {ref.abs().sum().item():.3f}, nonzero={int((ref!=0).sum())}")

capture_ok, capture_err = False, None
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        y.copy_(torch.mv(W, x))
    g.replay()
    torch.cuda.synchronize()
    capture_ok = True
except Exception as e:
    capture_err = f"{type(e).__name__}: {e}"
print(f"\n[1] CAPTURE  ok={capture_ok}  err={capture_err}")
R["capture_ok"], R["capture_err"] = capture_ok, capture_err

if capture_ok:
    same = torch.allclose(y, ref)
    print(f"    replay == eager : {same}  maxdiff={(y-ref).abs().max().item():.3e}")
    R["replay_matches_eager"] = bool(same)

    # (3a) 그래프 밖에서 입력 x 를 바꾸면 replay 가 반영하는가
    x.zero_()
    g.replay(); torch.cuda.synchronize()
    zero_in = float(y.abs().sum().item())
    x.index_fill_(0, spk_t, 1.0)
    g.replay(); torch.cuda.synchronize()
    back = torch.allclose(y, ref)
    print(f"    input x 반영: x=0 -> |y|={zero_in:.3e} ; x복구 -> ref일치={back}")
    R["graph_reads_live_input"] = bool(zero_in == 0.0 and back)

    # (3b) alive 마스크 (lesion 토글) 가 replay 에 반영되는가  -- 05 2.3
    alive = torch.ones(N, device="cuda")
    s = torch.zeros(N, device="cuda")
    y2 = torch.zeros(N, device="cuda")
    for _ in range(5):
        s.copy_(x * alive); y2.copy_(torch.mv(W, s))
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    tog_ok, tog_err = False, None
    try:
        with torch.cuda.graph(g2):
            s.copy_(x * alive)
            y2.copy_(torch.mv(W, s))
        g2.replay(); torch.cuda.synchronize()
        base = y2.clone()
        kill = spk_t[:len(spk_t)//2]
        alive.index_fill_(0, kill, 0.0)
        g2.replay(); torch.cuda.synchronize()
        d1 = (y2 - base).abs().max().item()
        alive.index_fill_(0, kill, 1.0)
        g2.replay(); torch.cuda.synchronize()
        d2 = (y2 - base).abs().max().item()
        tog_ok = (d1 > 0.0) and (d2 == 0.0)
        print(f"    alive.index_fill_ 토글: 끔 maxdiff={d1:.3f} / 복구 maxdiff={d2:.3e} -> {tog_ok}")
    except Exception as e:
        tog_err = f"{type(e).__name__}: {e}"
        print(f"    alive 토글 캡처 실패: {tog_err}")
    R["alive_toggle_works"], R["alive_toggle_err"] = tog_ok, tog_err

    # (3c) values 를 그래프 밖에서 갈아끼우면 replay 가 반영하는가 (rewire 토글, 05 2.7)
    if alias_val:
        save = val.clone()
        val.mul_(0.5)
        g.replay(); torch.cuda.synchronize()
        half = torch.allclose(y, ref * 0.5, atol=1e-3, rtol=1e-4)
        val.copy_(save)
        g.replay(); torch.cuda.synchronize()
        restored = torch.allclose(y, ref)
        print(f"    values 교체 반영: half={half} restore={restored}")
        R["values_swap_works"] = bool(half and restored)

    # 타이밍
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): g.replay()
    torch.cuda.synchronize()
    t_graph = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter()
    for _ in range(100): y.copy_(torch.mv(W, x))
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - t0) / 100 * 1000
    print(f"\n    mv 1회: eager {t_eager:.4f} ms / graph replay {t_graph:.4f} ms")
    R["mv_eager_ms"], R["mv_graph_ms"] = t_eager, t_graph

Path(__file__).resolve().parent.parent.joinpath("out").mkdir(exist_ok=True)
Path(__file__).resolve().parent.parent.joinpath("out/check_capture.json").write_text(
    json.dumps(R, indent=2), encoding="utf-8")
print("\nDONE")
