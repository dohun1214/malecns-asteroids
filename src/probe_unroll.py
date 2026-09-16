"""333 step 언롤 캡처가 실제로 되는지 + 스텝당 비용이 얼마인지 최소 확인.
(LIF 전체가 아니라 mv + elementwise 만. 2단계 착수 전 예산 확인용)"""
import os, json, time
from pathlib import Path
import numpy as np, torch

G = Path(os.environ.get("MALECNS_GRAPH",
         Path(__file__).resolve().parent.parent.parent / "malecns-song" / "graph"))
STEPS = int(os.environ.get("STEPS", "333"))
IDX = os.environ.get("IDX_DTYPE", "int32")
_np_idx = np.int64 if IDX == "int64" else np.int32

crow_np = np.load(G/"csrT_indptr.npy"); col_np = np.load(G/"csrT_indices.npy")
val_np  = np.load(G/"csrT_val.npy")
N, nnz = crow_np.size-1, col_np.size
crow = torch.from_numpy(crow_np.astype(_np_idx)).cuda()
col  = torch.from_numpy(col_np.astype(_np_idx)).cuda()
val  = torch.from_numpy(val_np.astype(np.float32)).cuda()
W = torch.sparse_csr_tensor(crow, col, val, size=(N,N), dtype=torch.float32, device="cuda")
print(f"N={N} nnz={nnz} idx={IDX} steps={STEPS}")
print(f"VRAM after W: {torch.cuda.memory_allocated()/2**20:.1f} MB")

v = torch.full((N,), -52.0, device="cuda")
g = torch.zeros(N, device="cuda")
alive = torch.ones(N, device="cuda")
inc = torch.zeros(N, device="cuda")
A,B,KBA,v0,vth,vrst = 0.99005, 0.96079, -0.03903, -52.0, -45.0, -52.0

def step():
    g.mul_(B).add_(inc)
    v.mul_(A).add_(v0*(1-A)).add_(g*KBA)
    s = (v > vth).float() * alive
    v.copy_(torch.where(s>0, torch.full_like(v, vrst), v))
    inc.copy_(torch.mv(W, s))

for _ in range(5): step()
torch.cuda.synchronize()

t0=time.perf_counter()
for _ in range(STEPS): step()
torch.cuda.synchronize()
t_eager=(time.perf_counter()-t0)*1000
print(f"eager  {STEPS} step: {t_eager:.2f} ms  ({t_eager/STEPS:.4f} ms/step)")

gr = torch.cuda.CUDAGraph()
t0=time.perf_counter()
try:
    with torch.cuda.graph(gr):
        for _ in range(STEPS): step()
    ok=True; err=None
except Exception as e:
    ok=False; err=f"{type(e).__name__}: {e}"
t_cap=(time.perf_counter()-t0)*1000
print(f"capture ok={ok} err={err}  (capture 소요 {t_cap:.0f} ms)")
res=dict(N=int(N),nnz=int(nnz),idx=IDX,steps=STEPS,eager_ms=t_eager,capture_ok=ok,capture_err=err)
if ok:
    gr.replay(); torch.cuda.synchronize()
    for _ in range(2): gr.replay()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(10): gr.replay()
    torch.cuda.synchronize()
    t_rep=(time.perf_counter()-t0)/10*1000
    print(f"graph replay {STEPS} step: {t_rep:.2f} ms  ({t_rep/STEPS:.4f} ms/step)")
    print(f"프레임 예산 66.7ms 대비: {t_rep/66.7:.2f}x   (목표 <50ms)")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/2**20:.1f} MB")
    res.update(replay_ms=t_rep, per_step_ms=t_rep/STEPS,
               vram_peak_mb=torch.cuda.max_memory_allocated()/2**20)
Path(__file__).resolve().parent.parent.joinpath("out/probe_unroll.json").write_text(
    json.dumps(res,indent=2),encoding="utf-8")
print("DONE")
