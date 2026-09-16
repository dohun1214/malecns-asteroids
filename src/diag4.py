import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy
from lif_rt import PARAMS

bp = BrainPolicy()
C = bp.C
sec = bp.sec
print(f"sec={sec:.4f}  circuit_idx 키 {len(C)}개")
for k in ("DNp01","DNp02","DNp04","DNp11"):
    print(f"  {k}: idx={C[k]} dtype={C[k].dtype}")

# 게이트 1(a)와 똑같은 자극을 그래프 경로로 준다
P = np.load(Path(__file__).resolve().parent.parent/"graph"/"lc4_position.npz", allow_pickle=True)
pos, side, valid = P["pos"], P["side"], P["valid"]
TH = float(P["theta_L"])
ax = pos[:,0]*np.cos(TH) + pos[:,1]*np.sin(TH)
m = valid & (side == "L"); a_h = ax[m]; ii = np.flatnonzero(m)
o = np.argsort(a_h); front16 = ii[o[:16]]

for tag, idx, hz in (("앞쪽16 @80 (게이트1a 동작점)", front16, 80.0),
                     ("앞쪽16 @150", front16, 150.0),
                     ("LC4 전체 @150", C["LC4"], 150.0)):
    bp.b.reset()
    bp.b.set_poisson_rates(idx, np.full(len(idx), hz, np.float32))
    bp.b.tally.zero_(); bp.b.seed.fill_(3); bp.b.graph.replay(); torch.cuda.synchronize()
    tal = bp.b.tally.cpu().numpy()
    bp._tal = tal
    line = f"  {tag:<28}"
    for k in ("DNp01","DNp02","DNp04","DNp11"):
        line += f" {k}={tal[C[k]].mean()/sec:7.1f}"
    line += f"  | 활성 {int((tal>0).sum()):,}"
    print(line)
    ch = bp.dec.channels(bp.rate_of)
    print(f"      decoder: p02_L={ch['p02_L']:.1f} p02_R={ch['p02_R']:.1f} "
          f"p11_L={ch['p11_L']:.1f} p11_R={ch['p11_R']:.1f} p04={ch['p04']:.1f} "
          f"intensity={ch['intensity']:.1f} norm={ch['norm']:.1f}")
