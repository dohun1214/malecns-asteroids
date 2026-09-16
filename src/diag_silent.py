"""그래프 replay 에서 자극이 왜 안 먹히는가. eager 와 나란히 놓고 가른다."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
C = dict(np.load(ROOT/"graph"/"circuit_idx.npz"))
LC4 = C["LC4"]
STEPS = 333
sec = STEPS*PARAMS["dt"]/1000.0

def report(tag, b):
    tal = b.tally
    print(f"  {tag:<34} tally>0 {int((tal>0).sum()):>7,}  "
          f"cnt합 {int(b.cnt.sum().item()):>5}  "
          f"LC4 발화 {int((tal[torch.as_tensor(LC4).cuda()]>0).sum()):>4}/{len(LC4)}  "
          f"DNp04 {float(tal[torch.as_tensor(C['DNp04']).cuda()].float().mean())/sec:7.1f}Hz")

print("=== A. eager (게이트 1에서 쓰던 경로) ===")
b = BrainRT(params=PARAMS)
for tag, idx, hz in (("LC4 전체 126 @ 15Hz", LC4, 15.0),
                     ("LC4 35개 @ 15Hz", LC4[:35], 15.0),
                     ("LC4 35개 @ 150Hz", LC4[:35], 150.0)):
    b.reset(); b.set_poisson(idx, hz); b.run_tally(STEPS); torch.cuda.synchronize()
    report(tag, b)

print("\n=== B. graph replay + set_poisson (캡처 때와 같은 API) ===")
b2 = BrainRT(params=PARAMS)
b2.reset(); b2.set_poisson(LC4, 15.0)
b2.capture(STEPS, tally=True)
for tag, idx, hz in (("LC4 전체 126 @ 15Hz", LC4, 15.0),
                     ("LC4 35개 @ 15Hz", LC4[:35], 15.0),
                     ("LC4 35개 @ 150Hz", LC4[:35], 150.0)):
    b2.reset(); b2.set_poisson(idx, hz)
    b2.tally.zero_(); b2.seed.fill_(1); b2.graph.replay(); torch.cuda.synchronize()
    report(tag, b2)

print("\n=== C. graph replay + set_poisson_rates (게임 루프가 쓰는 API) ===")
for tag, idx, hz in (("LC4 전체 126 @ 15Hz", LC4, 15.0),
                     ("LC4 35개 @ 15Hz", LC4[:35], 15.0),
                     ("LC4 35개 @ 150Hz", LC4[:35], 150.0)):
    b2.reset(); b2.set_poisson_rates(idx, np.full(len(idx), hz, np.float32))
    b2.tally.zero_(); b2.seed.fill_(1); b2.graph.replay(); torch.cuda.synchronize()
    report(tag, b2)

print("\n=== D. 차이 추적 ===")
b2.reset(); b2.set_poisson(LC4[:35], 150.0)
lam_a = b2.lam.clone(); rfc_a = b2.rfc.clone()
b2.reset(); b2.set_poisson_rates(LC4[:35], np.full(35, 150.0, np.float32))
lam_b = b2.lam.clone(); rfc_b = b2.rfc.clone()
print(f"  lam 최대차 {float((lam_a-lam_b).abs().max()):.3e}  "
      f"lam 비영 개수 {int((lam_a>0).sum())} vs {int((lam_b>0).sum())}")
print(f"  rfc 가 0 인 뉴런 {int((rfc_a==0).sum())} vs {int((rfc_b==0).sum())}"
      f"   <- set_poisson_rates 는 rfc 를 안 건드린다")
print(f"  (Shiu 방식: 자극 뉴런은 불응기 0 이어야 한다)")
