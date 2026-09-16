"""게이트 2 전 최종 점검: 뇌가 실제로 어떤 액션을 얼마나 내는가."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, GreedyPolicy, make_env, with_fire
from vision import Vision

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy()
print(f"th50={bp.gain}  cap={bp.cap}  min_intensity={bp.dec.min_intensity}")
for tag, pol in (("brain", bp), ("greedy", GreedyPolicy())):
    env.reset(seed=0); V.reset()
    if tag == "brain": bp.b.reset(); bp.frame = 0
    act = A.index("FIRE"); acts = []; hz = []; nrm = []; d02 = []; d11 = []
    f = 0
    while len(acts) < 250 and f < 8000:
        obs, rew, tr, te, info = env.step(act); f += 1
        if te or tr: env.reset(); V.reset(); continue
        if f % 4: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None: act = A.index("FIRE"); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player": ori = int(getattr(o,"orientation",0)); break
        if tag == "brain":
            idx, r = bp.map.rates(looms, bp.b.N, bp.gain, bp.cap)
            hz.append(float(r.max()) if len(r) else 0.0)
        a, ch = pol(looms, ori, A)
        acts.append(A[a]); nrm.append(ch["norm"])
        if tag == "brain":
            d02.append(ch["p02_L"]+ch["p02_R"]); d11.append(ch["p11_L"]+ch["p11_R"])
        act = with_fire(a, A)
    c = Counter(acts)
    print(f"\n[{tag}] 결정 {len(acts)}회")
    print(f"  액션: { {k: round(v/len(acts),3) for k,v in c.most_common()} }")
    print(f"  norm 중앙값 {np.median(nrm):.2f}  0인 비율 {np.mean(np.array(nrm)<1e-9)*100:.0f}%")
    if hz:
        print(f"  자극 Hz 중앙값 {np.median(hz):.1f} 최대 {max(hz):.1f}")
        print(f"  DNp02 평균 {np.mean(d02):.1f} Hz   DNp11 평균 {np.mean(d11):.1f} Hz")
