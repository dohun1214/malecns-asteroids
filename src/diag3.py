import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, with_fire
from vision import Vision

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy()   # th50, cap 은 기본값 사용
for FIRE in (False, True):
    env.reset(seed=0); V.reset(); bp.b.reset(); bp.frame = 0
    act = A.index("FIRE" if FIRE else "NOOP")
    print(f"\n=== FIRE={FIRE} ===")
    print(f"{'f':>5} {'운석':>4} {'루밍':>4} {'셀':>4} {'Hz최대':>7} {'lam합':>8} "
          f"{'활성':>7} {'p02L':>6} {'p02R':>6} {'p11L':>6} {'p11R':>6} {'p04':>6} {'액션':>8}")
    n = 0; f = 0
    while n < 12 and f < 3000:
        obs, rew, tr, te, info = env.step(act); f += 1
        if te or tr: env.reset(); V.reset(); continue
        if f % 4: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            act = A.index("FIRE" if FIRE else "NOOP"); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player": ori = int(getattr(o,"orientation",0)); break
        idx, rates = bp.map.rates(looms, bp.b.N, bp.gain, bp.cap)
        nl = sum(1 for L in looms if L["dtheta"] > 0)
        a, ch = bp(looms, ori, A)
        print(f"{f:>5} {len(looms):>4} {nl:>4} {len(idx):>4} "
              f"{(rates.max() if len(rates) else 0):>7.1f} {float(bp.b.lam.sum()):>8.4f} "
              f"{int((bp._tal>0).sum()):>7,} "
              f"{ch['p02_L']:>6.1f} {ch['p02_R']:>6.1f} {ch['p11_L']:>6.1f} {ch['p11_R']:>6.1f} "
              f"{ch['intensity']:>6.1f} {A[a]:>8}")
        act = with_fire(a, A) if FIRE else a
        n += 1
