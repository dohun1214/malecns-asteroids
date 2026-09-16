"""뇌가 왜 NOOP 만 내는가 + greedy 의 도피 부호가 맞는가."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, GreedyPolicy, run_episode, make_env
from vision import Vision

ROOT = Path(__file__).resolve().parent.parent
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
rng = np.random.default_rng(3)

# ---------------------------------------------------------------- 1. 자극이 가는가
bp = BrainPolicy()
env.reset(seed=0); V.reset()
act = A.index("NOOP")
print(f"{'f':>5} {'운석':>4} {'자극셀':>6} {'자극Hz최대':>10} | "
      f"{'DNp02':>7} {'DNp11':>7} {'DNp04':>7} {'전체활성':>8} | {'액션':>6}")
n_print = 0
for f in range(2000):
    obs, rew, tr, te, info = env.step(act)
    if te or tr: env.reset(); V.reset(); continue
    if f % 4: continue
    xy, head, looms = V.looming(env.objects)
    if xy is None: act = A.index("NOOP"); continue
    ori = 0
    for o in env.objects:
        if o and type(o).__name__ == "Player": ori = int(getattr(o, "orientation", 0)); break
    idx, rates = bp.map.rates(looms, bp.b.N, bp.gain, bp.cap)
    act, ch = bp(looms, ori, A)
    if n_print < 18:
        print(f"{f:>5} {len(looms):>4} {len(idx):>6} "
              f"{(rates.max() if len(rates) else 0):>10.2f} | "
              f"{ch['p02_L']+ch['p02_R']:>7.1f} {ch['p11_L']+ch['p11_R']:>7.1f} "
              f"{ch['intensity']:>7.1f} {int((bp._tal>0).sum()):>8,} | {A[act]:>6}")
        n_print += 1
print(f"\n팽창률 통계 (이 구간): ", end="")
dth = [L["dtheta"] for L in looms] if looms else []
print(f"{[round(x,3) for x in dth]}")
print(f"gain={bp.gain} cap={bp.cap}  -> 자극 Hz = gain x dtheta")

# ---------------------------------------------------------------- 2. greedy 부호
class GreedyFlip(GreedyPolicy):
    """도피 방향을 뒤집지 않는 버전 (= 위협 쪽으로 간다). 부호 확인용 대조."""
    def __call__(self, looms, orientation, actions):
        from vision import ship_heading_deg
        lat = fore = w = 0.0
        for L in looms:
            if L["dtheta"] <= 0: continue
            psi = np.radians(L["phi_rel"])
            lat += L["dtheta"]*np.sin(psi); fore += L["dtheta"]*np.cos(psi); w += L["dtheta"]
        ch = dict(lateral=lat, fore=fore, intensity=w, norm=float(np.hypot(lat, fore)))
        if w <= 0 or ch["norm"] < 1e-9: return actions.index("NOOP"), ch
        psi_e = np.degrees(np.arctan2(lat, fore))          # +180 없음
        head = ship_heading_deg(orientation)
        tgt = int(round((((head - psi_e) % 360.0) - 90.0)/22.5)) % 16
        diff = (tgt - orientation + 8) % 16 - 8
        if abs(diff) <= 1: return actions.index("UP"), ch
        return (actions.index("LEFT") if diff > 0 else actions.index("RIGHT")), ch

print(f"\n{'정책':<22} {'평균생존(f)':>11} {'최장':>8} {'목숨':>6}")
for name, pol in (("noop", lambda l,o,a:(a.index("NOOP"), {"norm":0.0})),
                  ("회전만(무작위)", lambda l,o,a:(int(np.random.default_rng().choice(
                       [a.index("NOOP"),a.index("LEFT"),a.index("RIGHT"),a.index("UP")])), {"norm":0.0})),
                  ("greedy 도피(+180)", GreedyPolicy()),
                  ("greedy 돌진(부호반대)", GreedyFlip())):
    r = np.random.default_rng(999)
    rs = [run_episode(env, pol, V, A, max_frames=6000, rng=r) for _ in range(4)]
    print(f"{name:<22} {np.mean([x['mean_life'] for x in rs]):>11.1f} "
          f"{np.mean([x['max_life'] for x in rs]):>8.1f} "
          f"{np.mean([x['n_lives'] for x in rs]):>6.1f}")
