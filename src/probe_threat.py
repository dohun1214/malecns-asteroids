"""위험 반경 R 을 눈대중으로 고르지 않기 위한 실측 (이슈 #6).

사건 정의가 쓸모 있으려면 '사건 안에 피격이 들어와야' 한다.
그래서 실제로 목숨을 잃기 직전, 접근 중인 운석이 얼마나 가까웠는지를 잰다.
R 은 그 분포에서 고른다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, ACT_EVERY, with_fire

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
rng = np.random.default_rng(12345)

all_min, pre_death, n_death, n_dec = [], [], 0, 0
for ep in range(N_EP):
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset()
    action = A.index("FIRE"); lives_prev = None; hist = []
    for f in range(MAXF):
        obs, rew, tr, te, info = env.step(action)
        if tr or te: break
        if f % ACT_EVERY: continue
        lives = info.get("lives")
        if lives_prev is not None and lives is not None and lives < lives_prev:
            n_death += 1
            # 죽기 직전 3번의 결정에서 가장 가까웠던 접근 운석
            w = [d for d in hist[-3:] if d is not None]
            if w: pre_death.append(min(w))
        lives_prev = lives
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            hist.append(None); action = A.index("FIRE"); continue
        n_dec += 1
        ds = [L["dist"] for L in looms if L["dtheta"] > 0]
        m = min(ds) if ds else None
        hist.append(m)
        if m is not None: all_min.append(m)
        action = A.index("FIRE")

a = np.array(all_min); p = np.array(pre_death)
print(f"결정 {n_dec}회, 목숨 잃음 {n_death}회, 죽기직전 표본 {len(p)}", flush=True)
print("전체 최소거리 분위수  " + "  ".join(
    f"{q}%:{np.percentile(a,q):.0f}" for q in (5,10,25,50,75,90)), flush=True)
if len(p):
    print("죽기직전 최소거리     " + "  ".join(
        f"{q}%:{np.percentile(p,q):.0f}" for q in (10,25,50,75,90,95)), flush=True)
    print(f"  평균 {p.mean():.1f}  최대 {p.max():.1f}", flush=True)
    for R in (10,15,20,25,30,40,50,60):
        print(f"  R={R:>3}: 죽기직전의 {100*(p<=R).mean():5.1f}% 를 포함, "
              f"전체 결정의 {100*(a<=R).mean():5.1f}% 가 반경 안", flush=True)
