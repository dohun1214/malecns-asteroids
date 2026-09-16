"""실제 플레이에서 팽창률 분포를 재서 자극 세기 상수를 '측정'으로 정한다.
추측한 gain 을 쓰면 순환논증 회계의 '부과한 것' 칸이 커진다."""
import sys, json
from pathlib import Path
import numpy as np
from ocatari.core import OCAtari
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision import Vision

ROOT = Path(__file__).resolve().parent.parent
env = OCAtari("ALE/Asteroids-v5", mode="ram", hud=False, frameskip=1,
              repeat_action_probability=0.0)
V = Vision()
rng = np.random.default_rng(0)
env.reset(seed=0); V.reset()
dth, ths, phis, nast, noship = [], [], [], [], 0
for f in range(12000):
    obs, rew, trunc, term, info = env.step(int(rng.integers(0, env.action_space.n)))
    if term or trunc:
        env.reset(); V.reset(); continue
    if f % 4: continue
    xy, head, looms = V.looming(env.objects)
    if xy is None: noship += 1; continue
    nast.append(len(looms))
    for L in looms:
        dth.append(L["dtheta"]); ths.append(L["theta"]); phis.append(L["phi_rel"])
dth = np.array(dth); ths = np.array(ths); phis = np.array(phis)
pos = dth[dth > 0]
print(f"결정 프레임 {len(nast):,}개 (Player 부재 {noship:,})  운석 표본 {len(dth):,}")
print(f"각크기 theta (도): 중앙값 {np.median(ths):.1f}  90%ile {np.percentile(ths,90):.1f}  최대 {ths.max():.1f}")
print(f"팽창률 dtheta (도/프레임): 양수 비율 {len(pos)/len(dth)*100:.0f}%")
for q in (50, 75, 90, 95, 99):
    print(f"   양수만 {q}%ile = {np.percentile(pos,q):8.3f}")
print(f"   최대 {pos.max():.3f}")
print(f"방위각 분포: |phi| 중앙값 {np.median(np.abs(phis)):.0f}도, "
      f"우측(phi>0) 비율 {np.mean(phis>0)*100:.0f}%")
print(f"화면당 운석 평균 {np.mean(nast):.1f}")
# 자극 세기 상수: 95%ile 팽창률이 게이트 1(a) 동작점(80Hz)이 되도록
ref = float(np.percentile(pos, 95))
gain = 80.0/ref
print(f"\n=> gain = 80 Hz / {ref:.3f}(95%ile) = {gain:.1f} Hz per (도/프레임)")
print(f"   상한 150 Hz (drive_sweep 에서 n=16 기준 포화 34%)")
(ROOT/"out"/"loom_stats.json").write_text(json.dumps(dict(
    n=len(dth), theta_med=float(np.median(ths)), pos_frac=float(len(pos)/len(dth)),
    p95=ref, gain=float(gain), ast_per_frame=float(np.mean(nast)),
    noship=int(noship), frames=len(nast)), indent=2), encoding="utf-8")
