"""게임 안에서 뇌의 디코딩이 실제 위협 방향을 따라가는가 — 결정적 측정.

기하학으로 계산한 위협 방위각(정답) 과 뇌가 낸 방위각을 나란히 놓고 원형 상관을 본다.
상관이 있으면 보정 문제, 없으면 구조적 문제다.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, with_fire
from vision import Vision

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy(); bp.dec.tau = 1.0     # 평활 끄고 원자료로
env.reset(seed=0); V.reset(); bp.b.reset(); bp.frame = 0
act = A.index("FIRE")
truth, dec, f, rows = [], [], 0, []
while len(truth) < 400 and f < 12000:
    obs, rew, tr, te, info = env.step(act); f += 1
    if te or tr: env.reset(); V.reset(); bp.dec.reset(); continue
    if f % 4: continue
    xy, head, looms = V.looming(env.objects)
    if xy is None: act = A.index("FIRE"); continue
    ori = 0
    for o in env.objects:
        if o and type(o).__name__ == "Player": ori = int(getattr(o,"orientation",0)); break
    # 정답: 자극에 실제로 쓰인 운석들의 세기 가중 방위각
    lat = fo = w = 0.0
    for L in looms:
        if L["dtheta"] <= 0: continue
        r = bp.cap*L["theta"]/(L["theta"] + bp.gain)
        psi = np.radians(L["phi_rel"])
        lat += r*np.sin(psi); fo += r*np.cos(psi); w += r
    if w <= 0: act = A.index("FIRE"); continue
    a, ch = bp(looms, ori, A)
    truth.append(np.arctan2(lat, fo))
    dec.append(np.arctan2(ch["lateral"], ch["fore"]))
    rows.append((ch["p02_L"], ch["p02_R"], ch["p11_L"], ch["p11_R"], lat/w, fo/w))
    act = with_fire(a, A)

t = np.array(truth); d = np.array(dec); M = np.array(rows)
def circ_corr(x, y):
    xs, ys = x - np.angle(np.mean(np.exp(1j*x))), y - np.angle(np.mean(np.exp(1j*y)))
    return float(np.sum(np.sin(xs)*np.sin(ys)) /
                 np.sqrt(np.sum(np.sin(xs)**2)*np.sum(np.sin(ys)**2)))
print(f"표본 {len(t)}개")
print(f"정답 방위각: 평균 {np.degrees(np.angle(np.mean(np.exp(1j*t)))):+.0f}도, "
      f"집중도 {abs(np.mean(np.exp(1j*t))):.2f}  (0=고름, 1=한쪽)")
print(f"뇌 방위각:   평균 {np.degrees(np.angle(np.mean(np.exp(1j*d)))):+.0f}도, "
      f"집중도 {abs(np.mean(np.exp(1j*d))):.2f}")
print(f"원형 상관 r = {circ_corr(t, d):+.3f}   <- 0 이면 뇌가 방향을 전혀 못 읽는 것")
err = np.degrees(np.angle(np.exp(1j*(d-t))))
print(f"각오차: 중앙값 {np.median(np.abs(err)):.0f}도  (무작위면 90도)")
print(f"\nDN 평균 Hz: p02_L {M[:,0].mean():.1f}  p02_R {M[:,1].mean():.1f}  "
      f"p11_L {M[:,2].mean():.1f}  p11_R {M[:,3].mean():.1f}")
print(f"  p02 가 p11 보다 큰 비율 {np.mean(M[:,0]+M[:,1] > M[:,2]+M[:,3])*100:.0f}%")
print(f"  R 이 L 보다 큰 비율     {np.mean(M[:,1]+M[:,3] > M[:,0]+M[:,2])*100:.0f}%")
print(f"정답 위협의 좌우 성분 평균 {M[:,4].mean():+.2f}, 전후 성분 평균 {M[:,5].mean():+.2f}")
# 정답을 구간별로 나눠 뇌 채널이 따라가는지
print(f"\n{'정답 |phi| 구간':<16} {'n':>4} {'p02합':>7} {'p11합':>7} {'fore채널':>9}")
ap = np.abs(np.degrees(t))
for lo, hi in ((0,45),(45,90),(90,135),(135,180)):
    m = (ap >= lo) & (ap < hi)
    if m.sum() < 3: continue
    print(f"  {lo:>3}~{hi:<3}도{'':<6} {int(m.sum()):>4} {M[m,0].mean()+M[m,1].mean():>7.1f} "
          f"{M[m,2].mean()+M[m,3].mean():>7.1f} "
          f"{(M[m,0]+M[m,1]-M[m,2]-M[m,3]).mean():>9.1f}")

# ---- 성분별 진단: 좌우와 전후 중 어느 쪽이 뒤집혔는가
lat_d = M[:,1] + M[:,3] - M[:,0] - M[:,2]      # 뇌의 좌우 채널
fore_d = M[:,0] + M[:,1] - M[:,2] - M[:,3]     # 뇌의 전후 채널
lat_t, fore_t = np.sin(t), np.cos(t)           # 정답 성분
print(f"\n성분별 상관 (양수여야 정상)")
print(f"  좌우: 뇌 vs 정답 r = {np.corrcoef(lat_d, lat_t)[0,1]:+.3f}"
      f"   (뇌 좌우 |평균| {np.abs(lat_d).mean():.1f})")
print(f"  전후: 뇌 vs 정답 r = {np.corrcoef(fore_d, fore_t)[0,1]:+.3f}"
      f"   (뇌 전후 |평균| {np.abs(fore_d).mean():.1f})")
print(f"\n정답 좌우 부호별 뇌 좌우 채널")
for lo, hi, lbl in ((-180,-5,"정답 왼쪽"), (5,180,"정답 오른쪽")):
    m = (np.degrees(t) >= lo) & (np.degrees(t) <= hi)
    if m.sum() < 3: continue
    print(f"  {lbl} (n={int(m.sum()):>3}): 뇌 좌우 채널 평균 {lat_d[m].mean():+7.2f}"
          f"   p02L {M[m,0].mean():5.1f} p02R {M[m,1].mean():5.1f}"
          f"   p11L {M[m,2].mean():5.1f} p11R {M[m,3].mean():5.1f}")
