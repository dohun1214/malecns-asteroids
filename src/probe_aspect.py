"""아타리 2600 픽셀은 정사각이 아니다. 방위각을 픽셀 좌표에서 그냥 atan2 로 재면 틀린다.

probe_heading 결과: orient 0/4/8/12(위/왼/아래/오른)는 정확한데 중간 각도가 최대 20도 어긋났고,
세로 변위가 가로의 약 2배였다. 배의 '진짜' heading 은 90 + 22.5*orient 여야 하므로,
그 가정 위에서 세로 팽창 계수 a 를 맞춘다:

    진짜 방향 = atan2(dy_math / a, dx)

이 a 를 안 쓰면 운석 방위각이 최대 20도 틀어지고, LC4 자극 위치가 통째로 밀린다.
05·02문서에 없는 함정이다.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT/"out"/"probe_heading.json").read_text(encoding="utf-8"))
ks, DX, DY = [], [], []
for k in range(16):
    v = d.get(str(k))
    if v and v["moved"] > 6:
        ks.append(k); DX.append(v["dx"]); DY.append(-v["dy"])   # 화면 y 아래 -> 수학 y 위
ks = np.array(ks); DX = np.array(DX); DY = np.array(DY)
nominal = np.radians(90 + 22.5*ks)

def resid(a):
    ang = np.arctan2(DY/a, DX)
    e = np.angle(np.exp(1j*(ang - nominal)))
    return np.degrees(e)

grid = np.linspace(1.0, 3.5, 2501)
rms = np.array([np.sqrt(np.mean(resid(a)**2)) for a in grid])
a_best = grid[int(np.argmin(rms))]
print(f"세로 팽창 계수 a 그리드 탐색: 최적 a = {a_best:.3f}, RMS 오차 {rms.min():.2f}도")
print(f"  a=1 (보정 안 함) 일 때 RMS {np.sqrt(np.mean(resid(1.0)**2)):.2f}도")
print(f"  a=2 일 때 RMS {np.sqrt(np.mean(resid(2.0)**2)):.2f}도")
print(f"\n{'orient':>7} {'명목각':>8} {'보정 전':>9} {'오차':>8} | {'보정 후':>9} {'오차':>8}")
r0, r1 = resid(1.0), resid(a_best)
for i, k in enumerate(ks):
    raw = np.degrees(np.arctan2(DY[i], DX[i])) % 360
    cor = np.degrees(np.arctan2(DY[i]/a_best, DX[i])) % 360
    print(f"{k:>7} {(90+22.5*k)%360:>8.1f} {raw:>9.1f} {r0[i]:>+8.1f} | "
          f"{cor:>9.1f} {r1[i]:>+8.1f}")
print(f"\n최대 오차: 보정 전 {np.abs(r0).max():.1f}도 -> 보정 후 {np.abs(r1).max():.1f}도")
print(f"\n참고: 아타리 2600 은 160x210 을 4:3 화면에 그린다.")
print(f"      픽셀 가로:세로 = (4/160):(3/210) = {(4/160)/(3/210):.2f} : 1  <- 측정값과 대조")
(ROOT/"out"/"aspect.json").write_text(json.dumps(
    dict(a=float(a_best), rms_deg=float(rms.min()),
         rms_uncorrected=float(np.sqrt(np.mean(r0**2))),
         max_err_uncorrected=float(np.abs(r0).max()),
         max_err_corrected=float(np.abs(r1).max()),
         theory_4_3=float((4/160)/(3/210))), indent=2), encoding="utf-8")
