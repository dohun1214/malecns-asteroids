"""격자 -> 시야각(도) 환산. 육각 좌표를 직교좌표로 다루면 안 된다.

(hex1, hex2) 는 육각 격자의 axial 좌표라 두 기저 벡터가 60도로 만난다.
직교좌표처럼 거리를 재면 축 방향에 따라 최대 15% 왜곡된다.

가정을 더 넣지 않기 위해, 격자 기저를 '측정된 3D 해부 변위'에서 직접 가져온다:
  hex1 +1, hex2 +1 이 각각 소마 위치를 3D 로 얼마나 옮기는지 회귀로 구하고,
  그 벡터의 크기와 사잇각으로 실제 격자 기하를 확정한다.

⚠️ 상관계수·단조성 결과는 이 문제와 무관하다. 최적 방향 탐색이 선형 사영 전체를
   훑기 때문에 기저를 바꿔도 같은 사영이 나온다 (각도 라벨만 바뀐다).
   영향을 받는 건 '몇 도인가' 뿐이다.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ann = pd.read_feather(ROOT.parent/"malecns-song"/"data"/"body-annotations.feather")
P = np.load(ROOT/"graph"/"lc4_position.npz", allow_pickle=True)
TH = float(P["theta_L"])
IO_DEG = 4.5     # 개안각 평균, Currea et al. 2023 (범위 3.4~5.5, 전방이 더 조밀)

typ = ann["type"].astype("string").fillna("")
h1 = ann["assignedOlHex1"].to_numpy(float); h2 = ann["assignedOlHex2"].to_numpy(float)
side = ann["somaSide"].astype("string").fillna("").to_numpy()
soma = ann["somaLocation"].to_numpy()
ok = np.array([isinstance(s, (list, np.ndarray)) for s in soma])
m0 = ~np.isnan(h1)
print(f"육각 좌표 보유 {int(m0.sum()):,}개")

for h in ("L", "R"):
    sel = m0 & (side == h)
    cols = np.unique(np.stack([h1[sel], h2[sel]], 1), axis=0)
    # --- 격자 기저를 3D 해부에서 측정
    msel = sel & ok & typ.isin(["Tm2","Tm4","Tm9","Tm20"]).to_numpy()
    S = np.stack([np.asarray(s, float) for s in soma[msel]])
    H = np.stack([h1[msel], h2[msel]], 1)
    A = np.c_[H, np.ones(len(H))]
    coef, *_ = np.linalg.lstsq(A, S, rcond=None)
    v1, v2 = coef[0], coef[1]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    ang = np.degrees(np.arccos(np.clip(v1 @ v2/(n1*n2), -1, 1)))
    spacing = (n1 + n2)/2
    print(f"\n{h}반구 고유 컬럼 {len(cols):,}개  (문헌: 초파리 한쪽 눈 약 750~800 개안)")
    print(f"  격자 기저 |hex1|={n1:.0f} |hex2|={n2:.0f} 사잇각 {ang:.1f}도"
          f"  -> {'육각 격자 맞음 (60 또는 120도)' if abs(ang-60)<12 or abs(ang-120)<12 else '격자 기하가 예상과 다름'}")
    # --- 두 축의 실제 시야 범위 (3D 거리 기준)
    for lbl, th in (("수평(앞뒤)", TH), ("배복(위아래)", TH + np.pi/2)):
        u = v1*np.cos(th) + v2*np.sin(th)
        scale = np.linalg.norm(u)/spacing          # 축 1단위 = 개안 몇 칸인가
        proj = cols[:,0]*np.cos(th) + cols[:,1]*np.sin(th)
        span_cols = proj.ptp()*scale
        print(f"  {lbl:<10} 범위 {proj.ptp():5.1f}단위 x {scale:.3f} = {span_cols:5.1f} 개안"
              f"  -> {span_cols*IO_DEG:5.0f}도")
    # --- LC4 중심의 범위
    pos, lside, valid = P["pos"], P["side"], P["valid"]
    s2 = valid & (lside == h)
    u = v1*np.cos(TH) + v2*np.sin(TH); scale = np.linalg.norm(u)/spacing
    axl = pos[:,0]*np.cos(TH) + pos[:,1]*np.sin(TH)
    print(f"  LC4 중심 수평 범위 {axl[s2].ptp():.1f}단위 = {axl[s2].ptp()*scale:.1f} 개안"
          f" -> {axl[s2].ptp()*scale*IO_DEG:.0f}도  (n={int(s2.sum())})")
    print(f"     (RF 중심이라 격자 전체보다 좁은 게 정상 — 가장자리 세포도 RF 는 바깥으로 뻗는다)")

print(f"\n문헌 대조: 초파리 한쪽 눈 수평 시야 약 150~180도, 개안 약 750~800개")
print(f"⚠️ 개안각 {IO_DEG}도는 '평균'이다. 전방이 더 조밀(최소 3.4도)하므로 전방 해상도는 과소평가된다.")
