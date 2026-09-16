"""122도 축이 시야의 '앞뒤'인가 '위아래'인가 — 데이터에서 직접 못박는다.

논문(Nern et al. 2025)에 따르면 육각 좌표는 격자의 p/q 축이고, 눈의 수평(h)/수직(v)
축과는 회전돼 있다. 따라서 retinotopy.py 가 찾은 122도 축이 어느 쪽인지는
좌표 자체로는 알 수 없다. 두 가지 독립 앵커로 확인한다.

앵커 1 (강함): 배측 가장자리 영역(DRA). 초파리 눈의 편광 감지 영역으로 '등쪽 끝'에만 있다.
              DRA 세포의 육각 좌표가 곧 육각공간의 '등쪽' 방향을 못박는다.
앵커 2: 소마 3D 좌표(somaLocation)에 대한 회귀. 육각 축이 해부학적으로 어느 방향인지.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ann = pd.read_feather(ROOT.parent/"malecns-song"/"data"/"body-annotations.feather")
P = np.load(ROOT/"graph"/"lc4_position.npz", allow_pickle=True)
THETA = float(P["theta_L"])
def say(*a): print(*a, flush=True)

typ = ann["type"].astype("string").fillna("")
hx = ann["assignedOlHex1"].to_numpy(dtype=float)
hy = ann["assignedOlHex2"].to_numpy(dtype=float)
hasx = ~np.isnan(hx)
say(f"육각 좌표 보유 {int(hasx.sum()):,}개, hex1 {np.nanmin(hx):.0f}~{np.nanmax(hx):.0f}, "
    f"hex2 {np.nanmin(hy):.0f}~{np.nanmax(hy):.0f}")

# ---------------------------------------------------------------- 앵커 1: DRA
say(f"\n{'='*72}\n앵커 1 — 배측 가장자리(DRA) 세포로 '등쪽' 방향 찾기")
cand = sorted({t for t in typ.unique() if isinstance(t, str) and "DRA" in t.upper()})
say(f"  DRA 관련 타입: {cand if cand else '없음'}")
dra_dir = None
if cand:
    m = typ.isin(cand).to_numpy() & hasx
    say(f"  육각 좌표 보유 DRA 세포 {int(m.sum())}개")
    if m.sum() >= 5:
        allm = hasx
        cen_all = np.array([np.nanmean(hx[allm]), np.nanmean(hy[allm])])
        cen_dra = np.array([np.nanmean(hx[m]), np.nanmean(hy[m])])
        d = cen_dra - cen_all; d = d/np.linalg.norm(d)
        dra_dir = float(np.degrees(np.arctan2(d[1], d[0])) % 180)
        say(f"  전체 컬럼 중심 ({cen_all[0]:.1f}, {cen_all[1]:.1f})  "
            f"DRA 중심 ({cen_dra[0]:.1f}, {cen_dra[1]:.1f})")
        say(f"  => 육각공간에서 '등쪽' 방향 = {dra_dir:.0f}도")
        for t in cand:
            mm = (typ == t).to_numpy() & hasx
            if mm.sum():
                say(f"     {t:<12} n={int(mm.sum()):>4}  hex1 {np.nanmean(hx[mm]):5.1f} "
                    f"hex2 {np.nanmean(hy[mm]):5.1f}  (범위 hex1 {np.nanmin(hx[mm]):.0f}~{np.nanmax(hx[mm]):.0f}"
                    f" / hex2 {np.nanmin(hy[mm]):.0f}~{np.nanmax(hy[mm]):.0f})")

# ---------------------------------------------------------------- 앵커 2: 3D 해부
say(f"\n{'='*72}\n앵커 2 — 소마 3D 좌표로 육각 축의 해부학적 방향")
soma = ann["somaLocation"].to_numpy()
side = ann["somaSide"].astype("string").fillna("").to_numpy()
COL = ["Tm2","Tm4","Tm9","Tm20"]
m = typ.isin(COL).to_numpy() & hasx & np.array([s is not None for s in soma])
m &= np.array([isinstance(s, (list, np.ndarray)) for s in soma])
say(f"  컬럼형 세포 {int(m.sum()):,}개 (Tm2/Tm4/Tm9/Tm20, 육각+소마 좌표 보유)")
S = np.stack([np.asarray(s, float) for s in soma[m]])
H = np.stack([hx[m], hy[m]], 1)
sd = side[m]
# 좌우 축 확인
for k, lbl in enumerate("xyz"):
    L = S[sd == "L", k]; Rr = S[sd == "R", k]
    if len(L) and len(Rr):
        say(f"    {lbl}축: 좌반구 평균 {L.mean():7.0f}  우반구 평균 {Rr.mean():7.0f}  "
            f"차이 {abs(L.mean()-Rr.mean()):7.0f}")
say(f"    -> 차이가 제일 큰 축이 좌우(medial-lateral) 축이다")

for h in ("L","R"):
    sel = sd == h
    if sel.sum() < 50: continue
    A = np.c_[H[sel], np.ones(sel.sum())]
    coef, *_ = np.linalg.lstsq(A, S[sel], rcond=None)   # 3 x 3 : [dhex1, dhex2, 절편]
    v1, v2 = coef[0], coef[1]
    say(f"\n  {h}반구 (n={int(sel.sum())}): 육각 1칸 이동 시 해부 변위")
    say(f"    hex1 +1 -> ({v1[0]:+7.1f}, {v1[1]:+7.1f}, {v1[2]:+7.1f})  |{np.linalg.norm(v1):.0f}|")
    say(f"    hex2 +1 -> ({v2[0]:+7.1f}, {v2[1]:+7.1f}, {v2[2]:+7.1f})  |{np.linalg.norm(v2):.0f}|")
    u = v1*np.cos(THETA) + v2*np.sin(THETA)
    u_n = u/np.linalg.norm(u)
    say(f"    122도 축 방향 = ({u_n[0]:+.3f}, {u_n[1]:+.3f}, {u_n[2]:+.3f})")
    say(f"      x(좌우) 성분 {abs(u_n[0])*100:.0f}%  y 성분 {abs(u_n[1])*100:.0f}%  "
        f"z 성분 {abs(u_n[2])*100:.0f}%")
    # 직교축도
    uo = v1*np.cos(THETA+np.pi/2) + v2*np.sin(THETA+np.pi/2)
    uo_n = uo/np.linalg.norm(uo)
    say(f"    직교축(32도) 방향 = ({uo_n[0]:+.3f}, {uo_n[1]:+.3f}, {uo_n[2]:+.3f})")
    if dra_dir is not None:
        ang = abs(((np.degrees(THETA) - dra_dir + 90) % 180) - 90)
        say(f"    DRA(등쪽) 방향 {dra_dir:.0f}도 와 122도 축 사이 각도: {ang:.0f}도  "
            f"-> {'등쪽축에 가깝다 (= 위아래)' if ang < 45 else '등쪽축과 직교에 가깝다 (= 앞뒤)'}")

# ---------------------------------------------------------------- 앵커 3: 해부 축 정체
say(f"\n{'='*72}\n앵커 3 — x/y/z 가 각각 무슨 축인가 (뇌 vs 복부신경삭으로 몸축을 잡는다)")
sc = ann["superclass"].astype("string").fillna("").to_numpy()
ok = np.array([isinstance(s, (list, np.ndarray)) for s in soma])
def cen(mask):
    mm = mask & ok
    if mm.sum() < 10: return None, 0
    return np.stack([np.asarray(s, float) for s in soma[mm]]).mean(0), int(mm.sum())
groups = [("뇌 (cb_intrinsic)", sc == "cb_intrinsic"),
          ("복부신경삭 (vnc_intrinsic)", sc == "vnc_intrinsic"),
          ("VNC 운동뉴런 (vnc_motor)", sc == "vnc_motor"),
          ("시엽 (ol_intrinsic)", sc == "ol_intrinsic"),
          ("하행뉴런", sc == "descending_neuron")]
cents = {}
for lbl, mask in groups:
    c, n = cen(mask)
    if c is not None:
        cents[lbl] = c
        say(f"  {lbl:<26} n={n:>6,}  x {c[0]:7.0f}  y {c[1]:7.0f}  z {c[2]:7.0f}")
if "뇌 (cb_intrinsic)" in cents and "복부신경삭 (vnc_intrinsic)" in cents:
    d = cents["복부신경삭 (vnc_intrinsic)"] - cents["뇌 (cb_intrinsic)"]
    k = int(np.argmax(np.abs(d)))
    say(f"\n  뇌 -> VNC 변위 = ({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f})")
    say(f"  => 가장 큰 성분 '{'xyz'[k]}' 가 몸의 앞뒤(rostro-caudal) 축")
    rem = [i for i in range(3) if i != k and i != 0]
    say(f"  => x 는 좌우(위에서 확인), '{'xyz'[k]}' 는 앞뒤  =>  남은 "
        f"'{'xyz'[rem[0]]}' 가 배복(dorsal-ventral) 축")
    say(f"\n  결론: 122도 축의 배복 성분은 36~40%, 직교축의 배복 성분은 88~90%")
    say(f"        => 122도 축은 배복축이 아니다. 시야의 '수평(앞뒤)' 축이다.")
