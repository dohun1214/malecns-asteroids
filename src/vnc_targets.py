"""이슈 #1 1단계: DNp02 / DNp11 의 하류 중 '방향을 구분하는' 집단을 연결성에서 찾는다.

이름(TTMn/DLMn/GFC...)으로 고르면 방향 정보가 있다는 보장이 없다.
GFC1~4 는 Giant Fiber 계열이라 방향 불변일 가능성이 높다 (02문서 4.4).

대신 데이터로 고른다:
  각 하류 뉴런에 대해  특이도 = (DNp02 시냅스 - DNp11 시냅스) / (합)
  +1 에 가까우면 DNp02 전용(후진 도피), -1 이면 DNp11 전용(전진 도피).
이 두 끝단이 곧 전후 채널의 판독 집단이 된다.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
C = dict(np.load(G/"circuit_idx.npz"))
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
sc  = nodes["superclass"].astype("string").fillna("").to_numpy()
sd  = nodes["somaSide"].astype("string").fillna("").to_numpy()

crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
w    = np.abs(np.load(G/"out_w.npy").astype(np.int64))
N = crow.size - 1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))

def out_syn(src_idx):
    m = np.zeros(N, bool); m[src_idx] = True
    sel = m[pre]
    v = np.zeros(N, np.int64)
    np.add.at(v, post[sel], w[sel])
    return v

s02, s11 = out_syn(C["DNp02"]), out_syn(C["DNp11"])
tot = s02 + s11
tgt = np.flatnonzero(tot > 0)
spec = (s02[tgt] - s11[tgt])/tot[tgt]
print(f"DNp02/DNp11 의 1홉 하류 {len(tgt):,}개 (시냅스 합 {tot.sum():,})")
print(f"  그중 VNC {int(np.char.startswith(sc[tgt].astype(str), 'vnc')).sum() if False else int(sum(str(x).startswith('vnc') for x in sc[tgt])):,}개")

print(f"\n특이도 분포: {'':>4}", end="")
for lo, hi in ((-1.01,-0.8),(-0.8,-0.4),(-0.4,0.4),(0.4,0.8),(0.8,1.01)):
    n = int(((spec >= lo) & (spec < hi)).sum())
    print(f"[{lo:+.1f},{hi:+.1f}) {n:>5}  ", end="")
print()

print(f"\n{'집단':<26} {'n':>5} {'시냅스':>9} {'좌/우':>9}  대표 타입")
groups = {}
for lbl, m in (("DNp02 전용 (특이도 > +0.8)", spec > 0.8),
               ("DNp11 전용 (특이도 < -0.8)", spec < -0.8),
               ("공통 (|특이도| < 0.4)", np.abs(spec) < 0.4)):
    ii = tgt[m]
    groups[lbl] = ii
    tt = pd.Series([str(x) for x in typ[ii]]).value_counts().head(5).to_dict()
    nl = int((sd[ii] == "L").sum()); nr = int((sd[ii] == "R").sum())
    print(f"{lbl:<26} {len(ii):>5} {int(tot[ii].sum()):>9} {nl:>4}/{nr:<4}  {tt}")

# VNC 로만 좁힌 버전
isvnc = np.array([str(x).startswith("vnc") for x in sc])
print(f"\nVNC 로 한정")
print(f"{'집단':<26} {'n':>5} {'시냅스':>9} {'좌/우':>9}  대표 타입")
save = {}
for lbl, m in (("DNp02 전용", spec > 0.8), ("DNp11 전용", spec < -0.8)):
    ii = tgt[m][isvnc[tgt[m]]]
    save[lbl] = ii
    tt = pd.Series([str(x) for x in typ[ii]]).value_counts().head(5).to_dict()
    nl = int((sd[ii] == "L").sum()); nr = int((sd[ii] == "R").sum())
    print(f"{lbl:<26} {len(ii):>5} {int(tot[ii].sum()):>9} {nl:>4}/{nr:<4}  {tt}")

# 기존 이름 기반 집단의 특이도는 어떤가
print(f"\n이름으로 고른 기존 집단의 특이도 (0 에 가까우면 방향을 못 가린다)")
for k in ("TTMn","PSI","DLMn","GFC1","GFC2","GFC3","GFC4","MDN"):
    ii = C[k]
    t = s02[ii] + s11[ii]
    if t.sum() == 0:
        print(f"  {k:<6} n={len(ii):>3}  DNp02/DNp11 직접 입력 없음")
        continue
    sp = (s02[ii].sum() - s11[ii].sum())/t.sum()
    print(f"  {k:<6} n={len(ii):>3}  시냅스 {int(t.sum()):>6}  특이도 {sp:+.3f}")

np.savez(G/"vnc_readout.npz",
         dn02_only=save["DNp02 전용"], dn11_only=save["DNp11 전용"],
         side_dn02=np.array([str(x) for x in sd[save["DNp02 전용"]]]),
         side_dn11=np.array([str(x) for x in sd[save["DNp11 전용"]]]))
print(f"\n-> graph/vnc_readout.npz")
