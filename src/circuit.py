"""A단계: 도피 회로 인덱스를 새 그래프 인덱스 공간에서 추출하고 02문서 3장 표로 검증한다.

인덱스 매핑이 틀리면 그 위에 쌓는 게 전부 무의미해지므로, 뽑기만 하지 않고
'이미 독립적으로 측정해둔 표를 재현하는가'로 확인한다.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G    = ROOT/"graph"
SRCG = ROOT.parent/"malecns-song"/"graph"
OUT  = ROOT/"out"; OUT.mkdir(exist_ok=True)

nodes = pd.read_feather(SRCG/"nodes.feather")
body  = np.load(SRCG/"body.npy")
print(f"nodes {nodes.shape}  body {body.shape}")
print(f"columns: {list(nodes.columns)}")
assert len(nodes) == len(body), "nodes/body 길이 불일치"
if "bodyId" in nodes.columns:
    assert (nodes["bodyId"].values == body).all(), "nodes 순서가 그래프 인덱스 순서가 아니다"
    print("nodes 행 순서 == 그래프 인덱스 순서  OK")

typ  = nodes["type"].astype("string").fillna("")
side = nodes["somaSide"].astype("string").fillna("") if "somaSide" in nodes.columns else None
inst = nodes["instance"].astype("string").fillna("") if "instance" in nodes.columns else None

def pick(name):
    return np.flatnonzero((typ == name).to_numpy())

VPN = ["LC4", "LPLC2", "LPLC1", "LPLC4", "LC6", "LC16"]
DN  = ["DNp01", "DNp02", "DNp03", "DNp04", "DNp05", "DNp06",
       "DNp09", "DNp10", "DNp11"]
VNC = ["TTMn", "PSI", "MDN", "GFC1", "GFC2", "GFC3", "GFC4"]

EXPECT_VPN = {"LC4": (126, 71, 55), "LPLC2": (185, 94, 91), "LPLC1": (134, 68, 66),
              "LPLC4": (97, 48, 49), "LC6": (124, 59, 65), "LC16": (182, 88, 94)}

idx = {}
print(f"\n{'='*64}\n시각 투사 뉴런 — 02문서 3장 표와 대조")
print(f"{'type':<8} {'n':>5} {'L':>4} {'R':>4}   {'기대 n/L/R':>14}  판정")
ok_all = True
for t in VPN:
    ii = pick(t); idx[t] = ii
    if side is not None:
        s = side.to_numpy()[ii]
        L = int((s == "L").sum()); Rr = int((s == "R").sum())
    else:
        L = Rr = -1
    idx[f"{t}_L"] = ii[side.to_numpy()[ii] == "L"]
    idx[f"{t}_R"] = ii[side.to_numpy()[ii] == "R"]
    e = EXPECT_VPN[t]
    ok = (len(ii), L, Rr) == e
    ok_all &= ok
    print(f"{t:<8} {len(ii):>5} {L:>4} {Rr:>4}   {str(e):>14}  {'OK' if ok else 'MISMATCH'}")

print(f"\n하행뉴런 (전부 각 2개여야 함)")
row = []
for t in DN:
    ii = pick(t); idx[t] = ii
    if side is not None:
        idx[f"{t}_L"] = ii[side.to_numpy()[ii] == "L"]
        idx[f"{t}_R"] = ii[side.to_numpy()[ii] == "R"]
    row.append(f"{t}:{len(ii)}")
    ok_all &= (len(ii) == 2)
print("  " + "  ".join(row))
if inst is not None and len(idx["DNp01"]):
    print(f"  DNp01 instance: {[inst.to_numpy()[i] for i in idx['DNp01']]}")

print(f"\nVNC 하류")
for t in VNC:
    ii = pick(t); idx[t] = ii
    print(f"  {t:<6} {len(ii)}", end="")
dlmn = np.flatnonzero(typ.str.startswith("DLMn").to_numpy())
idx["DLMn"] = dlmn
print(f"\n  DLMn* {len(dlmn)}")

# ---------------------------------------------------- 시냅스 표 재현
crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
w    = np.load(G/"out_w.npy").astype(np.int64)
N = crow.size - 1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))

EXPECT_SYN = {
    ("LC4","DNp01"):6362, ("LC4","DNp02"):4209, ("LC4","DNp03"):2507,
    ("LC4","DNp04"):11597, ("LC4","DNp06"):1152, ("LC4","DNp11"):3666,
    ("LPLC2","DNp01"):4862, ("LPLC2","DNp02"):5, ("LPLC2","DNp03"):1,
    ("LPLC2","DNp04"):3398, ("LPLC2","DNp06"):1719, ("LPLC2","DNp11"):71,
    ("LPLC1","DNp01"):1, ("LPLC1","DNp02"):0, ("LPLC1","DNp03"):3602,
    ("LPLC1","DNp04"):21, ("LPLC1","DNp06"):2773, ("LPLC1","DNp11"):1041,
    ("LPLC4","DNp01"):0, ("LPLC4","DNp02"):0, ("LPLC4","DNp03"):3740,
    ("LPLC4","DNp04"):0, ("LPLC4","DNp06"):0, ("LPLC4","DNp11"):305,
}
print(f"\n{'='*64}\nVPN -> DN 시냅스 총합")
print("02문서 3장 표는 w>=1 기준으로 확인됨 (check_table.py: 24/24 칸 일치).")
print("우리 그래프는 01문서 2장이 권장한 w>=3 이므로 값이 조금씩 낮은 게 정상이다.")
mask_cache = {}
def members(t):
    if t not in mask_cache:
        m = np.zeros(N, dtype=bool); m[idx[t]] = True; mask_cache[t] = m
    return mask_cache[t]

dns = ["DNp01","DNp02","DNp03","DNp04","DNp06","DNp11"]
print(f"{'':<8}" + "".join(f"{d:>9}" for d in dns))
n_mis = 0
for v in ["LC4","LPLC2","LPLC1","LPLC4"]:
    sel = members(v)[pre]
    line = f"{v:<8}"
    for d in dns:
        tot = int(np.abs(w[sel & members(d)[post]]).sum())
        exp = EXPECT_SYN[(v,d)]
        d_pct = (tot - exp) / exp * 100 if exp else 0.0
        if tot > exp: n_mis += 1          # w>=3 이 w>=1 보다 클 수는 없다
        line += f"{tot:>9}"
        if exp: line += f"({d_pct:+.0f}%)"
        else:   line += "      "
    print(line)
print(f"  (괄호는 w>=1 표 대비. 전부 <= 0% 여야 정상)")
ok_all &= (n_mis == 0)
if n_mis: print(f"  !! w>=1 보다 큰 칸이 {n_mis}개 — 있을 수 없는 일이므로 매핑 오류")

# 부호 확인 (LC4/LPLC2 는 콜린성이어야 함)
sign = np.load(G/"sign.npy")
for t in ["LC4","LPLC2","DNp02","DNp11","DNp04","DNp01"]:
    s = sign[idx[t]]
    print(f"  {t:<7} 부호 분포: +{int((s>0).sum())} / -{int((s<0).sum())} / 0:{int((s==0).sum())}")

# ---------------------------------------------------- 4장 경사/SVD 가 w>=3 에서 살아있는가
print(f"\n{'='*64}\n02문서 4장 재현 — w>=3 그래프에서")
def syn_matrix(vpn, dns):
    rows = idx[vpn]
    M = np.zeros((rows.size, len(dns)))
    pos = {b: i for i, b in enumerate(rows)}
    for j, d in enumerate(dns):
        dm = members(d)[post]
        sel = dm & members(vpn)[pre]
        for pp, ww_ in zip(pre[sel], np.abs(w[sel])):
            M[pos[pp], j] += ww_
    return M

DN12 = ["DNp01","DNp02","DNp03","DNp04","DNp05","DNp06","DNp09","DNp10","DNp11"]
for hemi in ("L", "R"):
    rows = idx[f"LC4_{hemi}"]
    sub = np.zeros((rows.size, 2))
    pos = {b: i for i, b in enumerate(rows)}
    for j, d in enumerate(["DNp02","DNp11"]):
        sel = members(d)[post] & np.isin(pre, rows)
        for pp, ww_ in zip(pre[sel], np.abs(w[sel])):
            sub[pos[pp], j] += ww_
    r = np.corrcoef(sub[:,0], sub[:,1])[0,1]
    exp_r = -0.690 if hemi == "L" else -0.746
    print(f"  LC4 {hemi}반구 (n={rows.size}): DNp02 vs DNp11 상관 r = {r:+.3f}  "
          f"(02문서 w>=1 기준 {exp_r:+.3f})  시냅스 범위 {sub.min():.0f}~{sub.max():.0f}")

M = syn_matrix("LC4", DN12)
for hemi in ("L","R"):
    rows = idx[f"LC4_{hemi}"]
    sel = np.isin(idx["LC4"], rows)
    X = M[sel] - M[sel].mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2 / (S**2).sum()
    top = np.argsort(np.abs(Vt[0]))[::-1][:2], np.argsort(np.abs(Vt[1]))[::-1][:2]
    print(f"  LC4 {hemi} SVD: PC1 {var[0]*100:.1f}% ({'/'.join(DN12[i] for i in top[0])}) "
          f"PC2 {var[1]*100:.1f}% ({'/'.join(DN12[i] for i in top[1])}) PC3 {var[2]*100:.1f}%"
          f"  PC1+PC2={var[0]*100+var[1]*100:.1f}%")

# 게이트 1(a) 용: LC4 세포를 전후축(DNp02 <-> DNp11) 점수로 정렬해 저장
i02, i11 = DN12.index("DNp02"), DN12.index("DNp11")
ap_raw = M[:, i02] - M[:, i11]
denom = M[:, i02] + M[:, i11]
ap = np.where(denom > 0, ap_raw / np.maximum(denom, 1), 0.0)
idx["LC4_ap_score"] = ap
print(f"  LC4 전후축 점수 (DNp02-DNp11)/(합): 범위 {ap.min():+.2f}~{ap.max():+.2f}, "
      f"0이 아닌 세포 {int((denom>0).sum())}/{len(ap)}")

np.savez(G/"circuit_idx.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in idx.items()})
(OUT/"circuit_report.json").write_text(json.dumps(
    {k: int(len(v)) for k, v in idx.items()} | {"table_match": bool(n_mis == 0), "all_ok": bool(ok_all)},
    indent=2), encoding="utf-8")
print(f"\n전체 판정: {'A단계 PASS — 인덱스 매핑 신뢰 가능' if ok_all else 'A단계 FAIL — 매핑 확인 필요'}")
print(f"-> graph/circuit_idx.npz, out/circuit_report.json")
