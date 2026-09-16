"""LC4 세포의 '위치'를 DNp02/DNp11 시냅스 비율과 무관하게 얻을 방법이 있는가?"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
ann = pd.read_feather(ROOT.parent/"malecns-song"/"data"/"body-annotations.feather")
print("컬럼:", [c for c in ann.columns])
lc4 = ann[ann["type"].astype("string") == "LC4"]
print(f"\nLC4 {len(lc4)}개")
for col in ("assignedOlHex1", "assignedOlHex2", "somaLocation", "somaNeuromere"):
    if col in ann.columns:
        nn = lc4[col].notna().sum()
        print(f"  {col:<16} 값 있는 LC4 {nn}/{len(lc4)}", end="")
        if nn: print(f"   예: {lc4[col].dropna().iloc[:3].tolist()}")
        else: print()
# LC4 의 시냅스 전 파트너 중 육각 좌표를 가진 컬럼형 세포가 얼마나 되는가
w = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
crow = np.load(ROOT/"graph"/"out_crow.npy").astype(np.int64)
post = np.load(ROOT/"graph"/"out_post.npy").astype(np.int64)
wt   = np.load(ROOT/"graph"/"out_w.npy").astype(np.int64)
N = crow.size-1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
C = np.load(ROOT/"graph"/"circuit_idx.npz")
lc4_idx = C["LC4"]
m = np.zeros(N, bool); m[lc4_idx] = True
sel = m[post]
upstream = np.unique(pre[sel])
bid = w["bodyId"].to_numpy()
up_bodies = set(bid[upstream].tolist())
hexed = ann[ann["assignedOlHex1"].notna()] if "assignedOlHex1" in ann.columns else ann.iloc[:0]
hex_bodies = set(hexed["bodyId"].tolist())
inter = up_bodies & hex_bodies
print(f"\nLC4 로 들어가는 전구체 뉴런 {len(up_bodies):,}개 중 육각 좌표 보유 {len(inter):,}개 "
      f"({len(inter)/max(len(up_bodies),1)*100:.0f}%)")
if inter:
    sub = hexed[hexed["bodyId"].isin(inter)]
    print(f"  타입 상위: {sub['type'].value_counts().head(8).to_dict()}")
    print(f"  hex1 범위 {sub['assignedOlHex1'].min()}~{sub['assignedOlHex1'].max()}, "
          f"hex2 범위 {sub['assignedOlHex2'].min()}~{sub['assignedOlHex2'].max()}")
