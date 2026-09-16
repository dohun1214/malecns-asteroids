"""02문서 3장의 VPN->DN 시냅스 표가 w>=1 기준인지 w>=3 기준인지 원본에서 확인한다.
(01문서 2장의 배치 읽기 관용구 — 통째로 읽으면 OOM 으로 죽는다)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.feather as ft, pyarrow.compute as pc

ROOT = Path(__file__).resolve().parent.parent
SRCG = ROOT.parent/"malecns-song"/"graph"
DATA = ROOT.parent/"malecns-song"/"data"

nodes = pd.read_feather(SRCG/"nodes.feather")
typ = nodes["type"].astype("string").fillna("")
bid = nodes["bodyId"].to_numpy()
VPN = ["LC4","LPLC2","LPLC1","LPLC4"]
DNS = ["DNp01","DNp02","DNp03","DNp04","DNp06","DNp11"]
grp = {t: bid[(typ == t).to_numpy()] for t in VPN + DNS}
want = np.unique(np.concatenate(list(grp.values())))
print(f"대상 body {want.size}개")

tbl = ft.read_table(DATA/"connectome-weights.feather", memory_map=True)
ids = pa.array(sorted(want.tolist()))
keep = []
for batch in tbl.to_batches(max_chunksize=2_000_000):
    m = pc.and_(pc.is_in(batch.column("body_pre"),  value_set=ids),
                pc.is_in(batch.column("body_post"), value_set=ids))
    f = batch.filter(m)
    if f.num_rows: keep.append(f)
del tbl
E = pa.Table.from_batches(keep).to_pandas()
print(f"해당 엣지 {len(E):,}행")

pre_b = E["body_pre"].to_numpy(); post_b = E["body_post"].to_numpy(); ww = E["weight"].to_numpy()
setof = {t: set(v.tolist()) for t, v in grp.items()}
def inset(arr, t):
    s = setof[t]
    return np.fromiter((x in s for x in arr), bool, arr.size)
pm = {t: inset(pre_b, t) for t in VPN}
qm = {t: inset(post_b, t) for t in DNS}

EXP = {("LC4","DNp01"):6362,("LC4","DNp02"):4209,("LC4","DNp03"):2507,("LC4","DNp04"):11597,
       ("LC4","DNp06"):1152,("LC4","DNp11"):3666,("LPLC2","DNp01"):4862,("LPLC2","DNp02"):5,
       ("LPLC2","DNp03"):1,("LPLC2","DNp04"):3398,("LPLC2","DNp06"):1719,("LPLC2","DNp11"):71,
       ("LPLC1","DNp01"):1,("LPLC1","DNp02"):0,("LPLC1","DNp03"):3602,("LPLC1","DNp04"):21,
       ("LPLC1","DNp06"):2773,("LPLC1","DNp11"):1041,("LPLC4","DNp01"):0,("LPLC4","DNp02"):0,
       ("LPLC4","DNp03"):3740,("LPLC4","DNp04"):0,("LPLC4","DNp06"):0,("LPLC4","DNp11"):305}

for thr, label in ((1, "w>=1 (원본 전체)"), (3, "w>=3 (우리 그래프)")):
    hit = miss = 0
    print(f"\n{label}")
    print(f"{'':<8}" + "".join(f"{d:>9}" for d in DNS))
    for v in VPN:
        line = f"{v:<8}"
        for d in DNS:
            sel = pm[v] & qm[d] & (ww >= thr)
            tot = int(ww[sel].sum())
            exp = EXP[(v,d)]
            ok = tot == exp
            hit += ok; miss += (not ok)
            line += f"{tot:>9}" + ("" if ok else "*")
        print(line)
    print(f"  02문서 표와 일치 {hit}/24 칸" + ("  <- 표의 기준" if miss == 0 else ""))
