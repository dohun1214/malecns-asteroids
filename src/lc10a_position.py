"""LC10a 의 시야 위치를 저장한다 — `graph/lc10a_position.npz`.

LC4 는 입력 컬럼(Tm2/Tm4)의 육각 좌표 무게중심으로 위치를 냈다 (retinotopy.py).
LC10a 는 입력의 0.9% 만 좌표를 가져서 그 방법이 안 통한다 -> **2단 전파** (20문서).

  1단계  좌표 없는 세포의 위치 = 그 세포의 '좌표 있는 입력' 의 시냅스 가중 무게중심
  2단계  LC10a 의 위치       = (좌표 있는 입력 + 1단계 결과) 의 무게중심

**축은 LC4 가 이미 찾은 theta_L 을 그대로 쓴다.** 같은 육각 좌표 틀이고, 새 축을 따로
고르면 그게 부과값이 된다. LC10a 만의 최적축(130도)은 AOTU019 경사에서 나온 값인데,
문헌상 AOTU019 는 LC10a 의 **부분집합만** 받으므로(08문서 §8.2) 축 추정에 쓰면 안 된다.

⚠️ 전후 축의 **부호**는 LC4 와 똑같이 부과값이다 (08문서 §2.1). 같은 부호를 쓰므로
   새로 지는 부과값은 없다.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SONG = ROOT.parent/"malecns-song"
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
def say(*a): print(*a, flush=True)

nodes = pd.read_feather(SONG/"graph"/"nodes.feather")
ann = pd.read_feather(SONG/"data"/"body-annotations.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
side = nodes["somaSide"].astype("string").fillna("").to_numpy()
N = len(nodes)
crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
wt = np.abs(np.load(G/"out_w.npy")).astype(np.float64)
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))

gidx = {int(b): i for i, b in enumerate(nodes["bodyId"].to_numpy())}
hx = np.full(N, np.nan); hy = np.full(N, np.nan)
h = ann[ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()]
for b, a1, a2 in zip(h["bodyId"].to_numpy(), h["assignedOlHex1"].to_numpy(),
                     h["assignedOlHex2"].to_numpy()):
    i = gidx.get(int(b))
    if i is not None: hx[i] = float(a1); hy[i] = float(a2)
HAS = ~np.isnan(hx)
say(f"육각 좌표 보유 {int(HAS.sum())} / {N}")


def centroid(known, px, py):
    """같은 타입끼리의 입력은 뺀다 (자기 타입 되먹임이 위치를 뭉갠다)."""
    m = known[pre] & (typ[pre] != typ[post])
    p, q, w = pre[m], post[m], wt[m]
    sw = np.bincount(q, weights=w, minlength=N)
    sx = np.bincount(q, weights=w*px[p], minlength=N)
    sy = np.bincount(q, weights=w*py[p], minlength=N)
    ok = sw >= MIN_SYN
    return (np.where(ok, sx/np.maximum(sw, 1e-9), np.nan),
            np.where(ok, sy/np.maximum(sw, 1e-9), np.nan), sw)


rx, ry, _ = centroid(HAS, hx, hy)
px1, py1 = hx.copy(), hy.copy()
fill = (~HAS) & (~np.isnan(rx)); px1[fill] = rx[fill]; py1[fill] = ry[fill]
say(f"1단계: {int(fill.sum())} 세포 추가 -> 좌표 보유 {int(np.sum(~np.isnan(px1)))}")
PX, PY, SW = centroid(~np.isnan(px1), px1, py1)

lc = np.where(typ == "LC10a")[0]
valid = ~np.isnan(PX[lc])
say(f"LC10a {len(lc)}세포 중 위치 복원 {int(valid.sum())}")

P4 = np.load(G/"lc4_position.npz", allow_pickle=True)
theta_L = float(P4["theta_L"])
say(f"축은 LC4 것을 그대로 쓴다: theta_L = {theta_L:.4f} rad = {np.degrees(theta_L):.1f}도")

pos = np.stack([PX[lc], PY[lc]], 1)
pos = np.nan_to_num(pos, nan=0.0)
np.savez(G/"lc10a_position.npz", idx=lc.astype(np.int64), pos=pos.astype(np.float64),
         side=side[lc].astype("U2"), valid=valid, theta_L=theta_L)

# 보고 — 축 위 범위가 두 반구에서 같은 틀인지 (LC4 와 같은 점검, 15문서)
ax = pos[:, 0]*np.cos(theta_L) + pos[:, 1]*np.sin(theta_L)
rep = {"min_syn": MIN_SYN, "n": int(len(lc)), "valid": int(valid.sum()),
       "theta_L": theta_L, "stage1_added": int(fill.sum())}
for s in ("L", "R"):
    m = valid & (side[lc] == s)
    a = ax[m]
    say(f"  {s} 반구 {int(m.sum()):3d}세포  축 범위 {a.min():7.2f} ~ {a.max():7.2f}"
        f"  중앙 {np.median(a):6.2f}  시냅스 중앙 {np.median(SW[lc][m]):7.0f}")
    rep[s] = dict(n=int(m.sum()), lo=float(a.min()), hi=float(a.max()),
                  med=float(np.median(a)))
say("\nLC4 와 비교 (같은 축 위에서)")
ax4 = P4["pos"][:, 0]*np.cos(theta_L) + P4["pos"][:, 1]*np.sin(theta_L)
for s in ("L", "R"):
    m = P4["valid"] & (P4["side"] == s)
    say(f"  LC4 {s} {int(m.sum()):3d}세포  축 범위 {ax4[m].min():7.2f} ~ {ax4[m].max():7.2f}")
    rep[f"lc4_{s}"] = dict(lo=float(ax4[m].min()), hi=float(ax4[m].max()))

json.dump(rep, open(OUT/"lc10a_position.json", "w"), ensure_ascii=False, indent=1)
say(f"\n저장: graph/lc10a_position.npz, out/lc10a_position.json")
