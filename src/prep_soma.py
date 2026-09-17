"""소마 3D 좌표를 대시보드용으로 뽑는다 (02문서 §10.2).

nodes.feather 의 somaLocation 에 좌표가 139,662 / 166,700 (84%) 들어 있다.
브라우저는 '전역 뉴런 번호'가 아니라 '점군의 몇 번째 점'을 알아야 하므로
  pt_of[전역번호] -> 점 인덱스 (좌표 없으면 -1)
매핑을 같이 저장한다. 서버가 스파이크 인덱스를 이걸로 변환해서 보낸다.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
SONG = ROOT.parent/"malecns-song"
nodes = pd.read_feather(SONG/"graph"/"nodes.feather")     # 시뮬 인덱스 <-> bodyId
N = len(nodes)
ann = pd.read_feather(SONG/"data"/"body-annotations.feather")
print("annotation 컬럼:", [c for c in ann.columns])
col = next((c for c in ann.columns if c.lower() in
            ("somalocation", "soma_location", "somaposition")), None)
assert col, f"소마 좌표 컬럼 없음: {list(ann.columns)}"
idcol = "bodyId" if "bodyId" in ann.columns else "body"
print(f"좌표 컬럼 '{col}', 식별자 '{idcol}'")

loc = dict(zip(ann[idcol].to_numpy(), ann[col].to_numpy()))
xyz = np.full((N, 3), np.nan, dtype=np.float32)
bad = 0
for i, bid in enumerate(nodes["bodyId"].to_numpy()):
    v = loc.get(bid)
    if v is None: continue
    try:
        a = np.asarray(v, dtype=np.float32).ravel()
        if a.size >= 3 and np.isfinite(a[:3]).all(): xyz[i] = a[:3]
        else: bad += 1
    except Exception:
        bad += 1
ok = np.isfinite(xyz).all(axis=1)
print(f"좌표 있는 뉴런 {int(ok.sum()):,} / {N:,} ({ok.mean()*100:.1f}%)  파싱실패 {bad}")

pt_of = np.full(N, -1, dtype=np.int32)
pt_of[ok] = np.arange(int(ok.sum()), dtype=np.int32)
pts = xyz[ok]
# 중심을 원점으로, 최대 반경 1 로 정규화 (브라우저에서 카메라 잡기 쉽게)
c = pts.mean(axis=0)
r = float(np.abs(pts - c).max())
norm = ((pts - c)/r).astype(np.float32)
print(f"원본 범위 x {pts[:,0].min():.0f}~{pts[:,0].max():.0f}  "
      f"y {pts[:,1].min():.0f}~{pts[:,1].max():.0f}  z {pts[:,2].min():.0f}~{pts[:,2].max():.0f}")

# 뉴런 그룹 라벨 (색칠용): 0=기타 1=LC4 2=LPLC2 3=DN 4=VNC판독
C = dict(np.load(G/"circuit_idx.npz"))
R_ = np.load(G/"vnc_readout.npz", allow_pickle=True)
grp = np.zeros(N, dtype=np.uint8)
mot = np.concatenate([R_["dn02_only"], R_["dn11_only"]])
grp[mot] = 4
for k in ("DNp01","DNp02","DNp03","DNp04","DNp05","DNp06","DNp09","DNp10","DNp11"):
    if k in C: grp[C[k]] = 3
grp[C["LPLC2"]] = 2
grp[C["LC4"]] = 1
print("그룹 분포:", {int(g): int((grp[ok] == g).sum()) for g in range(5)})

np.savez_compressed(G/"soma.npz", pts=norm, pt_of=pt_of, grp=grp[ok],
                    center=c, radius=np.float32(r))
# 브라우저가 바로 읽을 바이너리도 같이 (float32 xyz + uint8 group)
(ROOT/"web").mkdir(exist_ok=True)
norm.tofile(ROOT/"web"/"soma_xyz.bin")
grp[ok].astype(np.uint8).tofile(ROOT/"web"/"soma_grp.bin")
meta = dict(n_points=int(ok.sum()), n_neurons=int(N),
            center=[float(x) for x in c], radius=float(r))
(ROOT/"web"/"soma_meta.json").write_text(json.dumps(meta), encoding="utf-8")
print(f"-> graph/soma.npz, web/soma_xyz.bin ({norm.nbytes/1e6:.2f} MB), web/soma_grp.bin")
