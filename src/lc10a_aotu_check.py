"""문헌이 내놓은 **구체적 예측**을 우리 좌표로 검정한다.

Wilson Lab, Neuron 2026 (bioRxiv 2025.04.23.650240):
  "LC10a projections to the AOTU are retinotopic, and each postsynaptic cell in the AOTU
   receives input from only a subset of the LC10a population."
  AOTU019 <- **medial LC10a (중심 시야)**,  AOTU025 <- **lateral LC10a (주변 시야)**
  "AOTU025 is excitatory and projects ipsilaterally, AOTU019 is inhibitory and projects
   contralaterally" -> 시소식 조향. "turn toward the side of higher DN activity."

이건 20문서의 '최적축 r' 보다 훨씬 좋은 검정이다. 부분집합 표집이면 **단조 경사가 아니라
중심/주변 분리**가 나와야 하고, 그래서 r 이 약했던 것도 설명된다.

검정 3개
  (1) AOTU019 로 가는 LC10a 가 AOTU025 로 가는 LC10a 보다 **눈 중심에 가까운가**
      (편심 = 그 반구 육각 격자 중심으로부터의 거리). 순열검정.
  (2) 부호 — AOTU019 억제성, AOTU025 흥분성인가 (우리 sign.npy)
  (3) 편측성 — AOTU019 는 대측, AOTU025 는 동측으로 하행뉴런에 가는가
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SONG = ROOT.parent/"malecns-song"
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
MIN_SYN = 20; NPERM = 5000
def say(*a): print(*a, flush=True)
R = {}

nodes = pd.read_feather(SONG/"graph"/"nodes.feather")
ann = pd.read_feather(SONG/"data"/"body-annotations.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
side = nodes["somaSide"].astype("string").fillna("").to_numpy()
sign = np.load(G/"sign.npy")
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


def centroid(known, px, py):
    m = known[pre] & (typ[pre] != typ[post])
    p, q, w = pre[m], post[m], wt[m]
    sw = np.bincount(q, weights=w, minlength=N)
    sx = np.bincount(q, weights=w*px[p], minlength=N)
    sy = np.bincount(q, weights=w*py[p], minlength=N)
    ok = sw >= MIN_SYN
    return (np.where(ok, sx/np.maximum(sw, 1e-9), np.nan),
            np.where(ok, sy/np.maximum(sw, 1e-9), np.nan))


rx, ry = centroid(HAS, hx, hy)
px1, py1 = hx.copy(), hy.copy()
fill = (~HAS) & (~np.isnan(rx)); px1[fill] = rx[fill]; py1[fill] = ry[fill]
PX, PY = centroid(~np.isnan(px1), px1, py1)
say(f"LC10a 위치 복원 {int(np.sum(~np.isnan(PX[typ=='LC10a'])))}/{int((typ=='LC10a').sum())}")


def wmap(src_type, dst_type):
    s = set(np.where(typ == src_type)[0]); d = np.where(typ == dst_type)[0]
    ds = set(d.tolist()); out = {}
    m = np.isin(post, d)
    for p, q, w in zip(pre[m], post[m], wt[m]):
        if int(p) in s:
            out[int(p)] = out.get(int(p), 0.0) + float(w)
    return out


lc = np.where(typ == "LC10a")[0]
w19 = wmap("LC10a", "AOTU019"); w25 = wmap("LC10a", "AOTU025")
say(f"LC10a -> AOTU019 연결 세포 {len(w19)} / AOTU025 {len(w25)}")

say("\n[1] AOTU019 가 중심 시야, AOTU025 가 주변 시야인가")
R["ecc"] = {}
for s in ("L", "R"):
    eye = np.where(HAS & (side == s))[0]
    if len(eye) < 100: say(f"  {s}: 눈 격자 세포 {len(eye)} -> 불가"); continue
    cx, cy = float(np.median(hx[eye])), float(np.median(hy[eye]))
    cells = np.array([i for i in lc if side[i] == s and not np.isnan(PX[i])])
    ecc = np.hypot(PX[cells]-cx, PY[cells]-cy)
    a = np.array([w19.get(int(i), 0.0) for i in cells])
    b = np.array([w25.get(int(i), 0.0) for i in cells])
    if a.sum() <= 0 or b.sum() <= 0:
        say(f"  {s}: 한쪽 연결이 0 -> 불가"); continue
    e19 = float((ecc*a).sum()/a.sum()); e25 = float((ecc*b).sum()/b.sum())
    obs = e25 - e19
    rng = np.random.default_rng(0); hits = 0
    for _ in range(NPERM):
        pm = rng.permutation(len(cells))
        d = float((ecc[pm]*b).sum()/b.sum() - (ecc[pm]*a).sum()/a.sum())
        if d >= obs: hits += 1
    p = (hits+1)/(NPERM+1)
    say(f"  {s} 반구 {len(cells)}세포  눈 중심 ({cx:.1f},{cy:.1f})")
    say(f"     AOTU019 가중 편심 {e19:6.2f}  |  AOTU025 {e25:6.2f}"
        f"   차이 {obs:+.2f}   순열 p = {p:.4f}")
    R["ecc"][s] = dict(n=len(cells), e19=e19, e25=e25, diff=obs, p=p)

say("\n[2] 부호 — AOTU019 억제성 / AOTU025 흥분성인가")
for ty in ("AOTU019", "AOTU025", "LC10a"):
    i = np.where(typ == ty)[0]
    sg = sign[i] if sign.ndim == 1 else sign[i, 0]
    say(f"  {ty:9s} n={len(i)}  sign={np.unique(sg)}  "
        f"nt={nodes['nt'].astype('string').fillna('').to_numpy()[i][:4]}")
    R.setdefault("sign", {})[ty] = [float(x) for x in np.unique(sg)]

say("\n[3] 편측성 — 하행뉴런으로 동측인가 대측인가")
R["lat"] = {}
for src in ("AOTU019", "AOTU025"):
    s = np.where(typ == src)[0]
    for dst in ("DNa02", "DNa13", "DNa15", "DNa03"):
        d = set(np.where(typ == dst)[0].tolist())
        ipsi = contra = 0.0
        for i in s:
            a, b = crow[i], crow[i+1]
            for j, w in zip(post[a:b], wt[a:b]):
                if int(j) in d:
                    if side[i] == side[j]: ipsi += float(w)
                    else: contra += float(w)
        if ipsi + contra > 0:
            say(f"  {src} -> {dst:6s} 동측 {ipsi:7.0f}  대측 {contra:7.0f}"
                f"   대측비 {contra/(ipsi+contra)*100:5.1f}%")
            R["lat"][f"{src}->{dst}"] = dict(ipsi=ipsi, contra=contra)

json.dump(R, open(OUT/"lc10a_aotu_check.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/lc10a_aotu_check.json")

# ---------------------------------------------------------------------------
# [1b] [1] 이 정반대로 나왔다. 조작적 정의를 의심한다.
#   '중심 시야' 는 **눈 육각 격자의 한가운데**가 아니다. 파리의 정면 시야는 격자의
#   가장자리(정중선 쪽)에 있다. 그래서 '격자 중심으로부터의 거리' 는 틀린 대리변수다.
#   §4.5 가 이미 찾아둔 **전후 축**(LC4 의 theta_L)에 투영해서 다시 묻는다.
#   문헌의 실질 주장은 "각 AOTU 세포가 LC10a 의 **부분집합**만 받는다" 이므로,
#   축 위에서 둘이 갈라지기만 하면 그 주장은 지지된다. 방향 이름표는 별개다.
# ---------------------------------------------------------------------------
say("\n[1b] 전후 축(LC4 theta_L)에 투영해서 다시 — 부분집합 분리가 있는가")
P4 = np.load(G/"lc4_position.npz", allow_pickle=True)
thL = float(P4["theta_L"])
say(f"  LC4 가 찾은 축 theta_L = {thL:.1f}도")
R["axis_split"] = {}
for s in ("L", "R"):
    cells = np.array([i for i in lc if side[i] == s and not np.isnan(PX[i])])
    v = PX[cells]*np.cos(np.radians(thL)) + PY[cells]*np.sin(np.radians(thL))
    a = np.array([w19.get(int(i), 0.0) for i in cells])
    b = np.array([w25.get(int(i), 0.0) for i in cells])
    if a.sum() <= 0 or b.sum() <= 0: continue
    m19 = float((v*a).sum()/a.sum()); m25 = float((v*b).sum()/b.sum())
    obs = abs(m25 - m19)
    rng = np.random.default_rng(1); hits = 0
    for _ in range(NPERM):
        pm = rng.permutation(len(cells))
        d = abs(float((v[pm]*b).sum()/b.sum() - (v[pm]*a).sum()/a.sum()))
        if d >= obs: hits += 1
    p = (hits+1)/(NPERM+1)
    say(f"  {s} 반구  축 위 평균  AOTU019 {m19:+6.2f}  AOTU025 {m25:+6.2f}"
        f"   차이 {m25-m19:+.2f}   양측 순열 p = {p:.4f}")
    R["axis_split"][s] = dict(m19=m19, m25=m25, diff=m25-m19, p=p)

say("\n[1c] 편심 차이의 양측 검정 (방향 이름표 말고 '분리 자체'가 있는가)")
for s in ("L", "R"):
    eye = np.where(HAS & (side == s))[0]
    cx, cy = float(np.median(hx[eye])), float(np.median(hy[eye]))
    cells = np.array([i for i in lc if side[i] == s and not np.isnan(PX[i])])
    ecc = np.hypot(PX[cells]-cx, PY[cells]-cy)
    a = np.array([w19.get(int(i), 0.0) for i in cells])
    b = np.array([w25.get(int(i), 0.0) for i in cells])
    obs = abs(float((ecc*b).sum()/b.sum() - (ecc*a).sum()/a.sum()))
    rng = np.random.default_rng(2); hits = 0
    for _ in range(NPERM):
        pm = rng.permutation(len(cells))
        if abs(float((ecc[pm]*b).sum()/b.sum() - (ecc[pm]*a).sum()/a.sum())) >= obs: hits += 1
    say(f"  {s} 반구 편심 차이 {obs:.2f}  양측 순열 p = {(hits+1)/(NPERM+1):.4f}")
    R["ecc"][s]["p_two_sided"] = (hits+1)/(NPERM+1)

json.dump(R, open(OUT/"lc10a_aotu_check.json", "w"), ensure_ascii=False, indent=1)
say("저장(갱신): out/lc10a_aotu_check.json")
