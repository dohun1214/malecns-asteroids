"""게이트 4 타당성 — LC10a 를 시야에 놓을 수 있는가.

LC4 는 입력 컬럼(Tm2/Tm4)의 육각 좌표 무게중심으로 위치를 복원했다 (§4.5).
**LC10a 는 그게 안 된다** — 입력 시냅스의 0.9% 만 좌표를 가진다 (LC4 는 12.8%).
최대 입력원 Tm5Y/TmY21/Tm5a 가 전부 0% 다.

우회로: 좌표를 **한 단 더 전파**한다.
  1단계  좌표 없는 컬럼형 세포의 위치 = 그 세포의 좌표 있는 입력의 무게중심
  2단계  LC10a 의 위치       = (좌표 있는 입력 + 1단계 결과) 의 무게중심

이게 되는지 먼저 묻는다. 안 되면 **안 된다고 적고 멈춘다.**

검증 세 겹 — 순서가 중요하다
  (A) 1단계가 맞는지부터. **답을 아는 곳에서 시험한다** — Tm2(좌표 99.5% 보유)와
      Tm4(49.9%)의 좌표를 가리고 입력에서 복원해 진짜 좌표와 비교한다.
      여기서 안 맞으면 2단계는 볼 필요도 없다.
  (B) LC10a 피복률 — 몇 세포가, 입력 시냅스의 몇 %로 위치를 얻는가.
  (C) 독립 검정 — 복원한 위치가 **LC10a→AOTU019 시냅스 수**를 예측하는가.
      위치는 입력에서, 경사는 출력에서 왔으므로 순환이 아니다 (LC4/DNp02 와 같은 구조).
      held-out(반구 교차) + 순열검정으로 축 맞추기 반박을 막는다.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SONG = ROOT.parent/"malecns-song"
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NPERM   = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
def say(*a): print(*a, flush=True)
R = {"min_syn": MIN_SYN, "n_perm": NPERM}

nodes = pd.read_feather(SONG/"graph"/"nodes.feather")
ann   = pd.read_feather(SONG/"data"/"body-annotations.feather")
typ   = nodes["type"].astype("string").fillna("").to_numpy()
side  = nodes["somaSide"].astype("string").fillna("").to_numpy()
N = len(nodes)

crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
wt   = np.abs(np.load(G/"out_w.npy")).astype(np.float64)
pre  = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
say(f"그래프 {N} 세포 / {len(post)} 엣지")

gidx = {int(b): i for i, b in enumerate(nodes["bodyId"].to_numpy())}
hx = np.full(N, np.nan); hy = np.full(N, np.nan)
h = ann[ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()]
for b, a1, a2 in zip(h["bodyId"].to_numpy(), h["assignedOlHex1"].to_numpy(),
                     h["assignedOlHex2"].to_numpy()):
    i = gidx.get(int(b))
    if i is not None: hx[i] = float(a1); hy[i] = float(a2)
HAS = ~np.isnan(hx)
say(f"육각 좌표 보유 {int(HAS.sum())} 세포")


def centroid(known, px, py, drop_self_type=None):
    """각 세포에 대해 '좌표를 아는 입력'의 시냅스 가중 무게중심.
    drop_self_type: 같은 타입끼리의 입력을 빼고 싶을 때 (타입 배열)."""
    m = known[pre]
    if drop_self_type is not None:
        m &= (drop_self_type[pre] != drop_self_type[post])
    p, q, w = pre[m], post[m], wt[m]
    sw = np.bincount(q, weights=w, minlength=N)
    sx = np.bincount(q, weights=w*px[p], minlength=N)
    sy = np.bincount(q, weights=w*py[p], minlength=N)
    ok = sw >= MIN_SYN
    ox = np.where(ok, sx/np.maximum(sw, 1e-9), np.nan)
    oy = np.where(ok, sy/np.maximum(sw, 1e-9), np.nan)
    return ox, oy, sw


# ---------------------------------------------------------------- (A) 1단계 검정
say("\n[A] 1단계가 맞는가 — 답을 아는 곳에서 시험한다")
rx, ry, rw = centroid(HAS, hx, hy, drop_self_type=typ)
R["stage1_control"] = {}
for ty in ("Tm2", "Tm4", "Tm9", "Tm20", "T2", "TmY3", "Tm5Y", "TmY21", "Tm5a"):
    idx = np.where((typ == ty) & HAS)[0]
    if len(idx) < 10:
        say(f"  {ty:7s} 진짜 좌표 있는 세포 {len(idx):5d} -> 검정 불가"); continue
    good = idx[~np.isnan(rx[idx])]
    if len(good) < 10:
        say(f"  {ty:7s} 복원 {len(good)}/{len(idx)} -> 검정 불가"); continue
    r1 = float(np.corrcoef(rx[good], hx[good])[0, 1])
    r2 = float(np.corrcoef(ry[good], hy[good])[0, 1])
    err = float(np.median(np.hypot(rx[good]-hx[good], ry[good]-hy[good])))
    say(f"  {ty:7s} 복원 {len(good):5d}/{len(idx):5d}  r(hex1)={r1:+.3f} r(hex2)={r2:+.3f}"
        f"  중앙오차 {err:.2f} 컬럼")
    R["stage1_control"][ty] = dict(n=len(good), n_true=len(idx), r1=r1, r2=r2, err=err)

# 1단계 결과를 좌표 없는 세포에 채운다
px1, py1 = hx.copy(), hy.copy()
fill = (~HAS) & (~np.isnan(rx))
px1[fill] = rx[fill]; py1[fill] = ry[fill]
KNOWN2 = ~np.isnan(px1)
say(f"  -> 1단계로 {int(fill.sum())} 세포 추가. 좌표 보유 "
    f"{int(HAS.sum())} -> {int(KNOWN2.sum())}")
R["stage1_added"] = int(fill.sum()); R["known_after_stage1"] = int(KNOWN2.sum())

# ---------------------------------------------------------------- (B) 2단계 피복률
say("\n[B] 2단계 — LC10a 피복률")
px2, py2, sw2 = centroid(KNOWN2, px1, py1, drop_self_type=typ)


def coverage(ty):
    idx = np.where(typ == ty)[0]
    m = np.isin(post, idx)
    tot = float(wt[m].sum())
    used = float(wt[m & KNOWN2[pre] & (typ[pre] != typ[post])].sum())
    got = int(np.sum(~np.isnan(px2[idx])))
    say(f"  {ty:7s} {len(idx):4d}세포 중 위치 복원 {got:4d} ({got/len(idx)*100:5.1f}%)"
        f"   입력 시냅스 활용 {used/tot*100:5.1f}%")
    return dict(n=len(idx), got=got, syn_used=used/tot)


R["coverage"] = {ty: coverage(ty) for ty in ("LC10a", "LC4", "LC10d", "LC10c-1")}

# ---------------------------------------------------------------- (C) 독립 검정
say("\n[C] 복원한 위치가 LC10a->AOTU019 경사를 예측하는가")
tgt_idx = np.where(typ == "AOTU019")[0]
tgt_side = {i: side[i] for i in tgt_idx}
say(f"  AOTU019 {len(tgt_idx)}세포 side={[side[i] for i in tgt_idx]}")

lc = np.where(typ == "LC10a")[0]
m = np.isin(post, tgt_idx) & np.isin(pre, lc)
wmap = {}
for p, q, w in zip(pre[m], post[m], wt[m]):
    wmap[(int(p), int(q))] = wmap.get((int(p), int(q)), 0.0) + float(w)


def gradient(cells, same_side=True):
    g = np.zeros(len(cells))
    for k, i in enumerate(cells):
        for q in tgt_idx:
            if (side[i] == tgt_side[q]) == same_side:
                g[k] += wmap.get((int(i), int(q)), 0.0)
    return g


def best_axis(pos, g):
    """경사와 가장 잘 맞는 방향(도)과 그때의 r."""
    best = (0.0, 0.0)
    for a in np.arange(0, 180, 1.0):
        v = pos[:, 0]*np.cos(np.radians(a)) + pos[:, 1]*np.sin(np.radians(a))
        if np.std(v) < 1e-9: continue
        r = float(np.corrcoef(v, g)[0, 1])
        if abs(r) > abs(best[1]): best = (a, r)
    return best


R["axis"] = {}
store = {}
for s in ("L", "R"):
    cells = np.array([i for i in lc if side[i] == s and not np.isnan(px2[i])])
    if len(cells) < 20:
        say(f"  {s} 반구: 위치 있는 LC10a {len(cells)}개 -> 검정 불가"); continue
    pos = np.stack([px2[cells], py2[cells]], 1)
    g = gradient(cells)
    if np.std(g) < 1e-9:
        say(f"  {s} 반구: 동측 AOTU019 시냅스가 전부 같다 -> 검정 불가"); continue
    a, r = best_axis(pos, g)
    say(f"  {s} 반구 {len(cells):3d}세포  최적축 {a:5.1f}도  r = {r:+.3f}"
        f"   (경사 비영 {int((g>0).sum())}세포)")
    R["axis"][s] = dict(n=len(cells), axis=a, r=r, nonzero=int((g > 0).sum()))
    store[s] = (cells, pos, g, a)

if len(store) == 2:
    (cL, pL, gL, aL) = store["L"]; (cR, pR, gR, aR) = store["R"]
    say(f"  두 반구 최적축 차이 {abs(aL-aR):.1f}도 (LC4 는 4도 이내로 수렴했다)")
    v = pR[:, 0]*np.cos(np.radians(aL)) + pR[:, 1]*np.sin(np.radians(aL))
    rho = float(np.corrcoef(v, gR)[0, 1])
    say(f"  held-out: 좌반구 축으로 우반구를 재면 r = {rho:+.3f}")
    R["heldout_r"] = rho
    # 순열검정은 '최적축 탐색의 자유도'까지 보정한다 — 섞은 경사에도 축을 다시 찾아준다.
    ang = np.arange(0, 180, 1.0)
    P = (pR[:, 0][None, :]*np.cos(np.radians(ang))[:, None]
         + pR[:, 1][None, :]*np.sin(np.radians(ang))[:, None])   # (180, n)
    P = P - P.mean(1, keepdims=True)
    Pn = P/np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    obs = abs(best_axis(pR, gR)[1])
    rng = np.random.default_rng(0); hits = 0
    for _ in range(NPERM):
        gp = rng.permutation(gR); gp = gp - gp.mean()
        gn = np.linalg.norm(gp)
        if gn < 1e-12: continue
        if float(np.abs(Pn @ gp).max()/gn) >= obs: hits += 1
    p = (hits+1)/(NPERM+1)
    say(f"  순열검정 {NPERM}회 (우반구): p = {p:.4f}")
    R["perm_p"] = p

json.dump(R, open(OUT/"lc10a_feasible.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/lc10a_feasible.json")
