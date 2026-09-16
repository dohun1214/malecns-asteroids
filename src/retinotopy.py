"""LC4 의 시야 위치를 DNp02/DNp11 과 무관하게 복원한다.

MaleCNS 에는 LC4 의 망막위상 좌표가 없다(assignedOlHex 는 컬럼형 세포만).
그런데 LC4 로 들어오는 Tm2/Tm4 는 컬럼형이라 육각 좌표를 가진다.
=> LC4 의 시야 위치 = 그 입력 컬럼들의 (시냅스 수 가중) 육각 좌표 무게중심.

이 좌표는 DNp02/DNp11 시냅스 수와 아무 관련이 없다. 그래서 다음 질문이 성립한다:
  "진짜 위치가 앞쪽인 LC4 가, 알고 보니 DNp02 에 더 많이 연결돼 있는가?"
이건 구성상 참이 아니다. Dombrovski 2023 의 핵심 주장이고, 참일 이유가 없다.

검증 두 겹:
  (1) held-out  — 좌반구에서 축을 찾고 우반구에서 시험. 축 맞추기 반박 차단
  (2) 순열검정  — 위치를 LC4 끼리 섞은 귀무분포와 비교. 최적축 탐색의 자유도까지 보정
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NPERM   = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
R = {"min_syn": MIN_SYN, "n_perm": NPERM}
def say(*a): print(*a, flush=True)

ann   = pd.read_feather(ROOT.parent/"malecns-song"/"data"/"body-annotations.feather")
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
C     = np.load(G/"circuit_idx.npz")
crow  = np.load(G/"out_crow.npy").astype(np.int64)
post  = np.load(G/"out_post.npy").astype(np.int64)
wt    = np.abs(np.load(G/"out_w.npy").astype(np.int64))
N = crow.size - 1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))

# bodyId -> 그래프 인덱스
gidx = {int(b): i for i, b in enumerate(nodes["bodyId"].to_numpy())}
hexdf = ann[ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()]
hex1 = np.full(N, np.nan); hex2 = np.full(N, np.nan)
for b, h1, h2 in zip(hexdf["bodyId"].to_numpy(),
                     hexdf["assignedOlHex1"].to_numpy(),
                     hexdf["assignedOlHex2"].to_numpy()):
    i = gidx.get(int(b))
    if i is not None:
        hex1[i] = float(h1); hex2[i] = float(h2)
has_hex = ~np.isnan(hex1)
say(f"육각 좌표 보유 뉴런 {int(has_hex.sum()):,} / {N:,}")

# ---------------------------------------------- LC4 위치 = 입력 컬럼의 가중 무게중심
lc4 = C["LC4"]; lc4_set = {int(x): k for k, x in enumerate(lc4)}
is_lc4 = np.zeros(N, bool); is_lc4[lc4] = True
sel = is_lc4[post] & has_hex[pre]
p_, q_, w_ = pre[sel], post[sel], wt[sel]
pos = np.full((lc4.size, 2), np.nan); nsyn = np.zeros(lc4.size)
acc = np.zeros((lc4.size, 2)); tot = np.zeros(lc4.size)
for a, b, c in zip(p_, q_, w_):
    k = lc4_set[int(b)]
    acc[k, 0] += hex1[a]*c; acc[k, 1] += hex2[a]*c; tot[k] += c
ok = tot >= MIN_SYN
pos[ok] = acc[ok]/tot[ok, None]; nsyn = tot
side = nodes["somaSide"].astype("string").fillna("").to_numpy()[lc4]
say(f"\nLC4 위치 복원: {int(ok.sum())}/{lc4.size}개 (육각 입력 시냅스 >= {MIN_SYN})")
say(f"  입력 시냅스 수 중앙값 {np.median(tot):.0f}, 최소 {tot.min():.0f}, 최대 {tot.max():.0f}")
for h in ("L","R"):
    m = ok & (side == h)
    if m.sum():
        say(f"  {h}반구 {int(m.sum())}개: hex1 {pos[m,0].min():.1f}~{pos[m,0].max():.1f} "
            f"(퍼짐 {pos[m,0].std():.1f}), hex2 {pos[m,1].min():.1f}~{pos[m,1].max():.1f} "
            f"(퍼짐 {pos[m,1].std():.1f})")
R["recovered"] = int(ok.sum()); R["n_lc4"] = int(lc4.size)

# ---------------------------------------------- AP 점수 (DNp02 - DNp11)
def syn_to(dn):
    isd = np.zeros(N, bool); isd[C[dn]] = True
    s = isd[post] & is_lc4[pre]
    out = np.zeros(lc4.size)
    for a, c in zip(pre[s], wt[s]):
        out[lc4_set[int(a)]] += c
    return out
s02, s11 = syn_to("DNp02"), syn_to("DNp11")
den = s02 + s11
ap = np.where(den > 0, (s02 - s11)/np.maximum(den, 1), np.nan)
say(f"  AP 점수 계산 가능한 LC4 {int((den>0).sum())}개 "
    f"(DNp02+DNp11 시냅스 중앙값 {np.median(den[den>0]):.0f})")

# ---------------------------------------------- 축 찾기 / 검정
THETA = np.linspace(0, np.pi, 181, endpoint=False)
def proj(P, th): return P[:,0]*np.cos(th) + P[:,1]*np.sin(th)
def best_axis(P, y):
    rs = np.array([abs(np.corrcoef(proj(P, t), y)[0,1]) for t in THETA])
    k = int(np.nanargmax(rs)); return THETA[k], rs[k]

valid = ok & (den > 0) & ~np.isnan(ap)
say(f"\n{'='*70}\n위치 <-> DNp02/DNp11 경사 (Dombrovski 2023 의 핵심 주장)")
res = {}
for h in ("L","R"):
    m = valid & (side == h)
    P, y = pos[m], ap[m]
    r1 = np.corrcoef(P[:,0], y)[0,1]; r2 = np.corrcoef(P[:,1], y)[0,1]
    th, rb = best_axis(P, y)
    say(f"  {h}반구 n={int(m.sum())}:  hex1축 r={r1:+.3f}  hex2축 r={r2:+.3f}   "
        f"최적축 {np.degrees(th):.0f}도에서 |r|={rb:.3f}")
    res[h] = dict(n=int(m.sum()), r_hex1=r1, r_hex2=r2, theta=float(th), r_best=float(rb),
                  P=P, y=y)

# (1) held-out: 좌에서 찾은 축을 우에 그대로
thL = res["L"]["theta"]
rR_same = np.corrcoef(proj(res["R"]["P"], thL), res["R"]["y"])[0,1]
rR_mirr = np.corrcoef(proj(res["R"]["P"], np.pi - thL), res["R"]["y"])[0,1]
say(f"\n  [held-out] 좌반구 최적축({np.degrees(thL):.0f}도)을 우반구에 그대로 적용: r={rR_same:+.3f}")
say(f"             거울상 축({np.degrees(np.pi-thL):.0f}도) 적용:              r={rR_mirr:+.3f}")
say(f"             (시엽 2개는 거울 대칭이라 둘 중 하나가 맞아야 정상)")
R["heldout_R_same"], R["heldout_R_mirror"] = float(rR_same), float(rR_mirr)

# (2) 순열검정: 최적축 탐색의 자유도까지 보정
rng = np.random.default_rng(0)
say(f"\n  [순열검정] LC4 끼리 위치를 섞은 귀무분포 {NPERM}회 (최적축 탐색 포함)")
for h in ("L","R"):
    P, y = res[h]["P"], res[h]["y"]
    obs = res[h]["r_best"]
    null = np.empty(NPERM)
    for i in range(NPERM):
        null[i] = best_axis(P[rng.permutation(P.shape[0])], y)[1]
    p = (np.sum(null >= obs) + 1)/(NPERM + 1)
    say(f"    {h}반구: 실측 |r|={obs:.3f}, 귀무 중앙값 {np.median(null):.3f} "
        f"(95퍼센타일 {np.percentile(null,95):.3f})  ->  p = {p:.4f}"
        f"{'  유의' if p < 0.05 else '  유의하지 않음'}")
    R[f"perm_{h}"] = dict(obs=float(obs), null_med=float(np.median(null)),
                          null_p95=float(np.percentile(null,95)), p=float(p))

np.savez(G/"lc4_position.npz", idx=lc4, pos=pos, nsyn=nsyn, ap=ap,
         side=np.array([str(x) for x in side]), valid=valid,
         theta_L=res["L"]["theta"], theta_R=res["R"]["theta"])
(OUT/"retinotopy.json").write_text(json.dumps(
    {k: v for k, v in R.items()}, indent=2, default=float), encoding="utf-8")
say(f"\n-> graph/lc4_position.npz, out/retinotopy.json")
