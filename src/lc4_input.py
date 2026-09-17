"""LC4 **입력**의 나머지 81% 가 무엇이고, 거기에도 위치 정보가 있는가
(08문서 §4-5 / 07문서 §7 / 12문서 §6 의 미해결 항목).

적어둔 것: "Tm2/Tm4 가 LC4 입력의 19% 뿐이다. 나머지 81% 가 다른 위치 정보를 담을 수 있다."
한 번도 세어보지도, 검사하지도 않았다.

두 가지를 한다.
  A 회계 — LC4 가 받는 입력을 타입·상위분류별로, 그리고 **육각 좌표 보유 여부**로 나눈다.
    ⚠️ retinotopy.py 는 사실 Tm2/Tm4 만 쓴 게 아니라 **육각 좌표를 가진 입력 전부**를 쓴다
       (`sel = is_lc4[post] & has_hex[pre]`). 그러니 "19%" 부터 실측으로 확인해야 한다.
  B 검사 — 좌표가 **없는** 입력만으로 LC4 의 위치를 예측할 수 있는가.
    LC4 위치는 좌표 **있는** 입력으로 정의했으므로, 좌표 없는 입력으로 그걸 맞히면
    "나머지에도 위치 정보가 있다" 가 된다. 순환이 아니다 — 입력원이 서로 겹치지 않는다.
    교차검증(반구별 held-out)으로 잰다.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
SONG = ROOT.parent/"malecns-song"
crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
w = np.load(G/"out_w.npy")
C = dict(np.load(G/"circuit_idx.npz"))
P = np.load(G/"lc4_position.npz", allow_pickle=True)
nodes = pd.read_feather(SONG/"graph"/"nodes.feather")
TYPE = nodes["type"].astype("string").fillna("(없음)").to_numpy()
SUPER = nodes["superclass"].astype("string").fillna("(없음)").to_numpy()
N = crow.size - 1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
def say(*a): print(*a, flush=True)

LC4 = np.asarray(C["LC4"])
lc4_idx, pos, side, valid = P["idx"], P["pos"], P["side"], P["valid"]

# 육각 좌표 보유 뉴런 (retinotopy.py 와 같은 방식)
ann = pd.read_feather(SONG/"data"/"body-annotations.feather")
body = np.load(SONG/"graph"/"body.npy")
bid2i = {int(b): i for i, b in enumerate(body)}
hexdf = ann[ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()]
hex1 = np.full(N, np.nan); hex2 = np.full(N, np.nan)
for b, h1, h2 in zip(hexdf["bodyId"].to_numpy(), hexdf["assignedOlHex1"].to_numpy(),
                     hexdf["assignedOlHex2"].to_numpy()):
    i = bid2i.get(int(b))
    if i is not None: hex1[i] = float(h1); hex2[i] = float(h2)
HAS = ~np.isnan(hex1)
say(f"육각 좌표 보유 뉴런 {int(HAS.sum()):,} / {N:,}\n")

# ── A. 입력 회계 ────────────────────────────────────────────────────────────
m = np.isin(post, LC4)
ip, iw = pre[m], np.abs(w[m])
say("="*86)
say(f"A. LC4 {len(LC4)}세포가 받는 입력   엣지 {len(ip):,} / 시냅스 {iw.sum():,.0f}")
hm = HAS[ip]
say(f"   육각 좌표 **있는** 입력  엣지 {int(hm.sum()):,} ({hm.mean()*100:.1f}%)"
    f"  시냅스 {iw[hm].sum():,.0f} (**{iw[hm].sum()/iw.sum()*100:.1f}%**)")
say(f"   육각 좌표 **없는** 입력  엣지 {int((~hm).sum()):,}"
    f"  시냅스 {iw[~hm].sum():,.0f} ({iw[~hm].sum()/iw.sum()*100:.1f}%)")
out = dict(in_edge=int(len(ip)), in_syn=float(iw.sum()),
           hex_syn=float(iw[hm].sum()), nohex_syn=float(iw[~hm].sum()))


def table(mask, key, label, top=12):
    t = pd.DataFrame(dict(k=key[ip[mask]], w=iw[mask]))
    g = t.groupby("k").agg(엣지=("w", "size"), 시냅스=("w", "sum"))
    g["전체%"] = g["시냅스"]/iw.sum()*100
    g = g.sort_values("시냅스", ascending=False)
    say(f"\n{label}")
    say(f"   {'':<24}{'엣지':>8}{'시냅스':>10}{'전체%':>8}")
    for k, r in g.head(top).iterrows():
        say(f"   {str(k)[:24]:<24}{int(r['엣지']):>8,}{int(r['시냅스']):>10,}{r['전체%']:>7.1f}%")
    say(f"   {'(나머지 전부)':<24}{int(g['엣지'][top:].sum()):>8,}"
        f"{int(g['시냅스'][top:].sum()):>10,}{g['전체%'][top:].sum():>7.1f}%")
    return {str(k): float(r["시냅스"]) for k, r in g.head(top).iterrows()}


out["hex_types"] = table(hm, TYPE, "좌표 있는 입력의 타입 (상위 12)")
out["nohex_types"] = table(~hm, TYPE, "좌표 없는 입력의 타입 (상위 12)")
out["nohex_super"] = table(~hm, SUPER, "좌표 없는 입력의 상위분류", 8)

# ── B. 좌표 없는 입력만으로 위치를 맞힐 수 있나 ────────────────────────────
say("\n" + "="*86)
say("B. 좌표 **없는** 입력만으로 LC4 의 위치를 예측할 수 있는가")
say("   (위치는 좌표 **있는** 입력으로 정의했다. 입력원이 안 겹치므로 순환이 아니다)")

TH = float(P["theta_L"])
axis = pos[:, 0]*np.cos(TH) + pos[:, 1]*np.sin(TH)      # 전후 시야축
lc4_pos_of = {int(v): i for i, v in enumerate(lc4_idx)}

# 특징: LC4 세포 x (좌표 없는 입력원 타입) 시냅스 행렬
nh_pre = ip[~hm]; nh_w = iw[~hm]; nh_post = post[m][~hm]
tnames, tcode = np.unique(TYPE[nh_pre], return_inverse=True)
rows = np.array([lc4_pos_of.get(int(p_), -1) for p_ in nh_post])
keep = rows >= 0
X = np.zeros((len(lc4_idx), len(tnames)), dtype=np.float64)
np.add.at(X, (rows[keep], tcode[keep]), nh_w[keep])
say(f"   특징 행렬 {X.shape}  (LC4 세포 x 좌표없는 입력 타입)")
X = X/np.maximum(X.sum(1, keepdims=True), 1e-9)          # 세포별 정규화 (총량 효과 제거)


def cv_r(Xm, y, groups, lam=1.0):
    """반구를 나눠 교차검증. 한쪽으로 학습하고 다른 쪽을 맞힌다."""
    rs = []
    for h in np.unique(groups):
        tr = groups != h; te = ~tr
        if te.sum() < 5 or tr.sum() < 5: continue
        A = Xm[tr]; b = y[tr]
        A = np.c_[A, np.ones(len(A))]
        coef = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@b)
        p = np.c_[Xm[te], np.ones(int(te.sum()))]@coef
        if np.std(p) < 1e-12: rs.append(0.0); continue
        rs.append(float(np.corrcoef(p, y[te])[0, 1]))
    return rs


vm = valid
g = np.array([s for s in side])[vm]
r_real = cv_r(X[vm], axis[vm], g)
say(f"   반구 교차검증 r = " + " / ".join(f"{r:+.3f}" for r in r_real))
# 귀무분포: 위치를 섞는다
rng = np.random.default_rng(0)
null = []
for _ in range(500):
    y2 = axis[vm].copy(); rng.shuffle(y2)
    null.append(np.mean(cv_r(X[vm], y2, g)))
null = np.asarray(null); obs = float(np.mean(r_real))
p_emp = float((np.abs(null) >= abs(obs)).mean())
say(f"   평균 r = {obs:+.3f}   순열검정 500회 p = {p_emp:.4f}"
    f"  (귀무 |r| 평균 {np.abs(null).mean():.3f})")
out["cv_r"] = r_real; out["cv_mean"] = obs; out["p"] = p_emp

say("\n" + "="*86)
say("판정")
hexpct = iw[hm].sum()/iw.sum()*100
say(f"  좌표 있는 입력은 시냅스의 {hexpct:.1f}% 다 (문서의 '19%' 와 대조).")
if p_emp < 0.05 and abs(obs) > 0.2:
    say(f"  좌표 **없는** 입력만으로도 위치가 예측된다 (r={obs:+.3f}, p={p_emp:.4f}).")
    say("  -> 나머지에도 위치 정보가 있다. '19% 만으로 복원한 게 요행이 아니다' 는 뜻이고,")
    say("     동시에 '나머지 81% 가 위치와 무관하다' 는 가정도 틀렸다는 뜻이다.")
else:
    say(f"  좌표 없는 입력으로는 위치가 예측되지 않는다 (r={obs:+.3f}, p={p_emp:.4f}).")
    say("  -> 위치 정보는 좌표 있는 입력에 몰려 있다. 나머지는 다른 일을 한다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"lc4_input.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
say("-> out/lc4_input.json")
