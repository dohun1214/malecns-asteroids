"""LC4 의 출력 76% / 입력 81% 가 어디로 가는지 회계 (12문서 §6, 08문서 §4 미해결).

지금까지 적어둔 것: LC4 출력 엣지 12,253 중 하행뉴런행은 824 (6.7%) 뿐이고
**나머지가 어디로 가는지 모른다.** rewire 에서 "LC4 -> DN 만 섞으면 80% 만 무너지고
전체를 섞어야 132% 무너진다" 가 나온 것도 여기서 온다 — 방향 정보가 직접 시냅스에만
있지 않다는 뜻이다. 그런데 '나머지'가 무엇인지를 한 번도 세어본 적이 없다.

시뮬 그래프(w>=3) 그대로 센다. 타입·상위분류별로 엣지/시냅스를 집계하고,
**하행뉴런까지 2단계로 돌아오는 경로**가 얼마나 되는지도 본다.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
crow = np.load(G/"out_crow.npy"); post = np.load(G/"out_post.npy"); w = np.load(G/"out_w.npy")
C = dict(np.load(G/"circuit_idx.npz"))
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
TYPE = nodes["type"].astype("string").fillna("(없음)").to_numpy()
SUPER = nodes["superclass"].astype("string").fillna("(없음)").to_numpy()
SIGN = nodes["sign"].to_numpy()
def say(*a): print(*a, flush=True)

LC4 = np.asarray(C["LC4"])
DN = np.unique(np.concatenate([C[k] for k in C if k.startswith("DNp")
                               and not k.endswith(("_L", "_R"))]))

def out_of(srcs):
    ii, ww = [], []
    for s in np.atleast_1d(srcs):
        a, b = crow[int(s)], crow[int(s)+1]
        ii.append(post[a:b]); ww.append(w[a:b])
    return np.concatenate(ii), np.concatenate(ww)


tgt, wt = out_of(LC4)
say(f"LC4 {len(LC4)}세포의 출력 엣지 {len(tgt):,} / 시냅스 {wt.sum():,.0f}")
dn_m = np.isin(tgt, DN)
say(f"  하행뉴런(DNp*)행  엣지 {int(dn_m.sum()):,} ({dn_m.mean()*100:.1f}%)"
    f"  시냅스 {wt[dn_m].sum():,.0f} ({wt[dn_m].sum()/wt.sum()*100:.1f}%)")
say(f"  나머지            엣지 {int((~dn_m).sum()):,}  시냅스 {wt[~dn_m].sum():,.0f}\n")

out = {"n_edge": int(len(tgt)), "n_syn": float(wt.sum()),
       "dn_edge": int(dn_m.sum()), "dn_syn": float(wt[dn_m].sum())}


def table(mask, key, label, top=12):
    t = pd.DataFrame(dict(k=key[tgt[mask]], w=wt[mask]))
    g = t.groupby("k").agg(엣지=("w", "size"), 시냅스=("w", "sum"))
    g["시냅스%"] = g["시냅스"]/wt.sum()*100
    g = g.sort_values("시냅스", ascending=False)
    say(f"{label} (상위 {top})")
    say(f"   {'':<22}{'엣지':>8}{'시냅스':>10}{'시냅스%':>9}")
    for k, r in g.head(top).iterrows():
        say(f"   {str(k)[:22]:<22}{int(r['엣지']):>8,}{int(r['시냅스']):>10,}{r['시냅스%']:>8.1f}%")
    say(f"   {'(나머지 전부)':<22}{int(g['엣지'][top:].sum()):>8,}"
        f"{int(g['시냅스'][top:].sum()):>10,}{g['시냅스%'][top:].sum():>8.1f}%\n")
    return {str(k): dict(edge=int(r["엣지"]), syn=float(r["시냅스"]),
                         pct=float(r["시냅스%"])) for k, r in g.head(top).iterrows()}


out["by_super"] = table(np.ones(len(tgt), bool), SUPER, "상위분류별 (전체)", 10)
out["by_type_nondn"] = table(~dn_m, TYPE, "타입별 — 하행뉴런을 뺀 나머지", 15)

# 부호
say("부호별 (LC4 는 전부 콜린성이라 여기 부호는 **받는 쪽** 세포의 부호다)")
for s, lab in ((1, "흥분성 세포로"), (-1, "억제성 세포로"), (0, "부호 0")):
    m = SIGN[tgt] == s
    say(f"   {lab:<14} 엣지 {int(m.sum()):>7,}  시냅스 {wt[m].sum():>10,.0f}"
        f"  ({wt[m].sum()/wt.sum()*100:5.1f}%)")

# 2단계로 하행뉴런에 닿는가
say("\n2단계 경로 — LC4 -> (중간 1개) -> 하행뉴런")
mid = np.unique(tgt[~dn_m])
hit, hw = [], 0.0
for m_ in mid:
    a, b = crow[int(m_)], crow[int(m_)+1]
    sel = np.isin(post[a:b], DN)
    if sel.any(): hit.append(int(m_)); hw += float(w[a:b][sel].sum())
hit = np.asarray(hit)
in_w = wt[~dn_m][np.isin(tgt[~dn_m], hit)].sum()
say(f"   1단계 표적 {len(mid):,}세포 중 **{len(hit):,}세포**가 하행뉴런으로 직접 투사한다"
    f" ({len(hit)/max(len(mid),1)*100:.1f}%)")
say(f"   그 세포들이 LC4 에서 받는 시냅스 {in_w:,.0f}"
    f" (LC4 비-DN 출력의 {in_w/max(wt[~dn_m].sum(),1)*100:.1f}%)")
say(f"   그 세포들이 하행뉴런에 주는 시냅스 {hw:,.0f}")
top = pd.DataFrame(dict(k=TYPE[hit])).value_counts().head(10)
say("   그 중간 세포들의 타입 상위 10")
for (k,), v in top.items():
    say(f"     {str(k)[:26]:<26} {int(v):>5}세포")
out["two_step"] = dict(mid=int(len(mid)), relay=int(len(hit)),
                       lc4_to_relay_syn=float(in_w), relay_to_dn_syn=float(hw))

say("\n" + "="*84)
say("정리")
say(f"  LC4 출력의 {wt[dn_m].sum()/wt.sum()*100:.1f}% 만 하행뉴런으로 직접 간다.")
say(f"  나머지의 {in_w/max(wt[~dn_m].sum(),1)*100:.0f}% 는 **한 다리 건너 하행뉴런에 닿는 세포**로 간다.")
say("  -> 12문서의 'LC4->DN 만 섞으면 80%, 전체를 섞어야 132% 무너진다' 와 맞는 그림이다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"lc4_budget.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
say("-> out/lc4_budget.json")
