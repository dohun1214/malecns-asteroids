"""게이트 4 3단계: 추적 조향의 **판독 집단**을 연결성에서 찾는다.

도피 쪽(vnc_targets.py)과 같은 원칙이다:
  - **하행뉴런을 직접 읽지 않는다.** `alive` 가 출력만 막으므로 DNa02 끄고 DNa02 읽으면
    정의상 0 이 된다 — 동어반복이다 (10문서 §2).
  - 이름으로 고르지 않는다. **연결성 특이도**로 고른다.

도피는 DNp02 vs DNp11 (전후) 대비였다. 추적은 문헌이 **좌우 시소**라고 말한다:
  "AOTU025 is excitatory and projects ipsilaterally, AOTU019 is inhibitory and projects
   contralaterally" / "turn toward the side of higher DN activity"  (Neuron 2026)
그리고 우리 그래프에서 그 편측성이 100.0% / 0.0% 로 확인됐다 (08문서 §8.1).

그래서 특이도를 **좌 DN vs 우 DN** 으로 잡는다:
     특이도 = (좌 DNa 시냅스 - 우 DNa 시냅스) / (합)
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
sc = nodes["superclass"].astype("string").fillna("").to_numpy()
sd = nodes["somaSide"].astype("string").fillna("").to_numpy()
crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
w = np.abs(np.load(G/"out_w.npy").astype(np.int64))
N = crow.size - 1
pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
def say(*a): print(*a, flush=True)


def out_syn(src_idx):
    m = np.zeros(N, bool); m[src_idx] = True
    sel = m[pre]
    v = np.zeros(N, np.int64)
    np.add.at(v, post[sel], w[sel])
    return v


DN_SETS = {
    "DNa02": ["DNa02"],
    "DNa02+13+15+03": ["DNa02", "DNa13", "DNa15", "DNa03"],
}
rep = {}
save = {}
for label, types in DN_SETS.items():
    idx = np.concatenate([np.where(typ == t)[0] for t in types])
    iL = idx[sd[idx] == "L"]; iR = idx[sd[idx] == "R"]
    say(f"\n=== {label}: 좌 {len(iL)}세포 / 우 {len(iR)}세포 ===")
    sL, sR = out_syn(iL), out_syn(iR)
    tot = sL + sR
    tgt = np.flatnonzero(tot > 0)
    spec = (sL[tgt] - sR[tgt])/tot[tgt]
    say(f"  1홉 하류 {len(tgt):,}개 (시냅스 합 {tot.sum():,})")
    say(f"  {'집단':<22} {'n':>5} {'시냅스':>9} {'좌/우':>9}  대표 타입")
    grp = {}
    for lbl, m in (("좌 DN 전용 (>+0.8)", spec > 0.8),
                   ("우 DN 전용 (<-0.8)", spec < -0.8),
                   ("공통 (|s|<0.4)", np.abs(spec) < 0.4)):
        ii = tgt[m]; grp[lbl] = ii
        tt = pd.Series([str(x) for x in typ[ii]]).value_counts().head(4).to_dict()
        nl = int((sd[ii] == "L").sum()); nr = int((sd[ii] == "R").sum())
        say(f"  {lbl:<22} {len(ii):>5} {int(tot[ii].sum()):>9} {nl:>4}/{nr:<4}  {tt}")
    # VNC 로 한정 (도피 판독과 같은 기준 — 운동 출력만 본다)
    isvnc = np.array([str(x).startswith("vnc") for x in sc])
    L = tgt[spec > 0.8]; R = tgt[spec < -0.8]
    Lv = L[isvnc[L]]; Rv = R[isvnc[R]]
    say(f"  VNC 한정: 좌 전용 {len(Lv)}세포 / 우 전용 {len(Rv)}세포")
    rep[label] = dict(n_targets=int(len(tgt)), L=int(len(L)), R=int(len(R)),
                      Lv=int(len(Lv)), Rv=int(len(Rv)))
    save[label] = (L, R, Lv, Rv)

# 어느 집합을 쓸지 — 넓은 쪽(DNa02+13+15+03) 을 쓴다. DNa02 단독은 하류가 너무 적다.
L, R, Lv, Rv = save["DNa02+13+15+03"]
use_L, use_R = (Lv, Rv) if (len(Lv) >= 20 and len(Rv) >= 20) else (L, R)
say(f"\n사용: 좌 {len(use_L)}세포 / 우 {len(use_R)}세포"
    f"  ({'VNC 한정' if use_L is Lv else 'VNC 한정 아님 — 하류가 적어서'})")

# 🔴 교차 투사 점검 (10문서 §3 에서 도피 판독이 여기 걸렸다)
sL2 = out_syn(np.concatenate([np.where(typ == t)[0][sd[np.where(typ == t)[0]] == "L"]
                              for t in ["DNa02", "DNa13", "DNa15", "DNa03"]]))
sR2 = out_syn(np.concatenate([np.where(typ == t)[0][sd[np.where(typ == t)[0]] == "R"]
                              for t in ["DNa02", "DNa13", "DNa15", "DNa03"]]))
say(f"  좌 판독집단이 받는 시냅스: 좌DN {int(sL2[use_L].sum()):>7} / 우DN {int(sR2[use_L].sum()):>7}")
say(f"  우 판독집단이 받는 시냅스: 좌DN {int(sL2[use_R].sum()):>7} / 우DN {int(sR2[use_R].sum()):>7}")

np.savez(G/"pursuit_readout.npz", left=use_L.astype(np.int64), right=use_R.astype(np.int64),
         side_left=np.array([str(x) for x in sd[use_L]]),
         side_right=np.array([str(x) for x in sd[use_R]]))
rep["used"] = dict(left=int(len(use_L)), right=int(len(use_R)))
json.dump(rep, open(OUT/"pursuit_targets.json", "w"), ensure_ascii=False, indent=1)
say("\n-> graph/pursuit_readout.npz")
