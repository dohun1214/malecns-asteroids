"""게이트 4 (d) — rewire. **21문서 §7 의 1번 약점을 없애는 대조다.**

lesion 은 "LC10a 를 껐더니 조준이 0 이 됐다" 인데, 그건 이 모드의 **유일한 감각 입력을
끊은** 것이라 '방향이 틀어진다'가 아니라 '입력이 끊긴다'에 가깝다 (신호 있는 결정 0.0%).

rewire 는 **세포를 하나도 안 끄고, 차수·부호·가중치 다중집합을 전부 보존한 채**
`LC10a -> AOTU019/025` 의 연결 상대만 섞는다.
  -> 신호는 그대로 나온다. **방향만 뒤섞인다.** 그게 요점이다.

조건
  온전
  lc10a_aotu    LC10a -> AOTU019/AOTU025 배선 섞기. 25 / 50 / 100%, 시드 3개
  lc10a_other   LC10a 의 **다른** 출력(AOTU019/025 제외)을 같은 개수 섞기  ← 특이성 대조
  lc4_dn        도피 회로(LC4->DN) 섞기                                 ← 계통 간 특이성
  ctrl_matched  아무 곳의 흥분성 엣지를 같은 개수 섞기                    ← 크기 대조

판정은 **① 신호가 살아 있는가(|채널| 범위) ② 방향이 뒤섞였는가(방위와의 r)** 둘 다 본다.
신호까지 죽으면 lesion 과 같은 얘기가 되어 이 대조의 의미가 없다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS
from vision import LC10aMap
import rewire as RW

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
STEPS = 333
TH50, CAP, K = 7.2, 150.0, 16
PHIS = np.arange(-180, 181, 15.0)
SEEDS = (0, 1, 2)
def say(*a): print(*a, flush=True)

nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
LC10A = np.where(typ == "LC10a")[0]
AOTU = np.concatenate([np.where(typ == t)[0] for t in ("AOTU019", "AOTU025")])
C = dict(np.load(G/"circuit_idx.npz"))
P = np.load(G/"lc10a_position.npz", allow_pickle=True)
RD = dict(np.load(G/"pursuit_readout.npz", allow_pickle=True))
mp = LC10aMap(P["idx"], P["pos"], P["side"], P["valid"], float(P["theta_L"]), k=K)

crow, packed0, _, pre = RW.load()
N = crow.size - 1
post0, w0 = RW.unpack(packed0)

sel_aotu = RW.select(N, pre, packed0, LC10A, AOTU)
lc10a_mask = np.zeros(N, bool); lc10a_mask[LC10A] = True
aotu_mask = np.zeros(N, bool); aotu_mask[AOTU] = True
sel_other_all = np.flatnonzero(lc10a_mask[pre] & ~aotu_mask[post0])
DN = np.concatenate([C[k] for k in RW.DN_ALL if k in C])
sel_lc4dn = RW.select(N, pre, packed0, C["LC4"], DN)
rng0 = np.random.default_rng(99)
exc = np.flatnonzero((w0 > 0) & ~lc10a_mask[pre])
sel_other = sel_other_all[rng0.choice(len(sel_other_all), size=len(sel_aotu), replace=False)]
sel_ctrl = exc[rng0.choice(len(exc), size=len(sel_aotu), replace=False)]
say(f"전체 엣지 {len(packed0):,}")
say(f"  lc10a_aotu   {len(sel_aotu):6d}개 (전체의 {len(sel_aotu)/len(packed0)*100:.4f}%)")
say(f"  lc10a_other  {len(sel_other):6d}개 (LC10a 의 다른 출력 {len(sel_other_all):,} 중)")
say(f"  lc4_dn       {len(sel_lc4dn):6d}개")
say(f"  ctrl_matched {len(sel_ctrl):6d}개")

b = BrainRT(params=PARAMS)
b.reset(); b.set_poisson(LC10A, 1.0)
b.capture(STEPS, tally=True)
L, R = RD["left"], RD["right"]


def set_packed(arr):
    b.packed.copy_(torch.from_numpy(arr.astype(np.int32)).to(b.dev))


def sweep():
    out = []
    for n, phi in enumerate(PHIS):
        loom = [dict(phi_rel=float(phi), theta=7.2, dtheta=1.0, dist=50.0)]
        idx, rates = mp.rates(loom, b.N, TH50, CAP)
        b.set_poisson_rates(idx, rates)
        b.tally.zero_(); b.gacc.zero_()
        b.seed.fill_(n); b.graph.replay(); torch.cuda.synchronize()
        g = (b.gacc/STEPS).cpu().numpy()
        out.append(float(g[L].mean()) - float(g[R].mean()))
    return np.array(out)


def score(ch):
    m = np.abs(PHIS) > 1e-9
    r = float(np.corrcoef(PHIS, ch)[0, 1]) if np.std(ch) > 1e-12 else 0.0
    agree = float(np.mean(np.sign(ch[m]) == -np.sign(PHIS[m])))
    return dict(r=r, agree=agree, amp=float(np.abs(ch).max()),
                sd=float(np.std(ch)))


Rres = {}
set_packed(packed0); b.reset(); b.alive.fill_(1)
base = score(sweep())
Rres["온전"] = base
say(f"\n{'조건':<26} {'|채널|최대':>9} {'채널 sd':>8} {'방위와 r':>9} {'표적쪽 비율':>10}")
say("-"*68)
say(f"{'온전':<26} {base['amp']:9.4f} {base['sd']:8.4f} {base['r']:+9.3f} "
    f"{base['agree']*100:9.1f}%")

CONDS = [("lc10a_aotu", sel_aotu, (0.25, 0.5, 1.0)),
         ("lc10a_other", sel_other, (1.0,)),
         ("lc4_dn", sel_lc4dn, (1.0,)),
         ("ctrl_matched", sel_ctrl, (1.0,))]
for name, sel, fracs in CONDS:
    for fr in fracs:
        accs = []
        for sd in SEEDS:
            pk, moved = RW.rewire(packed0, sel, seed=sd, frac=fr)
            set_packed(pk); b.reset(); b.alive.fill_(1)
            accs.append(score(sweep()))
        g = lambda k: float(np.mean([a[k] for a in accs]))
        lbl = f"{name} {int(fr*100)}%"
        Rres[lbl] = dict(r=g("r"), agree=g("agree"), amp=g("amp"), sd=g("sd"),
                         r_sd=float(np.std([a["r"] for a in accs])))
        say(f"{lbl:<26} {g('amp'):9.4f} {g('sd'):8.4f} {g('r'):+9.3f} "
            f"{g('agree')*100:9.1f}%   (r sd {np.std([a['r'] for a in accs]):.3f})")

set_packed(packed0)
say("\n=== 판정 ===")
say(f"  온전 r = {base['r']:+.3f}, 무정보 = 0.000")
for k in Rres:
    if k == "온전": continue
    lost = (base["r"] - Rres[k]["r"])/base["r"]*100 if base["r"] != 0 else 0
    keep = Rres[k]["amp"]/base["amp"]*100
    say(f"  {k:<26} 방향 소실 {lost:6.1f}%   신호 크기 유지 {keep:6.1f}%")
json.dump(Rres, open(OUT/"gate4_rewire.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/gate4_rewire.json")
