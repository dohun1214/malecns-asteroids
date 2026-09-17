"""게이트 4 (d) 재측정 — 시드 10개, 용량 6단계.

첫 측정(시드 3개, 25/50/100%)에서 **용량 반응이 매우 비선형**으로 나왔다.
25/50% 는 거의 안 무너지는데 100% 에서 급락했고, 시드 간 변동도 100% 에서만 컸다
(r 표준편차 0.474 vs 0.008/0.009). 시드 3개로는 그게 진짜인지 표본 탓인지 모른다.

  - 시드 3 -> **10**
  - 용량 25/50/100 -> **10/25/50/75/90/100%**
  - 대조군도 시드 10개로

'섞는 엣지가 335개뿐이라 부분 치환 효과가 작다'는 가설이 맞으면, 낮은 용량에서
실제로 **바뀌는 엣지 수**가 적어야 한다 — 치환이 제자리로 돌아오는 비율까지 같이 센다.
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
STEPS = 333; TH50, CAP, K = 7.2, 150.0, 16
PHIS = np.arange(-180, 181, 15.0)
SEEDS = list(range(10))
DOSES = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
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
sel = RW.select(N, pre, packed0, LC10A, AOTU)
lm = np.zeros(N, bool); lm[LC10A] = True
am = np.zeros(N, bool); am[AOTU] = True
other_all = np.flatnonzero(lm[pre] & ~am[post0])
DN = np.concatenate([C[k] for k in RW.DN_ALL if k in C])
sel_lc4dn = RW.select(N, pre, packed0, C["LC4"], DN)
rng0 = np.random.default_rng(99)
exc = np.flatnonzero((w0 > 0) & ~lm[pre])
sel_other = other_all[rng0.choice(len(other_all), size=len(sel), replace=False)]
sel_ctrl = exc[rng0.choice(len(exc), size=len(sel), replace=False)]

b = BrainRT(params=PARAMS)
b.reset(); b.set_poisson(LC10A, 1.0); b.capture(STEPS, tally=True)
L, R = RD["left"], RD["right"]


def set_packed(a): b.packed.copy_(torch.from_numpy(a.astype(np.int32)).to(b.dev))


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


def rr(ch):
    return float(np.corrcoef(PHIS, ch)[0, 1]) if np.std(ch) > 1e-12 else 0.0


set_packed(packed0); b.reset(); b.alive.fill_(1)
base_ch = sweep(); base_r = rr(base_ch); base_amp = float(np.abs(base_ch).max())
say(f"섞는 엣지 {len(sel)}개.  온전 r = {base_r:+.3f}, |채널|최대 {base_amp:.4f}\n")
Rs = {"온전": dict(r=base_r, amp=base_amp)}

say(f"{'용량':>6} {'실제 바뀐 엣지':>12} {'방위와 r (평균±sd)':>22} {'방향 소실':>9} "
    f"{'신호 유지':>9} {'r>−0.5 시드':>11}")
say("-"*80)
for fr in DOSES:
    rsv, ampv, chg = [], [], []
    for sd in SEEDS:
        pk, moved = RW.rewire(packed0, sel, seed=sd, frac=fr)
        chg.append(int((pk != packed0).sum()))
        set_packed(pk); b.reset(); b.alive.fill_(1)
        ch = sweep(); rsv.append(rr(ch)); ampv.append(float(np.abs(ch).max()))
    m, s_ = float(np.mean(rsv)), float(np.std(rsv))
    lost = (base_r - m)/base_r*100
    keep = float(np.mean(ampv))/base_amp*100
    weak = int(sum(1 for x in rsv if x > -0.5))
    say(f"{int(fr*100):5d}% {np.mean(chg):12.0f} {m:+12.3f} ± {s_:5.3f} "
        f"{lost:8.1f}% {keep:8.1f}% {weak:8d}/10")
    Rs[f"lc10a_aotu {int(fr*100)}%"] = dict(r=m, r_sd=s_, lost=lost, keep=keep,
                                            changed=float(np.mean(chg)),
                                            rs=[float(x) for x in rsv])

say("\n대조군 (100%, 시드 10개)")
for name, sl in (("lc10a_other", sel_other), ("lc4_dn", sel_lc4dn),
                 ("ctrl_matched", sel_ctrl)):
    rsv, ampv = [], []
    for sd in SEEDS:
        pk, _ = RW.rewire(packed0, sl, seed=sd, frac=1.0)
        set_packed(pk); b.reset(); b.alive.fill_(1)
        ch = sweep(); rsv.append(rr(ch)); ampv.append(float(np.abs(ch).max()))
    m, s_ = float(np.mean(rsv)), float(np.std(rsv))
    say(f"  {name:<14} r = {m:+.3f} ± {s_:.3f}   방향 소실 {(base_r-m)/base_r*100:6.1f}%"
        f"   신호 유지 {np.mean(ampv)/base_amp*100:6.1f}%")
    Rs[name] = dict(r=m, r_sd=s_, lost=(base_r-m)/base_r*100,
                    keep=float(np.mean(ampv))/base_amp*100)

set_packed(packed0)
json.dump(Rs, open(OUT/"gate4_rewire2.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/gate4_rewire2.json")
