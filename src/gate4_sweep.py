"""게이트 4 (a) — 추적 회로가 표적의 방위를 조향 신호로 바꾸는가.

게이트 1(a)의 추적판이다. 폐루프 없이, 합성 표적 하나를 방위각으로 쓸면서
**LC10a 자극 -> AOTU019/025 -> DNa -> 하류 판독**이 좌우로 갈리는지만 본다.

판독 규칙은 우리가 고르지 않았다 — 문헌과 우리 그래프가 같이 말한 것이다 (08문서 §8.1):
  AOTU025 흥분성·동측 100% / AOTU019 억제성·대측 100% -> "turn toward the higher side"
그래서 조향 = (좌 판독 - 우 판독). 하류 판독 집단은 연결성 특이도로 뽑았다
(`pursuit_targets.py`, 좌 550 / 우 526 VNC 세포, 교차 투사 0.1% 미만).

대조 3종
  lesion:LC10a   LC10a 를 끄면 무너지는가
  무작위 자극    같은 수(275)의 무작위 세포를 대신 때리면 방위 정보가 나오는가
  무작위 판독    같은 크기의 무작위 하류 집단으로 읽으면 나오는가
"""
import sys, json
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS
from vision import LC10aMap

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"; OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
STEPS = 333
TH50, CAP, K = 7.2, 150.0, 16          # LC4 와 동일. 새 상수 없음
PHIS = np.arange(-180, 181, 15.0)
def say(*a): print(*a, flush=True)

nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
typ = nodes["type"].astype("string").fillna("").to_numpy()
LC10A = np.where(typ == "LC10a")[0]
P = np.load(G/"lc10a_position.npz", allow_pickle=True)
RD = dict(np.load(G/"pursuit_readout.npz", allow_pickle=True))
say(f"LC10a {len(LC10A)}세포 / 판독 좌 {len(RD['left'])} 우 {len(RD['right'])}")

mp = LC10aMap(P["idx"], P["pos"], P["side"], P["valid"], float(P["theta_L"]), k=K)
b = BrainRT(params=PARAMS)
b.reset(); b.set_poisson(LC10A, 1.0)        # rfc=0 까지 LC4 와 동일하게
b.capture(STEPS, tally=True)


def channel(gac, left, right):
    l = float(gac[left].mean()); r = float(gac[right].mean())
    return l, r, l - r


def sweep(lesion=None, drive_idx=None, readout=None, seed0=0):
    b.reset()
    if lesion is not None:
        b.lesion(torch.as_tensor(np.asarray(lesion), device="cuda"))
    L = RD["left"] if readout is None else readout[0]
    R = RD["right"] if readout is None else readout[1]
    out = []
    for n, phi in enumerate(PHIS):
        loom = [dict(phi_rel=float(phi), theta=7.2, dtheta=1.0, dist=50.0)]
        if drive_idx is None:
            idx, rates = mp.rates(loom, b.N, TH50, CAP)
        else:                                  # 무작위 자극: 같은 개수, 같은 세기
            rng = np.random.default_rng(1000 + n)
            idx = rng.choice(drive_idx, size=K, replace=False)
            rates = np.full(K, CAP*7.2/(7.2 + TH50), np.float32)
        b.set_poisson_rates(idx, rates)
        b.tally.zero_(); b.gacc.zero_()
        b.seed.fill_(n); b.graph.replay(); torch.cuda.synchronize()
        gac = (b.gacc/STEPS).cpu().numpy()
        out.append(channel(gac, L, R))
    return np.array(out)


def report(name, arr):
    ch = arr[:, 2]
    rng_ = float(np.abs(ch).max())
    # 방위 부호(오른쪽 표적 = phi>0)와 채널 부호가 일관되는가
    # 🔴 기대 부호는 **음수**다. 오른쪽 표적(phi>0) -> 오른쪽 눈 LC10a ->
    #    오른쪽 AOTU025(흥분·동측)가 오른쪽 DN 을 올리고,
    #    오른쪽 AOTU019(억제·대측)가 왼쪽 DN 을 내린다 -> (좌-우) < 0.
    #    그리고 "turn toward the side of higher DN activity" 이므로 오른쪽으로 돈다 = 표적 쪽.
    #    우리가 고른 규약이 아니라 배선에서 나온 부호다 (08문서 §8.1).
    m = np.abs(PHIS) > 1e-9
    agree = float(np.mean(np.sign(ch[m]) == -np.sign(PHIS[m])))
    r = float(np.corrcoef(PHIS, ch)[0, 1]) if np.std(ch) > 1e-12 else 0.0
    say(f"  {name:<22} |채널|최대 {rng_:8.4f}   방위와 r = {r:+.3f}"
        f"   표적 쪽으로 도는 비율 {agree*100:5.1f}%")
    return dict(max_abs=rng_, r=r, agree=agree, ch=[float(x) for x in ch])


say("\n방위각 스윕 (합성 표적 1개, 각크기 7.2도 고정)")
R = {}
R["intact"] = report("온전", sweep())
R["lesion_LC10a"] = report("lesion:LC10a", sweep(lesion=LC10A))
rng = np.random.default_rng(7)
R["random_drive"] = report("무작위 자극", sweep(drive_idx=np.arange(b.N)))
rl = rng.choice(b.N, size=len(RD["left"]), replace=False)
rr = rng.choice(b.N, size=len(RD["right"]), replace=False)
R["random_readout"] = report("무작위 판독", sweep(readout=(rl, rr)))

say("\n방위각별 조향 신호 (온전)")
say("  " + "  ".join(f"{p:+5.0f}" for p in PHIS))
say("  " + "  ".join(f"{c:+5.2f}" for c in np.array(R['intact']['ch'])))

json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in R.items()},
          open(OUT/"gate4_sweep.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/gate4_sweep.json")
