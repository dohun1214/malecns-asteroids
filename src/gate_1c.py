"""게이트 1(c): DNp11 만 끄면 '뒤쪽' 위협 반응이 무너지는가. + 편측성 채널 재측정.

핵심 설계: DNp11 을 끄고 DNp11 발화율을 재는 건 의미가 없다(당연히 0).
          하류 운동 출력(VNC: TTMn/DLMn/GFC/PSI)이 무너지는지를 봐야 한다.

2x2 가 진짜 증거다:
              앞쪽 위협      뒤쪽 위협
  DNp02 끔    무너져야 함    멀쩡해야 함
  DNp11 끔    멀쩡해야 함    무너져야 함
상호작용이 나오면 "많이 꺼서 무너진 것"이 원천 배제된다. 세포는 양쪽 다 2개뿐이다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"; OUT = ROOT/"out"
C = np.load(G/"circuit_idx.npz")
P = np.load(G/"lc4_position.npz", allow_pickle=True)
lc4, pos, side, valid = P["idx"], P["pos"], P["side"], P["valid"]
THETA = float(P["theta_L"])
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
nside = nodes["somaSide"].astype("string").fillna("").to_numpy()

N_STIM = int(sys.argv[1]) if len(sys.argv) > 1 else 16
RATE   = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
T_MS   = 300.0
p = dict(PARAMS); steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
R = {"n_stim": N_STIM, "rate": RATE}
def say(*a): print(*a, flush=True)
say(f"자극: LC4 {N_STIM}개 x {RATE:.0f} Hz")

axis = pos[:,0]*np.cos(THETA) + pos[:,1]*np.sin(THETA)
def stim_set(hemi, which):
    m = valid & (side == hemi); a_h = axis[m]; ii = lc4[m]
    o = np.argsort(a_h)
    return ii[o[:N_STIM]] if which == "front" else ii[o[-N_STIM:]]

MOTOR = ["TTMn","PSI","DLMn","GFC1","GFC2","GFC3","GFC4"]
def run(stim, lesion=None):
    b.reset(); b.alive.fill_(1)
    if lesion is not None and len(lesion):
        b.lesion(torch.as_tensor(np.asarray(lesion), device="cuda"))
    b.set_poisson(stim, RATE); b.run_tally(steps); torch.cuda.synchronize()
    g = lambda k: float(b.rates(C[k], steps).mean()) if len(C[k]) else 0.0
    motor = float(np.mean([g(k) for k in MOTOR]))
    return dict(DNp02=g("DNp02"), DNp11=g("DNp11"), DNp04=g("DNp04"), DNp01=g("DNp01"),
                motor=motor, **{k: g(k) for k in MOTOR})

# ------------------------------------------------ 편측성 채널 (DN 을 좌/우로 분리)
say(f"{'='*88}\n편측성 채널 — DNp02/DNp11 을 좌우 세포로 분리해서 측정")
say(f"{'자극':<22} {'DNp02_L':>8} {'DNp02_R':>8} {'DNp11_L':>8} {'DNp11_R':>8}  {'좌우 비대칭':>12}")
lat = {}
for hemi in ("L","R"):
    for which, lbl in (("front","앞"), ("rear","뒤")):
        b.reset(); b.set_poisson(stim_set(hemi, which), RATE); b.run_tally(steps)
        torch.cuda.synchronize()
        v = {}
        for dn in ("DNp02","DNp11"):
            for s in ("L","R"):
                cells = C[dn][nside[C[dn]] == s]
                v[f"{dn}_{s}"] = float(b.rates(cells, steps).mean()) if len(cells) else 0.0
        tot_L = v["DNp02_L"]+v["DNp11_L"]; tot_R = v["DNp02_R"]+v["DNp11_R"]
        asym = (tot_R-tot_L)/max(tot_R+tot_L,1e-9)
        say(f"{f'{hemi}반구 LC4 {lbl}쪽':<22} {v['DNp02_L']:>8.1f} {v['DNp02_R']:>8.1f} "
            f"{v['DNp11_L']:>8.1f} {v['DNp11_R']:>8.1f}  {asym:>+12.3f}")
        lat[f"{hemi}_{which}"] = dict(**v, asym=asym)
R["laterality"] = lat

# ------------------------------------------------ 게이트 1(c): 2x2
say(f"\n{'='*88}\n게이트 1(c) — DNp02/DNp11 단독 lesion x 앞/뒤 위협  (하류 운동 출력으로 판정)")
rng = np.random.default_rng(11)
rand2 = rng.choice(166700, size=2, replace=False)
base = {}
for which in ("front","rear"):
    base[which] = run(stim_set("L", which))
say(f"{'조건':<26} " + " ".join(f"{k:>7}" for k in ["DNp02","DNp11","DNp04"]) +
    f" | {'운동평균':>9} {'TTMn':>7} {'DLMn':>7} {'GFC2':>7} | {'운동 대비 기준':>14}")
res = {}
LES = [("온전", None), ("DNp11 2개 끔", C["DNp11"]), ("DNp02 2개 끔", C["DNp02"]),
       ("음성대조 무작위 2개", rand2), ("DNp04 2개 끔", C["DNp04"])]
for which, wl in (("front","앞쪽 위협"), ("rear","뒤쪽 위협")):
    say(f"  -- {wl} --")
    for name, les in LES:
        r = run(stim_set("L", which), les)
        ratio = r["motor"]/max(base[which]["motor"], 1e-9)
        say(f"  {name:<24} " + " ".join(f"{r[k]:>7.1f}" for k in ["DNp02","DNp11","DNp04"]) +
            f" | {r['motor']:>9.1f} {r['TTMn']:>7.1f} {r['DLMn']:>7.1f} {r['GFC2']:>7.1f} "
            f"| {ratio:>13.2f}x")
        res[f"{which}_{name}"] = dict(**r, ratio=ratio)

say(f"\n  2x2 요약 (운동 출력, 온전 대비 배율):")
say(f"  {'':<16} {'앞쪽 위협':>10} {'뒤쪽 위협':>10}")
for name in ("DNp02 2개 끔","DNp11 2개 끔","음성대조 무작위 2개"):
    say(f"  {name:<16} {res[f'front_{name}']['ratio']:>10.2f} {res[f'rear_{name}']['ratio']:>10.2f}")
i_f = res["front_DNp02 2개 끔"]["ratio"] - res["front_DNp11 2개 끔"]["ratio"]
i_r = res["rear_DNp02 2개 끔"]["ratio"] - res["rear_DNp11 2개 끔"]["ratio"]
say(f"\n  상호작용: 앞쪽에서 (DNp02끔 - DNp11끔) = {i_f:+.2f}, "
    f"뒤쪽에서 = {i_r:+.2f}   부호가 갈리면 특이적")
R["gate1c"] = res
(OUT/f"gate_1c_{N_STIM}_{int(RATE)}.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
say(f"\n-> out/gate_1c.json")
