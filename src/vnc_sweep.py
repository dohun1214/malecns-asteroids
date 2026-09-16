"""이슈 #1 2단계: VNC 판독 집단이 방위각 정보를 보존하는가 + lesion 이 '네트워크를 통해' 먹는가.

판독 집단은 vnc_targets.py 가 연결성에서 고른 것이다 (이름이 아니라 특이도 기준).
  전후 채널 = mean(DNp02 전용 하류) - mean(DNp11 전용 하류)
  좌우 채널 = mean(우측 하류)       - mean(좌측 하류)

핵심: 판독 세포가 lesion 대상과 다르므로, DNp02 를 꺼서 판독이 바뀌려면
      반드시 시냅스를 타고 내려와야 한다. 동어반복이 불가능하다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
C = dict(np.load(G/"circuit_idx.npz"))
R_ = np.load(G/"vnc_readout.npz", allow_pickle=True)
A02, A11 = R_["dn02_only"], R_["dn11_only"]
S02, S11 = R_["side_dn02"], R_["side_dn11"]
P = np.load(G/"lc4_position.npz", allow_pickle=True)
lc4, pos, side, valid = P["idx"], P["pos"], P["side"], P["valid"]
TH = float(P["theta_L"])
axis = pos[:,0]*np.cos(TH) + pos[:,1]*np.sin(TH)

N_STIM, RATE, T_MS = 16, 80.0, 300.0
p = dict(PARAMS); steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
def say(*a): print(*a, flush=True)
say(f"판독 집단: DNp02 하류 {len(A02)}개 / DNp11 하류 {len(A11)}개 "
    f"(기존 DN 판독은 종류당 2개)")

def run(stim, lesion=None):
    b.reset(); b.alive.fill_(1)
    if lesion is not None and len(lesion):
        b.lesion(torch.as_tensor(np.asarray(lesion), device="cuda"))
    b.set_poisson(stim, RATE); b.run_tally(steps); torch.cuda.synchronize()
    g = (b.gacc/steps).cpu().numpy()          # 등급 판독 (평균 시냅스 구동)
    t = b.tally.cpu().numpy()
    sec = steps*p["dt"]/1000.0
    a02, a11 = float(g[A02].mean()), float(g[A11].mean())
    lat = float(np.concatenate([g[A02][S02=="R"], g[A11][S11=="R"]]).mean()
                - np.concatenate([g[A02][S02=="L"], g[A11][S11=="L"]]).mean())
    return dict(a02=a02, a11=a11, fore=a02-a11, lateral=lat,
                dn02=float(t[C["DNp02"]].mean())/sec, dn11=float(t[C["DNp11"]].mean())/sec,
                motor=float(t[np.concatenate([A02, A11])].mean())/sec)

def stim_bin(hemi, k, nb=7):
    m = valid & (side == hemi); a = axis[m]; ii = lc4[m]; o = np.argsort(a)
    lo = int(round(k*(len(o)-N_STIM)/(nb-1)))
    return ii[o[lo:lo+N_STIM]]

say(f"\n{'='*78}\n방위각 스윕 — VNC 판독 집단이 방향을 보존하는가")
for hemi in ("L","R"):
    say(f"\n{hemi}반구   {'구간':>4} {'DNp02하류':>10} {'DNp11하류':>10} {'전후채널':>9} "
        f"{'정규화':>8} | {'DN 직접(참고)':>14}")
    rows = []
    for k in range(7):
        r = run(stim_bin(hemi, k))
        nrm = r["fore"]/max(r["a02"]+r["a11"], 1e-9)
        say(f"{'':>7} {k:>4} {r['a02']:>10.3f} {r['a11']:>10.3f} {r['fore']:>9.3f} "
            f"{nrm:>8.3f} | DNp02 {r['dn02']:5.1f} DNp11 {r['dn11']:5.1f}")
        rows.append((k, nrm, r))
    x = np.array([z[0] for z in rows]); y = np.array([z[1] for z in rows])
    rr = np.corrcoef(x, y)[0,1]
    mono = np.all(np.diff(y) <= 1e-4) or np.all(np.diff(y) >= -1e-4)
    say(f"{'':>7} 전후 채널 vs 위치: r = {rr:+.3f}  단조 {'예' if mono else '아니오'}  "
        f"범위 {y.min():+.3f} ~ {y.max():+.3f}")

say(f"\n{'='*78}\nlesion 이 네트워크를 통해 먹는가 (판독 세포는 lesion 대상이 아니다)")
rng = np.random.default_rng(11); rand2 = rng.choice(166700, size=2, replace=False)
say(f"{'조건':<24} {'앞쪽 위협':>28} | {'뒤쪽 위협':>28}")
say(f"{'':<24} {'DNp02하류':>10}{'DNp11하류':>10}{'전후':>8} | "
    f"{'DNp02하류':>10}{'DNp11하류':>10}{'전후':>8}")
base = {}
for name, les in (("온전", None), ("DNp02 2개 끔", C["DNp02"]), ("DNp11 2개 끔", C["DNp11"]),
                  ("음성대조 무작위2", rand2), ("DNp04 2개 끔", C["DNp04"])):
    line = f"{name:<24}"
    for which, k in (("front",0), ("rear",6)):
        r = run(stim_bin("L", k), les)
        if name == "온전": base[which] = r
        line += (f" {r['a02']:>10.3f}{r['a11']:>10.3f}{r['fore']:>8.3f} |"
                 if which == "front" else
                 f" {r['a02']:>10.3f}{r['a11']:>10.3f}{r['fore']:>8.3f}")
    say(line)

# ---------------------------------------------------------------- 좌우 채널 검증
say(f"\n{'='*78}\n좌우 채널 — VNC 하류가 교차 투사면 부호가 뒤집힌다 (미검증이었음)")
say(f"{'자극':<18} {'하류 좌':>9} {'하류 우':>9} {'좌우채널':>9} | "
    f"{'DNp02_L':>8}{'DNp02_R':>8}{'DNp11_L':>8}{'DNp11_R':>8}")
import pandas as pd
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
nsd = nodes["somaSide"].astype("string").fillna("").to_numpy()
for hemi in ("L", "R"):
    for k, lbl in ((0, "앞"), (6, "뒤")):
        b.reset(); b.alive.fill_(1)
        b.set_poisson(stim_bin(hemi, k), RATE); b.run_tally(steps); torch.cuda.synchronize()
        g = (b.gacc/steps).cpu().numpy()
        gl = float(np.concatenate([g[A02][S02=="L"], g[A11][S11=="L"]]).mean())
        gr = float(np.concatenate([g[A02][S02=="R"], g[A11][S11=="R"]]).mean())
        dn = {}
        for nm, arr in (("DNp02", C["DNp02"]), ("DNp11", C["DNp11"])):
            for sside in ("L","R"):
                cells = arr[nsd[arr] == sside]
                dn[f"{nm}_{sside}"] = float(g[cells].mean()) if len(cells) else 0.0
        say(f"{hemi}반구 LC4 {lbl}쪽{'':<6} {gl:>9.3f} {gr:>9.3f} {gr-gl:>+9.3f} | "
            f"{dn['DNp02_L']:>8.3f}{dn['DNp02_R']:>8.3f}"
            f"{dn['DNp11_L']:>8.3f}{dn['DNp11_R']:>8.3f}")
say(f"\n기대: 좌반구 LC4(= 왼쪽 시야) 자극 -> 좌우채널 음수여야 한다.")
say(f"      양수로 나오면 VNC 하류가 교차 투사이고, 디코더에서 부호를 뒤집어야 한다.")
