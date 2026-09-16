"""게이트 1(a): 위협의 방위각을 쓸면 방향 채널이 단조로 변하는가.

★ 자극할 LC4 를 '육각 좌표 위치'로만 고른다. DNp02/DNp11 시냅스 수는 선택에 쓰지 않는다.
   따라서 이 실험은 구성상 참이 아니다. 구조적 경사(|r|=0.85)가 실재해도
   16만 개 뉴런의 재귀·억제가 그걸 뭉개면 채널은 평평하게 나온다.

내부 대조 하나가 같이 붙는다:
   방향 채널(DNp02-DNp11)은 위치에 따라 변해야 하고,
   강도 채널(DNp04)은 변하지 말아야 한다 (02문서 4.3: 방향은 LC4, 강도는 LPLC2/DNp04).
   둘 다 변하면 "그냥 자극 위치 따라 전체가 흔들린 것"이므로 증거가 못 된다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"; OUT = ROOT/"out"
C = np.load(G/"circuit_idx.npz")
P = np.load(G/"lc4_position.npz", allow_pickle=True)
lc4, pos, side, valid = P["idx"], P["pos"], P["side"], P["valid"]
THETA = float(P["theta_L"])

N_STIM = int(sys.argv[1]) if len(sys.argv) > 1 else 16
RATE   = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
T_MS   = 300.0
p = dict(PARAMS); steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
RFC_MAX = 1000.0/p["t_rfc"]
R = {"n_stim": N_STIM, "rate": RATE, "theta_deg": float(np.degrees(THETA))}
def say(*a): print(*a, flush=True)

axis = pos[:,0]*np.cos(THETA) + pos[:,1]*np.sin(THETA)   # 전후 시야축 (122도)
say(f"자극: LC4 {N_STIM}개 x {RATE:.0f} Hz, {T_MS:.0f} ms. 선택 기준은 '육각 위치'뿐")
say(f"전후 시야축 = 육각공간 {np.degrees(THETA):.0f}도 방향 (retinotopy.py 가 좌반구에서 찾은 축)")

def run(idxs):
    b.reset(); b.set_poisson(idxs, RATE); b.run_tally(steps); torch.cuda.synchronize()
    g = lambda d: float(b.rates(C[d], steps).mean())
    return dict(DNp02=g("DNp02"), DNp11=g("DNp11"), DNp04=g("DNp04"), DNp01=g("DNp01"),
                act=int((b.tally>0).sum().item())/b.N*100)

NB = 7
out = {}
for hemi in ("L","R"):
    m = valid & (side == hemi)
    a_h = axis[m]; idx_h = lc4[m]
    order = np.argsort(a_h)
    say(f"\n{'='*84}\n{hemi}반구 (LC4 {int(m.sum())}개) — 시야 앞쪽 -> 뒤쪽으로 {NB}구간")
    say(f"{'구간':>4} {'축위치':>8} {'포화':>6} | {'DNp02':>7} {'DNp11':>7} "
        f"{'전후채널':>9} {'정규화':>8} | {'DNp04(강도)':>11} {'DNp01(GF)':>10}")
    rows = []
    for k in range(NB):
        lo = int(round(k*(len(order)-N_STIM)/(NB-1)))
        pick = idx_h[order[lo:lo+N_STIM]]
        apos = float(np.mean(a_h[order[lo:lo+N_STIM]]))
        r = run(pick)
        ch = r["DNp02"] - r["DNp11"]
        nrm = ch/max(r["DNp02"]+r["DNp11"], 1e-9)
        sat = max(r.values() - {r["act"]} if False else [r["DNp02"],r["DNp11"],r["DNp04"],r["DNp01"]])/RFC_MAX
        say(f"{k:>4} {apos:>8.1f} {sat*100:>5.0f}% | {r['DNp02']:>7.1f} {r['DNp11']:>7.1f} "
            f"{ch:>9.1f} {nrm:>8.3f} | {r['DNp04']:>11.1f} {r['DNp01']:>10.1f}")
        rows.append(dict(bin=k, axis=apos, **r, channel=ch, norm=nrm, sat=sat))
    ax = np.array([r["axis"] for r in rows]); ch = np.array([r["norm"] for r in rows])
    d4 = np.array([r["DNp04"] for r in rows])
    r_ch = np.corrcoef(ax, ch)[0,1]
    r_d4 = np.corrcoef(ax, d4)[0,1]
    mono = np.all(np.diff(ch) >= -0.02) or np.all(np.diff(ch) <= 0.02)
    say(f"  방향 채널 vs 위치: r = {r_ch:+.3f}   단조: {'예' if mono else '아니오'}   "
        f"범위 {ch.min():+.3f} ~ {ch.max():+.3f}")
    say(f"  강도 채널(DNp04) vs 위치: r = {r_d4:+.3f}   변동폭 "
        f"{(d4.max()-d4.min())/max(d4.mean(),1e-9)*100:.0f}%  <- 작아야 대조가 성립")
    out[hemi] = dict(rows=rows, r_channel=float(r_ch), r_dnp04=float(r_d4),
                     monotonic=bool(mono), span=float(ch.max()-ch.min()))

# 대조 1: 위치 무시하고 무작위로 고른 LC4
rng = np.random.default_rng(3)
say(f"\n{'='*84}\n대조 — 위치를 안 보고 무작위로 고른 LC4 {N_STIM}개 (10회)")
chs = []
for i in range(10):
    m = valid & (side == "L")
    pick = rng.choice(lc4[m], size=N_STIM, replace=False)
    r = run(pick)
    chs.append((r["DNp02"]-r["DNp11"])/max(r["DNp02"]+r["DNp11"],1e-9))
say(f"  정규화 방향 채널: 평균 {np.mean(chs):+.3f}, 표준편차 {np.std(chs):.3f}, "
    f"범위 {min(chs):+.3f}~{max(chs):+.3f}")
say(f"  (위치로 고른 경우의 범위 {out['L']['span']:.3f} 와 비교)")
R["random_ctrl"] = dict(mean=float(np.mean(chs)), std=float(np.std(chs)),
                        span=float(max(chs)-min(chs)))

# 대조 2: 좌 vs 우 같은 위치 (편측성 채널 — somaSide 기반, 경사와 무관)
say(f"\n{'='*84}\n편측성 채널 — 같은 시야 위치를 좌/우 반구에서 각각 자극")
for k, lbl in ((0, "앞쪽"), (NB-1, "뒤쪽")):
    line = f"  {lbl}: "
    for hemi in ("L","R"):
        m = valid & (side == hemi); a_h = axis[m]; idx_h = lc4[m]
        order = np.argsort(a_h)
        lo = int(round(k*(len(order)-N_STIM)/(NB-1)))
        r = run(idx_h[order[lo:lo+N_STIM]])
        line += f"{hemi}반구 자극 -> DNp02 {r['DNp02']:5.1f} DNp11 {r['DNp11']:5.1f}   "
    say(line)

R.update({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()})
R["rows"] = {k: v["rows"] for k, v in out.items()}
(OUT/"gate_1a.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
say(f"\n-> out/gate_1a.json")
