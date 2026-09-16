"""게이트 1(b) 보너스: 앞뒤 위협을 동시에 주면 방향 채널이 상쇄되고 강도는 유지되는가.

02문서 7장 경고: 이건 예측이지 관찰이 아니다. 선형 예측(경사 r=-0.7)은 이미 확정돼 있고,
게이트가 묻는 건 '전체 뇌 16만 개의 비선형성이 구해주는가'다.
=> 관측값을 선형 합 예측과 나란히 놓고, 결과를 원하는 쪽으로 몰지 않는다.
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
N_STIM = int(sys.argv[1]) if len(sys.argv) > 1 else 24
RATE   = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
p = dict(PARAMS); steps = int(round(300.0/p["dt"]))
b = BrainRT(params=p)
axis = pos[:,0]*np.cos(THETA) + pos[:,1]*np.sin(THETA)
def say(*a): print(*a, flush=True)

def sel(hemi, which):
    m = valid & (side == hemi); a = axis[m]; ii = lc4[m]; o = np.argsort(a)
    return ii[o[:N_STIM]] if which == "front" else ii[o[-N_STIM:]]

def run(stim):
    b.reset(); b.set_poisson(np.unique(np.concatenate(stim)), RATE)
    b.run_tally(steps); torch.cuda.synchronize()
    g = lambda k: float(b.rates(C[k], steps).mean())
    d02, d11, d04 = g("DNp02"), g("DNp11"), g("DNp04")
    return dict(DNp02=d02, DNp11=d11, DNp04=d04, DNp01=g("DNp01"),
                ch=d02-d11, norm=(d02-d11)/max(d02+d11,1e-9))

say(f"자극 {N_STIM}개 x {RATE:.0f} Hz\n")
F, Rr = run([sel("L","front")]), run([sel("L","rear")])
B = run([sel("L","front"), sel("L","rear")])
say(f"{'조건':<22} {'DNp02':>7} {'DNp11':>7} {'전후채널':>9} {'정규화':>8} {'DNp04(강도)':>11}")
for lbl, r in (("앞쪽만", F), ("뒤쪽만", Rr), ("앞+뒤 동시", B)):
    say(f"{lbl:<22} {r['DNp02']:>7.1f} {r['DNp11']:>7.1f} {r['ch']:>9.1f} "
        f"{r['norm']:>8.3f} {r['DNp04']:>11.1f}")
lin_ch = F["ch"] + Rr["ch"]
lin_04 = F["DNp04"] + Rr["DNp04"]
say(f"{'(선형 합 예측)':<22} {F['DNp02']+Rr['DNp02']:>7.1f} {F['DNp11']+Rr['DNp11']:>7.1f} "
    f"{lin_ch:>9.1f} {'':>8} {lin_04:>11.1f}")

say(f"\n  방향 채널: 앞 {F['ch']:+.1f} / 뒤 {Rr['ch']:+.1f} -> 동시 {B['ch']:+.1f}")
say(f"    선형 예측 {lin_ch:+.1f} 대비 {B['ch']-lin_ch:+.1f}")
say(f"    단독 최대 크기 대비 {abs(B['ch'])/max(abs(F['ch']),abs(Rr['ch'])):.3f}"
    f"  <- 작을수록 상쇄")
say(f"  강도 채널 DNp04: 동시 {B['DNp04']:.1f} vs 선형 {lin_04:.1f} "
    f"({B['DNp04']/lin_04:.2f}x, sublinear 면 <1)")
say(f"    단독 평균 {np.mean([F['DNp04'],Rr['DNp04']]):.1f} 대비 "
    f"{B['DNp04']/np.mean([F['DNp04'],Rr['DNp04']]):.2f}x  <- 1 이상이면 강도 유지")
say(f"  DNp01(GF): 앞 {F['DNp01']:.1f} 뒤 {Rr['DNp01']:.1f} 동시 {B['DNp01']:.1f} "
    f"(선형 {F['DNp01']+Rr['DNp01']:.1f}, {B['DNp01']/(F['DNp01']+Rr['DNp01']):.2f}x)")
say(f"    <- 02문서 4.4: Jang & von Reyn 2023 은 양측 동시 looming 에서 GF 가 sublinear 라고 함")

cancel = abs(B["norm"]) < 0.16
say(f"\n  판정: 정규화 채널 {B['norm']:+.3f} -> "
    f"{'상쇄 (06문서 3장 경고 임계 0.16 미만)' if cancel else '상쇄 아님'}")
(OUT/"gate_1b.json").write_text(json.dumps(
    dict(n_stim=N_STIM, rate=RATE, front=F, rear=Rr, both=B,
         lin_ch=lin_ch, lin_04=lin_04, cancel=bool(cancel)), indent=2, default=float),
    encoding="utf-8")
