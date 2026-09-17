"""온전한 뇌가 우회전으로 치우치는 원인 찾기 (13문서: 좌 16.5% vs 우 43.8%).

지금 결론들은 전부 조건 간 '차이'로 내서 이 편향에 오염되지 않는다.
하지만 화면에 그대로 보일 거라 원인은 알고 있어야 한다.

용의자 4개
  A 판독 자체의 무자극 기준선이 0이 아니다
  B 좌우 판독 집단의 '구성'이 다르다 (DNp02 하류 69/63, DNp11 하류 166/166 -> 혼합비가 다름)
  C LC4 개수 비대칭 (좌 71 / 우 55) 이 자극 강도를 좌우 비대칭으로 만든다
  D 디코더의 모듈러 산술이 비대칭이다 (diff 범위가 -8..+7 이라 음수 칸이 하나 많다)
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
axis = pos[:, 0]*np.cos(TH) + pos[:, 1]*np.sin(TH)
N_STIM, RATE, T_MS, NB = 16, 80.0, 300.0, 7
p = dict(PARAMS); steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
def say(*a): print(*a, flush=True)

say(f"판독 집단 구성")
say(f"  DNp02 하류 {len(A02):>3}개  좌 {int((S02=='L').sum()):>3} / 우 {int((S02=='R').sum()):>3}"
    f"   <- 좌우 개수가 다르다")
say(f"  DNp11 하류 {len(A11):>3}개  좌 {int((S11=='L').sum()):>3} / 우 {int((S11=='R').sum()):>3}")
L_all = np.concatenate([A02[S02 == "L"], A11[S11 == "L"]])
R_all = np.concatenate([A02[S02 == "R"], A11[S11 == "R"]])
say(f"  풀링 결과   좌 {len(L_all)}개 중 DNp02형 {int((S02=='L').sum())/len(L_all)*100:.1f}%"
    f"  /  우 {len(R_all)}개 중 DNp02형 {int((S02=='R').sum())/len(R_all)*100:.1f}%")
say(f"  -> 혼합비가 다르면 두 집단의 기저 발화율 차이가 좌우 채널에 상수 오프셋으로 샌다\n")
say(f"LC4 개수  좌 {int((valid&(side=='L')).sum())} / 우 {int((valid&(side=='R')).sum())}\n")


def run(stim=None):
    b.reset(); b.alive.fill_(1)
    if stim is None or len(stim) == 0:
        b.lam.zero_(); b.rfc.fill_(b.rfc_steps); b.has_poi = False
    else:
        b.set_poisson(stim, RATE)
    b.run_tally(steps); torch.cuda.synchronize()
    g = (b.gacc/steps).cpu().numpy()
    return g


def chans(g):
    l = float(np.concatenate([g[A02][S02 == "L"], g[A11][S11 == "L"]]).mean())
    r = float(np.concatenate([g[A02][S02 == "R"], g[A11][S11 == "R"]]).mean())
    # 집단별로 따로 뺀 뒤 평균 = 혼합비에 안 흔들리는 판독
    d02 = float(g[A02][S02 == "L"].mean() - g[A02][S02 == "R"].mean())
    d11 = float(g[A11][S11 == "L"].mean() - g[A11][S11 == "R"].mean())
    return dict(pooled=l-r, balanced=(d02+d11)/2.0, d02=d02, d11=d11,
                fore=float(g[A02].mean() - g[A11].mean()))


def stim_bin(hemi, k):
    m = valid & (side == hemi); a = axis[m]; ii = lc4[m]; o = np.argsort(a)
    lo = int(round(k*(len(o)-N_STIM)/(NB-1)))
    return ii[o[lo:lo+N_STIM]]


say("="*88)
say("A. 무자극 기준선 — 아무 자극도 안 넣었을 때 좌우 채널이 0인가")
c = chans(run(None))
say(f"   풀링 좌우채널 {c['pooled']:+.4f}   집단별균형 {c['balanced']:+.4f}   전후 {c['fore']:+.4f}")
say(f"   -> {'0이 아니다. 상수 오프셋이 있다' if abs(c['pooled'])>1e-4 else '0이다'}")
base = c

say("\n" + "="*88)
say("B. 대칭 자극 — 같은 시야 위치를 좌우 반구에서 동시에. 좌우 채널이 0이어야 한다")
say(f"   {'구간':>4} {'풀링':>12} {'집단별균형':>12} {'DNp02하류차':>12} {'DNp11하류차':>12}")
symm = []
for k in range(NB):
    st = np.concatenate([stim_bin("L", k), stim_bin("R", k)])
    c = chans(run(st)); symm.append(c)
    say(f"   {k:>4} {c['pooled']:>+12.4f} {c['balanced']:>+12.4f} {c['d02']:>+12.4f} {c['d11']:>+12.4f}")
mp = float(np.mean([x["pooled"] for x in symm])); mb = float(np.mean([x["balanced"] for x in symm]))
say(f"   평균 풀링 {mp:+.4f} / 집단별균형 {mb:+.4f}")
say(f"   -> 풀링이 {'편향돼 있다' if abs(mp) > abs(mb)*1.5 else '집단별균형과 비슷하다'}")

say("\n" + "="*88)
say("C. 좌/우 단독 자극 — 크기가 거울처럼 대칭인가 (LC4 71 vs 55 의 영향)")
say(f"   {'구간':>4} {'좌자극 풀링':>12} {'우자극 풀링':>12} {'합(0이어야)':>13} "
    f"{'좌자극 균형':>12} {'우자극 균형':>12} {'합':>10}")
asym = []
for k in range(NB):
    cl = chans(run(stim_bin("L", k))); cr = chans(run(stim_bin("R", k)))
    asym.append((cl, cr))
    say(f"   {k:>4} {cl['pooled']:>+12.4f} {cr['pooled']:>+12.4f} "
        f"{cl['pooled']+cr['pooled']:>+13.4f} {cl['balanced']:>+12.4f} "
        f"{cr['balanced']:>+12.4f} {cl['balanced']+cr['balanced']:>+10.4f}")
sp = float(np.mean([a[0]["pooled"]+a[1]["pooled"] for a in asym]))
sb = float(np.mean([a[0]["balanced"]+a[1]["balanced"] for a in asym]))
say(f"   평균 잔차: 풀링 {sp:+.4f} / 집단별균형 {sb:+.4f}  (0에 가까울수록 대칭)")

say("\n" + "="*88)
say("D. 디코더 산술 — 뇌 없이 순수 계산. 목표 방위가 균등분포일 때 좌/우가 반반인가")
for slop in (0, 1, 2):
    cnt = {"UP": 0, "LEFT": 0, "RIGHT": 0}
    for ori in range(16):
        for tgt in range(16):
            diff = (tgt - ori + 8) % 16 - 8        # -8 .. +7
            if abs(diff) <= slop: cnt["UP"] += 1
            elif diff > 0: cnt["LEFT"] += 1
            else: cnt["RIGHT"] += 1
    tot = cnt["LEFT"] + cnt["RIGHT"]
    say(f"   slop={slop}: 추진 {cnt['UP']:>3}  좌 {cnt['LEFT']:>3}  우 {cnt['RIGHT']:>3}"
        f"   -> 우 비율 {cnt['RIGHT']/tot*100:.1f}%  (50%여야 함)")
say("   원인: diff 범위가 -8..+7 이라 음수 칸이 하나 많다. diff=-8 은 정반대 방향이라")
say("         좌우 어느 쪽으로 돌아도 같은데 항상 RIGHT 로 간다.")

say("\n" + "="*88)
say("정리")
say(f"  A 무자극 오프셋      풀링 {base['pooled']:+.4f}  ->  집단별균형으로 바꾸면 {base['balanced']:+.4f}")
say(f"  B 대칭자극 잔차      풀링 {mp:+.4f}       ->  집단별균형 {mb:+.4f}")
say(f"  C 좌우 단독 합       풀링 {sp:+.4f}       ->  집단별균형 {sb:+.4f}")
say(f"  D 산술 편향          slop=1 에서 우 {100*7/13:.1f}% (구조적)")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"probe_bias.json").write_text(json.dumps(
    dict(baseline=base, symm_pooled=mp, symm_balanced=mb,
         resid_pooled=sp, resid_balanced=sb), indent=2, ensure_ascii=False), encoding="utf-8")
say("-> out/probe_bias.json")

# --- 수정 후 재검증 (연속 각도 판정) ---
say("\n" + "="*88)
say("D'. 수정판 — 반올림 전 연속 각도로 판정했을 때")
for slop in (0, 1, 2):
    cnt = {"UP": 0, "LEFT": 0, "RIGHT": 0}
    rngd = np.random.default_rng(0)
    for _ in range(200000):
        ori = int(rngd.integers(0, 16))
        world = float(rngd.uniform(0, 360))
        ddeg = ((world - 90.0) - 22.5*ori + 180.0) % 360.0 - 180.0
        if abs(ddeg) <= (slop+0.5)*22.5: cnt["UP"] += 1
        elif ddeg > 0: cnt["LEFT"] += 1
        else: cnt["RIGHT"] += 1
    tot = cnt["LEFT"] + cnt["RIGHT"]
    say(f"   slop={slop}: 추진 {cnt['UP']/2000:.1f}%  좌 {cnt['LEFT']/2000:.1f}%  "
        f"우 {cnt['RIGHT']/2000:.1f}%  -> 우 비율 {cnt['RIGHT']/tot*100:.2f}% (50%여야 함)")
