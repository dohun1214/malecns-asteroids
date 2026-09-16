"""게이트 3 (이슈 #4) 1부: 차수 보존 rewire 가 방향 정보를 무너뜨리는가 — 헤드리스.

lesion 은 "세포를 껐다"라서 '전선을 끊었다'는 반박이 가능하다.
rewire 는 **세포를 하나도 안 끄고, 차수·부호·가중치 다중집합을 전부 보존한 채**
연결 상대만 섞는다. 전체 엣지의 0.008% 만 바뀐다.

읽는 곳은 VNC 하류 운동 집단(507세포)이다. rewire 대상(LC4->DN 엣지)과 판독 집단이
겹치지 않으므로, 판독이 바뀌려면 반드시 시냅스를 타고 내려와야 한다.

조건
  온전        기준선
  lc4_dn      LC4 -> DN 배선만 섞기. 25 / 50 / 100%, 시드 3개
  lplc2_dn    LPLC2 -> DN 배선 섞기 (같은 종류의 엣지, 같은 표적, **방향은 안 나름**)
              -> 02문서 4.3 대로면 방향 채널이 안 무너져야 한다. **특이성 대조**
  ctrl_random LC4 가 아닌 흥분성 엣지를 같은 개수(824개) 섞기. **크기 대조**

⚠️ 주의: LC4->DN 결과는 '당연히 무너진다'에 가깝다 (섞은 것이 곧 재는 것).
   증거로서의 무게는 ① 용량 반응이 단조인가 ② lplc2_dn 이 안 무너지는가
   ③ ctrl_random 이 안 무너지는가 ④ 뇌가 침묵한 게 아니라 방향만 잃었는가 에 있다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS
import rewire as RW

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
SEEDS = (0, 1, 2, 3, 4, 5)
def say(*a): print(*a, flush=True)

b = BrainRT(params=p)
crow, packed0, _, pre = RW.load()
N = crow.size - 1
SEL = RW.conditions(C, N, pre, packed0)
say(f"전체 엣지 {len(packed0):,}   LC4->DN {len(SEL['lc4_dn'])}개 "
    f"(전체의 {len(SEL['lc4_dn'])/len(packed0)*100:.4f}%)")
say(f"판독: DNp02 하류 {len(A02)}개 / DNp11 하류 {len(A11)}개 (rewire 대상과 겹침 0)")


def set_packed(arr):
    b.packed.copy_(torch.from_numpy(arr.astype(np.int32)).to(b.dev))


def stim_bin(hemi, k):
    m = valid & (side == hemi); a = axis[m]; ii = lc4[m]; o = np.argsort(a)
    lo = int(round(k*(len(o)-N_STIM)/(NB-1)))
    return ii[o[lo:lo+N_STIM]], float(np.mean(a[o[lo:lo+N_STIM]]))


def run(stim):
    b.reset(); b.alive.fill_(1)
    b.set_poisson(stim, RATE); b.run_tally(steps); torch.cuda.synchronize()
    g = (b.gacc/steps).cpu().numpy()
    a02, a11 = float(g[A02].mean()), float(g[A11].mean())
    t = b.tally.cpu().numpy(); sec = steps*p["dt"]/1000.0
    return dict(a02=a02, a11=a11, fore=a02-a11,
                norm=(a02-a11)/max(abs(a02)+abs(a11), 1e-9),
                motor=float(t[np.concatenate([A02, A11])].mean())/sec,
                dn02=float(t[C["DNp02"]].mean())/sec,
                dn11=float(t[C["DNp11"]].mean())/sec)


def sweep():
    """양 반구 방위각 스윕 -> 방향 채널의 상관·범위·운동 출력."""
    out = {}
    for hemi in ("L", "R"):
        ax, nz, mo = [], [], []
        for k in range(NB):
            stim, a = stim_bin(hemi, k)
            r = run(stim)
            ax.append(a); nz.append(r["norm"]); mo.append(r["motor"])
        ax = np.array(ax); nz = np.array(nz)
        out[hemi] = dict(r=float(np.corrcoef(ax, nz)[0, 1]),
                         span=float(nz.max()-nz.min()),
                         mono=bool(np.all(np.diff(nz) >= -0.02) or np.all(np.diff(nz) <= 0.02)),
                         motor=float(np.mean(mo)), norms=[float(x) for x in nz])
    return out


def sweep_random(seed):
    """정보 없음 바닥선: 위치를 무시하고 무작위로 고른 LC4 묶음 7개.
    rewire 후 남은 범위가 '절반 살아남은 것'인지 '바닥에 닿은 것'인지 가르는 기준."""
    rng = np.random.default_rng(seed)
    out = {}
    for hemi in ("L", "R"):
        m = valid & (side == hemi); ii = lc4[m]; a = axis[m]
        ax, nz = [], []
        for k in range(NB):
            pick = rng.choice(len(ii), size=N_STIM, replace=False)
            r = run(ii[pick])
            ax.append(float(a[pick].mean())); nz.append(r["norm"])
        ax = np.array(ax); nz = np.array(nz)
        out[hemi] = dict(r=float(np.corrcoef(ax, nz)[0, 1]),
                         span=float(nz.max()-nz.min()), mono=False,
                         motor=0.0, norms=[float(x) for x in nz])
    return out


CONDS = [("온전", None, None, None)]
PLAN = [("lc4_p0211", (1.00,), "lc4_p0211 {p}% (경사 엣지만)"),
        ("lc4_dn",    (0.25, 0.50, 1.00), "lc4_dn {p}%"),
        ("lc4_all",   (0.25, 0.50, 1.00), "lc4_all {p}% (LC4 투사 전체)"),
        ("lplc2_dn",  (1.00,), "lplc2_dn {p}% (특이성대조)"),
        ("ctrl_random", (1.00,), "ctrl_random {p}% (엣지수 일치)"),
        ("ctrl_matched", (1.00,), "ctrl_matched {p}% (시냅스량 일치)"),
        ("ctrl_all",  (1.00,), "ctrl_all {p}% (lc4_all 짝 대조)")]
for key, fracs, lbl in PLAN:
    for frac in fracs:
        for sd in SEEDS:
            CONDS.append((lbl.format(p=int(frac*100)), key, frac, sd))

res = []
say(f"\n{'='*96}")
say(f"{'조건':<34}{'시드':>4} | {'r(좌)':>7}{'r(우)':>7} | {'범위(좌)':>9}{'범위(우)':>9} | "
    f"{'운동출력':>9} {'단조':>5}")
for name, key, frac, seed in CONDS:
    if key is None:
        set_packed(packed0); moved = 0
    else:
        newp, moved = RW.rewire(packed0, SEL[key], seed=seed, frac=frac)
        set_packed(newp)
    sw = sweep()
    mo = (sw["L"]["motor"] + sw["R"]["motor"])/2
    say(f"{name:<34}{'' if seed is None else seed:>4} | "
        f"{sw['L']['r']:>+7.3f}{sw['R']['r']:>+7.3f} | "
        f"{sw['L']['span']:>9.3f}{sw['R']['span']:>9.3f} | {mo:>9.2f} "
        f"{'예' if sw['L']['mono'] and sw['R']['mono'] else '아니오':>5}")
    res.append(dict(name=name, key=key, frac=frac, seed=seed, moved=moved,
                    L=sw["L"], R=sw["R"], motor=mo))
set_packed(packed0)

# 참고선: 온전한 뇌 + '위치를 안 보고' 무작위로 고른 LC4 묶음.
# ⚠️ 이건 '정보 0' 의 바닥선이 **아니다**. 무작위 묶음도 평균 축위치가 조금씩 다르므로
#    상관은 그대로 높게 나오고(|r| 0.85) 범위만 줄어든다. 대비가 약해진 경우일 뿐이다.
#    따라서 판정에는 쓰지 않고, 범위라는 지표가 대비에 얼마나 민감한지 보여주는 참고값으로만 둔다.
base_motor = float(np.mean([x["motor"] for x in res if x["key"] is None]))
floor = []
for sd in SEEDS:
    sw = sweep_random(100+sd)
    floor.append(dict(name="참고: 무작위 LC4 묶음", key="_floor", frac=None, seed=sd,
                      moved=0, L=sw["L"], R=sw["R"], motor=base_motor))
    say(f"{'바닥선 (무작위 LC4 묶음)':<34}{sd:>4} | {sw['L']['r']:>+7.3f}{sw['R']['r']:>+7.3f} | "
        f"{sw['L']['span']:>9.3f}{sw['R']['span']:>9.3f} |")
res += floor


def agg(key, frac=None):
    rs = [x for x in res if x["key"] == key and (frac is None or x["frac"] == frac)]
    if not rs: return None
    r = np.array([abs(x["L"]["r"]) for x in rs] + [abs(x["R"]["r"]) for x in rs])
    sp = np.array([x["L"]["span"] for x in rs] + [x["R"]["span"] for x in rs])
    mo = np.array([x["motor"] for x in rs])
    sr = np.array([x["L"]["r"] for x in rs] + [x["R"]["r"] for x in rs])
    return dict(n=len(rs), r_mean=float(r.mean()), r_sd=float(r.std()),
                r_signed=float(sr.mean()), sign_kept=float((sr < 0).mean()),
                span_mean=float(sp.mean()), span_sd=float(sp.std()),
                motor=float(np.nanmean(mo)), moved=int(rs[0]["moved"]))

base = agg(None)
say(f"\n{'='*96}\n요약 — 시드 {len(SEEDS)}개 평균 (|r| 은 좌우 {2*len(SEEDS)}개 합쳐서)")
say(f"{'조건':<34}{'치환엣지':>8}{'|r|':>15}{'부호있는 r':>11}{'부호유지':>8}"
    f"{'범위':>16}{'유지율':>9}{'운동출력':>9}")
def prow(name, a, denom):
    say(f"{name:<34}{a['moved']:>8,}{a['r_mean']:>9.3f}±{a['r_sd']:<5.3f}"
        f"{a['r_signed']:>+11.3f}{a['sign_kept']*100:>7.0f}%"
        f"{a['span_mean']:>10.3f}±{a['span_sd']:<5.3f}"
        f"{a['span_mean']/denom:>9.2f}{a['motor']:>9.2f}")
prow("온전", base, base["span_mean"])
summary = {"온전": base}
for key, fracs, lbl in PLAN:
    for frac in fracs:
        name = lbl.format(p=int(frac*100))
        a = agg(key, frac); summary[name] = a
        prow(name, a, base["span_mean"])
fl = agg("_floor"); summary["참고: 무작위 LC4 묶음"] = fl
prow("참고: 무작위 LC4 묶음", fl, base["span_mean"])

def keep(n): return summary[n]["span_mean"]/base["span_mean"]
say(f"\n{'='*96}\n판정")
lines = []
d25, d50, d100 = keep("lc4_dn 25%"), keep("lc4_dn 50%"), keep("lc4_dn 100%")
a25, a50, a100 = keep("lc4_all 25% (LC4 투사 전체)"), keep("lc4_all 50% (LC4 투사 전체)"), keep("lc4_all 100% (LC4 투사 전체)")
ok1 = d25 >= d50 >= d100 and a25 >= a50 >= a100
lines.append((f"① 용량 반응 단조   lc4_dn {d25:.2f} -> {d50:.2f} -> {d100:.2f} | "
              f"lc4_all {a25:.2f} -> {a50:.2f} -> {a100:.2f}", ok1))
ctrls = {n: keep(n) for n in ("lplc2_dn 100% (특이성대조)", "ctrl_random 100% (엣지수 일치)",
                              "ctrl_matched 100% (시냅스량 일치)", "ctrl_all 100% (lc4_all 짝 대조)")}
ok2 = all(v >= 0.85 for v in ctrls.values())
lines.append(("② 대조군은 안 무너진다   " +
              "  ".join(f"{n.split()[0]} {v:.2f}" for n, v in ctrls.items()), ok2))
worst_ctrl = min(ctrls.values())
ok3 = a100 < worst_ctrl - 0.15
lines.append((f"③ LC4 rewire 가 대조군보다 확실히 낮다   lc4_all {a100:.2f} vs "
              f"최저 대조군 {worst_ctrl:.2f}", ok3))
mr = summary["lc4_all 100% (LC4 투사 전체)"]["motor"]/base["motor"]
ok4 = mr > 0.5
lines.append((f"④ 침묵이 아니라 방향만 잃었다   운동 출력 {mr*100:.0f}% 유지", ok4))
la = summary["lc4_all 100% (LC4 투사 전체)"]
ok5 = la["sign_kept"] <= 0.6 and la["r_signed"] > base["r_signed"] + 0.8
lines.append((f"⑤ 약해진 게 아니라 뒤섞였다   부호있는 r {base['r_signed']:+.3f} -> "
              f"{la['r_signed']:+.3f}, 방향 부호 유지 {la['sign_kept']*100:.0f}% "
              f"(온전 100%, 우연 50%)", ok5))
for txt, ok in lines:
    say(f"  {'통과' if ok else '실패'}  {txt}")
say(f"\n  -> 게이트 3-1 {'통과' if all(o for _, o in lines) else '미통과'}")

(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"gate3_rewire.json").write_text(
    json.dumps(dict(summary=summary, rows=res), indent=2, ensure_ascii=False),
    encoding="utf-8")
say("\n-> out/gate3_rewire.json")
