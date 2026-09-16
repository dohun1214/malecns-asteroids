"""게이트 3 (이슈 #4) 3부: **짝지은(paired) 재생 실험** — 궤적 발산을 제거한 rewire 비교.

2부(gate3_game.py)의 약점: 배선을 바꾸면 궤적 자체가 달라져서, 조건마다 '본 화면'이
다르다. 조건 간 차이에 게임 상황의 차이가 섞인다 (에피소드 4개로는 ±7도쯤 흔들린다).

여기서는 **온전한 뇌가 실제로 본 자극 시퀀스를 테이프로 떠서**, 모든 조건에 **똑같이**
먹인다. 입력이 문자 그대로 동일하므로 출력 차이는 전부 배선에서 온다.
결정마다 짝이 지어지므로 짝지은 차이로 검정할 수 있다.

지표는 2부와 같다: 뇌가 읽어낸 회피 방위 vs 기하 정답의 각도 오차 (무정보 = 90도).
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, ACT_EVERY, with_fire, STEPS_PER_DECISION
import rewire as RW

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SEEDS = (0, 1, 2)
def say(*a): print(*a, flush=True)

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy()
crow, packed0, C, pre = RW.load()
N = crow.size - 1
SEL = RW.conditions(C, N, pre, packed0)


def set_packed(arr):
    bp.b.packed.copy_(torch.from_numpy(arr.astype(np.int32)).to(bp.b.dev))


def geo_escape(looms):
    x = y = w = 0.0
    for L in looms:
        if L["theta"] <= 0: continue
        psi = np.radians(L["phi_rel"])
        x += L["theta"]*np.sin(psi); y += L["theta"]*np.cos(psi); w += L["theta"]
    if w <= 0 or np.hypot(x, y) < 1e-9: return None
    return (np.degrees(np.arctan2(x, y)) + 180.0) % 360.0


# ---------------------------------------------------------------- 1. 테이프 뜨기
set_packed(packed0)
rng = np.random.default_rng(12345)
tape = []          # [(자극 idx, 자극 rate, 기하 정답, 에피소드 경계 플래그)]
for ep in range(N_EP):
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); bp.dec.reset(); bp.b.reset(); bp.frame = 0
    action = A.index("FIRE"); first = True
    for f in range(MAXF):
        obs, rew, tr, te, info = env.step(action)
        if tr or te: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            action = A.index("FIRE"); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        g = geo_escape(looms)
        idx, rates = bp.map.rates(looms, bp.b.N, bp.gain, bp.cap)
        if g is not None and len(idx):
            tape.append((np.asarray(idx), np.asarray(rates), float(g), first))
            first = False
        a, ch = bp(looms, ori, A)
        action = with_fire(a, A)
say(f"테이프 {len(tape):,} 결정 ({N_EP} 에피소드). 모든 조건에 이걸 똑같이 먹인다.")


# ---------------------------------------------------------------- 2. 재생
def replay():
    """테이프를 그대로 먹이고 결정마다 각도 오차를 낸다."""
    errs = np.empty(len(tape))
    bp.b.reset(); bp.dec.reset(); bp.frame = 0
    for i, (idx, rates, g, first) in enumerate(tape):
        if first:                      # 에피소드 경계에서 상태 초기화
            bp.b.reset(); bp.dec.reset(); bp.frame = 0
        bp.b.set_poisson_rates(idx, rates)
        bp.b.tally.zero_(); bp.b.gacc.zero_()
        bp.b.seed.fill_(bp.frame); bp.frame += 1
        bp.b.graph.replay(); torch.cuda.synchronize()
        bp._gac = (bp.b.gacc/STEPS_PER_DECISION).cpu().numpy()
        ch = bp.dec.channels(bp.rate_of)
        if ch["norm"] < 1e-9:
            errs[i] = np.nan; continue
        lat, fore = ch["unit"]
        brain = (np.degrees(np.arctan2(lat, fore)) + 180.0) % 360.0
        errs[i] = abs((brain - g + 180.0) % 360.0 - 180.0)
    return errs


PLAN = [("온전", None, None)]
for key in ("lc4_all", "lc4_dn", "lc4_p0211", "lplc2_dn",
            "ctrl_all", "ctrl_matched", "ctrl_random"):
    for sd in SEEDS:
        PLAN.append((f"{key} 100%", key, sd))

t0 = time.perf_counter()
E = {}
rows = []
say(f"\n{'조건':<20}{'시드':>4}{'유효':>7}{'평균|오차|':>11}{'중앙값':>8}{'±45도내':>9}"
    f"{'온전과 짝차이':>13}")
for name, key, sd in PLAN:
    if key is None:
        set_packed(packed0)
    else:
        newp, _ = RW.rewire(packed0, SEL[key], seed=sd, frac=1.0)
        set_packed(newp)
    e = replay(); E[(name, sd)] = e
    ok = ~np.isnan(e)
    base = E[("온전", None)]
    pair = e - base
    m = ok & ~np.isnan(base)
    dm = float(np.nanmean(pair[m])); ds = float(np.nanstd(pair[m])/max(np.sqrt(m.sum()), 1))
    rows.append(dict(name=name, key=key, seed=sd, n=int(ok.sum()),
                     mean=float(np.nanmean(e)), med=float(np.nanmedian(e)),
                     within45=float(np.nanmean(e[ok] < 45)), pair_mean=dm, pair_se=ds))
    say(f"{name:<20}{'' if sd is None else sd:>4}{int(ok.sum()):>7}{np.nanmean(e):>11.1f}"
        f"{np.nanmedian(e):>8.1f}{np.nanmean(e[ok] < 45)*100:>8.1f}%"
        f"{dm:>+9.1f}±{ds:<4.1f}")
set_packed(packed0)

say(f"\n{'='*84}\n요약 (시드 평균, 짝차이는 온전 대비 결정당 오차 증가)")
say(f"{'조건':<20}{'평균|오차|':>14}{'±45도내':>14}{'짝차이':>14}")
summ = {}
for key in (None, "lc4_all", "lc4_dn", "lc4_p0211", "lplc2_dn",
            "ctrl_all", "ctrl_matched", "ctrl_random"):
    rs = [r for r in rows if r["key"] == key]
    if not rs: continue
    nm = rs[0]["name"]
    mm = np.array([r["mean"] for r in rs]); w = np.array([r["within45"] for r in rs])
    pm = np.array([r["pair_mean"] for r in rs])
    summ[nm] = dict(mean=float(mm.mean()), sd=float(mm.std()),
                    within45=float(w.mean()), pair=float(pm.mean()),
                    pair_sd=float(pm.std()), n=len(rs))
    say(f"{nm:<20}{mm.mean():>8.1f}±{mm.std():<5.1f}{w.mean()*100:>8.1f}±{w.std()*100:<5.1f}%"
        f"{pm.mean():>+9.1f}±{pm.std():<4.1f}")

b_ = summ["온전"]
say(f"\n판정 (무정보 = 90도, 온전 = {b_['mean']:.1f}도)")
for nm in ("lc4_all 100%", "lc4_dn 100%", "lc4_p0211 100%",
           "ctrl_all 100%", "ctrl_matched 100%", "ctrl_random 100%", "lplc2_dn 100%"):
    a = summ[nm]
    frac = (a["mean"] - b_["mean"]) / max(90.0 - b_["mean"], 1e-9)
    say(f"  {nm:<20} 오차 {a['mean']:>5.1f}도  무정보까지의 거리 중 {frac*100:>5.0f}% 소실"
        f"   짝차이 {a['pair']:+.1f}도")

(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"gate3_replay.json").write_text(
    json.dumps(dict(n_decisions=len(tape), summary=summ, rows=rows),
               indent=2, ensure_ascii=False), encoding="utf-8")
say(f"\n-> out/gate3_replay.json  ({time.perf_counter()-t0:.0f}s)")
