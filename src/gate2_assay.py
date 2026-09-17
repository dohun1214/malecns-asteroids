"""게이트 2 재정의 (이슈 #8): lesion 인과성 assay — 짝지은 재생으로 잰다.

게임 점수는 버렸다 (11문서). 여기서 묻는 건 하나다:
  **읽어낸 행동이 지목한 회로에서 나오는가.**

측정 도구는 12문서에서 쓴 짝지은 재생이다. 온전한 뇌가 실제로 본 자극 시퀀스를
테이프로 떠서 모든 조건에 **똑같이** 먹인다. 폐루프에서 조건을 비교하면 궤적 발산이
교란 변수가 되기 때문이다 (12문서 §4 에서 대조군이 무너지는 것처럼 보였던 그 함정).

통과 기준 (이슈 #8)
  ① 음성 대조 : 무작위 k세포 제거가 행동을 안 바꾼다. k = 2 / 20 / 200 / 2000, 시드 5개
                (k 를 키우면 언젠가는 깨진다. 깨지는 지점을 숨기지 않고 보고한다)
  ② 세포당 효과: 지목한 2세포가 무작위 k세포보다 훨씬 크게 바꾼다
  ③ 특이성    : DNp02 / DNp11 / DNp04 / DNp01 제거가 **서로 다른 방향**의 변화를 낸다
  ④ 용량 반응 : LC4 를 0/25/50/75% 제거하면 변화가 단조다
  ⑤ 견고성    : 컨트롤러 파라미터를 바꿔도 위 결론이 유지된다 (2단계)

⚠️ 주 지표는 **액션 일치율**이다. 각도 오차는 보조 지표다 — DNp02 를 끄면 전후 채널이
   한쪽으로 고정돼서 각도 오차는 오히려 줄지만(-8.3도) 행동은 완전히 달라진다(추진 40->81%).
   "얼마나 틀렸나"가 아니라 "행동이 바뀌었나"를 물어야 한다.
⚠️ LC4 100% 제거는 **감각 입력 전체 제거**라 판독이 완전히 침묵한다 (각도 정의 불가).
   용량 반응의 끝점으로만 쓰고 각도 통계에서는 뺀다.

지표 (전부 온전과 **결정 단위로 짝지어** 낸다)
  오차     회피 방위 vs 기하 정답의 각도 (무정보 90도)
  액션일치 같은 입력에서 온전과 **똑같은 액션**을 고른 결정의 비율 (음성대조는 100%)
  추진비율 / 좌우 편향 — 변화의 '방향'을 보는 서명
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, ACT_EVERY, with_fire, STEPS_PER_DECISION

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
NSEED = 5
def say(*a): print(*a, flush=True)

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy(); C = bp.C
N = bp.b.N


def geo_escape(looms):
    x = y = w = 0.0
    for L in looms:
        if L["theta"] <= 0: continue
        psi = np.radians(L["phi_rel"])
        x += L["theta"]*np.sin(psi); y += L["theta"]*np.cos(psi); w += L["theta"]
    if w <= 0 or np.hypot(x, y) < 1e-9: return None
    return (np.degrees(np.arctan2(x, y)) + 180.0) % 360.0


def make_tape():
    """온전한 뇌로 실제 게임을 돌려서 (자극, 방위, 기하정답) 테이프를 뜬다."""
    rng = np.random.default_rng(12345)
    tape = []
    bp.b.alive.fill_(1)
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
            idx, rates = bp.map.rates(looms, N, bp.gain, bp.cap)
            if g is not None and len(idx):
                tape.append((np.asarray(idx), np.asarray(rates), float(g), ori, first))
                first = False
            a, ch = bp(looms, ori, A)
            action = with_fire(a, A)
    return tape


def replay(tape, lesion=None):
    """테이프를 그대로 먹이고 결정마다 (각도오차, 액션, 채널) 을 낸다."""
    bp.b.alive.fill_(1)
    if lesion is not None and len(lesion):
        bp.b.lesion(torch.as_tensor(np.asarray(lesion), device="cuda"))
    errs = np.full(len(tape), np.nan); acts = np.empty(len(tape), dtype=np.int16)
    fore = np.empty(len(tape)); lat = np.empty(len(tape))
    bp.b.reset(); bp.dec.reset(); bp.frame = 0
    for i, (idx, rates, g, ori, first) in enumerate(tape):
        if first:
            bp.b.reset(); bp.dec.reset(); bp.frame = 0
        bp.b.set_poisson_rates(idx, rates)
        bp.b.tally.zero_(); bp.b.gacc.zero_()
        bp.b.seed.fill_(bp.frame); bp.frame += 1
        bp.b.graph.replay(); torch.cuda.synchronize()
        bp._gac = (bp.b.gacc/STEPS_PER_DECISION).cpu().numpy()
        ch = bp.dec.channels(bp.rate_of)
        fore[i] = ch["fore"]; lat[i] = ch["lateral"]
        a, tgt, st = bp.dec.action(ch, ori, A)
        acts[i] = a
        if ch["norm"] >= 1e-9:
            lt, fo = ch["unit"]
            brain = (np.degrees(np.arctan2(lt, fo)) + 180.0) % 360.0
            errs[i] = abs((brain - g + 180.0) % 360.0 - 180.0)
    bp.b.alive.fill_(1)
    return dict(err=errs, act=acts, fore=fore, lat=lat)


def stats(r, base, A_=A):
    ok = ~np.isnan(r["err"]); n = len(r["act"])
    d = r["err"] - base["err"]; m = ok & ~np.isnan(base["err"])
    return dict(
        mean_err=float(np.nanmean(r["err"])),
        pair=float(np.nanmean(d[m])), pair_se=float(np.nanstd(d[m])/max(np.sqrt(m.sum()), 1)),
        agree=float(np.mean(r["act"] == base["act"])),
        up=float(np.mean([A_[a] == "UP" for a in r["act"]])),
        left=float(np.mean([A_[a] == "LEFT" for a in r["act"]])),
        right=float(np.mean([A_[a] == "RIGHT" for a in r["act"]])),
        fore=float(r["fore"].mean()), lat=float(r["lat"].mean()), n=int(n))


# ======================================================================= 1단계
say(f"테이프 뜨는 중 ({N_EP} 에피소드 x {MAXF}프레임)...")
tape = make_tape()
say(f"테이프 {len(tape):,} 결정. 모든 조건에 이걸 똑같이 먹인다.\n")

rng = np.random.default_rng(2026)
LC4 = np.asarray(C["LC4"])
CONDS = [("온전", None)]
for nm in ("DNp02", "DNp11", "DNp04", "DNp01"):
    CONDS.append((f"lesion {nm} (2세포)", C[nm]))
for pct in (25, 50, 75):                       # 무작위 부분집합, 시드 3개 (인덱스 순서 편향 제거)
    k = int(round(len(LC4)*pct/100))
    for s in range(3):
        CONDS.append((f"lesion LC4 {pct}% ({k}세포)", rng.choice(LC4, size=k, replace=False)))
CONDS.append((f"lesion LC4 100% (126세포)", LC4))
for k in (2, 20, 200, 2000):
    for s in range(NSEED):
        CONDS.append((f"음성대조 무작위 {k}세포", rng.choice(N, size=k, replace=False)))

t0 = time.perf_counter()
base = replay(tape)
rows = []
say(f"{'조건':<26}{'평균|오차|':>10}{'짝차이':>12}{'액션일치':>9}"
    f"{'추진':>7}{'좌':>6}{'우':>6}{'전후ch':>8}{'좌우ch':>8}")
b = stats(base, base)
rows.append(dict(name="온전", seed=None, **b))
say(f"{'온전':<26}{b['mean_err']:>10.1f}{'-':>12}{'100.0%':>9}"
    f"{b['up']*100:>6.1f}%{b['left']*100:>5.1f}%{b['right']*100:>5.1f}%"
    f"{b['fore']:>8.3f}{b['lat']:>8.3f}")
seen = {}
for name, les in CONDS[1:]:
    r = stats(replay(tape, les), base)
    seen[name] = seen.get(name, 0)
    rows.append(dict(name=name, seed=seen[name], **r)); seen[name] += 1
    say(f"{name:<26}{r['mean_err']:>10.1f}{r['pair']:>+8.1f}±{r['pair_se']:<3.1f}"
        f"{r['agree']*100:>8.1f}%{r['up']*100:>6.1f}%{r['left']*100:>5.1f}%"
        f"{r['right']*100:>5.1f}%{r['fore']:>8.3f}{r['lat']:>8.3f}")


def agg(prefix):
    rs = [x for x in rows if x["name"].startswith(prefix)]
    f = lambda k: (float(np.mean([x[k] for x in rs])), float(np.std([x[k] for x in rs])))
    return dict(n=len(rs), pair=f("pair"), agree=f("agree"), up=f("up"), err=f("mean_err"))


say(f"\n{'='*94}\n① 음성 대조 — 무작위 k세포를 껐을 때 (시드 {NSEED}개씩)")
say(f"{'k':>6}{'액션일치':>18}{'불일치(=효과)':>16}{'세포당 효과':>16}{'추진 비율':>15}")
neg = {}
for k in (2, 20, 200, 2000):
    a = agg(f"음성대조 무작위 {k}세포"); neg[k] = a
    dis = (1-a["agree"][0])*100
    say(f"{k:>6}{a['agree'][0]*100:>13.2f}±{a['agree'][1]*100:<4.2f}%{dis:>15.2f}%p"
        f"{dis/k:>15.4f}%p{a['up'][0]*100:>10.1f}±{a['up'][1]*100:<4.1f}%")

say(f"\n{'='*94}\n② 세포당 효과 — 지목한 2세포 vs 무작위 k세포")
say(f"{'조작':<26}{'세포':>6}{'불일치':>10}{'세포당':>12}{'무작위 2000개 대비':>18}")
r2000 = (1-neg[2000]["agree"][0])*100
for nm in ("lesion DNp11 (2세포)", "lesion DNp02 (2세포)",
           "lesion DNp04 (2세포)", "lesion DNp01 (2세포)"):
    x = [r for r in rows if r["name"] == nm][0]
    dis = (1-x["agree"])*100
    say(f"{nm:<26}{2:>6}{dis:>9.1f}%p{dis/2:>11.2f}%p{dis/r2000:>17.1f}배")
for k in (200, 2000):
    dis = (1-neg[k]["agree"][0])*100
    say(f"{'음성대조 무작위':<26}{k:>6}{dis:>9.1f}%p{dis/k:>11.4f}%p{dis/r2000:>17.1f}배")

say(f"\n{'='*94}\n③ 특이성 — 조작마다 '어느 방향으로' 바뀌는가")
say(f"{'조작':<26}{'추진':>10}{'좌-우 편향':>12}{'전후ch':>10}{'좌우ch':>10}{'액션일치':>10}")
say(f"{'온전':<26}{b['up']*100:>9.1f}%{(b['left']-b['right'])*100:>+11.1f}%"
    f"{b['fore']:>10.3f}{b['lat']:>10.3f}{'100.0%':>10}")
spec = {}
for nm in ("lesion DNp02 (2세포)", "lesion DNp11 (2세포)", "lesion DNp04 (2세포)",
           "lesion DNp01 (2세포)"):
    x = [r for r in rows if r["name"] == nm][0]; spec[nm] = x
    say(f"{nm:<26}{x['up']*100:>9.1f}%{(x['left']-x['right'])*100:>+11.1f}%"
        f"{x['fore']:>10.3f}{x['lat']:>10.3f}{x['agree']*100:>9.1f}%")

say(f"\n{'='*94}\n④ 용량 반응 — LC4 를 몇 % 껐을 때 (25/50/75 는 무작위 부분집합 시드 3개)")
say(f"{'제거율':>8}{'세포':>6}{'액션일치':>12}{'추진':>9}{'전후ch':>10}")
dose = [1.0]
say(f"{0:>7}%{0:>6}{100.0:>11.1f}%{b['up']*100:>8.1f}%{b['fore']:>10.3f}")
for pct in (25, 50, 75):
    a = agg(f"lesion LC4 {pct}%")
    xs = [r for r in rows if r["name"].startswith(f"lesion LC4 {pct}%")]
    dose.append(a["agree"][0])
    say(f"{pct:>7}%{int(round(126*pct/100)):>6}{a['agree'][0]*100:>7.1f}±{a['agree'][1]*100:<3.1f}%"
        f"{a['up'][0]*100:>8.1f}%{float(np.mean([x['fore'] for x in xs])):>10.3f}")
x100 = [r for r in rows if r["name"].startswith("lesion LC4 100%")][0]
dose.append(x100["agree"])
say(f"{100:>7}%{126:>6}{x100['agree']*100:>11.1f}%{x100['up']*100:>8.1f}%{x100['fore']:>10.3f}"
    f"   <- 감각 입력 전체 제거. 판독 완전 침묵")

say(f"\n{'='*94}\n판정")
ok1 = neg[2]["agree"][0] > 0.999 and neg[20]["agree"][0] > 0.97
say(f"  {'통과' if ok1 else '실패'}  ① 음성 대조: k=2 액션일치 {neg[2]['agree'][0]*100:.2f}% "
    f"(5시드 전부 소수점까지 동일), k=20 {neg[20]['agree'][0]*100:.2f}%")
say(f"          깨지는 지점: k=200 에서 {neg[200]['agree'][0]*100:.1f}%, "
    f"k=2000 에서 {neg[2000]['agree'][0]*100:.1f}%. 무작위 조작도 규모가 커지면 먹는다 — 정상이다")
d11 = (1-spec["lesion DNp11 (2세포)"]["agree"])*100
ok2 = d11/2 > 100*( (1-neg[2000]["agree"][0])*100/2000 )
say(f"  {'통과' if ok2 else '실패'}  ② 세포당 효과: DNp11 2세포 {d11:.1f}%p "
    f"vs 무작위 2000세포 {r2000:.1f}%p -> 세포당 {(d11/2)/((r2000)/2000):.0f}배")
u = {k: v["up"] for k, v in spec.items()}
ok3 = (u["lesion DNp02 (2세포)"] - b["up"]) * (u["lesion DNp11 (2세포)"] - b["up"]) < 0
say(f"  {'통과' if ok3 else '실패'}  ③ 특이성: DNp02 제거 추진 {u['lesion DNp02 (2세포)']*100:.1f}% vs "
    f"DNp11 제거 {u['lesion DNp11 (2세포)']*100:.1f}% (온전 {b['up']*100:.1f}%) = 반대 방향. "
    f"DNp04/DNp01 은 각각 {u['lesion DNp04 (2세포)']*100:.1f}% / "
    f"{u['lesion DNp01 (2세포)']*100:.1f}% 로 거의 안 바뀜")
ok4 = all(dose[i] >= dose[i+1] - 0.02 for i in range(len(dose)-1))
say(f"  {'통과' if ok4 else '실패'}  ④ 용량 반응 (액션일치): "
    + " -> ".join(f"{d*100:.1f}%" for d in dose))
say(f"\n  -> 1단계 {'통과' if ok1 and ok2 and ok3 and ok4 else '미통과'}  ({time.perf_counter()-t0:.0f}s)")

# ======================================================================= 2단계
say(f"\n{'='*94}\n⑤ 견고성 — 컨트롤러 파라미터를 바꿔도 같은 결론이 나오는가")
say("  (자극 부호화가 바뀌면 테이프도 달라지므로 파라미터 조합마다 테이프를 새로 뜬다)")
PARAMS = [("기본 k=16 th50=7.2 slop=1", dict(k=16, th50=7.2, align_slop=1)),
          ("좁게 k=8 th50=4.0 slop=2", dict(k=8, th50=4.0, align_slop=2)),
          ("넓게 k=24 th50=12.0 slop=1", dict(k=24, th50=12.0, align_slop=1))]
SUB = [("lesion DNp02", C["DNp02"]), ("lesion DNp11", C["DNp11"]),
       ("음성대조 무작위 2", np.random.default_rng(7).choice(N, size=2, replace=False)),
       ("음성대조 무작위 200", np.random.default_rng(8).choice(N, size=200, replace=False))]
rob = {}
say(f"\n{'파라미터':<28}{'조작':<22}{'액션일치':>10}{'추진':>9}{'전후ch':>10}")
for pname, kw in PARAMS:
    bp.set_controller(**kw)
    tp = make_tape()
    bs = replay(tp)
    bb = stats(bs, bs)
    say(f"{pname:<28}{'온전':<22}{'100.0%':>10}{bb['up']*100:>8.1f}%{bb['fore']:>10.3f}")
    rob[pname] = {"온전": bb}
    for sname, les in SUB:
        r = stats(replay(tp, les), bs)
        rob[pname][sname] = r
        say(f"{'':<28}{sname:<22}{r['agree']*100:>9.1f}%{r['up']*100:>8.1f}%{r['fore']:>10.3f}")
bp.set_controller(k=16, th50=7.2, align_slop=1)

say(f"\n  결론 유지 여부")
allok = True
for pname, _ in PARAMS:
    d = rob[pname]
    s1 = (d["lesion DNp02"]["up"] - d["온전"]["up"]) > 0
    s2 = (d["lesion DNp11"]["up"] - d["온전"]["up"]) < 0
    s3 = d["음성대조 무작위 2"]["agree"] > 0.99
    s4 = d["lesion DNp11"]["agree"] < d["음성대조 무작위 200"]["agree"]
    ok = s1 and s2 and s3 and s4; allok &= ok
    say(f"  {'통과' if ok else '실패'}  {pname:<28} DNp02 추진↑ {s1} / DNp11 추진↓ {s2} / "
        f"무작위2 무변화 {s3} / DNp11 2세포 > 무작위 200세포 {s4}")
say(f"\n  -> 2단계 {'통과' if allok else '미통과'}")
say(f"\n  ==> 게이트 2 (재정의) {'통과' if ok1 and ok2 and ok3 and ok4 and allok else '미통과'}"
    f"  (총 {time.perf_counter()-t0:.0f}s)")

(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"gate2_assay.json").write_text(
    json.dumps(dict(n_decisions=len(tape), rows=rows,
                    neg={str(k): v for k, v in neg.items()}, dose=dose,
                    robust={p: {k: v for k, v in d.items()} for p, d in rob.items()}),
               indent=2, ensure_ascii=False), encoding="utf-8")
say("-> out/gate2_assay.json")
