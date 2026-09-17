"""크기 맞춘 MLP 대조군 (02문서 §7 게이트 3의 남은 항목).

**막으려는 반박**: "커넥톰이라서 방향이 나오는 게 아니라, 파라미터가 그만큼 많으면
아무 망이나 나오는 것 아니냐."

rewire(12문서)가 이미 더 센 대조다 — 같은 망에서 차수·부호·가중치를 보존한 채 연결
상대만 섞으면 방향이 무너진다. 하지만 "다른 구조의 같은 크기 망"은 안 해봤다.

설계 — 입력을 문자 그대로 똑같이 준다.
  짝지은 재생 테이프(게이트 3과 같은 코드)를 떠서, 각 결정의 **LC4 126세포 자극률**을
  126차원 벡터로 만든다. 정답은 같은 결정의 **기하 정답 방위**다.
  초파리: 그 자극을 뇌에 넣고 판독 507세포에서 (좌우, 전후) 를 읽는다 -> 각도
  MLP:    같은 126차원을 넣고 (좌우, 전후) 2개를 뱉는다 -> 각도

두 팔을 다 잰다.
  ① 무작위 초기화 MLP  — 학습 안 함. 파라미터 수만 맞춘다.
     여기서 무정보선(90도)이 나와야 "파라미터 수 때문이 아니다"가 선다.
  ② 지도학습 MLP      — 정답을 보여주고 학습시킨다. 당연히 잘한다.
     ⚠️ 이건 반박이 아니다. **초파리는 정답을 한 번도 안 봤다.**
        학습한 망과의 비교는 "성능 경쟁"이 아니라 "무엇이 공짜로 주어졌나"의 대조다.

파라미터 맞추기 (w>=3 그래프 실측)
  직접 경로  LC4->DN 824 엣지 + DN->판독 609 엣지 = 1,433  -> 은닉 11
  LC4 출력 전체                              12,253       -> 은닉 96
  과잉                                                    -> 은닉 256
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, make_env, Vision, ACT_EVERY, with_fire,
                  frame_action, STEPS_PER_DECISION)

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
SEEDS = (0, 1, 2, 3, 4)
DEV = "cuda"
def say(*a): print(*a, flush=True)

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy()
C = dict(np.load(G/"circuit_idx.npz"))
LC4 = np.asarray(C["LC4"])                 # 전역 뉴런 번호 126개
POS = {int(v): i for i, v in enumerate(LC4)}


def geo_escape(looms):
    x = y = w = 0.0
    for L in looms:
        if L["theta"] <= 0: continue
        psi = np.radians(L["phi_rel"])
        x += L["theta"]*np.sin(psi); y += L["theta"]*np.cos(psi); w += L["theta"]
    if w <= 0 or np.hypot(x, y) < 1e-9: return None
    return (np.degrees(np.arctan2(x, y)) + 180.0) % 360.0


# ── 1. 테이프 (게이트 3과 같은 방식) ─────────────────────────────────────────
rng = np.random.default_rng(12345)
tape = []
for ep in range(N_EP):
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); bp.dec.reset(); bp.b.reset(); bp.frame = 0
    action = (A.index("NOOP"), A.index("FIRE")); first = True
    for f in range(MAXF):
        obs, rew, tr, te, info = env.step(frame_action(action[0], action[1], f))
        if tr or te: break
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            action = (A.index("NOOP"), A.index("FIRE")); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player" and o.wh[0] > 0:
                ori = int(getattr(o, "orientation", 0)); break
        g = geo_escape(looms)
        idx, rates = bp.map.rates(looms, bp.b.N, bp.gain, bp.cap)
        if g is not None and len(idx):
            tape.append((np.asarray(idx), np.asarray(rates), float(g), ep, first))
            first = False
        a, ch = bp(looms, ori, A, vel=V.ship_v)
        action = (a, with_fire(a, A))
say(f"테이프 {len(tape):,} 결정 ({N_EP} 에피소드)")

# 126차원 입력 + 정답
X = np.zeros((len(tape), len(LC4)), dtype=np.float32)
Y = np.zeros(len(tape), dtype=np.float32)
EP = np.zeros(len(tape), dtype=np.int64)
drop = 0
for i, (idx, rates, g, ep, first) in enumerate(tape):
    for j, v in zip(idx, rates):
        p = POS.get(int(j))
        if p is None: drop += 1; continue
        X[i, p] = v
    Y[i] = g; EP[i] = ep
say(f"입력 {X.shape} (LC4 126세포 자극률). LC4 밖 자극 {drop}개  <- 0이어야 정상")
say(f"자극이 들어간 세포 수 중앙값 {np.median((X>0).sum(1)):.0f}/126")

# ── 2. 초파리 기준선 (같은 테이프) ──────────────────────────────────────────
def fly():
    errs = np.full(len(tape), np.nan)
    bp.b.alive.fill_(1); bp.b.reset(); bp.dec.reset(); bp.frame = 0
    for i, (idx, rates, g, ep, first) in enumerate(tape):
        if first: bp.b.reset(); bp.dec.reset(); bp.frame = 0
        bp.b.set_poisson_rates(idx, rates)
        bp.b.tally.zero_(); bp.b.gacc.zero_()
        bp.b.seed.fill_(bp.frame); bp.frame += 1
        bp.b.graph.replay(); torch.cuda.synchronize()
        bp._gac = (bp.b.gacc/STEPS_PER_DECISION).cpu().numpy()
        ch = bp.dec.channels(bp.rate_of)
        if ch["norm"] < 1e-9: continue
        lat, fo = ch["unit"]
        brain = (np.degrees(np.arctan2(lat, fo)) + 180.0) % 360.0
        errs[i] = abs((brain - g + 180.0) % 360.0 - 180.0)
    return errs


fe = fly()
say(f"\n초파리 뇌 (같은 테이프): 평균 {np.nanmean(fe):.1f}도  중앙 {np.nanmedian(fe):.1f}도"
    f"  ±45도 안 {float(np.nanmean(fe<=45))*100:.1f}%")

# ── 3. MLP ─────────────────────────────────────────────────────────────────
HID = [(11, "직접 경로 1,433엣지 일치"), (96, "LC4 출력 12,253엣지 일치"),
       (256, "과잉 (32,768 파라미터)")]
xt = torch.from_numpy(X).to(DEV)
# 정답을 단위벡터로 (각도를 직접 회귀하면 ±180 경계에서 터진다)
ang = np.radians(Y)
yt = torch.from_numpy(np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32)).to(DEV)


def mk(h, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(),
                               torch.nn.Linear(h, 2)).to(DEV)


def err_of(model, sel):
    with torch.no_grad():
        o = model(xt[sel]).cpu().numpy()
    pred = (np.degrees(np.arctan2(o[:, 0], o[:, 1]))) % 360.0
    return np.abs((pred - Y[sel.cpu().numpy() if torch.is_tensor(sel) else sel]
                   + 180.0) % 360.0 - 180.0)


allsel = np.arange(len(tape))
say("\n① 무작위 초기화 MLP (학습 안 함) — 파라미터 수만 맞춘다")
say(f"   {'은닉':>5} {'파라미터':>9}  {'평균':>7} {'중앙':>7} {'±45도 안':>9}   맞춘 대상")
out = {"fly": dict(mean=float(np.nanmean(fe)), median=float(np.nanmedian(fe)),
                   in45=float(np.nanmean(fe <= 45)))}
for h, tag in HID:
    es = [err_of(mk(h, s), allsel) for s in SEEDS]
    m = np.mean([e.mean() for e in es]); md = np.mean([np.median(e) for e in es])
    i45 = np.mean([(e <= 45).mean() for e in es])
    npar = (X.shape[1]+1)*h + (h+1)*2
    say(f"   {h:>5} {npar:>9,}  {m:>7.1f} {md:>7.1f} {i45*100:>8.1f}%   {tag}")
    out[f"random_h{h}"] = dict(mean=float(m), median=float(md), in45=float(i45), par=npar)

say("\n② 지도학습 MLP — 정답을 보여주고 학습. 에피소드 단위로 나눠서 누수 차단")
say("   ⚠️ 초파리 숫자도 **같은 검증셋에서만** 다시 낸다. 안 그러면 사과와 오렌지다.")
say("   ⚠️ 학습 오차도 같이 낸다. 학습 오차마저 안 내려가면 그건 학습 부족이 아니라")
say("      **이 입력이 담고 있는 정보의 한계**라는 뜻이다.")
# 마지막 4 에피소드를 검증셋으로 (1개면 검증 표본이 너무 적다)
te_set = set(range(max(0, int(EP.max())-3), int(EP.max())+1))
is_te = np.array([e in te_set for e in EP])
tr_m = torch.from_numpy(~is_te).to(DEV)
tr_i = np.where(~is_te)[0]; te_i = np.where(is_te)[0]
fe_te = fe[te_i]
say(f"\n   초파리 뇌 (검증셋 {len(te_i)}결정만): 평균 {np.nanmean(fe_te):.1f}도"
    f"  중앙 {np.nanmedian(fe_te):.1f}도  ±45도 안 {float(np.nanmean(fe_te<=45))*100:.1f}%")
out["fly_test"] = dict(mean=float(np.nanmean(fe_te)), median=float(np.nanmedian(fe_te)),
                       in45=float(np.nanmean(fe_te <= 45)), n=int(len(te_i)))
say(f"   {'은닉':>5} {'학습 평균':>9} {'검증 평균':>9} {'검증 중앙':>9} {'±45도 안':>9}  수렴")
for h, tag in HID:
    tes, trs, last = [], [], []
    for sd in SEEDS[:3]:
        mdl = mk(h, sd); opt = torch.optim.Adam(mdl.parameters(), 3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)
        hist = []
        for it in range(20000):        # 수렴할 때까지 충분히 (3,000 은 모자랐다)
            opt.zero_grad()
            o = mdl(xt[tr_m])
            o = o/o.norm(dim=1, keepdim=True).clamp_min(1e-6)
            loss = (1.0 - (o*yt[tr_m]).sum(1)).mean()      # 코사인 손실
            loss.backward(); opt.step(); sch.step()
            if it % 2000 == 0: hist.append(float(loss.detach()))
        tes.append(err_of(mdl, te_i)); trs.append(err_of(mdl, tr_i))
        last.append(hist[-1] - hist[-2] if len(hist) > 1 else 0.0)
    m = np.mean([e.mean() for e in tes]); md = np.mean([np.median(e) for e in tes])
    i45 = np.mean([(e <= 45).mean() for e in tes]); mtr = np.mean([e.mean() for e in trs])
    say(f"   {h:>5} {mtr:>9.1f} {m:>9.1f} {md:>9.1f} {i45*100:>8.1f}%  손실변화 {np.mean(last):+.5f}")
    out[f"trained_h{h}"] = dict(train=float(mtr), mean=float(m), median=float(md),
                                in45=float(i45))

# ── 판정 ───────────────────────────────────────────────────────────────────
say("\n" + "="*92)
say("판정")
f_m = out["fly"]["mean"]
r_worst = min(out[f"random_h{h}"]["mean"] for h, _ in HID)
say(f"  초파리 뇌            평균 {f_m:.1f}도   (무정보 = 90도)")
say(f"  무작위 MLP (최고)     평균 {r_worst:.1f}도")
if r_worst > 80:
    say("  -> 같은 크기 망을 무작위로 놓으면 **무정보선에 붙는다.**")
    say("     방향 정보는 파라미터 수가 아니라 **배선**에서 온다. 반박이 막힌다.")
else:
    say("  -> 무작위 망도 무정보선보다 낫다. 이유를 규명해야 한다.")
t_best = min(out[f"trained_h{h}"]["mean"] for h, _ in HID)
tr_at_best = [out[f"trained_h{h}"]["train"] for h, _ in HID
              if out[f"trained_h{h}"]["mean"] == t_best][0]
ft = out["fly_test"]["mean"]
gap = t_best - tr_at_best
say(f"  같은 검증셋에서  초파리 {ft:.1f}도  vs  학습한 MLP {t_best:.1f}도")
say(f"                  그 MLP 의 학습 오차 {tr_at_best:.1f}도  ->  학습-검증 격차 {gap:+.1f}도")
# 🔴 학습 오차가 낮은데 검증이 높으면 그건 '정보의 한계'가 아니라 **과적합**이다.
#    처음에 이 분기를 안 나눠서 "정보 한계에 도달했다"는 틀린 문장을 뱉었다.
if gap > 15:
    say("  -> **과적합이다.** 학습셋은 잘 맞추는데 새 에피소드에서 무너진다.")
    say(f"     즉 입력에는 정보가 더 있지만 **{len(tr_i):,}결정으로는 일반화가 안 된다.**")
    if t_best >= ft - 3:
        say(f"     이 데이터량에서는 **지도학습이 초파리를 못 이긴다** ({t_best:.1f} vs {ft:.1f}).")
        say("     ⚠️ 다만 이건 '초파리가 최적'이 아니라 '학습 데이터가 적다' 는 뜻이다.")
        say("        데이터를 더 주면 학습한 망이 이길 수 있다. 그래도 반박은 안 된다 —")
        say("        **초파리는 정답을 한 번도 안 봤다.**")
    else:
        say(f"     그래도 학습한 망이 더 낫다 ({t_best:.1f} vs {ft:.1f}). 정직하게 적는다.")
elif t_best >= ft - 3:
    say("  -> 학습-검증 격차가 작은데도 초파리를 못 이긴다.")
    say("     **이 입력이 담을 수 있는 정보의 한계**에 가깝다는 뜻이다.")
else:
    say(f"  -> 학습한 망이 더 낫다 ({t_best:.1f} vs {ft:.1f}). 반박은 아니지만 정직하게 적는다.")
say("")
say("  어느 경우든 ①이 핵심이다: **같은 크기 망을 무작위로 놓으면 무정보선이다.**")
say("  방향 정보는 파라미터 수가 아니라 배선에서 온다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"mlp_control.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
say("-> out/mlp_control.json")
