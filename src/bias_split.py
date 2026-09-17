"""좌우 편향 −9.1% 의 원인을 둘로 분리한다 (이슈 #32).

이슈 #11 에서 우회전 편향의 2/3 은 디코더 모듈러 산술 버그였고 고쳤다.
남은 −9.1% 을 "진짜 네트워크 비대칭"이라고 적어뒀는데, 그 안에 용의자가 둘 섞여 있다.

  용의자 1  LC4 개수 비대칭 (좌 71 / 우 55)
  용의자 2  구간 나누기가 반구마다 덮는 시야 범위가 다르다  <- 교란

probe_bias.py 의 stim_bin 은 **순위**로 16개씩 고른다. 세포 수가 다르면 같은 번호의 구간이
덮는 범위가 달라진다 (L bin6 폭 5.67 vs R bin6 폭 6.43). 두 반구의 axis 범위 자체는 거의
같으므로(L −4.05~18.89 / R −4.06~18.60) 창을 **절대 좌표**로 잡으면 분리할 수 있다.

  W1 순위 창    자극 수 같음(16)  덮는 범위 다름   <- 현행. 기준선
  W2 절대 창    자극 수 다름      덮는 범위 같음   <- 개수만 다를 때
  W3 절대 창 + 개수 맞춤 (시드 5)  둘 다 같음      <- 결정적 조건

W3 에서도 대칭 자극에 좌우 채널이 0 이 아니면 하류 회로 자체의 비대칭이다.
"""
import sys, json, time
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
SEEDS = (0, 1, 2, 3, 4)
p = dict(PARAMS); steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
def say(*a): print(*a, flush=True)


def run(stim):
    b.reset(); b.alive.fill_(1)
    if stim is None or len(stim) == 0:
        b.lam.zero_(); b.rfc.fill_(b.rfc_steps); b.has_poi = False
    else:
        b.set_poisson(np.asarray(stim), RATE)
    b.run_tally(steps); torch.cuda.synchronize()
    return (b.gacc/steps).cpu().numpy()


def lr(g):
    """좌우 채널. 집단별로 따로 빼서 평균 = 혼합비에 안 흔들리는 판독 (이슈 #11 B)."""
    d02 = float(g[A02][S02 == "L"].mean() - g[A02][S02 == "R"].mean())
    d11 = float(g[A11][S11 == "L"].mean() - g[A11][S11 == "R"].mean())
    pooled = float(np.concatenate([g[A02][S02 == "L"], g[A11][S11 == "L"]]).mean()
                   - np.concatenate([g[A02][S02 == "R"], g[A11][S11 == "R"]]).mean())
    return pooled, (d02 + d11)/2.0


IL = lc4[valid & (side == "L")]; AL = axis[valid & (side == "L")]
IR = lc4[valid & (side == "R")]; AR = axis[valid & (side == "R")]
say(f"LC4  좌 {len(IL)} / 우 {len(IR)}"
    f"   axis  좌 {AL.min():+.2f}~{AL.max():+.2f} / 우 {AR.min():+.2f}~{AR.max():+.2f}\n")


def win_rank(I, A, k):
    """순위 창 — 각 반구에서 N_STIM 개. 자극 수는 같고 덮는 범위가 달라진다."""
    o = np.argsort(A); lo = int(round(k*(len(o)-N_STIM)/(NB-1)))
    return I[o[lo:lo+N_STIM]], A[o[lo:lo+N_STIM]]


LO, HI = max(AL.min(), AR.min()), min(AL.max(), AR.max())
W = (HI - LO)/NB * 1.6                      # 창끼리 조금 겹치게 (창당 세포 수 확보)
CEN = np.linspace(LO + W/2, HI - W/2, NB)


def win_abs(I, A, k):
    """절대 창 — 같은 axis 구간. 덮는 범위는 같고 자극 수가 달라진다."""
    m = (A >= CEN[k]-W/2) & (A <= CEN[k]+W/2)
    return I[m], A[m]


say(f"절대 창: 중심 {CEN[0]:+.2f}~{CEN[-1]:+.2f}, 폭 {W:.2f}")
say(f"   {'창':>3} {'좌 n':>6} {'우 n':>6} {'맞춤 n':>7}   (맞춤 = 둘 중 작은 쪽)")
NMATCH = []
for k in range(NB):
    nl, nr = len(win_abs(IL, AL, k)[0]), len(win_abs(IR, AR, k)[0])
    NMATCH.append(min(nl, nr))
    say(f"   {k:>3} {nl:>6} {nr:>6} {NMATCH[-1]:>7}")
say("")

out = {}
GAMEONLY = "--gameonly" in sys.argv

if not GAMEONLY:          # 게임 부분만 다시 돌릴 때
    # ── W1 / W2 ─────────────────────────────────────────────────────────────────
    for tag, fn, note in (("W1 순위 창", win_rank, "자극 수 같음 / 범위 다름"),
                          ("W2 절대 창", win_abs,  "범위 같음 / 자극 수 다름")):
        say("="*92)
        say(f"{tag} — {note}")
        say(f"   {'창':>3} {'좌 n':>5} {'우 n':>5} {'대칭자극 좌우채널':>18} {'좌단독':>10} {'우단독':>10} {'합':>10}")
        rows = []
        for k in range(NB):
            sl, _ = fn(IL, AL, k); sr, _ = fn(IR, AR, k)
            _, sym = lr(run(np.concatenate([sl, sr])))
            _, cl = lr(run(sl)); _, cr = lr(run(sr))
            rows.append(dict(k=k, nl=len(sl), nr=len(sr), sym=sym, l=cl, r=cr))
            say(f"   {k:>3} {len(sl):>5} {len(sr):>5} {sym:>+18.4f} {cl:>+10.4f} {cr:>+10.4f} {cl+cr:>+10.4f}")
        m = float(np.mean([abs(x["sym"]) for x in rows]))
        say(f"   |대칭자극 좌우채널| 평균 {m:.4f}")
        out[tag] = dict(rows=rows, mean_abs=m)

    # ── W3 절대 창 + 개수 맞춤 ───────────────────────────────────────────────────
    say("="*92)
    say("W3 절대 창 + 개수 맞춤 — 범위도 자극 수도 같다. **결정적 조건**")
    say(f"   {'창':>3} {'n':>4} {'대칭자극 좌우채널 (시드 5 평균±sd)':>36} {'좌단독':>10} {'우단독':>10}")
    rows3 = []
    for k in range(NB):
        sl, _ = win_abs(IL, AL, k); sr, _ = win_abs(IR, AR, k); n = NMATCH[k]
        if n < 4:
            say(f"   {k:>3} {n:>4}   세포가 너무 적어 건너뜀"); continue
        ss, ll, rr = [], [], []
        for sd in SEEDS:
            g = np.random.default_rng(sd)
            a = sl[g.choice(len(sl), n, replace=False)]
            c = sr[g.choice(len(sr), n, replace=False)]
            ss.append(lr(run(np.concatenate([a, c])))[1])
            ll.append(lr(run(a))[1]); rr.append(lr(run(c))[1])
        rows3.append(dict(k=k, n=n, sym=float(np.mean(ss)), sd=float(np.std(ss)),
                          l=float(np.mean(ll)), r=float(np.mean(rr))))
        say(f"   {k:>3} {n:>4} {np.mean(ss):>+28.4f} ± {np.std(ss):<5.4f}"
            f" {np.mean(ll):>+10.4f} {np.mean(rr):>+10.4f}")
    m3 = float(np.mean([abs(x["sym"]) for x in rows3]))
    say(f"   |대칭자극 좌우채널| 평균 {m3:.4f}")
    out["W3 절대창+개수맞춤"] = dict(rows=rows3, mean_abs=m3)

    # ── 전역 다운샘플: 좌 LC4 71 -> 55 ──────────────────────────────────────────
    say("\n" + "="*92)
    say(f"D 전역 다운샘플 — 좌 LC4 {len(IL)} -> {len(IR)} 개로 줄이고 W1(순위 창) 재측정")
    say("   (개수가 원인이면 W1 의 비대칭이 줄어야 한다)")
    rowsD = []
    for k in range(NB):
        ss = []
        for sd in SEEDS:
            g = np.random.default_rng(100+sd)
            keep = np.sort(g.choice(len(IL), len(IR), replace=False))
            il, al = IL[keep], AL[keep]
            o = np.argsort(al); lo = int(round(k*(len(o)-N_STIM)/(NB-1)))
            sl = il[o[lo:lo+N_STIM]]
            sr, _ = win_rank(IR, AR, k)
            ss.append(lr(run(np.concatenate([sl, sr])))[1])
        rowsD.append(dict(k=k, sym=float(np.mean(ss)), sd=float(np.std(ss))))
        say(f"   {k:>3} {np.mean(ss):>+12.4f} ± {np.std(ss):.4f}"
            f"   (원래 {out['W1 순위 창']['rows'][k]['sym']:+.4f})")
    mD = float(np.mean([abs(x["sym"]) for x in rowsD]))
    say(f"   |대칭자극 좌우채널| 평균 {mD:.4f}  (원래 {out['W1 순위 창']['mean_abs']:.4f})")
    out["D 좌 다운샘플"] = dict(rows=rowsD, mean_abs=mD)


    # ── 2단계: 비대칭이 어느 층에서 생기는가 ────────────────────────────────────
    #   층 1  LC4 -> DN           (감각 -> 하행뉴런)
    #   층 2  DN  -> VNC 판독집단  (하행뉴런 -> 우리가 고른 판독)
    #   거울쌍을 비교한다: 좌반구만 자극했을 때의 좌측 값 vs 우반구만 자극했을 때의 우측 값.
    #   회로가 대칭이면 이 둘이 같아야 한다.
    DN02L, DN02R = C["DNp02_L"], C["DNp02_R"]
    DN11L, DN11R = C["DNp11_L"], C["DNp11_R"]
    RO02L, RO02R = A02[S02 == "L"], A02[S02 == "R"]
    RO11L, RO11R = A11[S11 == "L"], A11[S11 == "R"]

    # 🔴 (L−R)/(L+R) 은 쓰면 안 된다. DNp11 은 앞쪽 창에서 구동이 0 근처(음수도 된다)라
    #   분모가 사라지면서 지수가 ±1.4 로 터진다 — 실제로 한 번 그렇게 찍고 잘못 읽을 뻔했다.
    #   창별로 나누지 말고 **창을 가로질러 합산한 비**를 쓴다. 분모가 0 이 될 일이 없다.
    def pooled_asym(ll, rr):
        """Σ|좌−우| / Σ(|좌|+|우|).  0 이면 완전 대칭, 1 이면 한쪽만 반응."""
        ll, rr = np.asarray(ll), np.asarray(rr)
        den = float(np.abs(ll).sum() + np.abs(rr).sum())
        return float(np.abs(ll-rr).sum())/den if den > 1e-12 else 0.0

    say("\n" + "="*92)
    say("2단계 — 비대칭이 어느 층에서 생기는가 (절대 창 + 개수 맞춤, 시드 5 평균)")
    say("   거울쌍: 좌반구만 자극했을 때의 **좌측** 값  vs  우반구만 자극했을 때의 **우측** 값")
    say("   회로가 대칭이면 이 둘이 같아야 한다. 구동(mV)을 그대로 적는다.")
    say(f"   {'창':>3} {'DNp02 좌|우':>18} {'DNp11 좌|우':>18} {'판독02 좌|우':>18} {'판독11 좌|우':>18}")
    pair = {k: ([], []) for k in ("dn02", "dn11", "ro02", "ro11")}
    lev = []
    for k in range(NB):
        sl, _ = win_abs(IL, AL, k); sr, _ = win_abs(IR, AR, k); n = NMATCH[k]
        if n < 4: continue
        v = {key: [[], []] for key in pair}
        for sd in SEEDS:
            g = np.random.default_rng(sd)
            gl = run(sl[g.choice(len(sl), n, replace=False)])
            gr = run(sr[g.choice(len(sr), n, replace=False)])
            for key, il, ir in (("dn02", DN02L, DN02R), ("dn11", DN11L, DN11R),
                                ("ro02", RO02L, RO02R), ("ro11", RO11L, RO11R)):
                v[key][0].append(float(gl[il].mean())); v[key][1].append(float(gr[ir].mean()))
        row = {"k": k}
        for key in pair:
            a_, b_ = float(np.mean(v[key][0])), float(np.mean(v[key][1]))
            pair[key][0].append(a_); pair[key][1].append(b_)
            row[key] = (a_, b_)
        lev.append(row)
        say(f"   {k:>3}" + "".join(
            f" {row[key][0]:>+8.3f}|{row[key][1]:<+8.3f} " for key in ("dn02","dn11","ro02","ro11")))
    mean_lev = {key: pooled_asym(*pair[key]) for key in pair}
    say(f"   합산 비대칭   DNp02 {mean_lev['dn02']:.3f}  DNp11 {mean_lev['dn11']:.3f}"
        f"   |  판독02 {mean_lev['ro02']:.3f}  판독11 {mean_lev['ro11']:.3f}")
    out["2단계 층별"] = dict(rows=[{k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in r.items()} for r in lev],
                            pooled=mean_lev)

    # 배선을 직접 세어본다 (시뮬 없이. 위 측정의 기전이 여기 있다)
    crow = np.load(G/"out_crow.npy"); post = np.load(G/"out_post.npy"); wgt = np.load(G/"out_w.npy")
    def wsum(srcs, dst):
        t = 0.0; m = 0
        for s_ in np.atleast_1d(srcs):
            a_, b_ = crow[int(s_)], crow[int(s_)+1]
            sel = np.isin(post[a_:b_], dst)
            t += float(wgt[a_:b_][sel].sum()); m += int(sel.sum())
        return t, m

    say("\n" + "="*92)
    say("배선 회계 (w>=3 그래프. 가중치 합 / 엣지 수)")
    say("   LC4 -> DN   *동측만 있다. 교차 0개*")
    wire = {}
    for t, dl, dr in (("DNp02", DN02L, DN02R), ("DNp11", DN11L, DN11R)):
        tl, nl = wsum(C["LC4_L"], dl); tr, nr = wsum(C["LC4_R"], dr)
        wire[f"lc4_{t}"] = dict(l=tl, nl=nl, r=tr, nr=nr)
        say(f"     LC4_L -> {t}_L {tl:>7.0f}/{nl:<4d} (세포당 {tl/max(nl,1):.1f})"
            f"   LC4_R -> {t}_R {tr:>7.0f}/{nr:<4d} (세포당 {tr/max(nr,1):.1f})"
            f"   합계비 {tl/max(tr,1e-9):.2f}  세포당비 {(tl/max(nl,1))/max(tr/max(nr,1),1e-9):.2f}")
    say("   DN -> 판독집단   *거울쌍이면 두 줄의 비가 서로 역수여야 한다*")
    for t, sl_, sr_, rl, rr in (("DNp02", DN02L, DN02R, RO02L, RO02R),
                                ("DNp11", DN11L, DN11R, RO11L, RO11R)):
        al_, _ = wsum(sl_, rl); ar_, _ = wsum(sl_, rr)
        bl_, _ = wsum(sr_, rl); br_, _ = wsum(sr_, rr)
        ratio_l = al_/max(ar_, 1e-9)          # 좌 DN 의 (동측/대측)
        ratio_r = br_/max(bl_, 1e-9)          # 우 DN 의 (동측/대측)
        wire[f"dn_{t}"] = dict(ll=al_, lr=ar_, rl=bl_, rr=br_,
                               ipsi_l=ratio_l, ipsi_r=ratio_r)
        say(f"     {t}_L 동측/대측 {ratio_l:.2f}   {t}_R 동측/대측 {ratio_r:.2f}"
            f"   -> 어긋남 {abs(ratio_l-ratio_r)/max((ratio_l+ratio_r)/2,1e-9)*100:.0f}%")
    out["배선"] = wire


# ── 3단계: 게임에서도 같은 답이 나오는가 ────────────────────────────────────
#   위는 전부 고정 자극 실험이다. 실제로 보이는 증상은 "게임에서 오른쪽으로 더 돈다" 이므로
#   같은 조작을 게임에 걸어서 좌/우 회전 비율이 바뀌는지 본다.
if GAMEONLY:
    prev = json.loads((ROOT/"out"/"bias_split.json").read_text(encoding="utf-8"))
    for k_ in ("W1 순위 창", "W2 절대 창", "W3 절대창+개수맞춤", "D 좌 다운샘플", "2단계 층별", "배선"):
        if k_ in prev: out[k_] = prev[k_]
    m3 = out["W3 절대창+개수맞춤"]["mean_abs"]; mD = out["D 좌 다운샘플"]["mean_abs"]
    mean_lev = out["2단계 층별"]["pooled"]

if "--nogame" not in sys.argv:
    say("\n" + "="*92)
    say("3단계 — 게임에서 좌/우 회전 비율 (에피소드 10개, 무작위 no-op 시작)")
    say("   * 결정적인 증거는 1~2단계다. 게임 숫자는 증상의 크기를 보여줄 뿐,\n     위협이 실제로 한쪽에서 더 많이 오면 한쪽으로 더 도는 게 맞는 행동이다.")
    del b                                   # 뇌를 두 개 띄우지 않는다 (VRAM)
    torch.cuda.empty_cache()
    from play import BrainPolicy, GreedyPolicy, run_episode, make_env, Vision
    env = make_env(); ACT = env.unwrapped.get_action_meanings(); V = Vision()
    pol = BrainPolicy()

    def turns(tag, off=None, policy=None):
        pol.b.alive.fill_(1); pol.dec.reset()
        if off is not None and len(off):
            pol.b.lesion(torch.as_tensor(np.asarray(off), device="cuda"))
        p_ = policy if policy is not None else pol
        rng = np.random.default_rng(4242)
        rs = [run_episode(env, p_, V, ACT, max_frames=4000, rng=rng) for _ in range(10)]
        L = sum(r["n_left"] for r in rs); R = sum(r["n_right"] for r in rs)
        D = sum(r["n_dec"] for r in rs)
        bias = (L-R)/max(L+R, 1)
        # 13문서는 '전체 결정 대비 좌% − 우%' (퍼센트포인트) 로 적었다. 둘 다 낸다.
        pp = (L-R)/max(D, 1)
        per = [r["turn_bias"] for r in rs]        # 에피소드별 편향 -> 흔들림을 같이 낸다
        se = float(np.std(per, ddof=1)/np.sqrt(len(per)))
        say(f"   {tag:<28} 좌 {L:>5}  우 {R:>5}   (좌−우)/(좌+우) {bias*100:>+6.1f}% ± {se*100:.1f}"
            f"   13문서식 {pp*100:>+6.1f}%p   (점수 {np.mean([r['score'] for r in rs]):.0f})")
        return dict(left=L, right=R, dec=D, bias=bias, pp=pp, se=se, per=per)

    gm = {}
    # 🔴 대조군 먼저. 규칙 기반 정책은 **같은 기하**를 쓰고 뇌를 안 쓴다.
    #   이게 같이 치우쳐 있으면 편향은 회로가 아니라 게임/디코더 기하에서 온다.
    rnd = np.random.default_rng(9)
    gm["무작위 (바닥값)"] = turns("무작위 정책 (바닥값)",
                              policy=lambda l, o, a: (int(rnd.integers(0, len(a))), {"norm": 0.0}))
    gm["규칙 기반 (뇌 없음)"] = turns("규칙 기반 — 뇌 없이 같은 기하", policy=GreedyPolicy())
    gm["온전"] = turns("온전 (뇌)")
    for sd in (0, 1):
        g = np.random.default_rng(300+sd)
        drop = IL[g.choice(len(IL), len(IL)-len(IR), replace=False)]   # 좌 71 -> 55
        gm[f"좌 LC4 55개 (시드{sd})"] = turns(f"좌 LC4 를 55개로 (시드{sd})", drop)
    g = np.random.default_rng(400)
    drop_r = IR[g.choice(len(IR), len(IL)-len(IR), replace=False)]     # 우 55 -> 39 (반대 방향)
    gm["우 LC4 39개"] = turns("우 LC4 를 39개로 (반대 방향)", drop_r)
    out["3단계 게임"] = gm
    d0 = gm["온전"]["bias"]
    dl = float(np.mean([gm[f"좌 LC4 55개 (시드{sd})"]["bias"] for sd in (0, 1)]))
    say(f"   온전 {d0*100:+.1f}%  ->  좌를 55개로 {dl*100:+.1f}%"
        f"   (개수가 원인이면 0 쪽으로 가야 한다)")
    out["3단계 게임"]["요약"] = dict(intact=d0, left_down=dl)

# ── 판정 ────────────────────────────────────────────────────────────────────
say("\n" + "="*92)
say("판정")
w1, w2 = out["W1 순위 창"]["mean_abs"], out["W2 절대 창"]["mean_abs"]
say(f"  W1 순위 창 (현행)             {w1:.4f}   자극 수만 맞춤")
say(f"  W2 절대 창                    {w2:.4f}   덮는 범위만 맞춤")
say(f"  W3 절대 창 + 개수 맞춤         {m3:.4f}   둘 다 맞춤")
say(f"  D  좌 LC4 를 55개로 줄임       {mD:.4f}   (W1 대비 {(mD-w1)/w1*100:+.0f}%)")
say("")
red = (w1 - m3)/w1*100 if w1 > 0 else 0.0
say(f"  ① 자극 설계를 둘 다 맞추면 비대칭이 {red:.0f}% 만 준다. 대부분 남는다.")
if abs(mD - w1) < w1*0.15:
    say(f"  ② 좌 LC4 를 우와 같은 55개로 줄여도 비대칭이 그대로다 ({w1:.4f} -> {mD:.4f}).")
    say(f"     **LC4 개수 비대칭(71 vs 55)은 원인이 아니다.**")
else:
    say(f"  ② 좌 LC4 를 55개로 줄이면 비대칭이 바뀐다 ({w1:.4f} -> {mD:.4f}). 개수가 기여한다.")
dn = max(mean_lev["dn02"], mean_lev["dn11"]); ro = max(mean_lev["ro02"], mean_lev["ro11"])
say(f"  ③ 층별: 하행뉴런 단계 {dn:.3f}  /  판독 단계 {ro:.3f}")
if "3단계 게임" in out:
    q = out["3단계 게임"]["요약"]
    say(f"  ④ 게임: 온전 {q['intact']*100:+.1f}%  ->  좌 LC4 를 55개로 {q['left_down']*100:+.1f}%")
    gr = out["3단계 게임"]["규칙 기반 (뇌 없음)"]["bias"]
    rr = out["3단계 게임"]["무작위 (바닥값)"]["bias"]
    say(f"     대조: 규칙 기반(뇌 없음) {gr*100:+.1f}%   무작위 {rr*100:+.1f}%")
    if abs(gr) > abs(q['intact'])*0.5:
        say("     -> **뇌 없이도 비슷하게 치우친다.** 게임 기하/디코더가 상당 부분을 만든다.")
    else:
        say("     -> 규칙 기반은 안 치우친다. 게임 편향은 **회로에서 온다.**")
if ro > dn*1.5:
    say("     -> 하행뉴런까지는 거의 대칭인데 **판독 단계에서 벌어진다.**")
    say("        판독 집단은 우리가 고른 것이므로, 이건 커넥톰의 사실이 아니라 **판독 설계의 성질**이다.")
elif dn > ro*1.5:
    say("     -> **하행뉴런 단계에서 이미 벌어져 있다.** 감각->하행 배선 자체의 비대칭이다.")
else:
    say("     -> 두 단계가 비슷하게 기여한다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"bias_split.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
say("-> out/bias_split.json")
