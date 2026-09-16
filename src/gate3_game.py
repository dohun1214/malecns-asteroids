"""게이트 3 (이슈 #4) 2부: rewire 가 **게임 안에서의 행동**을 무너뜨리는가.

점수로 재지 않는다. 점수는 '가만히 있기'가 이기는 지표라 회로의 성질을 못 잰다 (11문서).
대신 **뇌가 고른 회피 방향이 기하학적으로 옳은 방향과 얼마나 맞는가**를 잰다.

  기하 정답 : 화면의 운석들에서 각크기 가중으로 낸 위협 방위 + 180도
  뇌의 답   : VNC 하류 판독 3채널에서 나온 회피 방위
  지표      : 두 각의 차이. 정보가 없으면 평균 |오차| 90도, 균등분포.

이 지표는 대조군을 이길 필요가 없고, 게임을 잘할 필요도 없다.
'뇌가 화면을 보고 방향을 계산하고 있는가' 만 묻는다.
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, ACT_EVERY, with_fire
from vision import ship_heading_deg
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
    """각크기 가중 위협 방위(배 기준) + 180도. vision 의 phi_rel 규약(오른쪽 +)."""
    x = y = w = 0.0
    for L in looms:
        if L["theta"] <= 0: continue
        psi = np.radians(L["phi_rel"])
        x += L["theta"]*np.sin(psi); y += L["theta"]*np.cos(psi); w += L["theta"]
    if w <= 0 or np.hypot(x, y) < 1e-9: return None
    return (np.degrees(np.arctan2(x, y)) + 180.0) % 360.0


def episode(rng):
    obs = env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); bp.dec.reset(); bp.b.reset(); bp.frame = 0
    action = A.index("FIRE"); errs = []; acts = []
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
        a, ch = bp(looms, ori, A)
        acts.append(A[a])
        g = geo_escape(looms)
        if g is not None and ch["norm"] > 1e-9:
            lat, fore = ch["unit"]
            brain = (np.degrees(np.arctan2(lat, fore)) + 180.0) % 360.0   # 판독 = 위협 -> +180
            errs.append(abs((brain - g + 180.0) % 360.0 - 180.0))
        action = with_fire(a, A)
    return np.array(errs), acts


def bench(seed=12345):
    rng = np.random.default_rng(seed)
    E, AC = [], []
    for _ in range(N_EP):
        e, a = episode(rng); E.append(e); AC += a
    e = np.concatenate(E)
    n = len(AC) or 1
    return dict(n=int(len(e)), mean_err=float(e.mean()), med_err=float(np.median(e)),
                within45=float((e < 45).mean()), within90=float((e < 90).mean()),
                up=AC.count("UP")/n, left=AC.count("LEFT")/n, right=AC.count("RIGHT")/n,
                noop=AC.count("NOOP")/n)


PLAN = [("온전", None, None)]
for key, frac in (("lc4_all", 1.00), ("lc4_dn", 1.00), ("lc4_p0211", 1.00),
                  ("ctrl_all", 1.00), ("ctrl_matched", 1.00)):
    for sd in SEEDS:
        PLAN.append((f"{key} {int(frac*100)}%", key, (frac, sd)))

rows = []
t0 = time.perf_counter()
say(f"에피소드 {N_EP} x {MAXF}프레임.  |오차| 는 뇌의 회피 방위 vs 기하 정답 (정보 없으면 90도)")
say(f"\n{'조건':<22}{'시드':>4}{'결정':>7}{'평균|오차|':>11}{'중앙값':>8}"
    f"{'±45도내':>9}{'±90도내':>9} | {'추진':>7}{'좌':>6}{'우':>6}")
for name, key, fs in PLAN:
    if key is None:
        set_packed(packed0); sd = ""
    else:
        frac, sd = fs
        newp, _ = RW.rewire(packed0, SEL[key], seed=sd, frac=frac)
        set_packed(newp)
    r = bench()
    rows.append(dict(name=name, key=key, seed=(None if key is None else sd), **r))
    say(f"{name:<22}{sd:>4}{r['n']:>7}{r['mean_err']:>11.1f}{r['med_err']:>8.1f}"
        f"{r['within45']*100:>8.1f}%{r['within90']*100:>8.1f}% | "
        f"{r['up']*100:>6.1f}%{r['left']*100:>5.1f}%{r['right']*100:>5.1f}%")
set_packed(packed0)

say(f"\n{'='*86}\n요약 (시드 평균)")
say(f"{'조건':<22}{'평균|오차|':>14}{'±45도내':>14}{'추진':>9}")
summ = {}
for key in (None, "lc4_all", "lc4_dn", "lc4_p0211", "ctrl_all", "ctrl_matched"):
    rs = [x for x in rows if x["key"] == key]
    if not rs: continue
    nm = rs[0]["name"]
    me = np.array([x["mean_err"] for x in rs]); w45 = np.array([x["within45"] for x in rs])
    up = np.array([x["up"] for x in rs])
    summ[nm] = dict(mean_err=float(me.mean()), mean_err_sd=float(me.std()),
                    within45=float(w45.mean()), within45_sd=float(w45.std()),
                    up=float(up.mean()), n=len(rs))
    say(f"{nm:<22}{me.mean():>8.1f}±{me.std():<5.1f}{w45.mean()*100:>8.1f}±{w45.std()*100:<5.1f}%"
        f"{up.mean()*100:>8.1f}%")

b_, l_, c_ = summ["온전"], summ["lc4_all 100%"], summ["ctrl_all 100%"]
say(f"\n판정")
say(f"  ① LC4 배선을 섞으면 방향이 무너진다 : 평균 |오차| {b_['mean_err']:.1f}도 -> "
    f"{l_['mean_err']:.1f}도 (무정보 90도)  "
    f"-> {'통과' if l_['mean_err'] > b_['mean_err'] + 10 else '실패'}")
say(f"  ② 같은 규모 대조는 안 무너진다      : {c_['mean_err']:.1f}도  "
    f"-> {'통과' if abs(c_['mean_err'] - b_['mean_err']) < 5 else '실패'}")
say(f"  ③ ±45도 안에 드는 비율              : {b_['within45']*100:.1f}% -> "
    f"{l_['within45']*100:.1f}% (대조 {c_['within45']*100:.1f}%)")

(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"gate3_game.json").write_text(
    json.dumps(dict(n_ep=N_EP, max_frames=MAXF, summary=summ, rows=rows),
               indent=2, ensure_ascii=False), encoding="utf-8")
say(f"\n-> out/gate3_game.json  ({time.perf_counter()-t0:.0f}s)")
