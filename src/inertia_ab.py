"""관성 보정 전/후 A/B (이슈 #36).

컨트롤러가 바뀌면 궤적이 갈라지므로 짝지은 재생으로는 못 잰다. 폐루프로 직접 돌린다.
뇌와 규칙 기반 **둘 다** 같은 보정을 받는다 — 한쪽만 주면 비교가 아니라 편들기가 된다.

지표
  점수 / 목숨당 생존 / 추진 비율
  회피 방위 오차   가려는 방향과 배가 실제 가는 방향의 각도 차 (관성이 있으면 이게 핵심)
  피격            사건 기반 (이슈 #6)
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, GreedyPolicy, run_episode, make_env, Vision,
                  with_fire, frame_action, ACT_EVERY)
from vision import ship_heading_deg, ASPECT

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
def say(*a): print(*a, flush=True)


def wrap(d, span): return d - span*round(d/span)


def ship_xy():
    for o in env.objects:
        if o and type(o).__name__ == "Player" and o.wh[0] > 0:
            return float(o.xy[0]), float(o.xy[1])
    return None


def run(pol, seed0):
    """폐루프 1세트. run_episode 로 게임 지표를, 별도 루프로 방향 오차를 낸다."""
    rng = np.random.default_rng(seed0)
    rs = [run_episode(env, pol, V, A, max_frames=MAXF, rng=rng) for _ in range(N_EP)]
    f = lambda k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
    ev = sum(r["events"] for r in rs); hit = sum(r["hits"] for r in rs)
    return dict(score=f("score"), life=f("mean_life"), up=f("up_frac"),
                turn_bias=f("turn_bias"), events=ev, hits=hit,
                escape=1.0 - hit/max(ev, 1),
                # 🔴 조건 간 평균을 σ 로 비교하면 에피소드 간 분산에 묻힌다.
                #   ALE 는 같은 no-op 시작이면 결정론적이라 **에피소드가 짝지어진다.**
                #   에피소드별 값을 그대로 남겨서 짝지은 차이로 검정한다.
                per_score=[float(r["score"]) for r in rs],
                per_life=[float(r["mean_life"]) for r in rs])


def heading_gap(pol, seed0):
    """가려는 방향 vs 실제 가는 방향. 결정마다 잰다."""
    rng = np.random.default_rng(seed0)
    gaps = []
    for _ in range(N_EP):
        env.reset(seed=int(rng.integers(0, 2**31)))
        for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
        V.reset()
        if hasattr(pol, "dec"): pol.dec.reset()
        if hasattr(pol, "b"): pol.b.reset(); pol.frame = 0
        action = (A.index("NOOP"), A.index("FIRE")); prev = ship_xy()
        for f in range(MAXF):
            env.step(frame_action(action[0], action[1], f))
            if f % ACT_EVERY: continue
            xy, head, looms = V.looming(env.objects)
            if xy is None:
                prev = None; action = (A.index("NOOP"), A.index("FIRE")); continue
            ori = 0
            for o in env.objects:
                if o and type(o).__name__ == "Player" and o.wh[0] > 0:
                    ori = int(getattr(o, "orientation", 0)); break
            try:    a, ch = pol(looms, ori, A, vel=V.ship_v)
            except TypeError: a, ch = pol(looms, ori, A)
            if ch.get("norm", 0) > 1e-9 and prev is not None:
                # '가려는 방향' 은 보정 전의 **회피 방향**으로 고정해서 잰다.
                # 보정판을 자기 보정 목표로 채점하면 당연히 잘 나온다 (자기 채점 금지).
                psi_esc = np.degrees(np.arctan2(ch["lateral"], ch["fore"])) + 180.0
                want = (ship_heading_deg(ori) - psi_esc) % 360.0
                dx = wrap(xy[0]-prev[0], 160.0); dy = -wrap(xy[1]-prev[1], 210.0)/ASPECT
                if np.hypot(dx, dy) > 0.3:
                    go = np.degrees(np.arctan2(dy, dx)) % 360.0
                    gaps.append(abs((want - go + 180.0) % 360.0 - 180.0))
            prev = xy
            action = (a, with_fire(a, A))
    g = np.asarray(gaps)
    return dict(n=len(g), median=float(np.median(g)), mean=float(g.mean()),
                over90=float((g > 90).mean()))



def paired(a, b, key):
    """짝지은 차이 (같은 시드·같은 no-op 시작). 평균 차이 ± 표준오차."""
    d = np.asarray(b[key]) - np.asarray(a[key])
    se = float(d.std(ddof=1)/np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se, float(d.mean()/se) if se > 0 else float("nan")

t0 = time.perf_counter()
say(f"에피소드 {N_EP} x 최대 {MAXF}프레임, 무작위 no-op 시작 (시드 고정)\n")
out = {}
for name, make in (("전체 뇌", lambda on: BrainPolicy(inertia=on)),
                   ("규칙 기반", lambda on: GreedyPolicy(inertia=on))):
    say("="*96)
    say(f"{name}")
    say(f"   {'':12}{'점수':>16}{'목숨당 생존':>16}{'추진%':>9}{'회피율':>9}"
        f"{'방향오차 중앙':>14}{'>90도':>8}")
    for on in (False, True):
        pol = make(on)
        r = run(pol, 12345)
        if hasattr(pol, "b"): pol.b.alive.fill_(1)
        hg = heading_gap(pol, 12345)
        tag = "관성 보정 O" if on else "관성 무시"
        out[f"{name}|{on}"] = dict(**r, gap=hg)
        say(f"   {tag:<12}{r['score'][0]:>9.0f}±{r['score'][1]:<6.0f}"
            f"{r['life'][0]:>9.0f}±{r['life'][1]:<6.0f}{r['up'][0]*100:>9.1f}"
            f"{r['escape']*100:>9.1f}{hg['median']:>14.1f}{hg['over90']*100:>8.1f}")

say("\n" + "="*96)
say("판정")
for name in ("전체 뇌", "규칙 기반"):
    a = out[f"{name}|False"]; b = out[f"{name}|True"]
    say(f"  {name}")
    say(f"    방향 오차   {a['gap']['median']:.1f}도 -> {b['gap']['median']:.1f}도"
        f"   (>90도 {a['gap']['over90']*100:.0f}% -> {b['gap']['over90']*100:.0f}%)")
    say(f"    점수        {a['score'][0]:.0f} -> {b['score'][0]:.0f}"
        f"   목숨당 생존 {a['life'][0]:.0f} -> {b['life'][0]:.0f}"
        f"   회피율 {a['escape']*100:.1f}% -> {b['escape']*100:.1f}%")
    for key, lab in (("per_score", "점수"), ("per_life", "목숨당 생존")):
        m, se, t = paired(a, b, key)
        mark = "유의" if abs(t) >= 2 else "잡음 안"
        say(f"    짝지은 차이 {lab:<8} {m:>+8.0f} ± {se:<7.0f} (t={t:+.2f})  {mark}")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"inertia_ab.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
say(f"\n-> out/inertia_ab.json  ({time.perf_counter()-t0:.0f}s)")
