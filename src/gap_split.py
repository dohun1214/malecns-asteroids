"""회피 방향 오차 82.8도를 분해한다 (17문서 §7 남은 항목).

관성을 넣고도 '가려는 방향'과 '배가 실제 가는 방향'이 중앙값 82.8도 어긋난다.
관성만의 문제가 아니라는 뜻인데, 무엇이 남았는지 안 갈라봤다.

용의자
  V1 회전 해상도   한 결정에 22.5도씩만 돈다. 목표까지 90도면 4결정(0.27초)이 걸린다.
  V2 목표 변동성   목표 방위가 결정마다 크게 흔들리면 아무리 빨라도 못 따라간다.
  V3 판독 잡음     뇌가 읽어낸 방향 자체가 기하 정답에서 벗어난다 (게이트 3의 43.6도).
  V4 관성 잔여     보정을 넣었어도 남는 부분.

잰다
  a) 목표 방위의 **결정 간 변화량** 분포 — 22.5도보다 크면 구조적으로 못 따라간다
  b) **정렬 오차** (몸이 목표를 얼마나 향하고 있나) — 회전으로 좁혀지는 부분
  c) **이동 방향 지연** — 실제 이동 방향이 과거 몇 결정 전의 목표와 가장 잘 맞나
  d) 상한: 목표가 고정돼 있었다면(마지막 목표를 유지) 오차가 얼마였을까
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, make_env, Vision, ACT_EVERY, with_fire,
                  frame_action)
from vision import ship_heading_deg, ASPECT

ROOT = Path(__file__).resolve().parent.parent
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXF = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
pol = BrainPolicy()
def say(*a): print(*a, flush=True)
def wrap180(d): return abs((d + 180.0) % 360.0 - 180.0)
def signed(d): return (d + 180.0) % 360.0 - 180.0


def ship_xy():
    for o in env.objects:
        if o and type(o).__name__ == "Player" and o.wh[0] > 0:
            return float(o.xy[0]), float(o.xy[1])
    return None


rng = np.random.default_rng(777)
rec = []          # 결정마다 (목표방위, 몸방위, 실제이동방위 or None, 액션)
for ep in range(N_EP):
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); pol.dec.reset(); pol.b.reset(); pol.frame = 0
    action = (A.index("NOOP"), A.index("FIRE")); prev = ship_xy(); mark = len(rec)
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
        a, ch = pol(looms, ori, A, vel=V.ship_v)
        go = None
        if prev is not None:
            dx = xy[0]-prev[0]; dy = xy[1]-prev[1]
            dx -= 160.0*round(dx/160.0); dy -= 210.0*round(dy/210.0)
            dy = -dy/ASPECT
            if np.hypot(dx, dy) > 0.3: go = np.degrees(np.arctan2(dy, dx)) % 360.0
        if ch["norm"] > 1e-9:
            psi = np.degrees(np.arctan2(ch["lateral"], ch["fore"])) + 180.0
            want = (ship_heading_deg(ori) - psi) % 360.0
            rec.append(dict(want=want, head=ship_heading_deg(ori), go=go,
                            act=A[a], ep=ep, i=len(rec)-mark))
        prev = xy
        action = (a, with_fire(a, A))

say(f"결정 {len(rec):,}개 ({N_EP} 에피소드)\n")
want = np.array([r["want"] for r in rec]); head = np.array([r["head"] for r in rec])
ep = np.array([r["ep"] for r in rec]); ii = np.array([r["i"] for r in rec])
go = np.array([np.nan if r["go"] is None else r["go"] for r in rec])
ok = ~np.isnan(go)
out = {}

# (a) 목표 변동성
cons = (np.diff(ep) == 0) & (np.diff(ii) == 1)
dwant = np.abs(signed(np.diff(want)))[cons]
say("a) 목표 방위가 결정마다 얼마나 흔들리나  (회전은 결정당 22.5도까지만 가능)")
for q in (25, 50, 75, 90):
    say(f"     {q}분위 {np.percentile(dwant, q):6.1f}도")
say(f"   22.5도를 넘는 비율 **{float((dwant > 22.5).mean())*100:.1f}%**"
    f"   45도 넘음 {float((dwant > 45).mean())*100:.1f}%")
say("   -> 이만큼 흔들리면 회전이 아무리 정확해도 구조적으로 따라갈 수 없다\n")
out["target_jitter"] = dict(p50=float(np.median(dwant)),
                            over225=float((dwant > 22.5).mean()),
                            over45=float((dwant > 45).mean()))

# (b) 정렬 오차
align = wrap180(want - head)
say(f"b) 정렬 오차 (몸이 목표를 향하고 있나)  중앙 {np.median(align):.1f}도"
    f"   ±22.5도 안 {float((align <= 22.5).mean())*100:.1f}%")
out["align"] = dict(median=float(np.median(align)), in225=float((align <= 22.5).mean()))

# (c) 이동 방향이 '몇 결정 전 목표'와 가장 잘 맞나
say("\nc) 실제 이동 방향은 **몇 결정 전** 목표와 가장 잘 맞나 (관성 지연)")
say(f"     {'지연':>4} {'중앙 오차':>9}")
best = (0, 1e9)
for lag in range(0, 9):
    m = ok.copy(); m[:lag] = False
    same = np.zeros(len(rec), bool)
    same[lag:] = (ep[lag:] == ep[:-lag]) if lag else True
    m &= same
    e = wrap180(want[np.roll(np.arange(len(rec)), lag)][m] - go[m]) if lag else wrap180(want[m]-go[m])
    md = float(np.median(e))
    say(f"     {lag:>4} {md:>9.1f}")
    if md < best[1]: best = (lag, md)
say(f"   -> 가장 잘 맞는 지연 **{best[0]}결정** ({best[0]*4/60*1000:.0f}ms), 그때 오차 {best[1]:.1f}도")
out["lag"] = dict(best=best[0], err=best[1])

# (d) 상한 — 목표가 안 흔들렸다면
say("\nd) 상한 — 목표가 그 에피소드 내내 고정이었다면 (흔들림을 0으로)")
fix_err = []
for e_ in range(N_EP):
    m = (ep == e_) & ok
    if m.sum() < 10: continue
    # 그 에피소드의 '평균 목표' 하나로 고정
    a_ = np.radians(want[m]); mu = np.degrees(np.arctan2(np.sin(a_).mean(), np.cos(a_).mean()))
    fix_err.append(np.median(wrap180(mu - go[m])))
say(f"   고정 목표 기준 중앙 오차 {np.mean(fix_err):.1f}도  (실제 {np.median(wrap180(want[ok]-go[ok])):.1f}도)")
out["fixed_target"] = dict(err=float(np.mean(fix_err)))

say("\n" + "="*84)
say("판정")
say(f"  실제 방향 오차 중앙 {np.median(wrap180(want[ok]-go[ok])):.1f}도")
say(f"  ① 목표가 결정마다 중앙 {np.median(dwant):.1f}도 흔들리고, **{float((dwant>22.5).mean())*100:.0f}%**"
    f" 는 한 결정에 돌 수 있는 22.5도를 넘는다")
say(f"  ② 몸이 목표를 ±22.5도 안으로 향한 결정은 {float((align<=22.5).mean())*100:.0f}% 뿐이다")
say(f"  ③ 이동 방향은 {best[0]}결정 전 목표와 가장 잘 맞는다 (오차 {best[1]:.1f}도)")
say("  -> 남은 오차는 관성이 아니라 **목표가 회전 속도보다 빨리 바뀌는 것**이 주범이다.")
say("     회전 해상도(22.5도/결정)를 못 바꾸는 한 구조적으로 줄지 않는다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"gap_split.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
say("-> out/gap_split.json")
