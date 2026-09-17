"""배에 관성이 있는가, 있다면 우리 제어가 그걸 무시하고 있는가 (사용자 질문).

현재 디코더는 **회피 방향으로 몸을 돌리고 추진**한다. 속도항이 전혀 없다.
vision.py 가 배 속도(svx, svy)를 계산은 하는데 **반환하지도 쓰지도 않는다** (죽은 변수).

1. 관성이 실제로 있는가 — 추진을 끊고 얼마나 미끄러지는지
2. 있다면 얼마나 어긋나는가 — '가려는 방향' vs '실제로 가는 방향'
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, with_fire, frame_action, ACT_EVERY
from vision import ship_heading_deg, ASPECT

ROOT = Path(__file__).resolve().parent.parent
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings()
out = {}


def ship_xy():
    for o in env.objects:
        if o and type(o).__name__ == "Player" and o.wh[0] > 0:
            return float(o.xy[0]), float(o.xy[1])
    return None


def wrap(d, span):
    return d - span*round(d/span)


# ── 1. 관성이 있는가 ────────────────────────────────────────────────────────
say("="*78)
say("1. 추진을 끊고 미끄러지는가 (뇌 없이 순수 물리)")
env.reset(seed=7)
for _ in range(40): env.step(A.index("NOOP"))
THRUST = 30   # 8프레임으로는 속도가 거의 안 붙어서 '안 움직인다'로 잘못 읽힌다
for _ in range(THRUST): env.step(A.index("UP"))
p = ship_xy(); traj = []
for f in range(180):
    env.step(A.index("NOOP"))
    q = ship_xy()
    if p is not None and q is not None:
        dx = wrap(q[0]-p[0], 160.0); dy = wrap(q[1]-p[1], 210.0)/ASPECT
        traj.append(float(np.hypot(dx, dy)))
    p = q
tr = np.asarray(traj)
if len(tr) >= 120:
    say(f"   추진 {THRUST}프레임 뒤 NOOP 만 눌렀을 때 프레임당 이동거리")
    for lab, sl in (("직후 1~10", tr[:10]), ("30~40", tr[30:40]),
                    ("60~70", tr[60:70]), ("120~130", tr[120:130]),
                    ("170~180", tr[170:])):
        say(f"     {lab:>10}프레임  평균 {sl.mean():6.3f} px/frame")
    keeps = tr[120:130].mean() > tr[:10].mean()*0.5
    say(f"   -> {'관성이 있다. 추진을 끊어도 계속 미끄러진다' if keeps else '금방 멈춘다'}")
    out["coast"] = dict(first=float(tr[:10].mean()), f30=float(tr[30:40].mean()),
                        f60=float(tr[60:70].mean()), f120=float(tr[120:130].mean()),
                        f170=float(tr[170:].mean()))

# 회전해도 속도가 유지되는가
say("\n   회전 중에도 속도가 유지되는가 (관성이면 유지돼야 한다)")
env.reset(seed=7)
for _ in range(40): env.step(A.index("NOOP"))
for _ in range(THRUST): env.step(A.index("UP"))
p = ship_xy(); sp = []
for f in range(120):
    env.step(A.index("LEFT"))                 # 회전만. 추진 없음
    q = ship_xy()
    if p is not None and q is not None:
        sp.append(float(np.hypot(wrap(q[0]-p[0],160.0), wrap(q[1]-p[1],210.0)/ASPECT)))
    p = q
sp = np.asarray(sp)
say(f"     회전만 120프레임: 처음 10 {sp[:10].mean():.3f} -> 마지막 10 {sp[-10:].mean():.3f} px/frame")
out["rotate_coast"] = dict(first=float(sp[:10].mean()), last=float(sp[-10:].mean()))

# ── 2. 가려는 방향 vs 실제로 가는 방향 ─────────────────────────────────────
say("\n" + "="*78)
say("2. 뇌가 '가려는 방향' 과 배가 '실제로 가는 방향' 이 얼마나 어긋나나")
V = Vision(); pol = BrainPolicy()
rng = np.random.default_rng(11)
gaps, gaps_up, speeds, headgap = [], [], [], []
want_log, vx_log, vy_log = [], [], []
for ep in range(3):
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); pol.dec.reset(); pol.b.reset(); pol.frame = 0
    action = (A.index("NOOP"), A.index("FIRE")); prev = ship_xy()
    for f in range(3000):
        env.step(frame_action(action[0], action[1], f))
        if f % ACT_EVERY: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            prev = None; action = (A.index("NOOP"), A.index("FIRE")); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player" and o.wh[0] > 0:
                ori = int(getattr(o, "orientation", 0)); break
        a, ch = pol(looms, ori, A)
        if ch["norm"] > 1e-9 and prev is not None:
            # 뇌가 가리키는 회피 방향 (세계 좌표)
            psi_esc = np.degrees(np.arctan2(ch["lateral"], ch["fore"])) + 180.0
            want = (ship_heading_deg(ori) - psi_esc) % 360.0
            # 실제 이동 방향 (수학 좌표, 종횡비 보정)
            dx = wrap(xy[0]-prev[0], 160.0); dy = -wrap(xy[1]-prev[1], 210.0)/ASPECT
            spd = float(np.hypot(dx, dy))
            want_log.append(want); vx_log.append(dx); vy_log.append(dy)
            if spd > 0.3:                      # 거의 멈춰 있으면 방향이 무의미
                go = np.degrees(np.arctan2(dy, dx)) % 360.0
                g = abs((want - go + 180.0) % 360.0 - 180.0)
                gaps.append(g)
                if A[a] == "UP": gaps_up.append(g)
                # 참고: 몸이 향한 방향과 실제 이동 방향의 차이
                headgap.append(abs((ship_heading_deg(ori) - go + 180.0) % 360.0 - 180.0))
            speeds.append(spd)
        prev = xy
        action = (a, with_fire(a, A))

g = np.asarray(gaps); gu = np.asarray(gaps_up); hg = np.asarray(headgap)
sp2 = np.asarray(speeds)
say(f"   표본 {len(g)}개 (움직이는 순간만)")
say(f"   가려는 방향 vs 실제 이동 방향   중앙값 {np.median(g):5.1f}도   평균 {g.mean():5.1f}도")
say(f"     ↑ 추진을 실제로 누른 순간만   중앙값 {np.median(gu):5.1f}도   (n={len(gu)})")
say(f"   몸이 향한 방향 vs 실제 이동 방향 중앙값 {np.median(hg):5.1f}도")
say(f"   속도: 중앙값 {np.median(sp2):.2f} px/frame, 90분위 {np.quantile(sp2,0.9):.2f}")
say(f"   90도 넘게 어긋난 비율 (사실상 반대로 가고 있다) {float((g>90).mean())*100:.1f}%")
out["gap"] = dict(n=len(g), median=float(np.median(g)), mean=float(g.mean()),
                  median_up=float(np.median(gu)), median_headgap=float(np.median(hg)),
                  over90=float((g > 90).mean()),
                  speed_median=float(np.median(sp2)))


# ── 3. 관성을 넣으면 명령이 얼마나 달라지나 (제어는 안 바꾼다. 차이만 잰다) ──
say("\n" + "="*78)
say("3. 관성을 고려한 목표 방향과 지금 목표 방향의 차이")
say("   지금:   회피 방향으로 몸을 돌리고 추진 (속도항 없음)")
say("   관성판: (원하는 속도 − 현재 속도) 방향으로 추진 = 감속까지 하는 방향")
diffs = []
for want_deg, vx, vy in zip(want_log, vx_log, vy_log):
    wx, wy = np.cos(np.radians(want_deg)), np.sin(np.radians(want_deg))
    spd = np.hypot(vx, vy)
    # 목표 속도를 현재 속도 크기만큼 회피 방향으로 둔다 (배율은 상쇄돼 방향만 남는다)
    tx, ty = wx*max(spd, 1.0) - vx, wy*max(spd, 1.0) - vy
    if np.hypot(tx, ty) < 1e-6: continue
    d = np.degrees(np.arctan2(ty, tx))
    diffs.append(abs((d - want_deg + 180.0) % 360.0 - 180.0))
d = np.asarray(diffs)
say(f"   표본 {len(d)}개   중앙값 {np.median(d):5.1f}도   평균 {d.mean():5.1f}도")
say(f"   45도 넘게 다른 비율 {float((d>45).mean())*100:.1f}%   90도 넘게 {float((d>90).mean())*100:.1f}%")
out["cmd_diff"] = dict(n=len(d), median=float(np.median(d)), mean=float(d.mean()),
                       over45=float((d > 45).mean()), over90=float((d > 90).mean()))

say("\n" + "="*78)
say("판정")
say(f"  관성: 추진 {THRUST}프레임 뒤 120프레임(2초)이 지나도"
    f" {out['coast']['f120']/max(out['coast']['first'],1e-9)*100:.0f}% 속도가 남는다")
say(f"  제어: 우리 디코더에는 속도항이 **없다**. 몸을 돌리고 추진할 뿐이다.")
say(f"  결과: 가려는 방향과 실제 가는 방향이 중앙값 {np.median(g):.0f}도 어긋난다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"inertia.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
say("-> out/inertia.json")
