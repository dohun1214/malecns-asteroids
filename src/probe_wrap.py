"""🔴 화면 랩어라운드를 방위각 계산이 무시하고 있는가.

`vision._rel()` 은 상대 위치를 그냥 (ax-sx, ay-sy) 로 계산한다. **감싸기 보정이 없다.**
반면 배 속도(line 77)와 운석 추적(line 111)에는 `dx -= 160*round(dx/160)` 보정이 있다.

Asteroids 는 화면 끝에서 반대쪽으로 나온다. 배가 x=5, 운석이 x=155 면 실제 최단 변위는
**−10** 인데 우리는 **+150** 으로 본다. 그러면 방위각·거리·각크기가 전부 틀린다.

재는 것
  0 실제 좌표 범위 (랩 주기가 정말 160 / 210 인가)
  1 감싸기가 더 가까운 (배, 운석) 쌍의 비율
  2 방위각 오차 분포 — 결정당 회전량 22.5도를 넘는 비율
  3 **가장 가까운 운석의 정체가 바뀌는** 결정의 비율   <- 이게 제일 중요하다
  4 위협 반경(30px) 안에 드는 운석 수가 달라지는가
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, with_fire, frame_action, ACT_EVERY
from vision import ASPECT

FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEEDS = [11, 33, 44]
W, H = 160.0, 210.0
def say(*a): print(*a, flush=True)

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
pol = BrainPolicy(inertia=True)


def wrap(d, p):
    return d - p*np.round(d/p)


xs, ys = [], []
n_pair = n_wrapx = n_wrapy = n_wrap = 0
dphi = []
n_dec = 0; n_near_change = 0; n_big = 0
n_r_change = 0
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    env.reset(seed=0)
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); pol.b.reset(); pol.frame = 0; pol.dec.reset()
    act = (A.index("NOOP"), A.index("FIRE"))
    for f in range(FRAMES):
        _, rew, tr, te, _ = env.step(frame_action(act[0], act[1], f))
        if tr or te: break
        if f % ACT_EVERY: continue
        ship, asts = V.parse(env.objects)
        if ship is None: continue
        sx, sy = float(ship.xy[0]), float(ship.xy[1])
        xs.append(sx); ys.append(sy)
        if not asts: continue
        ax = np.array([float(a.xy[0]) for a in asts])
        ay = np.array([float(a.xy[1]) for a in asts])
        for a in asts: xs.append(float(a.xy[0])); ys.append(float(a.xy[1]))
        # 지금 방식
        rx0, ry0 = ax - sx, -(ay - sy)/ASPECT
        # 감싸기 보정
        rx1, ry1 = wrap(ax - sx, W), -wrap(ay - sy, H)/ASPECT
        n_pair += len(asts)
        n_wrapx += int(np.sum(np.abs(rx1 - rx0) > 1e-9))
        n_wrapy += int(np.sum(np.abs(ry1 - ry0) > 1e-9))
        n_wrap += int(np.sum((np.abs(rx1-rx0) > 1e-9) | (np.abs(ry1-ry0) > 1e-9)))
        p0 = np.degrees(np.arctan2(ry0, rx0)); p1 = np.degrees(np.arctan2(ry1, rx1))
        d = np.abs((p1 - p0 + 180) % 360 - 180)
        dphi.extend(d.tolist())
        n_big += int(np.sum(d > 22.5))
        d0 = np.hypot(rx0, ry0); d1 = np.hypot(rx1, ry1)
        n_dec += 1
        if int(np.argmin(d0)) != int(np.argmin(d1)): n_near_change += 1
        if int(np.sum(d0 <= 30.0)) != int(np.sum(d1 <= 30.0)): n_r_change += 1
        ori = int(getattr(ship, "orientation", 0))
        xy, head, looms = V.looming(env.objects)
        a_, _c = pol(looms, ori, A, vel=V.ship_v)
        act = (a_, with_fire(a_, A))

xs = np.array(xs); ys = np.array(ys); dphi = np.array(dphi)
say(f"[0] 좌표 범위  x {xs.min():.0f}~{xs.max():.0f}   y {ys.min():.0f}~{ys.max():.0f}"
    f"   (가정한 랩 주기 {W:.0f} x {H:.0f})")
say(f"[1] (배,운석) 쌍 {n_pair:,}개 중 감싸기가 더 가까운 것")
say(f"      x 축 {n_wrapx:,} ({n_wrapx/n_pair*100:.1f}%)   "
    f"y 축 {n_wrapy:,} ({n_wrapy/n_pair*100:.1f}%)   "
    f"둘 중 하나 {n_wrap:,} (**{n_wrap/n_pair*100:.1f}%**)")
say(f"[2] 방위각 오차  중앙 {np.median(dphi):.1f}도  평균 {dphi.mean():.1f}도  "
    f"90분위 {np.percentile(dphi,90):.1f}도")
say(f"      22.5도(결정당 회전량) 초과 {n_big:,} / {n_pair:,} = **{n_big/n_pair*100:.1f}%**")
say(f"[3] 결정 {n_dec:,}회 중 **가장 가까운 운석의 정체가 바뀌는** 경우 "
    f"{n_near_change:,} (**{n_near_change/max(n_dec,1)*100:.1f}%**)")
say(f"[4] 위협 반경 30px 안의 운석 수가 달라지는 결정 "
    f"{n_r_change:,} ({n_r_change/max(n_dec,1)*100:.1f}%)")
json.dump(dict(n_pair=n_pair, wrap_frac=n_wrap/n_pair, big_frac=n_big/n_pair,
               near_change=n_near_change/max(n_dec,1), n_dec=n_dec,
               dphi_med=float(np.median(dphi)), dphi_mean=float(dphi.mean()),
               xmin=float(xs.min()), xmax=float(xs.max()),
               ymin=float(ys.min()), ymax=float(ys.max())),
          open("out/probe_wrap.json", "w"), ensure_ascii=False, indent=1)
say("\n저장: out/probe_wrap.json")
