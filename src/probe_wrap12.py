"""★ 마지막 시도 — **개루프**로. 폐루프 스캔은 원리적으로 안 된다.

probe_wrapfit(행동 스캔)은 실패했다. 편향 최소는 166, 각거리 최대는 174 로 갈렸고
인접 후보끼리 11.5% vs 27.1% 처럼 튀었다. 이유는 12문서에 이미 적혀 있다 —
**"폐루프 조건 비교에서는 궤적 발산 자체가 교란 변수다."** 주기를 1px 바꾸면 궤적이
갈라지고, 그 차이가 주기의 효과보다 크다. *우리가 적어둔 함정에 우리가 걸렸다.*
(다만 **210 은 두 지표 모두에서 최악**이라, 고친 방향이 맞다는 것은 확인됐다.)

개루프로 간다. 배를 정지시키고 **운석의 감싸기 사건**만 본다. 운석은 등속이고
화면 밖 좌표가 0% 다. probe_wrap11 이 158 을 낸 이유는 이음매 근처에서 검출이 끊겨
트랙이 일찍 끝나는데 **간격 창을 8프레임으로 좁게** 잡았기 때문이다.

이번엔
  - 간격 창을 **80프레임**까지 연다 (이음매에서 오래 안 보일 수 있다)
  - **x 가 맞아떨어지는 것**을 조건으로 건다: |wrap(ex + vx·k - sx, 160)| < 3
    x 주기는 확정값이므로 이건 공짜 제약이고, 가짜 짝을 거의 다 걸러낸다
"""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
STEP = 2; GAP = 80


def wrapx(d): return d - 160.0*np.round(d/160.0)


def collect(frames=60000):
    env.reset(seed=0)
    for _ in range(17): env.step(A.index("NOOP"))
    tracks = {}; nxt = 0
    for f in range(frames):
        env.step(A.index("NOOP"))
        if f % STEP: continue
        _, asts = V.parse(env.objects)
        cur = collections.defaultdict(list)
        for a in asts:
            cur[tuple(a.wh)].append((float(a.xy[0]), float(a.xy[1])))
        used = set()
        for k, pts in cur.items():
            for (x, y) in pts:
                best, bd = None, 1e9
                for tid, t in tracks.items():
                    if t["wh"] != k or tid in used: continue
                    lf, lx, ly = t["last"]
                    if f - lf > STEP: continue
                    dx = wrapx(x - lx); dy = y - ly
                    if abs(dx) > 8 or abs(dy) > 8: continue
                    d = dx*dx + dy*dy
                    if d < bd: bd, best = d, tid
                if best is None:
                    tracks[nxt] = dict(wh=k, last=(f, x, y), pts=[(f, x, y)]); nxt += 1
                else:
                    used.add(best); tracks[best]["last"] = (f, x, y)
                    tracks[best]["pts"].append((f, x, y))
    return tracks


tr = collect()
vel = {}
for tid, t in tr.items():
    p = t["pts"]
    if len(p) < 10: continue
    vx = np.mean([wrapx(b[1]-a[1])/(b[0]-a[0]) for a, b in zip(p, p[1:]) if b[0]-a[0] == STEP])
    vy = np.mean([(b[2]-a[2])/(b[0]-a[0]) for a, b in zip(p, p[1:]) if b[0]-a[0] == STEP])
    vel[tid] = (float(vx), float(vy), p)
say(f"트랙 {len(tr)}개 중 등속 추정 {len(vel)}개")

ev = []
for eid, (evx, evy, ep) in vel.items():
    ef, ex, ey = ep[-1]
    for sid, (svx, svy, sp) in vel.items():
        if sid == eid or tr[sid]["wh"] != tr[eid]["wh"]: continue
        sf, sx, sy = sp[0]
        k = sf - ef
        if not (0 < k <= GAP): continue
        if abs(evx - svx) > 0.08 or abs(evy - svy) > 0.08: continue
        if abs(wrapx(ex + evx*k - sx)) > 3.0: continue        # ★ x 가 맞아떨어져야 한다
        P = (ey + evy*k) - sy
        if abs(P) > 80: ev.append(abs(P))

if not ev:
    say("감싸기 사건 0개 — 더 길게 돌려야 한다")
else:
    c = collections.Counter(int(round(v)) for v in ev)
    say(f"감싸기 사건 {len(ev)}개")
    say(f"   최빈 {c.most_common(8)}")
    say(f"   중앙 {np.median(ev):.1f}   평균 {np.mean(ev):.1f} ± {np.std(ev):.1f}")
    say(f"   -> **y 주기 = {c.most_common(1)[0][0]}**")
# ★ 자가시험 — 방향을 뒤집는다. 방금 찾은 y=178 을 제약으로 걸고 x 를 추정하면 **160** 이어야 한다.
WRAPY = 178.0
def wrapy(d): return d - WRAPY*np.round(d/WRAPY)
evx = []
for eid, (evx_, evy_, ep) in vel.items():
    ef, ex, ey = ep[-1]
    for sid, (svx, svy, sp) in vel.items():
        if sid == eid or tr[sid]["wh"] != tr[eid]["wh"]: continue
        sf, sx, sy = sp[0]
        k = sf - ef
        if not (0 < k <= GAP): continue
        if abs(evx_ - svx) > 0.08 or abs(evy_ - svy) > 0.08: continue
        if abs(wrapy(ey + evy_*k - sy)) > 3.0: continue       # y 가 맞아떨어져야 한다
        P = (ex + evx_*k) - sx
        if abs(P) > 80: evx.append(abs(P))
if not evx:
    say("\n★ 자가시험: x 감싸기 사건 0개 (운석은 세로로만 움직인다 — 09문서). 판별 불가")
else:
    cx = collections.Counter(int(round(v)) for v in evx)
    say(f"\n★ 자가시험: x 사건 {len(evx)}개  최빈 {cx.most_common(5)}"
        f"  -> {cx.most_common(1)[0][0]} (정답 160)")
json.dump(dict(n=len(ev), vals=[float(v) for v in ev[:200]],
               x_n=len(evx), x_vals=[float(v) for v in evx[:200]]),
          open("out/probe_wrap12.json", "w"))
