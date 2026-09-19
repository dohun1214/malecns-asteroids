"""y 랩 주기 확정 — **운석의 감싸기 사건을 직접 잡는다.**

앞선 시도가 전부 실패한 이유
  #7  계속 추진 = 가속 (자가시험 10.6)
  #8  전이 구간(배 y 520~528)을 랩으로 오인 (18~34)
  #9  픽셀 행 범위 -> 177. 맞는 자릿수지만 '칠해진 행 수'지 좌표 주기가 아니다
  #10 좌표 집합 범위 -> 배 x 가 164 (정답 160). **스프라이트가 이음매를 걸치면 좌표가
      범위 밖으로 나간다.** 범위는 주기가 아니다

이번엔 **운석**을 쓴다. 운석은 화면 밖 좌표가 **0%** 다 (probe_wrap3). 그리고 등속이다.
배는 정지(NOOP)시켜 두고, 크기 + x(주기 160, 이미 확정) 로 운석을 추적한다.
y 가 갑자기 크게 튀면 그게 감싸기다:

    P = (튀기 직전 y) + (프레임당 dy) x (경과 프레임) - (튀고 난 뒤 y)

자가시험: **같은 코드로 x 의 주기를 재면 160 이 나와야 한다.**
"""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
STEP = 2                      # 스프라이트 깜빡임 위상 고정


def collect(frames=40000):
    """(크기, 트랙id) -> [(프레임, x, y)]"""
    env.reset(seed=0)
    for _ in range(17): env.step(A.index("NOOP"))
    tracks = {}          # key -> dict(last=(f,x,y), vel=(vx,vy), pts=[])
    nxt = 0
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
                    if f - lf > STEP*2: continue
                    dx = x - lx; dx -= 160.0*round(dx/160.0)      # x 는 160 확정
                    dy = y - ly
                    if abs(dx) > 8 or abs(dy) > 8: continue       # 작은 이동만 연결
                    d = dx*dx + dy*dy
                    if d < bd: bd, best = d, tid
                if best is None:
                    tracks[nxt] = dict(wh=k, last=(f, x, y), pts=[(f, x, y)]); nxt += 1
                else:
                    used.add(best)
                    tracks[best]["last"] = (f, x, y)
                    tracks[best]["pts"].append((f, x, y))
        # 오래 끊긴 트랙은 버리되 '마지막 지점'은 남겨 감싸기 후보로 쓴다
    return tracks


tracks = collect()
say(f"트랙 {len(tracks)}개,  점 {sum(len(t['pts']) for t in tracks.values()):,}개")

# 각 트랙의 등속 속도
vel = {}
for tid, t in tracks.items():
    p = t["pts"]
    if len(p) < 8: continue
    dy = [(b[2]-a[2])/(b[0]-a[0]) for a, b in zip(p, p[1:]) if b[0]-a[0] == STEP]
    dx = [( (b[1]-a[1]) - 160.0*round((b[1]-a[1])/160.0) )/(b[0]-a[0])
          for a, b in zip(p, p[1:]) if b[0]-a[0] == STEP]
    if dy: vel[tid] = (float(np.mean(dx)), float(np.mean(dy)), p)

say(f"등속 추정 가능한 트랙 {len(vel)}개")

# 끝난 트랙의 마지막 점 <-> 새로 생긴 트랙의 첫 점을 이어 감싸기를 찾는다
ends = [(t[2][-1], t[0], t[1], tid) for tid, t in vel.items()]
starts = [(t[2][0], t[0], t[1], tid) for tid, t in vel.items()]
evy, evx = [], []
for (ef, ex, ey), evx_, evy_, eid in ends:
    for (sf, sx, sy), svx, svy, sid in starts:
        if sid == eid or not (0 < sf - ef <= STEP*4): continue
        if tracks[eid]["wh"] != tracks[sid]["wh"]: continue
        if abs(evx_ - svx) > 0.4 or abs(evy_ - svy) > 0.4: continue   # 같은 속도
        k = sf - ef
        py = ey + evy_*k - sy
        px = (ex + evx_*k - sx)
        if abs(py) > 60: evy.append(abs(py))
        if abs(px) > 60: evx.append(abs(px))


def mode(v, name, known=None):
    if not v:
        say(f"{name}: 감싸기 사건 0개"); return None
    c = collections.Counter(int(round(x)) for x in v)
    say(f"{name}: 사건 {len(v)}개  중앙 {np.median(v):.1f}  최빈 {c.most_common(5)}")
    m = c.most_common(1)[0][0]
    if known is not None:
        say(f"   -> 자가시험 {'통과' if abs(m-known) <= 2 else '실패'} (최빈 {m}, 정답 {known})")
    return m


mx = mode(evx, "x 주기", known=160)
my = mode(evy, "y 주기")
json.dump(dict(x=mx, y=my), open("out/probe_wrap11.json", "w"))
