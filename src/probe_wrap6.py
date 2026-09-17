"""y 랩 주기를 **스캔**으로 확정한다.

전이 구간(화면 밖 520~528)의 길이가 들쭉날쭉해서 역산이 180~206 으로 흩어졌다.
대신 운석을 쓴다 — 운석은 등속이므로, **맞는 주기 P 로 감싸면 프레임당 변위가 작아야 한다.**
후보 P 를 훑어 '작은 변위' 비율이 최대인 값을 고른다. x 도 같은 방법으로 검증한다
(정답 160 이 나와야 한다 — 방법 자체의 자가시험이다).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()

dxs, dys = [], []
for seed in (0, 1, 2):
    env.reset(seed=0)
    for _ in range(11 + seed*7): env.step(A.index("NOOP"))
    prev = {}
    for f in range(4000):
        env.step(A.index("NOOP"))
        if f % 2: continue                       # 깜빡임 위상 고정
        _, asts = V.parse(env.objects)
        cur = {}
        for a in asts:
            cur.setdefault(tuple(a.wh), []).append((float(a.xy[0]), float(a.xy[1])))
        for k, v in cur.items():
            # **크기가 같은 운석이 정확히 1개일 때만** 쓴다. 매칭 모호성 제거.
            if len(v) == 1 and k in prev and len(prev[k]) == 1:
                dxs.append(v[0][0] - prev[k][0][0])
                dys.append(v[0][1] - prev[k][0][1])
        prev = cur
dxs = np.array(dxs); dys = np.array(dys)
say(f"모호하지 않은 연속 관측 {len(dxs):,}쌍")


def scan(d, lo, hi, small):
    best = None
    rows = []
    for P in range(lo, hi+1):
        w = d - P*np.round(d/P)
        frac = float(np.mean(np.abs(w) <= small))
        rows.append((P, frac))
        if best is None or frac > best[1]: best = (P, frac)
    return best, rows


for nm, d, lo, hi, small in (("x", dxs, 140, 200, 6.0), ("y", dys, 140, 220, 6.0)):
    say(f"\n[{nm}] 변위 범위 {d.min():.0f}~{d.max():.0f},  "
        f"|d|>{small} 인 관측 {int(np.sum(np.abs(d)>small)):,} "
        f"({np.mean(np.abs(d)>small)*100:.1f}%)")
    (P, fr), rows = scan(d, lo, hi, small)
    top = sorted(rows, key=lambda r: -r[1])[:6]
    say(f"   최적 주기 **{P}** (작은 변위 비율 {fr*100:.2f}%)")
    say(f"   상위 후보: " + "  ".join(f"{p}:{f*100:.2f}%" for p, f in top))
    say(f"   현재 코드 값 {160 if nm=='x' else 210} 일 때: "
        f"{dict(rows)[160 if nm=='x' else 210]*100:.2f}%")
json.dump(dict(n=len(dxs)), open("out/probe_wrap6.json", "w"))
