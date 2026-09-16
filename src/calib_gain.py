"""자극 세기 상수를 실전 분포 위에서 고른다.

앞선 실수: 95%ile 팽창률(1.454)이 80Hz 가 되게 잡았더니 '전형적인' 위협(중앙값 0.093)이
5Hz 밖에 안 됐고, DN 이 프레임당 0~2 스파이크라 방향이 안 읽혔다.

기준(중립): 전형적 위협이 게이트 1(a)에서 검증된 동작점(DN 이 포화 30% 미만에서
등급을 내는 구간)에 오도록 한다. '점수가 잘 나오는 값'으로 고르지 않는다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env
from vision import Vision

ROOT = Path(__file__).resolve().parent.parent
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
RFC_MAX = 1000.0/0.2/11    # 불응 11 step -> 455 Hz
bp = BrainPolicy()

def sample(gain, n_dec=150, seed=0):
    bp.gain = gain; bp.frame = 0
    env.reset(seed=seed); V.reset()
    act = A.index("FIRE"); rows = []; acts = []
    f = 0
    while len(rows) < n_dec and f < 6000:
        obs, rew, tr, te, info = env.step(act); f += 1
        if te or tr: env.reset(); V.reset(); continue
        if f % 4: continue
        xy, head, looms = V.looming(env.objects)
        if xy is None: act = A.index("FIRE"); continue
        ori = 0
        for o in env.objects:
            if o and type(o).__name__ == "Player": ori = int(getattr(o,"orientation",0)); break
        a, ch = bp(looms, ori, A)
        from play import with_fire
        act = with_fire(a, A)
        rows.append(ch); acts.append(A[a])
    d02 = np.array([r["p02_L"]+r["p02_R"] for r in rows])
    d11 = np.array([r["p11_L"]+r["p11_R"] for r in rows])
    d04 = np.array([r["intensity"] for r in rows])
    nrm = np.array([r["norm"] for r in rows])
    from collections import Counter
    c = Counter(acts)
    return dict(gain=gain, d02=d02.mean(), d11=d11.mean(), d04=d04.mean(),
                sat=max(d02.max(), d11.max(), d04.max())/RFC_MAX,
                zero=float(np.mean(nrm < 1e-9)),
                acts={k: v/len(acts) for k, v in c.most_common()})

print(f"{'gain':>7} {'DNp02':>8} {'DNp11':>8} {'DNp04':>8} {'포화':>6} {'무반응%':>8}  액션 분포")
R = []
for g in (2.0, 7.2, 20.0, 50.0):
    r = sample(g)
    R.append(r)
    print(f"{g:>7} {r['d02']:>8.1f} {r['d11']:>8.1f} {r['d04']:>8.1f} "
          f"{r['sat']*100:>5.0f}% {r['zero']*100:>7.0f}%  "
          f"{ {k: round(v,2) for k,v in r['acts'].items()} }")
(ROOT/"out"/"calib_gain.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
print(f"\n선택 기준: DN 이 0 이 아니고 포화 30% 미만, 회전/추진이 고루 나오는 구간")
