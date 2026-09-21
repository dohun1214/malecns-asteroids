"""❓ [이슈 #63] 09문서·06문서 §7 의 '배가 안 보이는 구간 20%' 는 무엇이었나.

결정 시점마다 Player 상태를 셋으로 나눈다.
  없음   : wh>0 인 Player 객체가 아예 없다
  띠     : Player 는 있는데 y 가 놀이터(18~194) 밖 — 520~528 띠
  보임   : 놀이터 안
예전 코드(수정 전)는 '띠'도 배로 받았으므로 예전의 '배 부재' = '없음' 뿐이다.
'없음' 구간이 목숨 감소(피격) 직후에 몰리는지도 센다. 규칙 정책(GPU 없음), 시드 5개.
"""
import sys, json, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, GreedyPolicy, frame_action, ACT_EVERY, with_fire

FRAMES = 6000; SEEDS = [0, 1, 2, 3, 4]
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings()
res = []
for sd in SEEDS:
    V = Vision(); pol = GreedyPolicy(); rng = np.random.default_rng(100 + sd)
    env.reset(seed=sd)
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset()
    action = (A.index("NOOP"), A.index("FIRE"))
    st = []; lives_prev = None; death_dec = []
    for f in range(FRAMES):
        _, _, tr, te, info = env.step(frame_action(action[0], action[1], f))
        if tr or te: break
        if f % ACT_EVERY: continue
        ps = [o for o in env.objects if o is not None and type(o).__name__ == "Player"
              and o.wh[0] > 0 and o.wh[1] > 0]
        if not ps: s = "없음"
        elif any(14 <= float(o.xy[1]) <= 198 for o in ps): s = "보임"
        else: s = "띠"
        st.append(s)
        lives = info.get("lives")
        if lives_prev is not None and lives < lives_prev: death_dec.append(len(st) - 1)
        lives_prev = lives
        xy, head, looms = V.looming(env.objects)
        if xy is None:
            action = (A.index("NOOP"), A.index("FIRE")); continue
        ori = int(getattr(ps[0], "orientation", 0)) if ps else 0
        a, _ = pol(looms, ori, A, vel=V.ship_v)
        action = (a, with_fire(a, A))
    n = len(st); c = collections.Counter(st)
    # 구간 길이
    runs = collections.defaultdict(list); cur = st[0]; L = 0
    for s in st:
        if s == cur: L += 1
        else: runs[cur].append(L); cur = s; L = 1
    runs[cur].append(L)
    # '없음' 결정 중 피격 후 60결정(4초) 안에 있는 비율
    near = set()
    for d in death_dec:
        for k in range(max(0, d - 5), min(n, d + 60)): near.add(k)
    none_idx = [i for i, s in enumerate(st) if s == "없음"]
    frac_near = sum(1 for i in none_idx if i in near) / max(len(none_idx), 1)
    r = dict(seed=sd, n=n, deaths=len(death_dec),
             없음=c["없음"]/n, 띠=c["띠"]/n, 보임=c["보임"]/n,
             없음_평균구간프레임=float(np.mean(runs["없음"]))*ACT_EVERY if runs["없음"] else 0,
             없음_최대구간프레임=int(max(runs["없음"]))*ACT_EVERY if runs["없음"] else 0,
             띠_평균구간프레임=float(np.mean(runs["띠"]))*ACT_EVERY if runs["띠"] else 0,
             없음중_피격직후=frac_near)
    res.append(r); say(r)
keys = ["없음", "띠", "보임", "없음중_피격직후"]
summ = {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res]))) for k in keys}
say("평균±표준편차", summ)
json.dump(dict(runs=res, summary=summ), open("out/probe_absent.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
