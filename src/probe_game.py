"""05·02문서가 주장만 해놓고 측정 안 한 것들을 확인한다. 컨트롤러를 이 위에 짓는다.

  1. ALE Asteroids 가 네이티브 4프레임당 정확히 22.5도(16방위 중 1칸) 회전하는가  (05 3.4)
  2. OCAtari Player.orientation 이 16방위인가                                  (02 2장)
  3. step() 반환 순서가 obs, reward, truncated, terminated, info 인가           (05 3.2)
  4. Player.dx/dy 가 항상 0인가                                                (05 3.2)
  5. 물체 약 1% 가 h <= 0 인가 / 운석 크기 3종 / 화면당 개수                      (02 11장)
  6. 스텝 비용 (raw ALE vs OCAtari)                                            (05 3.1)
"""
import sys, json, time, inspect
from pathlib import Path
import numpy as np
from ocatari.core import OCAtari

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
R = {}
def say(*a): print(*a, flush=True)

def mk():
    return OCAtari("ALE/Asteroids-v5", mode="ram", hud=False, frameskip=1,
                   repeat_action_probability=0.0)

env = mk()
say(f"액션 {env.action_space}  의미: {env.unwrapped.get_action_meanings()}")
R["actions"] = env.unwrapped.get_action_meanings()

# ---------------------------------------------------------------- 3. 반환 순서
src = inspect.getsource(type(env).step)
say(f"\n[3] OCAtari.step() 반환문:")
for line in src.splitlines():
    if "return" in line:
        say(f"    {line.strip()}")
R["step_return_src"] = [l.strip() for l in src.splitlines() if "return" in l]

out = env.reset(seed=0)
say(f"    reset() 반환 길이 {len(out)}")
res = env.step(0)
say(f"    step() 반환 길이 {len(res)}, 타입 {[type(x).__name__ for x in res]}")

# ---------------------------------------------------------------- 2. orientation
env.reset(seed=0)
for _ in range(120): env.step(0)          # 게임 시작 대기
def player(e):
    for o in e.objects:
        if o and type(o).__name__ == "Player": return o
    return None
p = player(env)
say(f"\n[2] Player 속성: orientation={getattr(p,'orientation',None)} "
    f"xy={getattr(p,'xy',None)} wh={getattr(p,'wh',None)}")

# ---------------------------------------------------------------- 1. 회전 해상도
LEFT = env.unwrapped.get_action_meanings().index("LEFT") if "LEFT" in env.unwrapped.get_action_meanings() else 4
ors, frames = [], []
for f in range(400):
    env.step(LEFT)
    p = player(env)
    ors.append(None if p is None else int(getattr(p, "orientation", -1)))
    frames.append(f)
valid = [o for o in ors if o is not None and o >= 0]
uniq = sorted(set(valid))
say(f"\n[1] LEFT 400프레임 연속: orientation 고유값 {len(uniq)}개 -> {uniq}")
# 값이 바뀐 프레임 간격
chg = [i for i in range(1, len(ors)) if ors[i] != ors[i-1] and ors[i] is not None and ors[i-1] is not None]
gaps = np.diff(chg) if len(chg) > 1 else np.array([])
if gaps.size:
    vals, cnts = np.unique(gaps, return_counts=True)
    say(f"    값이 바뀐 프레임 간격 분포: {dict(zip(vals.tolist(), cnts.tolist()))}")
    say(f"    중앙값 {np.median(gaps):.0f} 프레임/칸"
        f"  -> 05문서 주장 '4프레임당 1칸' {'일치' if np.median(gaps)==4 else '불일치'}")
    if len(uniq) > 1:
        say(f"    한 칸 = 360/{len(uniq)} = {360/len(uniq):.1f}도"
            f"  -> 05문서 주장 22.5도 {'일치' if abs(360/len(uniq)-22.5)<0.1 else '불일치'}")
R["orientation_values"] = uniq
R["frames_per_step"] = float(np.median(gaps)) if gaps.size else None

# ---------------------------------------------------------------- 4. dx/dy
env.reset(seed=0)
for _ in range(120): env.step(0)
UP = env.unwrapped.get_action_meanings().index("UP")
pdx, adx, prev = [], [], None
for _ in range(200):
    env.step(UP)
    p = player(env)
    if p is not None:
        pdx.append((getattr(p,"dx",None), getattr(p,"dy",None)))
    for o in env.objects:
        if o and type(o).__name__ == "Asteroid":
            adx.append((getattr(o,"dx",None), getattr(o,"dy",None)))
nz_p = sum(1 for d in pdx if d[0] not in (0, None) or d[1] not in (0, None))
nz_a = sum(1 for d in adx if d[0] not in (0, None) or d[1] not in (0, None))
say(f"\n[4] Player dx/dy 표본 {len(pdx)}개 중 0이 아닌 것 {nz_p}개  "
    f"-> 05문서 주장 '항상 0' {'일치' if nz_p==0 else '불일치'}")
say(f"    Asteroid dx/dy 표본 {len(adx)}개 중 0이 아닌 것 {nz_a}개 ({nz_a/max(len(adx),1)*100:.0f}%)  "
    f"-> 05문서 주장 '운석은 정상' {'일치' if nz_a>0 else '불일치'}")
R["player_dxdy_nonzero"], R["asteroid_dxdy_nonzero_pct"] = nz_p, nz_a/max(len(adx),1)*100

# ---------------------------------------------------------------- 5. 물체 통계
env.reset(seed=0)
sizes, bad, counts, cats = [], 0, [], {}
rng = np.random.default_rng(0)
for _ in range(3000):
    env.step(int(rng.integers(0, env.action_space.n)))
    objs = [o for o in env.objects if o]
    n_ast = 0
    for o in objs:
        c = type(o).__name__
        cats[c] = cats.get(c, 0) + 1
        w, h = o.wh
        if w <= 0 or h <= 0: bad += 1
        if c == "Asteroid":
            n_ast += 1; sizes.append((w, h))
    counts.append(n_ast)
tot = sum(cats.values())
say(f"\n[5] 3000프레임: 물체 {tot:,}개, 종류 {cats}")
say(f"    w<=0 or h<=0 인 물체 {bad}개 ({bad/tot*100:.2f}%)  -> 02문서 주장 '약 1%' "
    f"{'일치' if 0.2 < bad/tot*100 < 3 else '불일치'}")
su, sc = np.unique(np.array(sizes), axis=0, return_counts=True)
top = sorted(zip(su.tolist(), sc.tolist()), key=lambda z: -z[1])[:6]
say(f"    운석 크기 상위: {top}")
say(f"    화면당 운석 수: 평균 {np.mean(counts):.1f}, 중앙값 {np.median(counts):.0f}, "
    f"범위 {min(counts)}~{max(counts)}  -> 02문서 주장 '보통 4~8개'")
R["bad_wh_pct"] = bad/tot*100; R["ast_per_frame"] = float(np.mean(counts))
R["ast_sizes_top"] = [(list(s), int(c)) for s, c in top]

# ---------------------------------------------------------------- 6. 스텝 비용
import ale_py, gymnasium as gym
raw = gym.make("ALE/Asteroids-v5", frameskip=1, repeat_action_probability=0.0)
raw.reset(seed=0)
t0 = time.perf_counter()
for _ in range(3000): raw.step(0)
t_raw = (time.perf_counter()-t0)/3000*1000
env.reset(seed=0)
t0 = time.perf_counter()
for _ in range(3000): env.step(0)
t_oc = (time.perf_counter()-t0)/3000*1000
say(f"\n[6] 스텝 비용: raw ALE {t_raw:.3f} ms / OCAtari {t_oc:.3f} ms  "
    f"(05문서 0.303 / 0.724)")
say(f"    뇌 예산 9.2 ms 대비 OCAtari {t_oc/9.2*100:.1f}%")
R["ms_raw"], R["ms_ocatari"] = t_raw, t_oc

(OUT/"probe_game.json").write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
say(f"\n-> out/probe_game.json")
