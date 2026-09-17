"""발사가 먹히는가 — RAM 과 운석 개수로 결정적으로 본다 (화면 비교는 깜빡임에 속는다)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env

env = make_env(); A = env.unwrapped.get_action_meanings()
ale = env._env.env.env.ale
FIREOF = {"NOOP":"FIRE","LEFT":"LEFTFIRE","RIGHT":"RIGHTFIRE","UP":"UPFIRE"}


def n_ast():
    return sum(1 for o in env.objects
               if o and type(o).__name__ == "Asteroid" and o.wh[0] > 0)


def run(mode, n=1800, seed=11):
    env.reset(seed=seed)
    for _ in range(40): env.step(A.index("NOOP"))
    rng = np.random.default_rng(0)
    rams = []; score = 0.0; counts = []; blend = []
    for f in range(n):
        rot = rng.choice(["NOOP", "LEFT", "RIGHT", "UP"], p=[.55, .15, .15, .15])
        a = rot if mode == "none" else (FIREOF[rot] if (mode == "hold" or f % 8 == 0) else rot)
        s1 = np.asarray(ale.getScreenRGB(), dtype=np.uint8)
        o, r, tr, te, info = env.step(A.index(a))
        score += float(r)
        if f % 7 == 0:
            rams.append(ale.getRAM().copy())
            counts.append(n_ast())
            blend.append(np.maximum(s1, np.asarray(ale.getScreenRGB(), dtype=np.uint8)).copy())
        if tr or te: break
    return rams, score, counts, blend


b_ram, b_s, b_c, b_f = run("none")
print(f"발사 없음: 점수 {b_s:.0f}, 운석 수 시퀀스 앞 {b_c[:12]}", flush=True)
for m in ("hold", "pulse8"):
    r_, s_, c_, f_ = run(m)
    n = min(len(b_ram), len(r_))
    dram = [i for i in range(n) if not np.array_equal(b_ram[i], r_[i])]
    dscr = [i for i in range(n) if not np.array_equal(b_f[i], f_[i])]
    print(f"{m:<7} 점수 {s_:6.0f}  RAM 다른 표본 {len(dram)}/{n} (첫 {dram[0] if dram else '-'})"
          f"  합성화면 다른 표본 {len(dscr)}/{n}", flush=True)
    print(f"        운석 수 앞 {c_[:12]}", flush=True)

# 발사 직후 화면에서 탄환이 보이는가 — 한 스텝씩 들여다본다
env.reset(seed=11)
for _ in range(120): env.step(A.index("NOOP"))
st = ale.cloneSystemState()
ale.restoreSystemState(st); base = []
for k in range(14):
    env.step(A.index("NOOP")); base.append(np.asarray(ale.getScreenRGB(), dtype=np.uint8).copy())
ale.restoreSystemState(st); fire = []
env.step(A.index("FIRE"))
fire.append(np.asarray(ale.getScreenRGB(), dtype=np.uint8).copy())
for k in range(13):
    env.step(A.index("NOOP")); fire.append(np.asarray(ale.getScreenRGB(), dtype=np.uint8).copy())
print("\nFIRE 1회 후 프레임별 차이 픽셀 수 (NOOP 기준)", flush=True)
print("  " + " ".join(f"{int((fire[k]!=base[k]).any(axis=2).sum()):>4}" for k in range(14)), flush=True)
d = (fire[6] != base[6]).any(axis=2)
ys, xs = np.nonzero(d)
if len(xs): print(f"  6프레임째 차이 위치 x {xs.min()}~{xs.max()} y {ys.min()}~{ys.max()}", flush=True)
