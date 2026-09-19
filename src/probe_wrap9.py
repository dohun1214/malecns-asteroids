"""y 랩 주기 확정 — **화면 픽셀로 직접.** 앞선 시도 2개는 자가시험에서 떨어졌다.

  probe_wrap7 : 계속 추진 = 가속. 등속 가정이 깨져 x 자가시험이 10.6 (정답 160) 로 실패.
  probe_wrap8 : 관성 주행. 그런데 '랩'으로 잡힌 것이 전이 구간(화면 밖 520~528) 진입/이탈이라
                주기가 18~34 로 나왔다. 역시 자가시험 실패.

이번엔 객체 좌표를 안 믿는다. **렌더된 RGB 화면에서 배경이 아닌 픽셀이 어느 행에 나타나는지**
장시간 누적한다. 놀이터(playfield)의 세로 범위가 곧 y 랩 주기다.
점수 HUD 는 위쪽 별도 띠로 분리돼 보일 것이다.
가로도 같이 재서 **160 이 나오는지**로 방법 자체를 자가시험한다.
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, Vision, with_fire, frame_action, ACT_EVERY
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()


def grab():
    return np.asarray(env._env.env.env.ale.getScreenRGB(), dtype=np.uint8)


rng = np.random.default_rng(5)
env.reset(seed=0)
for _ in range(17): env.step(A.index("NOOP"))
scr = grab()
H, W = scr.shape[0], scr.shape[1]
bg = np.bincount(scr.reshape(-1, 3).astype(np.int64) @ np.array([65536, 256, 1])).argmax()
say(f"화면 {W} x {H},  배경색 코드 {bg}")
row = np.zeros(H, np.int64); col = np.zeros(W, np.int64)
act = (A.index("NOOP"), A.index("FIRE"))
for f in range(9000):
    _, _, tr, te, _ = env.step(frame_action(act[0], act[1], f))
    if tr or te:
        env.reset(seed=0)
        for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
        continue
    if f % 3: continue
    s = grab().astype(np.int64) @ np.array([65536, 256, 1])
    m = s != bg
    row += m.sum(1); col += m.sum(0)
    if f % ACT_EVERY == 0:
        b = int(rng.integers(0, 4))
        base = [A.index(x) for x in ("NOOP", "LEFT", "RIGHT", "UP")][b]
        act = (base, with_fire(base, A))

nz = np.flatnonzero(row > 0)
say(f"\n비배경 픽셀이 한 번이라도 나온 행: {nz.min()} ~ {nz.max()}")
# 연속 구간(띠)으로 쪼갠다
bands = []
st = nz[0]; prev = nz[0]
for r in nz[1:]:
    if r - prev > 3:
        bands.append((st, prev)); st = r
    prev = r
bands.append((st, prev))
say("행 띠(간격 3 이상으로 분리):")
for a, b in bands:
    say(f"   {a:3d} ~ {b:3d}  (높이 {b-a+1:3d})  누적 픽셀 {row[a:b+1].sum():,}")
say("\n행별 누적 픽셀 (0~40, 180~209) — 점수 HUD 와 놀이터의 경계를 본다")
for lo, hi in ((0, 41), (180, 210)):
    say("   " + " ".join(f"{r}:{row[r]}" for r in range(lo, hi) if row[r] > 0))
nzc = np.flatnonzero(col > 0)
say(f"\n열 범위: {nzc.min()} ~ {nzc.max()}  (폭 {nzc.max()-nzc.min()+1})  "
    f"<- 자가시험: 160 이어야 한다")
json.dump(dict(rows=[[int(a), int(b)] for a, b in bands],
               col_lo=int(nzc.min()), col_hi=int(nzc.max())),
          open("out/probe_wrap9.json", "w"), ensure_ascii=False, indent=1)
say("저장: out/probe_wrap9.json")
