"""❓ [이슈 #63] Player y 520~528 띠의 정체.
OCAtari: y = 100 + 2*(ram[74] - 41). ram[74] 가 251~255 면 y 520~528.
부호 있는 바이트로 읽으면 -5~-1 → y = 18 - 2k = 8~16 (놀이터 윗가장자리 바로 위).
띠 프레임에서 ram[74] 와, 배의 x 열 부근 화면 픽셀이 실제로 몇 행에 칠해져 있는지 본다.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env, frame_action
def say(*a): print(*a, flush=True)
env = make_env(); A = env.unwrapped.get_action_meanings(); ale = env._env.env.env.ale
env.reset(seed=0)
seen = 0; ok = 0; rows_all = []
up = A.index("UP")
for f in range(20000):
    a = up if (f // 60) % 3 == 0 else A.index("NOOP")
    _, _, tr, te, info = env.step(a)
    if tr or te: env.reset(seed=f)
    ps = [o for o in env.objects if o is not None and type(o).__name__ == "Player" and o.wh[0] > 0]
    if not ps: continue
    y = float(ps[0].xy[1]); x = int(ps[0].xy[0])
    if y < 400: continue
    ram = ale.getRAM(); r74 = int(ram[74]); s = r74 - 256
    img = np.asarray(ale.getScreenRGB()); bg = img[0, 0]
    img2 = np.asarray(ale.getScreenRGB())
    col = img[:, max(0, x-1):x+8]
    rows = np.where((np.abs(col.astype(int) - bg.astype(int)).sum(-1) > 30).any(1))[0]
    rows = [int(r) for r in rows if r < 60 or r > 180]
    seen += 1
    if seen <= 12:
        say(f"f={f} ocatari_y={y:.0f} ram74={r74} (부호={s}) → 보정 y={100+2*(s-41)}  x={x}  칠해진 행={rows}")
    rows_all.append((100 + 2*(s-41), rows))
    if seen >= 60: break
say(f"띠 프레임 {seen}개")
