"""고친 뒤 실제 정책 루프에서 발사가 되는지 확인."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import run_episode, make_env, Vision, GreedyPolicy

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
r2 = np.random.default_rng(7)
POLS = [("가만히 있기", lambda l,o,a:(a.index("NOOP"), {"norm":0.0})),
        ("규칙 기반", GreedyPolicy())]
print(f"{'정책':<14}{'fire':>6}{'점수':>9}{'평균 생존':>10}{'프레임':>8}", flush=True)
for name, pol in POLS:
    for fire in (True, False):
        rng = np.random.default_rng(12345)
        rs = [run_episode(env, pol, V, A, max_frames=6000, rng=rng, fire=fire)
              for _ in range(4)]
        print(f"{name:<14}{str(fire):>6}{np.mean([r['score'] for r in rs]):>9.0f}"
              f"{np.mean([r['mean_life'] for r in rs]):>10.0f}"
              f"{np.mean([r['frames'] for r in rs]):>8.0f}", flush=True)
