"""분기 녹화 전에 '무엇을 저장해야 같은 시작점이 되는가'를 실측한다 (05문서 §3.6).

cloneState() 는 RNG 를 포함하지 않아 재개 시 발산한다 -> cloneSystemState().
그리고 ALE 상태만 복원하면 OCAtari 객체가 step 0 부터 어긋난다고 05문서가 경고한다.
**어긋나는지 직접 확인한다.**
"""
import sys, copy
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import make_env

env = make_env(); A = env.unwrapped.get_action_meanings()
u = env.unwrapped
print("OCAtari 속성:", [a for a in dir(env) if not a.startswith("__")][:40], flush=True)
print("unwrapped:", type(u).__name__, flush=True)
# OCAtari 는 ale 를 _env.env.env.ale 로 들고 있고 _clone_state/_restore_state 를 제공한다
ale = env._env.env.env.ale
print("ale:", type(ale).__name__, " cloneSystemState:",
      hasattr(ale, "cloneSystemState"), flush=True)
for name in ("_state_buffer_ns", "_state_buffer_rgb", "_state_buffer_dqn", "objects",
             "_slots", "_ns_state", "buffer_window_size", "_env"):
    print(f"  env.{name}: {hasattr(env, name)}", flush=True)
print("  u._np_random:", hasattr(u, "_np_random"), flush=True)


def obj_sig(e):
    return tuple((type(o).__name__, tuple(o.xy), tuple(o.wh))
                 for o in e.objects if o)


def run(e, acts):
    sig = []
    for a in acts:
        e.step(a)
        sig.append(obj_sig(e))
    return sig


rng = np.random.default_rng(0)
acts = [int(rng.integers(0, 14)) for _ in range(60)]
env.reset(seed=7)
for _ in range(80): env.step(A.index("FIRE"))

# --- 스냅샷 A: ALE 시스템 상태만
s_ale = env._clone_state()
base = run(env, acts)

def restore_ale_only():
    env._restore_state(s_ale)

restore_ale_only()
r1 = run(env, acts)
print(f"\nALE 시스템 상태만 복원 -> 객체 시퀀스 일치: {r1 == base}", flush=True)
if r1 != base:
    d = next(i for i in range(len(base)) if r1[i] != base[i])
    print(f"  첫 불일치 step {d}: base {len(base[d])}개 vs 복원 {len(r1[d])}개", flush=True)

# --- 스냅샷 B: ALE + OCAtari 내부까지
restore_ale_only()
run(env, acts[:1])            # 한 스텝 진행해 상태를 흐트러뜨린다
restore_ale_only()
extra = {}
for name in ("_state_buffer_ns", "_state_buffer_rgb", "_state_buffer_dqn",
             "objects", "_slots", "_ns_state"):
    if hasattr(env, name):
        try: extra[name] = copy.deepcopy(getattr(env, name))
        except Exception as ex: print(f"  deepcopy 실패 {name}: {ex}", flush=True)
nprng = copy.deepcopy(getattr(u, "_np_random", None))

def restore_full():
    env._restore_state(s_ale)
    for k, v in extra.items():
        setattr(env, k, copy.deepcopy(v))
    if nprng is not None: u._np_random = copy.deepcopy(nprng)

restore_full()
r2 = run(env, acts)
print(f"ALE + OCAtari 내부까지 복원 -> 일치: {r2 == base}", flush=True)
print(f"저장한 OCAtari 필드: {list(extra.keys())}", flush=True)
