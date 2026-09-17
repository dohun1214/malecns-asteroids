"""02문서 §10.4 분기 녹화 — **같은 시작점**에서 정상 뇌와 병변 뇌가 갈라지는 걸 보여준다.

이 데모의 본체다. 점수가 아니라 '같은 상황에서 다르게 행동한다'가 주장이다.

같은 시작점을 만드는 법 (05문서 §3.6 + probe_snapshot.py 실측)
  ⚠️ `cloneState()` 는 RNG 를 포함하지 않아 재개 시 발산한다 -> `cloneSystemState()`.
  ⚠️ **ALE 상태만 복원하면 OCAtari 객체가 step 0 부터 어긋난다** (실측 확인).
     같이 복원해야 하는 것: _state_buffer_ns / _state_buffer_rgb / _state_buffer_dqn /
     objects / _slots / _ns_state / unwrapped._np_random
  ⚠️ 뇌 상태도 전부: v, g, refr, inc, alive, rfc, lam, sp, cnt, over, seed, gacc, tally, _t
     + 디코더 평활 상태 + Vision 의 이전 프레임 캐시

⚠️ **MediaRecorder 금지** (05문서 §3.7). 실시간 사양이라 프레임을 버리는데,
   두 분기가 서로 다르게 버려지면 '같은 시작점' 주장이 깨진다.
   -> 서버에서 프레임을 step 번호로 인덱싱해 raw 로 덤프하고 ffmpeg 으로 합친다.
   -> 확대는 반드시 `flags=neighbor`. bilinear 로 키우면 아타리 스프라이트가 뭉갠다.

검증: '온전'을 두 번 돌려서 **바이트 단위로 같은지** 확인한다. 이게 통과해야
      나머지 분기의 차이가 '조작 때문'이라고 말할 수 있다.
"""
import sys, copy, json, subprocess, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import BrainPolicy, make_env, Vision, ACT_EVERY, with_fire, frame_action
import rewire as RW

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT/"out"/"rec"; REC.mkdir(parents=True, exist_ok=True)
PREFIX = int(sys.argv[1]) if len(sys.argv) > 1 else 45      # 공통 구간 (결정 수)
TAIL   = int(sys.argv[2]) if len(sys.argv) > 2 else 135     # 분기 구간
SEED   = int(sys.argv[3]) if len(sys.argv) > 3 else 20260917
W, H = 160, 210
def say(*a): print(*a, flush=True)

env = make_env(); A = env.unwrapped.get_action_meanings(); V = Vision()
bp = BrainPolicy(); C = bp.C
U = env.unwrapped
OC_FIELDS = ("_state_buffer_ns", "_state_buffer_rgb", "_state_buffer_dqn",
             "objects", "_slots", "_ns_state")
packed0 = bp.b.packed.clone()
crow, p0, Cg, pre = RW.load()
sel = RW.conditions(Cg, crow.size-1, pre, p0)
packed_rw = torch.from_numpy(RW.rewire(p0, sel["lc4_all"], seed=0, frac=1.0)[0]
                             .astype(np.int32)).to(bp.b.dev)


def screen():
    return np.asarray(env._env.env.env.ale.getScreenRGB(), dtype=np.uint8)


# ------------------------------------------------------------------ 스냅샷
BRAIN_T = ("v", "g", "refr", "inc", "alive", "rfc", "drive", "lam",
           "sp", "cnt", "over", "seed", "gacc", "tally", "first", "packed")

def snapshot():
    b = bp.b
    return dict(
        ale=env._clone_state(),
        oc={k: copy.deepcopy(getattr(env, k)) for k in OC_FIELDS if hasattr(env, k)},
        rng=copy.deepcopy(getattr(U, "_np_random", None)),
        brain={k: getattr(b, k).clone() for k in BRAIN_T},
        t=b._t, has_poi=b.has_poi,
        frame=bp.frame, hold=bp._hold_left,
        dec=copy.deepcopy(bp.dec._s),
        # ⚠️ ship_v 도 넣어야 한다. 관성 보정이 이 값을 쓰므로 빠뜨리면
        #   '같은 시작점' 이 깨지고 바이트 단위 동일 검증이 실패한다 (이슈 #36).
        vis=(copy.deepcopy(V.prev_ship), copy.deepcopy(V.prev), V._next_id,
             tuple(V.ship_v)))


def restore(s):
    b = bp.b
    env._restore_state(s["ale"])
    for k, v in s["oc"].items(): setattr(env, k, copy.deepcopy(v))
    if s["rng"] is not None: U._np_random = copy.deepcopy(s["rng"])
    for k in BRAIN_T: getattr(b, k).copy_(s["brain"][k])
    b._t = s["t"]; b.has_poi = s["has_poi"]
    bp.frame = s["frame"]; bp._hold_left = s["hold"]
    bp.dec._s = copy.deepcopy(s["dec"])
    V.prev_ship, V.prev, V._next_id = (copy.deepcopy(s["vis"][0]),
                                       copy.deepcopy(s["vis"][1]), s["vis"][2])
    V.ship_v = tuple(s["vis"][3]) if len(s["vis"]) > 3 else (0.0, 0.0)


# ------------------------------------------------------------------ 주행
def ship_xy():
    for o in env.objects:
        if o and type(o).__name__ == "Player" and o.wh[0] > 0:
            return float(o.xy[0]), float(o.xy[1])
    return None


def decide_and_step(sink, n_dec, track=None):
    """n_dec 번의 결정 동안 진행하면서 매 프레임을 sink 에 쓴다.
    track 이 있으면 프레임마다 배 위치도 기록한다 (영상에서 궤적을 그리려고).
    화면만 보면 분기가 잘 안 보인다 — 운석은 탄도라 총에 맞기 전까진 같은 길을 간다.
    실제로 갈라지는 건 **배의 궤적**이다."""
    acts = []
    action = (A.index("NOOP"), A.index("FIRE"))
    for d in range(n_dec):
        for k in range(ACT_EVERY):
            env.step(frame_action(action[0], action[1], k))
            sink.write(screen().tobytes())
            if track is not None:
                p = ship_xy()
                track.append((p[0], p[1], 1.0) if p else (0.0, 0.0, 0.0))
        objs = env.objects
        xy, head, looms = V.looming(objs)
        ori = 0
        for o in objs:
            if o and type(o).__name__ == "Player":
                ori = int(getattr(o, "orientation", 0)); break
        a, ch = bp(looms, ori, A, vel=V.ship_v)   # 배가 없어도 뇌는 항상 돌린다 (06문서 §7)
        acts.append(A[a])
        action = ((A.index("NOOP"), A.index("FIRE")) if xy is None
                  else (a, with_fire(a, A)))
    return acts


def apply_cond(name):
    b = bp.b
    b.alive.fill_(1); b.packed.copy_(packed0)
    if name == "온전" or name == "온전(재현확인)": return
    if name == "무작위 2세포":
        b.lesion(torch.as_tensor(np.random.default_rng(5).choice(b.N, 2, replace=False),
                                 device="cuda"))
    elif name == "LC4 배선 섞기":
        b.packed.copy_(packed_rw)
    else:
        b.lesion(torch.as_tensor(np.asarray(C[name]), device="cuda"))


def fresh_start():
    rng = np.random.default_rng(SEED)
    env.reset(seed=int(rng.integers(0, 2**31)))
    for _ in range(int(rng.integers(1, 31))): env.step(A.index("NOOP"))
    V.reset(); bp.dec.reset(); bp.b.reset(); bp.frame = 0
    bp.b.alive.fill_(1); bp.b.packed.copy_(packed0)


# ---------------------------------------------------- 0. 분기점 정찰
# [실측/06문서 §7] 배가 안 보이는 구간이 평균 265프레임짜리 덩어리로 전체의 20% 나온다.
# 하필 거기서 갈라지면 세 화면이 똑같아 보여서 데모가 아무것도 안 보여준다.
# 전부 결정론이므로 **같은 시드로 한 번 정찰해서 배가 잘 보이는 구간을 고르고**
# 두 번째 주행에서 그 지점에 스냅샷을 찍는다.
class Null:
    def write(self, b): pass


t0 = time.perf_counter()
fresh_start()
SCOUT = PREFIX + TAIL + 400
seen = []
null = Null()
for d in range(SCOUT):
    decide_and_step(null, 1)
    seen.append(1 if any(o and type(o).__name__ == "Player" and o.wh[0] > 0
                         for o in env.objects) else 0)
seen = np.asarray(seen)
cum = np.concatenate([[0], np.cumsum(seen)])
best, best_v = PREFIX, -1.0
for sp_ in range(PREFIX, SCOUT - TAIL):
    v = (cum[sp_+TAIL] - cum[sp_]) / TAIL
    if v > best_v: best_v, best = v, sp_
say(f"정찰 {SCOUT} 결정: 배 보임 {seen.mean()*100:.0f}%. "
    f"분기점 = {best}번째 결정 (이후 {TAIL}결정 동안 배 보임 {best_v*100:.0f}%)")
PREFIX_USED = best

fresh_start()
pre_path = REC/"_prefix.rgb"
SHOW = 45                      # 영상에 넣을 공통 구간 (앞부분은 버린다)
pre_track = []
with open(pre_path, "wb") as f:
    decide_and_step(Null(), max(0, PREFIX_USED - SHOW))
    pre_acts = decide_and_step(f, min(SHOW, PREFIX_USED), track=pre_track)
np.save(REC/"_prefix_ship.npy", np.asarray(pre_track, dtype=np.float32))
say(f"공통 구간 {PREFIX_USED} 결정 중 마지막 {min(SHOW, PREFIX_USED)}개를 영상에 담았다. "
    f"{pre_path.stat().st_size/1e6:.1f} MB")

SNAP = snapshot()
say("분기점 스냅샷 저장 (ALE 시스템상태 + OCAtari 6필드 + RNG + 뇌 16텐서 + 디코더/Vision)")

# ------------------------------------------------------------------ 2. 분기
BRANCHES = ["온전", "온전(재현확인)", "DNp11", "DNp02", "무작위 2세포", "LC4 배선 섞기"]
out = {}
for name in BRANCHES:
    restore(SNAP)
    apply_cond(name)
    p = REC/f"{name.replace(' ', '_')}.rgb"
    tr = []
    with open(p, "wb") as f:
        acts = decide_and_step(f, TAIL, track=tr)
    np.save(REC/f"{name.replace(' ', '_')}_ship.npy", np.asarray(tr, dtype=np.float32))
    up = sum(a == "UP" for a in acts)/max(len(acts), 1)
    out[name] = dict(path=str(p), up=up, bytes=p.stat().st_size,
                     acts="".join({"UP": "^", "LEFT": "<", "RIGHT": ">",
                                   "NOOP": ".", "FIRE": "."}.get(a, "?") for a in acts))
    say(f"  {name:<16} 추진 {up*100:>5.1f}%   {p.stat().st_size/1e6:.1f} MB")
restore(SNAP); apply_cond("온전")

# ------------------------------------------------------------------ 3. 검증
a = (REC/"온전.rgb").read_bytes()
b = (REC/"온전(재현확인).rgb").read_bytes()
same = (a == b)
say(f"\n🔍 같은 시작점 검증: '온전' 두 번 주행이 바이트 단위로 동일 = {same}")
if not same:
    n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), n)
    say(f"   첫 불일치 바이트 {d} (프레임 {d//(W*H*3)})  길이 {len(a)} vs {len(b)}")
    say("   -> 스냅샷이 불완전하다. 이게 통과하기 전엔 분기 비교를 신뢰하면 안 된다.")

for n1, n2 in (("온전", "DNp11"), ("온전", "DNp02"),
               ("온전", "무작위 2세포"), ("온전", "LC4 배선 섞기")):
    x = (REC/f"{n1.replace(' ','_')}.rgb").read_bytes()
    y = (REC/f"{n2.replace(' ','_')}.rgb").read_bytes()
    n = min(len(x), len(y))
    d = next((i for i in range(n) if x[i] != y[i]), None)
    fr = "끝까지 동일" if d is None else f"{d//(W*H*3)}프레임째부터 갈라짐"
    say(f"   {n1} vs {n2:<14} {fr}")

(REC/"branches.json").write_text(json.dumps(
    {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in out.items()},
    indent=2, ensure_ascii=False), encoding="utf-8")
say(f"\n주행 {time.perf_counter()-t0:.0f}s  -> out/rec/*.rgb")
