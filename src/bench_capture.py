"""05문서 5장 2~3단계.

2단계: 333 step 언롤 CUDA Graph 캡처, replay 시간 측정 (목표 < 50 ms/frame)
3단계: alive.index_fill_ 토글이 replay 출력을 실제로 바꾸는가  <- 데모의 핵심 기능
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"out"; OUT.mkdir(exist_ok=True)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 333
R = {"steps": STEPS}
def say(*a): print(*a, flush=True)

crow = np.load(ROOT/"graph"/"out_crow.npy")
deg = np.diff(crow)
rng = np.random.default_rng(1)
STIM = rng.choice(166700, size=300, replace=False)      # 임시 자극 (실제 LC4 는 5단계)

def timed_replay(b, n=20):
    for _ in range(3): b.graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n): b.graph.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n


# ------------------------------------------------------------------ 2단계
say(f"{'='*70}\n2단계: {STEPS} step 언롤 CUDA Graph 캡처  (dt={PARAMS['dt']}ms -> "
    f"프레임 {STEPS*PARAMS['dt']:.1f}ms 시뮬)")
say(f"  게임 프레임 예산 66.7ms / 목표 50ms\n")

best = None
say(f"{'nprog':>6} {'lanes':>6} {'eblock':>7} {'캡처ms':>8} {'replay ms/frame':>16} "
    f"{'ms/step':>9} {'실시간배':>9}")
for nprog, lanes, eblock in [(512,16,64), (1024,8,64), (2048,4,128),
                             (512,32,32), (256,32,64), (1024,16,32)]:
    b = BrainRT(params=PARAMS, nprog=nprog, lanes=lanes, eblock=eblock)
    b.reset(); b.set_poisson(STIM, 150.0)
    t0 = time.perf_counter()
    try:
        b.capture(STEPS)
    except Exception as ex:
        say(f"{nprog:>6} {lanes:>6} {eblock:>7}   캡처 실패: {type(ex).__name__}: {ex}")
        del b; torch.cuda.empty_cache(); continue
    cap_ms = (time.perf_counter()-t0)*1000
    b.reset(); b.set_poisson(STIM, 150.0)
    ms = timed_replay(b)
    rt_ratio = (STEPS*PARAMS["dt"])/ms
    say(f"{nprog:>6} {lanes:>6} {eblock:>7} {cap_ms:8.0f} {ms:16.2f} {ms/STEPS:9.4f} "
        f"{rt_ratio:8.2f}x")
    if best is None or ms < best[0]:
        best = (ms, nprog, lanes, eblock)
    R[f"cfg_{nprog}_{lanes}_{eblock}"] = dict(capture_ms=cap_ms, replay_ms=ms,
                                              per_step_ms=ms/STEPS, realtime=rt_ratio)
    del b; torch.cuda.empty_cache()

ms, nprog, lanes, eblock = best
say(f"\n  최적: nprog={nprog} lanes={lanes} eblock={eblock}  ->  {ms:.2f} ms/frame")
say(f"  프레임 예산 66.7ms 대비 {ms/66.7:.2f}x   목표 50ms 대비 {ms/50:.2f}x   "
    f"{'통과' if ms < 50 else ('예산은 통과' if ms < 66.7 else '초과')}")
R["best"] = dict(replay_ms=ms, nprog=nprog, lanes=lanes, eblock=eblock,
                 vs_budget=ms/66.7, vs_target=ms/50)

b = BrainRT(params=PARAMS, nprog=nprog, lanes=lanes, eblock=eblock)
b.reset(); b.set_poisson(STIM, 150.0)
b.capture(STEPS)
say(f"  VRAM: 할당 {torch.cuda.memory_allocated()/2**20:.0f} MB / "
    f"예약 {torch.cuda.memory_reserved()/2**20:.0f} MB (8GB 중)")
R["vram_mb"] = torch.cuda.memory_reserved()/2**20

# replay 가 eager 와 같은가
b.reset(); b.set_poisson(STIM, 150.0); b.graph.replay(); torch.cuda.synchronize()
v_g, g_g = b.v.clone(), b.g.clone()
b.reset(); b.set_poisson(STIM, 150.0); b.run_eager(STEPS)
dv = float((b.v-v_g).abs().max().item()); dg = float((b.g-g_g).abs().max().item())
say(f"  graph replay vs eager 루프:  |dv|max={dv:.3e}  |dg|max={dg:.3e}  "
    f"{'일치' if dv==0 and dg==0 else '불일치'}")
R["replay_vs_eager_dv"], R["replay_vs_eager_dg"] = dv, dg

# 프레임 반복 시 링버퍼 위상이 맞는가 (STEPS % L == 0 이어야 함).
# Poisson 으로는 못 잰다: 그래프는 매 프레임 같은 시드(baked step)를 재생하는데
# eager 는 step 이 계속 증가한다. 결정론적 주입으로 확인한다.
thr = PARAMS["v_th"] - PARAMS["v_0"]
b.reset(); b.set_poisson(None, 0); b.set_drive(STIM, thr*1.2)
for _ in range(4): b.graph.replay()
torch.cuda.synchronize()
v4 = b.v.clone()
b.reset(); b.set_drive(STIM, thr*1.2); b.run_eager(STEPS*4)
d4 = float((b.v-v4).abs().max().item())
say(f"  4 프레임 연속 replay vs eager {STEPS*4} step (결정론적 주입): |dv|max={d4:.3e}  "
    f"{'링버퍼 위상 OK' if d4==0 else '위상 어긋남'}")
R["four_frame_dv"] = d4
b.set_drive(None, 0); b.set_poisson(STIM, 150.0)


# ------------------------------------------------------------------ 3단계
say(f"\n{'='*70}\n3단계: alive.index_fill_ 토글이 replay 출력을 바꾸는가")

def frames(b, n, lesion_idx=None, seedbase=0):
    b.reset(); b.set_poisson(STIM, 150.0)
    b.alive.fill_(1)
    if lesion_idx is not None and len(lesion_idx):
        b.lesion(torch.as_tensor(np.asarray(lesion_idx), device="cuda"))
    tot = []
    for f in range(n):
        b.seed.fill_(seedbase + f)
        b.graph.replay()
        torch.cuda.synchronize()
        tot.append(int(b.cnt.sum().item()))
    return np.array(tot), b.v.clone(), b.g.clone()

NF = 6
base_pop, base_v, _ = frames(b, NF)
say(f"  기준(온전한 뇌) 프레임별 마지막-9스텝 스파이크 합: {base_pop.tolist()}")
R["baseline_pop"] = base_pop.tolist()

# 재현성: 같은 조건 두 번
rep_pop, rep_v, _ = frames(b, NF)
same = bool((base_pop == rep_pop).all() and float((base_v-rep_v).abs().max().item()) == 0.0)
say(f"  같은 조건 재실행 -> {'비트 단위 동일' if same else '다름(문제)'}")
R["repeatable"] = same

hubs = np.argsort(deg)[::-1][:500].copy()
silent = np.flatnonzero(deg == 0)[:500]
say(f"  (음성 대조용 출력엣지 0개 뉴런: 전체 {int((deg==0).sum()):,}개 중 500개 사용)")
cases = [("허브 500개 (출차수 상위)", hubs),
         ("무작위 500개", rng.choice(166700, size=500, replace=False)),
         ("음성 대조: 출력 엣지 0개인 500개", silent)]
say(f"\n  {'조건':<32} {'스파이크 변화':>14} {'|dv|max':>12} {'판정':>10}")
for name, idx in cases:
    pop, v, _ = frames(b, NF, lesion_idx=idx)
    dpop = (pop.sum() - base_pop.sum()) / max(base_pop.sum(), 1) * 100
    dvm = float((v-base_v).abs().max().item())
    verdict = "바뀜" if dvm > 0 else "동역학 변화 0"
    say(f"  {name:<32} {dpop:>13.2f}% {dvm:>12.4f} {verdict:>10}")
    R[f"lesion_{name}"] = dict(dpop_pct=dpop, dv=dvm, n=len(idx))

# 토글 ON -> OFF 복구
pop_off, v_off, _ = frames(b, NF, lesion_idx=hubs)
b.alive.fill_(1)
pop_on, v_on, _ = frames(b, NF)
back = bool((pop_on == base_pop).all() and float((v_on-base_v).abs().max().item()) == 0.0)
say(f"\n  토글 OFF->ON 복구: {'기준과 비트 단위 동일' if back else '복구 실패'}")
R["toggle_restore"] = back

# 토글 비용
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    b.alive.index_fill_(0, torch.as_tensor(hubs, device="cuda").long(), 0)
torch.cuda.synchronize()
say(f"  alive.index_fill_ 1회 비용: {(time.perf_counter()-t0)*1e6/1000:.1f} us "
    f"(프레임 예산의 {(time.perf_counter()-t0)*1000/1000/66.7*100:.3f}%)")

(OUT/"bench_capture.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
say(f"\n-> out/bench_capture.json")
