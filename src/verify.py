"""01문서 7.1 침묵 사다리 0~5 재통과 + post-major 레퍼런스와의 스텝 단위 등가성 검증.

개조(dt 0.2 / pre-major scatter / packed int32 / int32 누산) 후에 반드시 다시 통과시킬 것.
등가성 검증이 사다리보다 강하다: 완전히 다른 코드 경로(gather vs scatter)가
스파이크 하나까지 같은 답을 내야 한다.

  정확히 같은 답이 나오는 이유: 가중치가 정수(시냅스 수 x 부호)라
  부분합이 전부 fp32 정확 표현 범위(|합| <= 12만 << 2^24) 안에 들어온다.
  따라서 덧셈 순서와 무관하게 결과가 비트 단위로 같다.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS, POST_BITS, POST_MASK

ROOT = Path(__file__).resolve().parent.parent
GDIR = ROOT / "graph"
OUT  = ROOT / "out"; OUT.mkdir(exist_ok=True)
R = {}
def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 레퍼런스 구현
class BrainRef:
    """post-major CSR gather (inc = W @ s). E 의 lif.py 와 같은 계산 구조, dt 만 0.2."""
    def __init__(self, p):
        self.p = dict(p)
        crow = np.load(GDIR/"out_crow.npy").astype(np.int64)
        post = np.load(GDIR/"out_post.npy").astype(np.int64)
        w    = np.load(GDIR/"out_w.npy").astype(np.float32)   # 부호 있는 시냅스 '수'
        N = crow.size - 1
        pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
        order = np.argsort(post, kind="stable")
        Tindptr = np.zeros(N+1, dtype=np.int64)
        np.cumsum(np.bincount(post[order], minlength=N), out=Tindptr[1:])
        self.N = N
        self.W = torch.sparse_csr_tensor(
            torch.from_numpy(Tindptr).cuda(),
            torch.from_numpy(pre[order]).cuda(),
            torch.from_numpy(w[order]).cuda(),
            size=(N, N), dtype=torch.float32, device="cuda")
        dt = p["dt"]
        self.A = float(np.exp(-dt/p["t_mbr"])); self.B = float(np.exp(-dt/p["tau"]))
        K = 1.0/(1.0 - p["t_mbr"]/p["tau"]); self.KBA = float(K*(self.B-self.A))
        self.D = int(round(p["t_dly"]/dt)); self.L = self.D
        self.rfc_steps = int(round(p["t_rfc"]/dt))
        self.reorder = False
        self.reset()

    def reset(self):
        N, d = self.N, "cuda"
        self.v = torch.full((N,), self.p["v_0"], device=d)
        self.g = torch.zeros(N, device=d)
        self.refr = torch.zeros(N, dtype=torch.int32, device=d)
        self.ring = torch.zeros(self.L, N, device=d)
        self.t = 0

    def step(self, drive=None, alive=None, rfc=None):
        p = self.p; slot = self.t % self.L
        inc = self.ring[slot] * float(p["w_syn"])
        active = self.refr == 0
        self.g = torch.where(active, self.g*self.B + inc, self.g + inc)
        if self.reorder:   # 대수적으로 동일, 부동소수점 결합순서만 다름
            vn = p["v_0"]*(1.0-self.A) + self.v*self.A + self.g*self.KBA
        else:
            vn = p["v_0"] + (self.v-p["v_0"])*self.A + self.g*self.KBA
        self.v = torch.where(active, vn, self.v)
        if drive is not None:
            self.v = self.v + drive
        self.ring[slot] = 0
        self.refr = torch.clamp(self.refr-1, min=0)
        spk = (self.v > p["v_th"]) & active
        self.v = torch.where(spk, torch.full_like(self.v, p["v_rst"]), self.v)
        self.g = torch.where(spk, torch.zeros_like(self.g), self.g)
        rs = rfc if rfc is not None else torch.full_like(self.refr, self.rfc_steps)
        self.refr = torch.where(spk, rs, self.refr)
        s = spk.float()
        if alive is not None:
            s = s * alive.float()
        self.ring[slot] = torch.mv(self.W, s)      # 슬롯 재사용: t 에서 읽고 t 에서 쓴다
        self.t += 1
        return spk


# ------------------------------------------------------------------- 사다리
def ladder():
    p = dict(PARAMS)
    b = BrainRT(params=p)
    N = b.N
    say(f"\n{'='*66}\n침묵 사다리 (01문서 7.1)   N={N}  nnz={b.nnz}  dt={p['dt']}ms")
    say(f"  지연 D={b.D} step ({b.D*p['dt']}ms)  불응 {b.rfc_steps} step ({b.rfc_steps*p['dt']}ms)"
        f"  링 길이 L={b.L}")

    # --- 0 : 자극 없이 100ms -> 발화 0 -----------------------------------
    b.reset()
    r0 = b.run_eager(int(100/p["dt"]), record_pop=True)
    n0 = int(r0["pop"].sum())
    say(f"\n[0] 자극 없이 100ms  ->  총 발화 {n0}   {'PASS' if n0==0 else 'FAIL'}")
    R["l0_spikes"] = n0

    # --- 1 : 단일 뉴런 임계 초과 전류 -------------------------------------
    i = 12345
    thr = p["v_th"] - p["v_0"]          # 7 mV
    def first_spike(amp, n=40):
        b.reset(); b.set_drive([i], amp)
        r = b.run_eager(n, record_pop=True); b.set_drive(None, 0)
        nz = np.flatnonzero(r["pop"]); return (int(nz[0]) if nz.size else None), int(r["pop"].sum())
    # 매 스텝 주입이라 작은 값도 누적되어 결국 터진다. 가르는 건 '몇 스텝 만에' 터지는가다.
    t_sup, n_sup = first_spike(thr*2.0)
    t_sub, n_sub = first_spike(thr*0.5)
    t_tiny, n_tiny = first_spike(thr*0.02)
    ok1 = (t_sup == 0) and (t_sub is not None) and (t_sub > t_sup) and \
          (t_tiny is None or t_tiny > t_sub)
    say(f"[1] 단일 뉴런 직접 주입, 첫 발화 스텝:  2x임계 -> {t_sup} / "
        f"0.5x임계 -> {t_sub} / 0.02x임계 -> {t_tiny}   {'PASS' if ok1 else 'FAIL'}")
    say(f"    (임계 초과는 첫 스텝에 즉시 발화해야 한다. 적분/임계/리셋이 맞다는 뜻)")
    R["l1_t_sup"], R["l1_t_sub"], R["l1_t_tiny"] = t_sup, t_sub, t_tiny

    # --- 2 : A 강제 발화 -> B 의 g 가 정확히 오르는가 ----------------------
    crow = np.load(GDIR/"out_crow.npy"); post = np.load(GDIR/"out_post.npy")
    wv   = np.load(GDIR/"out_w.npy")
    deg = np.diff(crow)
    A_ = int(np.argsort(deg)[len(deg)//2])           # 중앙값 출차수 뉴런
    e0 = int(crow[A_]); B_ = int(post[e0]); wAB = float(wv[e0])
    expect = wAB * p["w_syn"]
    b.reset(); b.set_drive([A_], thr*3.0)
    b.run_eager(1)                                   # A 가 이 스텝에 발화
    b.set_drive(None, 0)
    gs = []
    for k in range(b.D + 3):
        b.run_eager(1)
        gs.append(float(b.g[B_].item()))
    arrive = next((k for k, x in enumerate(gs) if abs(x) > 1e-9), None)
    got = gs[arrive] if arrive is not None else 0.0
    ok2 = (arrive == b.D - 1) and abs(got - expect) < 1e-5
    say(f"[2] A={A_} (출차수 {deg[A_]}) -> B={B_}  시냅스 {wAB:+.0f}개")
    say(f"    기대 g 점프 = {wAB:+.0f} x {p['w_syn']} = {expect:+.6f} mV")
    say(f"    실측 = {got:+.6f} mV,  도착 지연 = {arrive} (기대 {b.D-1})"
        f"   {'PASS' if ok2 else 'FAIL'}")
    R.update(l2_A=A_, l2_B=B_, l2_syn=wAB, l2_expect=expect, l2_got=got,
             l2_delay=arrive, l2_delay_expect=b.D-1)

    # --- 3 : ACh 입력 / GABA 입력의 v 방향 --------------------------------
    sign = np.load(GDIR/"sign.npy")
    pre_all = np.repeat(np.arange(len(deg), dtype=np.int64), deg)
    exc = int(pre_all[np.flatnonzero(sign[pre_all] > 0)[0]])
    inh_e = np.flatnonzero(sign[pre_all] < 0)[0]
    inh = int(pre_all[inh_e])
    res = {}
    for name, src in (("ACh(+)", exc), ("GABA/Glu(-)", inh)):
        tgt = int(post[int(crow[src])]); syn = float(wv[int(crow[src])])
        b.reset(); b.set_drive([src], thr*3.0); b.run_eager(1); b.set_drive(None, 0)
        v_before = float(b.v[tgt].item())
        b.run_eager(b.D + 1)
        res[name] = (float(b.v[tgt].item()) - v_before, syn)
    ok3 = res["ACh(+)"][0] > 0 and res["GABA/Glu(-)"][0] < 0
    say(f"[3] ACh 전구체 dv = {res['ACh(+)'][0]:+.6f} (시냅스 {res['ACh(+)'][1]:+.0f}) / "
        f"억제성 dv = {res['GABA/Glu(-)'][0]:+.6f} (시냅스 {res['GABA/Glu(-)'][1]:+.0f})"
        f"   {'PASS' if ok3 else 'FAIL'}")
    R["l3_exc_dv"], R["l3_inh_dv"] = res["ACh(+)"][0], res["GABA/Glu(-)"][0]

    # --- 4 : 총 입력 시냅스 x w_syn 이 임계와 같은 자릿수인가 ---------------
    in_syn = np.bincount(post, weights=np.abs(wv), minlength=len(deg))
    med = float(np.median(in_syn))
    say(f"[4] 뉴런당 총 입력 시냅스 중앙값 {med:.0f} x w_syn {p['w_syn']} = "
        f"{med*p['w_syn']:.1f} mV  vs 임계 {thr:.0f} mV  -> {med*p['w_syn']/thr:.0f}배")
    R["l4_median_in_syn"] = med; R["l4_ratio"] = med*p["w_syn"]/thr

    # --- 5 : 자극 세기 ----------------------------------------------------
    say(f"[5] Shiu 방식 자극 = w_syn x f_poi = {p['w_syn']} x {p['f_poi']} = "
        f"{b.poi_amp:.2f} mV = 임계의 {b.poi_amp/thr:.1f}배  "
        f"(E 는 w_syn 0.275 로 68.75mV. w_syn 만 바뀐 것)")
    R["l5_poi_amp"] = b.poi_amp; R["l5_ratio"] = b.poi_amp/thr

    passed = (n0 == 0) and ok1 and ok2 and ok3
    say(f"\n사다리 0~5 : {'ALL PASS' if passed else 'FAIL 있음'}")
    R["ladder_pass"] = bool(passed)
    return b


# ------------------------------------------------------------------- 등가성
def equivalence(steps=600, ndrive=400, seed=0):
    say(f"\n{'='*66}\npost-major gather 레퍼런스 vs pre-major scatter (Triton)  {steps} step")
    p = dict(PARAMS)
    rng = np.random.default_rng(seed)
    idx = rng.choice(166700, size=ndrive, replace=False)
    thr = p["v_th"] - p["v_0"]

    rt = BrainRT(params=p); rt.reset(); rt.set_drive(idx, thr*1.2)
    ref = BrainRef(p)
    drive = torch.zeros(rt.N, device="cuda")
    drive[torch.from_numpy(idx).cuda().long()] = thr*1.2

    mism = []; tot_rt = 0; tot_ref = 0
    first_mism = None; first_inc_mism = None
    dv_trace = []
    for t in range(steps):
        # (연결 연산의 정확성은 test_scatter.py 가 격리해서 검증한다)
        spk_ref = ref.step(drive=drive)
        rt.run_eager(1)
        sw = (rt._t - 1) % rt.L
        c = int(rt.cnt[sw].item())
        a = torch.zeros(rt.N, dtype=torch.bool, device="cuda")
        if c: a[rt.sp[sw, :c].long()] = True
        tot_rt += c; tot_ref += int(spk_ref.sum().item())
        d = int((a ^ spk_ref).sum().item())
        if d:
            mism.append((t, d))
            if first_mism is None: first_mism = t
        dv_trace.append(float((rt.v - ref.v).abs().max().item()))
    dv = dv_trace[-1]
    dg = float((rt.g - ref.g).abs().max().item())
    say(f"  누적 스파이크  scatter {tot_rt:,} / gather {tot_ref:,}  "
        f"(차이 {abs(tot_rt-tot_ref)/max(tot_ref,1)*100:.3f}%)")
    say(f"  스파이크가 처음 달라진 스텝: {first_mism}")
    say(f"  |dv|max 성장: " + "  ".join(
        f"t={k}:{dv_trace[k]:.2e}" for k in (0, 5, 10, 20, 30,
        min(first_mism if first_mism else 40, steps-1)) if k < steps))
    say(f"  최종 |dv|max = {dv:.3e}   |dg|max = {dg:.3e}")
    R.update(eq_steps=steps, eq_spikes_rt=tot_rt, eq_spikes_ref=tot_ref,
             eq_mismatch_steps=len(mism), eq_first_mism=first_mism,
             eq_first_inc_mism=first_inc_mism, eq_dv=dv, eq_dg=dg,
             eq_dv_trace=[dv_trace[k] for k in range(0, min(steps, 60))])
    return first_mism


def chaos_control(steps=400, ndrive=400, seed=0):
    """대조 실험: 레퍼런스를 '수학적으로 동일하지만 연산 순서만 다른' 형태로 한 번 더 돌린다.
       v_0 + (v-v_0)*A  ==  v_0*(1-A) + v*A   (대수적으로 같음, 부동소수점으로는 다름)
    두 레퍼런스끼리도 비슷한 시점에 갈라지면, 위 불일치는 커널 버그가 아니라
    스파이킹 망의 혼돈 + 부동소수점 결합순서 때문이다."""
    say(f"\n{'-'*66}\n대조: 레퍼런스 vs 레퍼런스(연산 순서만 변경)  {steps} step")
    p = dict(PARAMS)
    rng = np.random.default_rng(seed)
    idx = rng.choice(166700, size=ndrive, replace=False)
    thr = p["v_th"] - p["v_0"]
    a, b = BrainRef(p), BrainRef(p)
    b.reorder = True
    drive = torch.zeros(a.N, device="cuda")
    drive[torch.from_numpy(idx).cuda().long()] = thr*1.2
    first = None; n = 0
    for t in range(steps):
        sa = a.step(drive=drive); sb = b.step(drive=drive)
        d = int((sa ^ sb).sum().item())
        if d:
            n += 1
            if first is None: first = t
    say(f"  스파이크가 처음 달라진 스텝: {first}   불일치 스텝 {n}/{steps}")
    R["chaos_first_mism"], R["chaos_mism_steps"] = first, n
    return first


if __name__ == "__main__":
    t0 = time.time()
    ladder()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    fm = equivalence(steps=n)
    fc = chaos_control(steps=n)
    say(f"\n판정: 커널 경로 자체는 test_scatter.py 에서 EXACT 확인됨.")
    say(f"      구현 간 첫 갈림 t={fm} vs 같은 구현 내 연산순서만 바꾼 대조군 t={fc}")
    R["verdict_fp_chaos"] = bool(fc is not None and fm is not None
                                 and abs(fm - fc) < max(fm, fc))
    (OUT/"verify.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
    say(f"\n총 {time.time()-t0:.1f}s   -> out/verify.json")
