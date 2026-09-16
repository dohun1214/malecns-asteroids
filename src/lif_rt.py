"""실시간 전체 뇌 LIF — 이벤트 구동 Triton 커널 + CUDA Graph 언롤 캡처.

E 의 src/lif.py (post-major sparse CSR, dt=0.1) 를 개조한 것이다. 수식은 동일하다:

    active = (refr == 0)
    g <- active ? g*B + inc : g + inc          inc = w_syn * sum(signed synapse count)
    v <- active ? v_0 + (v-v_0)*A + g*K(B-A) : v
    v += drive (+ poisson)
    refr <- max(refr-1, 0)
    spk = (v > v_th) & active
    on spk: v=v_rst, g=0, refr=rfc_steps
    스파이크는 t_dly 뒤에 도착

바뀐 것 4가지:
 1. dt 0.1 -> 0.2 ms   (선형계 해석적 적분이라 정확도 손실 0. 지연 9 / 불응 11 스텝 정수)
 2. post-major gather (inc = W @ s, 매 스텝 nnz 전체 = 84 MB) 
    -> pre-major scatter (발화한 뉴런의 출력 엣지만 = 약 0.8 MB)
 3. 가중치를 packed int32 (signed count 상위 14bit | post 하위 18bit) 로. clip 0개
 4. inc 누산을 int32 로. 가중치가 정수라 원자 덧셈 순서와 무관하게 '정확히' 결정론적이다
    (fp32 atomicAdd 였으면 순서마다 반올림이 달라져 분기 비교 데모가 깨진다)

⚠️ 실측으로 잡은 함정: 커널 안에서 같은 주소를 읽고 쓰면 Triton 이 순서를 보장하지
   않는다. 도착 전류 버퍼를 "읽고 0으로 되돌리기"는 atomic_xchg 로 해야 한다.
   에러가 안 나고 매 실행 결과만 달라지므로 재현성 테스트 없이는 못 잡는다.

CUDA Graph 제약 (05문서 2.3):
 - 모든 텐서를 사전 할당하고 절대 재할당하지 않는다. 그래프는 주소를 굽는다
 - 그래프 안에서 .item()/.cpu() 금지, shape 은 전부 정적
 - 스파이크 개수는 런타임 값이므로 그리드를 고정하고 커널 안에서 조기 return 한다
"""
import os, math, json
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent.parent
GDIR = Path(os.environ.get("ASTEROIDS_GRAPH", ROOT / "graph"))

POST_BITS = 18
POST_MASK = (1 << POST_BITS) - 1

PARAMS = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,
    t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8,
    w_syn=0.15,          # 01문서 4장: w>=3 그래프면 0.15 에서 시작
    f_poi=250.0,
    dt=0.2,
)


# --------------------------------------------------------------------- kernels
@triton.jit
def _membrane(V, G, R, INC, ALIVE, RFC, DRIVE, LAM, SP, CNT, OVER,
              n, cap, SEED, step, poi_amp,
              A, B, KBA, v_0, v_th, v_rst, w_syn, HAS_POI: tl.constexpr,
              BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n

    v   = tl.load(V + off, mask=m, other=0.0)
    g   = tl.load(G + off, mask=m, other=0.0)
    r   = tl.load(R + off, mask=m, other=1).to(tl.int32)
    # ⚠️ 같은 주소를 tl.load 로 읽고 tl.store 로 0을 쓰면 Triton 이 두 연산의
    #    순서를 보장하지 않는다. 일부 프로그램이 '이미 0이 된 값'을 읽어
    #    도착 전류가 통째로 사라지고, 실행할 때마다 결과가 달라진다.
    #    (실측: 뉴런 111759 가 inc=15 인데 g 가 0 / 2.25 / 2.25 로 갈렸음)
    #    atomic_xchg 는 읽기와 0쓰기를 한 연산으로 묶어 이 위험을 없앤다.
    inc = tl.atomic_xchg(INC + off, tl.zeros([BLOCK], dtype=tl.int32), mask=m)

    incf   = inc.to(tl.float32) * w_syn
    active = r == 0
    g = tl.where(active, g * B + incf, g + incf)
    v = tl.where(active, v_0 + (v - v_0) * A + g * KBA, v)

    v = v + tl.load(DRIVE + off, mask=m, other=0.0)

    if HAS_POI:
        lam = tl.load(LAM + off, mask=m, other=0.0)
        u = tl.rand(tl.load(SEED) * 1000003 + step, off)
        # 작은 lambda 용 역CDF 4항. lam = 150Hz*0.2ms/1000 = 0.03 이면 P(n>=4) ~ 3e-8
        pk = tl.exp(-lam); c = pk
        k = tl.where(u > c, 1.0, 0.0)
        pk = pk * lam;        c = c + pk;  k = tl.where(u > c, k + 1.0, k)
        pk = pk * lam * 0.5;  c = c + pk;  k = tl.where(u > c, k + 1.0, k)
        pk = pk * lam / 3.0;  c = c + pk;  k = tl.where(u > c, k + 1.0, k)
        k = tl.where(lam > 0.0, k, 0.0)
        v = v + k * poi_amp

    r = tl.maximum(r - 1, 0)
    s = (v > v_th) & active & m
    v = tl.where(s, v_rst, v)
    g = tl.where(s, 0.0, g)
    r = tl.where(s, tl.load(RFC + off, mask=m, other=0).to(tl.int32), r)

    tl.store(V + off, v, mask=m)
    tl.store(G + off, g, mask=m)
    tl.store(R + off, r.to(tl.int8), mask=m)

    alive = tl.load(ALIVE + off, mask=m, other=0)
    e  = s & (alive != 0)                      # lesion: 발화는 하되 출력이 안 나간다
    ei = e.to(tl.int32)
    num = tl.sum(ei, axis=0)
    if num > 0:
        base = tl.atomic_add(CNT, num)
        pos = base + tl.cumsum(ei, axis=0) - ei
        tl.store(SP + pos, off.to(tl.int32), mask=e & (pos < cap))
        if base + num > cap:
            tl.atomic_max(OVER, base + num)


@triton.jit
def _scatter(SP, CNT, CROW, PACKED, INC,
             POST_MASK: tl.constexpr, PB: tl.constexpr, NPROG: tl.constexpr,
             LANES: tl.constexpr, EBLOCK: tl.constexpr):
    """그리드 (NPROG, LANES) 고정. 스파이크 개수는 런타임이라 커널 안에서 판단한다.
    한 뉴런의 출력 엣지 리스트를 LANES 개가 나눠 맡는다 (출차수 최대 7,749 / 중앙값 45,
    172배 불균형이라 이게 없으면 한 프로그램이 커널 전체를 붙잡는다)."""
    cnt  = tl.load(CNT)
    lane = tl.program_id(1)
    pid  = tl.program_id(0)
    while pid < cnt:
        src = tl.load(SP + pid)
        lo = tl.load(CROW + src).to(tl.int64)
        hi = tl.load(CROW + src + 1).to(tl.int64)
        start = lo + lane * EBLOCK
        while start < hi:
            e = start + tl.arange(0, EBLOCK)
            em = e < hi
            p = tl.load(PACKED + e, mask=em, other=0)
            tgt = p & POST_MASK          # 하위 18bit, 마스크라 부호 문제 없음
            w = p >> PB                  # 산술 시프트가 14bit 부호 확장을 대신한다
            tl.atomic_add(INC + tgt, w, mask=em)
            start += LANES * EBLOCK
        pid += NPROG


@triton.jit
def _tally(SP, CNT, TALLY, NPROG: tl.constexpr, BLOCK: tl.constexpr):
    """분석 전용: 뉴런별 누적 스파이크 수. 핫 루프에 부담을 주지 않도록 별도 커널."""
    cnt = tl.load(CNT)
    pid = tl.program_id(0)
    base = pid * BLOCK
    while base < cnt:
        o = base + tl.arange(0, BLOCK)
        m = o < cnt
        i = tl.load(SP + o, mask=m, other=0)
        tl.atomic_add(TALLY + i, 1, mask=m)
        base += NPROG * BLOCK


# ----------------------------------------------------------------------- Brain
class BrainRT:
    def __init__(self, params=None, device="cuda", cap=32768,
                 nprog=512, lanes=16, eblock=64, block=1024, graph_dir=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        p = self.p
        self.dev = torch.device(device)
        gd = Path(graph_dir) if graph_dir else GDIR

        crow = np.load(gd / "out_crow.npy")
        packed = np.load(gd / "out_packed.npy")
        self.N = int(crow.size - 1)
        self.nnz = int(packed.size)
        self.crow = torch.from_numpy(crow.astype(np.int32)).to(self.dev)
        self.packed = torch.from_numpy(packed.astype(np.int32)).to(self.dev)

        dt = p["dt"]
        self.A   = float(math.exp(-dt / p["t_mbr"]))
        self.B   = float(math.exp(-dt / p["tau"]))
        K        = 1.0 / (1.0 - p["t_mbr"] / p["tau"])
        self.KBA = float(K * (self.B - self.A))
        self.D   = int(round(p["t_dly"] / dt))
        # 링 길이 = D. 스텝 t 에서 슬롯 t%D 를 '먼저 읽고(= t-D 의 스파이크) 그 다음 덮어쓴다'
        # 이므로 D+1 이 필요 없다. dt=0.2 에서 D=9 이고 333 = 9*37 이라
        # 한 프레임 언롤이 슬롯 위상을 정확히 한 바퀴 돌려놓는다 (05문서 2.4).
        self.L   = self.D
        self.rfc_steps = int(round(p["t_rfc"] / dt))
        self.poi_amp = float(p["w_syn"] * p["f_poi"])

        N = self.N
        d = self.dev
        # --- 사전 할당. 절대 재할당하지 않는다 (05문서 2.3) -------------------
        self.v     = torch.full((N,), p["v_0"], dtype=torch.float32, device=d)
        self.g     = torch.zeros(N, dtype=torch.float32, device=d)
        self.refr  = torch.zeros(N, dtype=torch.int8,   device=d)
        self.inc   = torch.zeros(N, dtype=torch.int32,  device=d)
        self.alive = torch.ones(N,  dtype=torch.uint8,  device=d)
        self.rfc   = torch.full((N,), self.rfc_steps, dtype=torch.int8, device=d)
        self.drive = torch.zeros(N, dtype=torch.float32, device=d)
        self.lam   = torch.zeros(N, dtype=torch.float32, device=d)
        self.cap   = int(cap)
        self.sp    = torch.zeros(self.L, self.cap, dtype=torch.int32, device=d)
        self.cnt   = torch.zeros(self.L, dtype=torch.int32, device=d)
        self.over  = torch.zeros(1, dtype=torch.int32, device=d)
        self.seed  = torch.zeros(1, dtype=torch.int32, device=d)
        self.tally = torch.zeros(N, dtype=torch.int32, device=d)   # 분석용 누적 발화수
        self.first = torch.zeros(N, dtype=torch.int32, device=d)   # 분석용 첫 발화 스텝
        self._cnt_view = [self.cnt.narrow(0, i, 1) for i in range(self.L)]

        self.nprog, self.lanes, self.eblock, self.block = nprog, lanes, eblock, block
        self.nblk = triton.cdiv(N, block)
        self.has_poi = False
        self.graph = None
        self.graph_steps = 0
        self._t = 0

    # ------------------------------------------------------------------ state
    def reset(self):
        self.v.fill_(self.p["v_0"]); self.g.zero_(); self.refr.zero_()
        self.inc.zero_(); self.sp.zero_(); self.cnt.zero_(); self.over.zero_()
        self.seed.zero_(); self._t = 0

    def set_poisson(self, idx, rate_hz):
        """Shiu 방식: w_syn*f_poi 를 v 에 직접 더하고, 자극 뉴런은 불응기 0."""
        self.lam.zero_(); self.rfc.fill_(self.rfc_steps)
        if idx is not None and len(idx):
            i = torch.as_tensor(np.asarray(idx), device=self.dev).long()
            self.lam[i] = float(rate_hz) * self.p["dt"] / 1000.0
            self.rfc[i] = 0
            self.has_poi = True
        else:
            self.has_poi = False

    def set_drive(self, idx, mv):
        self.drive.zero_()
        if idx is not None and len(idx):
            i = torch.as_tensor(np.asarray(idx), device=self.dev).long()
            self.drive[i] = float(mv)

    def lesion(self, idx, on=True):
        """05문서 2.3: 그래프 밖에서 같은 주소의 내용만 바꾼다. 동기화 없음."""
        i = idx if torch.is_tensor(idx) else torch.as_tensor(np.asarray(idx), device=self.dev)
        self.alive.index_fill_(0, i.long(), 0 if on else 1)

    # ------------------------------------------------------------------- step
    def _step(self, t):
        sw = t % self.L
        sr = (t - self.D) % self.L
        _scatter[(self.nprog, self.lanes)](
            self.sp[sr], self._cnt_view[sr], self.crow, self.packed, self.inc,
            POST_MASK=POST_MASK, PB=POST_BITS,
            NPROG=self.nprog, LANES=self.lanes, EBLOCK=self.eblock,
            num_warps=4)
        self._cnt_view[sw].zero_()
        _membrane[(self.nblk,)](
            self.v, self.g, self.refr, self.inc, self.alive, self.rfc,
            self.drive, self.lam, self.sp[sw], self._cnt_view[sw], self.over,
            self.N, self.cap, self.seed, t, self.poi_amp,
            self.A, self.B, self.KBA, self.p["v_0"], self.p["v_th"], self.p["v_rst"],
            float(self.p["w_syn"]), HAS_POI=self.has_poi, BLOCK=self.block,
            num_warps=4)

    def run_tally(self, steps, reset_tally=True):
        """뉴런별 누적 발화수와 첫 발화 스텝을 기록하며 eager 로 진행 (GPU 동기화 없음)."""
        if reset_tally:
            self.tally.zero_(); self.first.fill_(-1)
        for k in range(steps):
            t = self._t
            self._step(t)
            sw = t % self.L
            _tally[(64,)](self.sp[sw], self._cnt_view[sw], self.tally,
                          NPROG=64, BLOCK=256, num_warps=4)
            fresh = (self.tally > 0) & (self.first < 0)
            self.first = torch.where(fresh, torch.full_like(self.first, t), self.first)
            self._t += 1
        return self.tally

    def rates(self, idx, steps):
        """Hz. steps 는 run_tally 에 준 스텝 수."""
        sec = steps * self.p["dt"] / 1000.0
        i = torch.as_tensor(np.asarray(idx), device=self.dev).long()
        return (self.tally[i].float() / sec).cpu().numpy()

    def run_eager(self, steps, record_pop=False, record=None):
        pop = torch.zeros(steps, dtype=torch.int32, device=self.dev) if record_pop else None
        raster = None
        if record is not None:
            ridx = torch.as_tensor(np.asarray(record), device=self.dev).long()
            raster = torch.zeros(steps, ridx.numel(), dtype=torch.uint8, device=self.dev)
        for k in range(steps):
            t = self._t
            self._step(t)
            if pop is not None:
                pop[k] = self.cnt[t % self.L]
            if raster is not None:
                sw = t % self.L
                c = self.cnt[sw]
                hit = torch.zeros(self.N, dtype=torch.uint8, device=self.dev)
                hit[self.sp[sw, :c].long()] = 1
                raster[k] = hit[ridx]
            self._t += 1
        out = {}
        if pop is not None:
            out["pop"] = pop.cpu().numpy()
        if raster is not None:
            out["raster"] = raster.cpu().numpy()
        out["overflow"] = int(self.over.item())
        return out

    # ------------------------------------------------------- CUDA Graph 캡처
    def capture(self, steps, warmup=3):
        """steps 개를 통째로 언롤해서 캡처한다. 링버퍼 슬롯이 캡처 시점 상수가 되는 게
        맞으려면 steps 가 L 의 배수여야 한다 (05문서 2.4 의 함정)."""
        assert steps % self.L == 0, (
            f"steps({steps}) 가 링버퍼 길이 L({self.L}) 의 배수가 아니다. "
            f"프레임 경계에서 지연 슬롯 위상이 어긋나 조용히 틀린 결과가 나온다.")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                for k in range(steps):
                    self._step(k)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.reset()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            for k in range(steps):
                self._step(k)
        self.graph = gph
        self.graph_steps = steps
        return gph

    def replay(self):
        self.graph.replay()
        self._t += self.graph_steps


def load_params_report():
    return json.loads((GDIR / "prep_report.json").read_text(encoding="utf-8"))
