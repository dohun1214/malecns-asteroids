"""실시간 대시보드 서버 — 게임 + 뇌를 60Hz 로 돌리면서 브라우저로 쏜다.

구조 (05문서 §1)
  [시뮬 스레드]  60Hz 페이싱. 매 프레임 env.step, 4프레임마다 뇌 결정 1회.
  [메일박스]     단일 슬롯. 이전 프레임이 안 나갔으면 덮어쓴다 (큐를 쌓으면 지연 누적).
  [asyncio]      websockets 로 브로드캐스트. compression=None.

메시지 3종
  FLY1 (binary)  헤더 12B + uint16 델타 인코딩된 '점 인덱스' — 3D 뇌 발화
  FLY2 (binary)  헤더 12B + raw RGB 210x160x3 — 게임 화면
  text  (JSON)   수치·채널·운석 박스·병변 상태·성능

⚠️ 스파이크는 '전역 뉴런 번호'가 아니라 **점군의 몇 번째 점**으로 보낸다 (prep_soma.py).
⚠️ 토글은 결정 경계에서만 적용한다. CUDA Graph 는 포인터를 굽기 때문에
   alive 텐서 내용만 바꾸면 되지만, 한 결정 중간에 바뀌면 그 결정이 반쪽이 된다.
"""
import sys, os, re, json, time, struct, threading, asyncio, collections, socket
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import (BrainPolicy, make_env, Vision, ACT_EVERY, with_fire,
                  frame_action, STEPS_PER_DECISION)
from vision import ship_heading_deg
import rewire as RW

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT/"web"
HZ = 60.0
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
VIDEO_EVERY = 1                     # 결정당 게임 화면 전송 횟수 (1 = 매 결정)
MAGIC_SPK, MAGIC_VID = 0x31594C46, 0x32594C46



from static_server import RangeStatic, Dual   # web/ 를 내보내는 작은 HTTP 서버

class Mailbox:
    """단일 슬롯. 덮어쓰기. 큐를 쌓지 않는다."""
    def __init__(self):
        self._v = None
        self._lk = threading.Lock()
        self._ev = threading.Event()

    def put(self, v):
        with self._lk:
            self._v = v
        self._ev.set()

    def take(self):
        with self._lk:
            v, self._v = self._v, None
        self._ev.clear()
        return v


class Sim(threading.Thread):
    GROUPS = ("DNp11", "DNp02", "DNp01", "LC4half", "rewire")

    def __init__(self, box):
        super().__init__(daemon=True)
        self.box = box
        self.cmds = collections.deque()
        self.stop_flag = threading.Event()
        self.lesion = set()
        S = np.load(ROOT/"graph"/"soma.npz")
        self.pt_of = S["pt_of"]
        self.n_pts = int((self.pt_of >= 0).sum())

    def setup(self):
        self.bp = BrainPolicy()
        self.C = self.bp.C
        # 래스터에 쓸 '실제' 그룹별 스파이크 집계용 인덱스.
        # [버그 이력] 예전엔 LC4 밴드에 운석 각크기를, LPLC2 밴드에 판독 강도를 그렸다.
        #   라벨과 내용이 달라서 "LC4 를 끄면 밴드가 사라진다"가 성립하지 않았다.
        #   집계는 tally 에 이미 다 있다. 그대로 센다.
        R_ = np.load(ROOT/"graph"/"vnc_readout.npz", allow_pickle=True)
        # 밴드 선택: 우리가 자극하는 건 LC4 뿐이라 **LPLC2 는 항상 0 이다.**
        # 늘 비어 있는 밴드를 라벨만 붙여 두면 화면이 고장난 것처럼 보인다.
        # 대신 이 데모의 이야기 그대로 LC4 -> DNp02 / DNp11 -> VNC 판독을 쌓는다.
        self.grp_idx = dict(lc4=np.asarray(self.C["LC4"]),
                            dnp02=np.asarray(self.C["DNp02"]),
                            dnp11=np.asarray(self.C["DNp11"]),
                            vnc=np.concatenate([R_["dn02_only"], R_["dn11_only"]]))
        self.sec = STEPS_PER_DECISION*0.2/1000.0
        self._dead_v = 0
        self._dead_pts = []
        self.env = make_env()
        self.A = self.env.unwrapped.get_action_meanings()
        self.V = Vision()
        self.packed0 = self.bp.b.packed.clone()
        crow, packed0, C, pre = RW.load()
        sel = RW.conditions(C, crow.size-1, pre, packed0)
        newp, _ = RW.rewire(packed0, sel["lc4_all"], seed=0, frac=1.0)
        self.packed_rw = torch.from_numpy(newp.astype(np.int32)).to(self.bp.b.dev)
        self.rng = np.random.default_rng(1234)

    def grab(self):
        """RGB 화면 1장. OCAtari 의 obs 는 mode 에 따라 RAM/축소화면일 수 있어서 ALE 에서 직접 뽑는다."""
        try:
            return np.asarray(self.env._env.env.env.ale.getScreenRGB(), dtype=np.uint8)
        except Exception:
            return None

    def screen(self):
        """[실측] 아타리 2600 은 한 주사선에 여러 스프라이트를 못 그려서 **프레임마다 번갈아
        그린다.** 켜진 픽셀 수가 1416 / 194 로 매 프레임 진동하고, 운석 스프라이트는
        **짝수 프레임에만** 그려진다 (probe_flicker.py: 위상 0·2 는 68% 칠해짐, 1·3 은 0%).

        대시보드는 4프레임에 한 장만 보내므로 위상이 고정된다. 하필 안 그리는 위상에 잠기면
        **운석이 영원히 안 보이고 우리가 덧그린 박스만 남는다.** 실제로 그렇게 보였다.
        (운석 dx/dy 가 절반의 프레임에서 0이던 것, 배가 20% 안 보이던 것과 같은 계열의 함정)

        -> 연속 2프레임의 **최댓값 합성**. 아타리 전처리의 표준(max over last two frames)이고,
           CRT + 잔상이 하는 일과 같아서 사람이 실제로 보는 화면에 더 가깝다.
           실측: 합성 전 60프레임 중 30장이 운석 0% -> 합성 후 0장."""
        a, b = self._scr2
        if a is None: return b
        if b is None: return a
        return np.maximum(a, b)

    def cells(self, g):
        if g == "LC4half": return self.C["LC4"][::2]
        return self.C.get(g)

    def apply(self, g, on):
        if g == "rewire":
            self.bp.b.packed.copy_(self.packed_rw if on else self.packed0)
        else:
            idx = self.cells(g)
            if idx is None or not len(idx): return
            self.bp.b.lesion(torch.as_tensor(np.asarray(idx), device="cuda"), on=on)
        (self.lesion.add if on else self.lesion.discard)(g)

    def restore(self):
        self.bp.b.alive.fill_(1)
        self.bp.b.packed.copy_(self.packed0)
        self.lesion.clear()

    def refresh_dead(self):
        """꺼진 뉴런의 **점 인덱스**를 다시 계산한다.
        [버그 이력] 클라이언트가 '그룹 단위'로 죽은 색을 칠하던 걸 고친다.
          DNp11 2개를 껐는데 하행뉴런 18개가 전부 죽은 색이 됐다.
          이 데모의 주장이 '2개만 껐다'인데 화면이 18개라고 말하면 안 된다.
        alive 텐서가 유일한 진실이므로 거기서 직접 뽑는다 (그룹 조합·복구 전부 자동으로 맞는다)."""
        al = self.bp.b.alive.cpu().numpy()
        pts = self.pt_of[np.flatnonzero(al == 0)]
        self._dead_pts = sorted(int(x) for x in pts[pts >= 0])
        self._dead_v += 1

    def drain(self):
        n = 0
        while self.cmds:
            c = self.cmds.popleft()
            k = c.get("cmd")
            if k == "lesion": self.apply(c["group"], bool(c["on"])); n += 1
            elif k == "restore": self.restore(); n += 1
            elif k == "newgame": self._new = True
        if n: self.refresh_dead()

    def run(self):
        self.setup()
        A, env, V, bp = self.A, self.env, self.V, self.bp
        period = 1.0/HZ
        self._new = True
        self._objs = None; self._objs_age = 0
        self._scr2 = [None, None]
        action = (A.index("NOOP"), A.index("FIRE"))
        step = 0; frame = 0; score = 0.0
        t_next = time.perf_counter()
        ms_brain = ms_frame = 0.0
        fps_t = time.perf_counter(); fps_n = 0; fps = 0.0
        while not self.stop_flag.is_set():
            if self._new:
                self._new = False
                env.reset(seed=int(self.rng.integers(0, 2**31)))
                for _ in range(int(self.rng.integers(1, 31))): env.step(A.index("NOOP"))
                V.reset(); bp.dec.reset(); bp.b.reset(); bp.frame = 0
                self._objs = None; self._objs_age = 0
                self._scr2 = [None, None]
                score = 0.0; action = (A.index("NOOP"), A.index("FIRE"))
            t0 = time.perf_counter()
            obs, rew, tr, te, info = env.step(frame_action(action[0], action[1], frame))
            score += float(rew); frame += 1
            self._scr2 = [self._scr2[1], self.grab()]      # 깜빡임 합성용 2프레임 링
            # [실측/09문서] Player 객체가 **프레임의 절반에서 없다.** 4프레임마다 결정하는
            # 고정 위상 루프는 그 '없는 위상'에 잠겨서 영원히 뇌를 안 돌릴 수 있다.
            # (헤드리스에서는 타이밍이 흔들려 안 걸렸는데 60Hz 고정 페이싱에서 드러났다.
            #  운석 dx/dy 가 절반의 프레임에서 0이던 것과 같은 종류의 함정이다.)
            # -> 매 프레임 '배가 보이는 마지막 객체 목록'을 캐시해두고 결정 때 그걸 쓴다.
            if any(o and type(o).__name__ == "Player" for o in env.objects):
                self._objs = list(env.objects); self._objs_age = 0
            elif self._objs is not None:
                self._objs_age += 1
                if self._objs_age > 3: self._objs = None
            if tr or te:
                self._new = True
            elif frame % ACT_EVERY == 0:
                self.drain()
                objs = self._objs if self._objs is not None else env.objects
                xy, head, looms = V.looming(objs)
                ori = 0
                for o in objs:
                    if o and type(o).__name__ == "Player":
                        ori = int(getattr(o, "orientation", 0)); break
                # [실측] 배가 없는 구간이 **평균 265프레임(최대 459 = 7.6초)** 짜리 덩어리로
                # 전체의 20% 나온다 (probe_ship.py. 위상 문제가 아니라 진짜 긴 공백이다).
                # 그동안 뇌를 안 돌리면 3D 뇌 화면이 통째로 얼어붙어서 고장난 것처럼 보인다.
                # -> 자극만 비우고 뇌는 **항상** 돌린다. 액션만 무시한다.
                tb = time.perf_counter()
                a, ch = bp(looms, ori, A, vel=V.ship_v)
                ms_brain = (time.perf_counter()-tb)*1000.0
                action = ((A.index("NOOP"), A.index("FIRE")) if xy is None
                          else (a, with_fire(a, A)))
                step += 1
                self.emit(step, self.screen(), ch, looms if xy else [], xy, ori,
                          score, info, ms_brain, ms_frame, fps, action)
            ms_frame = (time.perf_counter()-t0)*1000.0
            fps_n += 1
            if time.perf_counter()-fps_t >= 0.5:
                fps = fps_n/(time.perf_counter()-fps_t); fps_t = time.perf_counter(); fps_n = 0
            # 페이싱: 데드라인 누산 + 스핀 (Windows sleep 해상도 15.6ms)
            t_next += period
            d = t_next - time.perf_counter()
            if d > 0.002: time.sleep(d - 0.001)
            while time.perf_counter() < t_next: pass
            if time.perf_counter() - t_next > 0.25: t_next = time.perf_counter()

    def emit(self, step, obs, ch, looms, xy, ori, score, info, ms_brain, ms_frame, fps, action):
        tal = self.bp._tal if hasattr(self.bp, "_tal") else None
        if tal is None: return
        fired = np.flatnonzero(tal > 0)
        pts = self.pt_of[fired]
        pts = np.sort(pts[pts >= 0]).astype(np.int64)
        d = np.diff(pts, prepend=np.int64(0))
        d = np.clip(d, 0, 65535).astype(np.uint16)
        spk = struct.pack("<III", MAGIC_SPK, step, len(d)) + d.tobytes()
        vid = None
        if step % VIDEO_EVERY == 0 and obs is not None:
            a = np.ascontiguousarray(obs, dtype=np.uint8)
            h, w = a.shape[0], a.shape[1]
            vid = struct.pack("<III", MAGIC_VID, step, (w << 16) | h) + a.tobytes()
        rates = {k: round(float(tal[v].sum())/len(v)/self.sec, 1)
                 for k, v in self.grp_idx.items()}          # 그룹 평균 발화율 (Hz)
        st = dict(step=step, score=score, ship_age=int(self._objs_age),
                  rates=rates, dead_v=self._dead_v, lives=info.get("lives") if isinstance(info, dict) else None,
                  fps=round(fps, 1), ms_brain=round(ms_brain, 2), ms_frame=round(ms_frame, 2),
                  spikes=int(len(pts)), n_fired=int(fired.size),
                  action=self.A[action[0] if isinstance(action, tuple) else action], ori=ori, heading=round(ship_heading_deg(ori), 1),
                  lesion=sorted(self.lesion),
                  ship=[round(float(xy[0]), 1), round(float(xy[1]), 1)] if xy else None,
                  looms=[dict(x=round(float(L["x"]), 1), y=round(float(L["y"]), 1),
                              w=int(L["w"]), h=int(L["h"]),
                              th=round(float(L["theta"]), 2), dth=round(float(L["dtheta"]), 3),
                              phi=round(float(L["phi_rel"]), 1)) for L in looms[:16]])
        if ch is not None:
            st["ch"] = {k: (round(float(ch[k]), 4) if not isinstance(ch[k], tuple) else
                            [round(float(x), 4) for x in ch[k]])
                        for k in ("fore", "lateral", "intensity", "norm", "unit",
                                  "p02_L", "p02_R", "p11_L", "p11_R", "p04",
                                  "a02", "a11", "l", "r")}
        if self._dead_v != getattr(self, "_dead_sent", -1):
            st["dead"] = self._dead_pts          # 바뀔 때만 보낸다
            self._dead_sent = self._dead_v
        self.box.put((spk, vid, json.dumps(st, ensure_ascii=False)))


async def main():
    import websockets
    from websockets.asyncio.server import serve
    box = Mailbox()
    sim = Sim(box); sim.start()
    clients = set()

    async def handler(ws):
        clients.add(ws)
        sim._dead_sent = -1          # 새 클라이언트에 죽은 점 목록을 다시 보낸다
        try:
            meta = json.loads((WEB/"soma_meta.json").read_text(encoding="utf-8"))
            await ws.send(json.dumps(dict(type="hello", **meta,
                                          groups=list(Sim.GROUPS))))
            async for m in ws:
                try: c = json.loads(m)
                except Exception: continue
                sim.cmds.append(c)
        except Exception:
            pass
        finally:
            clients.discard(ws)

    async def send_one(ws, spk, vid, txt):
        """⚠️ 타임아웃이 없으면 죽은 소켓 하나가 send 에서 영원히 멈춰서 **전체 송출이 얼어붙는다.**
        (브라우저가 페이지를 떠나면 반쯤 열린 소켓이 남는다. 실제로 화면이 통째로 정지했다.)
        느린 클라이언트는 프레임을 쌓지 말고 끊는다."""
        try:
            await asyncio.wait_for(ws.send(spk), 1.0)
            if vid is not None: await asyncio.wait_for(ws.send(vid), 1.0)
            await asyncio.wait_for(ws.send(txt), 1.0)
            return None
        except Exception:
            try: await ws.close()
            except Exception: pass
            return ws

    async def pump():
        while True:
            v = box.take()
            if v is None:
                await asyncio.sleep(0.004); continue
            spk, vid, txt = v
            if clients:
                res = await asyncio.gather(
                    *[send_one(ws, spk, vid, txt) for ws in list(clients)],
                    return_exceptions=True)
                for r in res:
                    if isinstance(r, Exception) or r is None: continue
                    clients.discard(r)
            await asyncio.sleep(0)

    # 정적 파일 서버 — 별도 스레드 (듀얼스택·Range 는 static_server.py 참고)
    import functools
    httpd = Dual(("::", PORT+1), functools.partial(RangeStatic, directory=str(WEB)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"화면:  http://localhost:{PORT+1}/index.html", flush=True)
    print(f"소켓:  ws://localhost:{PORT}", flush=True)
    async with serve(handler, None, PORT, compression=None, max_size=None):
        await pump()


if __name__ == "__main__":
    asyncio.run(main())
