"""60Hz 페이싱 루프 — 게임은 뇌를 기다리지 않는다 (05문서 3.3).

세 개의 시계가 다르다: 게임 60Hz / 뇌 결정 15Hz / 웹 송출 15~30Hz.
하나로 맞추려 하면 전부 무너진다. 뇌가 늦으면 게임은 이전 액션을 유지하고 계속 간다.

규칙 3개:
  1. 데드라인 누산기 (next_t += PERIOD). sleep(1/60) 반복은 드리프트가 쌓인다
  2. 밀렸으면 빚을 탕감 (next_t = max(next_t, now)). 안 그러면 따라잡으려고 폭주한다
  3. 늦은 뇌를 join() 하지 않는다

⚠️ Windows 함정: time.sleep() 해상도가 기본 15.6ms 다. 60Hz(16.7ms)엔 못 쓴다.
   데드라인 1.5ms 전까지만 자고 나머지는 스핀한다.
"""
import sys, time, threading, json
from pathlib import Path
import numpy as np

PERIOD = 1.0/60.0
ACT_EVERY = 4                      # 4프레임 = 66.7ms = 회전 22.5도 한 칸


def sleep_until(t_target, spin=0.0015):
    """Windows 에서 60Hz 를 맞추려면 자다가 마지막은 스핀해야 한다."""
    while True:
        dt = t_target - time.perf_counter()
        if dt <= 0: return
        if dt > spin: time.sleep(dt - spin)
        else:
            while time.perf_counter() < t_target: pass
            return


class Mailbox:
    """단일 슬롯. 이전 것이 아직 안 나갔으면 덮어쓴다 (05문서 4.7).
    큐를 쌓으면 지연이 누적된다."""
    def __init__(self):
        self._v = None; self._lk = threading.Lock(); self._ev = threading.Event()
    def put(self, v):
        with self._lk:
            self._v = v; self._ev.set()
    def take(self, timeout=None):
        if not self._ev.wait(timeout): return None
        with self._lk:
            v, self._v = self._v, None; self._ev.clear()
            return v
    def peek_take(self):
        if not self._ev.is_set(): return None
        return self.take(0)


class PacingLoop:
    def __init__(self, env, decide_mailbox, action_mailbox, act_every=ACT_EVERY):
        self.env = env; self.to_brain = decide_mailbox; self.from_brain = action_mailbox
        self.act_every = act_every
        self.action = 0
        self.stop = threading.Event()
        self.frame = 0
        self.stats = dict(frames=0, late=0, max_late_ms=0.0, dt=[])

    def snapshot(self):
        objs = [o for o in self.env.objects if o]
        return dict(frame=self.frame, objs=[(type(o).__name__, *o.xy, *o.wh,
                    getattr(o, "orientation", -1)) for o in objs])

    def run(self, seconds):
        env = self.env
        env.reset(seed=0)
        next_t = time.perf_counter()
        t_end = next_t + seconds
        last = next_t
        while not self.stop.is_set() and time.perf_counter() < t_end:
            obs, rew, trunc, term, info = env.step(self.action)   # ← OCAtari 순서
            if term or trunc:
                env.reset()
            self.frame += 1; self.stats["frames"] += 1
            if self.frame % self.act_every == 0:
                self.to_brain.put(self.snapshot())
            a = self.from_brain.peek_take()
            if a is not None:
                self.action = a
            next_t += PERIOD
            now = time.perf_counter()
            if now > next_t:                       # 빚 탕감
                self.stats["late"] += 1
                self.stats["max_late_ms"] = max(self.stats["max_late_ms"], (now-next_t)*1000)
                next_t = now
            else:
                sleep_until(next_t)
            t = time.perf_counter()
            self.stats["dt"].append((t-last)*1000); last = t
        return self.stats


def dummy_brain(to_brain, to_game, delay_ms, stop, n_actions=14):
    rng = np.random.default_rng(0)
    while not stop.is_set():
        s = to_brain.take(timeout=0.2)
        if s is None: continue
        time.sleep(delay_ms/1000.0)               # 뇌 대신 더미 지연
        to_game.put(int(rng.integers(0, n_actions)))


if __name__ == "__main__":
    from ocatari.core import OCAtari
    SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    R = {}
    print(f"{'뇌 지연':>8} {'스레드':>8} {'평균 fps':>9} {'프레임 간격 중앙값':>18} "
          f"{'p99':>8} {'늦은 프레임':>11} {'최대 지각':>10}")
    for delay in (0, 9, 30, 66, 100):
        for mode in ("thread", "single"):
            env = OCAtari("ALE/Asteroids-v5", mode="ram", hud=False, frameskip=1,
                          repeat_action_probability=0.0)
            mb_in, mb_out = Mailbox(), Mailbox()
            stop = threading.Event()
            loop = PacingLoop(env, mb_in, mb_out)
            if mode == "thread":
                th = threading.Thread(target=dummy_brain,
                                      args=(mb_in, mb_out, delay, stop), daemon=True)
                th.start()
                st = loop.run(SECS)
                stop.set(); th.join(timeout=1)
            else:
                # 싱글스레드: 결정 프레임마다 뇌를 인라인으로 돌린다
                orig = loop.snapshot
                def blocking_snapshot(_o=orig, _d=delay):
                    s = _o()
                    time.sleep(_d/1000.0)
                    mb_out.put(0)
                    return s
                loop.snapshot = blocking_snapshot
                st = loop.run(SECS)
            dt = np.array(st["dt"][10:])
            fps = st["frames"]/SECS
            print(f"{delay:>6}ms {mode:>8} {fps:>9.1f} {np.median(dt):>17.2f}ms "
                  f"{np.percentile(dt,99):>7.2f}ms {st['late']:>11,} "
                  f"{st['max_late_ms']:>9.1f}ms")
            R[f"{delay}_{mode}"] = dict(fps=fps, median_ms=float(np.median(dt)),
                                        p99_ms=float(np.percentile(dt,99)),
                                        late=st["late"], max_late_ms=st["max_late_ms"])
            env.close()
    Path(__file__).resolve().parent.parent.joinpath("out/pacing.json").write_text(
        json.dumps(R, indent=2), encoding="utf-8")
    print(f"\n목표: 60.0 fps, 프레임 간격 16.67ms.  05문서 주장 = 뇌 지연 66ms 까지 흔들림 없음")
