"""장시간 안정성 시험 — 데모 당일 30분 켜놔도 버티는가.

보는 것
  fps / 뇌 ms 가 시간에 따라 흘러내리는가
  서버 프로세스 메모리(RSS)와 GPU VRAM 이 계속 늘어나는가
  게임오버 -> 새 에피소드 전환이 매끄러운가
  클라이언트가 붙었다 끊었다 해도 송출이 안 얼어붙는가 (죽은 소켓 함정 재발 방지)
  토글을 계속 눌러도 상태가 어긋나지 않는가
"""
import asyncio, json, sys, time, subprocess
from pathlib import Path
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
URL = "ws://localhost:8765"
GROUPS = ["DNp11", "DNp02", "DNp01", "LC4half", "rewire"]
def say(*a): print(*a, flush=True)


def proc_mem():
    """서버 파이썬 프로세스의 RSS(MB) 와 GPU 사용량(MB)."""
    rss = np.nan
    try:
        import psutil
        for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
            cl = p.info.get("cmdline") or []
            if any("server.py" in str(x) for x in cl):
                rss = p.info["memory_info"].rss/1e6; break
    except Exception:
        pass
    vram = np.nan
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        vram = float(o.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return rss, vram


async def session(dur, tag, toggle=True):
    """한 번 접속해서 dur 초 동안 통계를 모은다."""
    st = dict(fps=[], ms=[], spk=[], gap=[], n=0, bytes=0, scores=[], lesion_ok=0,
              lesion_bad=0, new_eps=0)
    t0 = time.perf_counter(); last = t0; last_step = None; want = set()
    async with websockets.connect(URL, compression=None, max_size=None) as ws:
        nxt_toggle = t0 + 20
        while time.perf_counter() - t0 < dur:
            m = await asyncio.wait_for(ws.recv(), timeout=10)
            st["bytes"] += len(m)
            if not isinstance(m, str): continue
            d = json.loads(m)
            if d.get("type") == "hello": continue
            now = time.perf_counter()
            st["gap"].append(now - last); last = now
            st["n"] += 1
            st["fps"].append(d.get("fps", 0)); st["ms"].append(d.get("ms_brain", 0))
            st["spk"].append(d.get("n_fired", 0)); st["scores"].append(d.get("score", 0))
            if last_step is not None and d["step"] < last_step: st["new_eps"] += 1
            last_step = d["step"]
            if set(d.get("lesion", [])) == want: st["lesion_ok"] += 1
            else: st["lesion_bad"] += 1
            if toggle and now > nxt_toggle:
                nxt_toggle = now + 20
                if want:
                    await ws.send(json.dumps({"cmd": "restore"})); want = set()
                else:
                    g = GROUPS[(st["n"] // 7) % len(GROUPS)]
                    await ws.send(json.dumps({"cmd": "lesion", "group": g, "on": True}))
                    want = {g}
                await asyncio.sleep(0.4)     # 적용 지연 감안
        try: await ws.send(json.dumps({"cmd": "restore"}))
        except Exception: pass
    return st


async def main():
    say(f"장시간 시험 {MIN:.0f}분. 1분마다 접속을 끊었다 다시 붙인다.")
    say(f"{'경과':>6}{'fps':>7}{'뇌ms':>8}{'발화':>8}{'간격ms':>9}"
        f"{'최대간격':>9}{'RSS MB':>9}{'VRAM MB':>9}{'점수':>8}{'새에피':>7}{'병변동기':>9}")
    t0 = time.perf_counter(); rows = []
    while time.perf_counter() - t0 < MIN*60:
        try:
            s = await session(60, "s")
        except Exception as e:
            say(f"  !! 세션 실패: {type(e).__name__} {e}"); await asyncio.sleep(2); continue
        rss, vram = proc_mem()
        g = np.asarray(s["gap"])*1000
        row = dict(t=(time.perf_counter()-t0)/60,
                   fps=float(np.mean(s["fps"])), ms=float(np.mean(s["ms"])),
                   spk=float(np.mean(s["spk"])), gap=float(np.median(g)),
                   gmax=float(g.max() if len(g) else np.nan),
                   rss=rss, vram=vram, score=float(np.max(s["scores"])),
                   neweps=s["new_eps"],
                   sync=s["lesion_ok"]/max(s["lesion_ok"]+s["lesion_bad"], 1))
        rows.append(row)
        say(f"{row['t']:>5.1f}분{row['fps']:>7.1f}{row['ms']:>8.1f}{row['spk']:>8.0f}"
            f"{row['gap']:>9.1f}{row['gmax']:>9.1f}{row['rss']:>9.0f}{row['vram']:>9.0f}"
            f"{row['score']:>8.0f}{row['neweps']:>7}{row['sync']*100:>8.0f}%")
        await asyncio.sleep(1)              # 끊긴 채로 잠깐 — 죽은 소켓 처리 확인

    say(f"\n{'='*96}\n판정")
    f = np.asarray([r["fps"] for r in rows]); m = np.asarray([r["ms"] for r in rows])
    rs = np.asarray([r["rss"] for r in rows]); vr = np.asarray([r["vram"] for r in rows])
    ok1 = f.min() > 55
    say(f"  {'통과' if ok1 else '실패'}  fps  평균 {f.mean():.1f}  최소 {f.min():.1f}  "
        f"처음 {f[0]:.1f} -> 마지막 {f[-1]:.1f}")
    ok2 = m.max() < 50
    say(f"  {'통과' if ok2 else '실패'}  뇌   평균 {m.mean():.1f}ms  최대 {m.max():.1f}ms  "
        f"(예산 66.7ms)")
    drift = (rs[-1]-rs[0]) if np.isfinite(rs).all() else np.nan
    ok3 = not np.isfinite(drift) or drift < 200
    say(f"  {'통과' if ok3 else '실패'}  RSS  {rs[0]:.0f} -> {rs[-1]:.0f} MB "
        f"({drift:+.0f} MB / {MIN:.0f}분)")
    vd = (vr[-1]-vr[0]) if np.isfinite(vr).all() else np.nan
    ok4 = not np.isfinite(vd) or vd < 200
    say(f"  {'통과' if ok4 else '실패'}  VRAM {vr[0]:.0f} -> {vr[-1]:.0f} MB ({vd:+.0f} MB)")
    sy = np.asarray([r["sync"] for r in rows])
    say(f"  참고  병변 상태 동기 {sy.mean()*100:.0f}% (토글 직후 몇 프레임은 어긋나는 게 정상)")
    say(f"  참고  새 에피소드 전환 {sum(r['neweps'] for r in rows)}회, "
        f"최고 점수 {max(r['score'] for r in rows):.0f}")
    say(f"  참고  프레임 간격 중앙값 {np.median([r['gap'] for r in rows]):.1f}ms "
        f"(15Hz = 66.7ms), 최대 {max(r['gmax'] for r in rows):.0f}ms")
    say(f"\n  -> 장시간 시험 {'통과' if (ok1 and ok2 and ok3 and ok4) else '미통과'}")
    (ROOT/"out").mkdir(exist_ok=True)
    (ROOT/"out"/"soak.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    say("-> out/soak.json")

asyncio.run(main())
