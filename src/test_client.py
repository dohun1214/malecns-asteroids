"""서버가 실제로 프레임을 쏘는지 확인하는 최소 클라이언트."""
import asyncio, json, struct, sys, time
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0


async def main():
    async with websockets.connect(URL, compression=None, max_size=None) as ws:
        t0 = time.perf_counter(); n_spk = n_vid = n_txt = 0
        bytes_ = 0; last = None; toggled = False
        while time.perf_counter() - t0 < SEC:
            m = await asyncio.wait_for(ws.recv(), timeout=5)
            bytes_ += len(m)
            if isinstance(m, str):
                d = json.loads(m); n_txt += 1
                if d.get("type") == "hello":
                    print(f"hello: 점 {d['n_points']:,} / 뉴런 {d['n_neurons']:,} "
                          f"groups={d['groups']}", flush=True)
                else:
                    last = d
                    if n_txt % 15 == 0:
                        print(f"  step {d['step']:>5}  fps {d['fps']:>5.1f}  "
                              f"뇌 {d['ms_brain']:>5.1f}ms  프레임 {d['ms_frame']:>4.1f}ms  "
                              f"발화 {d['n_fired']:>6,}  점 {d['spikes']:>6,}  "
                              f"액션 {d['action']:<9} 점수 {d['score']:>5.0f}  "
                              f"병변 {d['lesion']}", flush=True)
                    if not toggled and time.perf_counter()-t0 > SEC*0.5:
                        toggled = True
                        await ws.send(json.dumps({"cmd": "lesion", "group": "DNp11", "on": True}))
                        print("  --- DNp11 끄기 전송 ---", flush=True)
            else:
                mg = struct.unpack_from("<I", m)[0]
                if mg == 0x31594C46: n_spk += 1
                elif mg == 0x32594C46:
                    n_vid += 1
                    if n_vid == 1:
                        wh = struct.unpack_from("<I", m, 8)[0]
                        print(f"video: {wh>>16}x{wh & 0xffff}  {len(m):,} B", flush=True)
        el = time.perf_counter()-t0
        print(f"\n{el:.1f}초: 스파이크프레임 {n_spk} ({n_spk/el:.1f}/s), "
              f"영상 {n_vid}, 상태 {n_txt}, 총 {bytes_/1e6:.1f} MB "
              f"({bytes_/el/1e6:.2f} MB/s)", flush=True)
        if last: print("마지막 상태:", json.dumps(
            {k: last[k] for k in ("step","fps","ms_brain","n_fired","action","lesion","score")},
            ensure_ascii=False), flush=True)

asyncio.run(main())
