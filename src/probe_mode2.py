"""LC10a 를 끈 뒤 조향 신호가 얼마나 빨리 0 으로 가는가.

헤드리스(gate4_sweep)에서는 lesion 직후 `b.reset()` 을 하므로 정확히 0 이 나온다.
대시보드는 **살아 있는 뇌에 그대로 병변을 건다** — 재귀 활동이 남아 있을 수 있다.
실측해서 '얼마 만에 조용해지는가'를 적어둔다. 안 적으면 화면에서 "안 꺼졌네"로 보인다.
"""
import asyncio, json
import websockets
URL = "ws://127.0.0.1:8765"


async def main():
    async with websockets.connect(URL, max_size=None) as ws:
        async def collect(n):
            out = []
            while len(out) < n:
                m = await asyncio.wait_for(ws.recv(), timeout=15)
                if isinstance(m, str):
                    d = json.loads(m)
                    if "step" in d: out.append(d)
            return out
        await ws.send(json.dumps({"cmd": "restore"}))
        await ws.send(json.dumps({"cmd": "mode", "mode": "pursuit"}))
        a = await collect(40)
        base = [abs(x.get("ch", {}).get("steer", 0)) for x in a[-20:]]
        print(f"온전 |조향| 평균 {sum(base)/len(base):.4f}  최대 {max(base):.4f}")
        await ws.send(json.dumps({"cmd": "lesion", "group": "LC10a", "on": True}))
        b = await collect(150)
        for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 100), (100, 150)):
            w = [abs(x.get("ch", {}).get("steer", 0)) for x in b[lo:hi]]
            sp = [x.get("n_fired", 0) for x in b[lo:hi]]
            print(f"  병변 후 결정 {lo:3d}~{hi:3d}  |조향| 평균 {sum(w)/len(w):.4f}"
                  f"  최대 {max(w):.4f}   발화 뉴런 평균 {sum(sp)/len(sp):.0f}")
        await ws.send(json.dumps({"cmd": "restore"}))
        print("OK")

asyncio.run(main())
