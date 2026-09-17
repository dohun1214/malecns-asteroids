"""대시보드 모드 전환 점검 — 서버를 띄운 채 붙어서 도망/쫓기를 오가며 상태를 확인한다."""
import asyncio, json, sys
import websockets

URL = "ws://127.0.0.1:8765"


async def main():
    async with websockets.connect(URL, max_size=None) as ws:
        seen = {}
        async def collect(n=40):
            out = []
            while len(out) < n:
                m = await asyncio.wait_for(ws.recv(), timeout=10)
                if isinstance(m, str):
                    d = json.loads(m)
                    if "step" in d: out.append(d)
            return out
        a = await collect(25)
        print(f"[도망]  mode={a[-1].get('mode')}  ch keys={sorted(a[-1].get('ch',{}))}")
        print(f"        rates={a[-1].get('rates')}  action={a[-1].get('action')}")
        await ws.send(json.dumps({"cmd": "mode", "mode": "pursuit"}))
        b = await collect(40)
        print(f"[쫓기]  mode={b[-1].get('mode')}  ch keys={sorted(b[-1].get('ch',{}))}")
        print(f"        rates={b[-1].get('rates')}  action={b[-1].get('action')}")
        await ws.send(json.dumps({"cmd": "lesion", "group": "LC10a", "on": True}))
        c = await collect(30)
        print(f"[쫓기+LC10a끄기] lesion={c[-1].get('lesion')} "
              f"steer={c[-1].get('ch',{}).get('steer')} action={c[-1].get('action')}")
        await ws.send(json.dumps({"cmd": "mode", "mode": "escape"}))
        d = await collect(30)
        print(f"[도망+LC10a끄기] mode={d[-1].get('mode')} lesion={d[-1].get('lesion')} "
              f"action={d[-1].get('action')} fps={d[-1].get('fps')} "
              f"ms_brain={d[-1].get('ms_brain')}")
        await ws.send(json.dumps({"cmd": "restore"}))
        e = await collect(20)
        print(f"[복구]  lesion={e[-1].get('lesion')} fps={e[-1].get('fps')}")
        print("OK")

asyncio.run(main())
