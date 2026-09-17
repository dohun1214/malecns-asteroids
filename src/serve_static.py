"""web/ 만 내보내는 정적 서버. 화면·영상만 볼 때 GPU 를 띄우지 않으려고 둔다.
  python src/serve_static.py 8790
"""
import sys, functools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_server import RangeStatic, Dual   # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
WEB = Path(__file__).resolve().parents[1]/"web"
print(f"http://localhost:{PORT}/index.html", flush=True)
Dual(("::", PORT), functools.partial(RangeStatic, directory=str(WEB))).serve_forever()
