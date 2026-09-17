"""web/ 를 브라우저로 내보내는 작은 정적 서버.

server.py 가 이걸 같은 프로세스 안에서 띄우고, 정적 파일만 손볼 때는
GPU 를 띄우지 않고 `python src/serve_static.py 8790` 으로 따로 띄운다.
"""
import os, re, socket
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

class _Slice:
    """copyfileobj 가 EOF 까지 읽어버리지 않게 길이를 잘라서 내보낸다."""
    def __init__(self, f, n): self.f, self.n = f, n
    def read(self, k=-1):
        if self.n <= 0: return b""
        if k is None or k < 0: k = self.n
        b = self.f.read(min(k, self.n)); self.n -= len(b); return b
    def close(self): self.f.close()

class RangeStatic(SimpleHTTPRequestHandler):
    # HTTP/1.0 이면 응답마다 연결을 닫는다. 영상은 Range 요청을 여러 번 보내므로
    # 연결을 재사용하게 1.1 로 올린다 (Content-Length 는 기본 핸들러가 붙인다).
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def end_headers(self):      # 개발 중 캐시 때문에 고친 화면이 안 뜨는 걸 막는다
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # 🔴 [실측] 파이썬 기본 정적 핸들러는 Range 를 무시하고 200 으로 전부 보낸다.
    #   그러면 크롬의 <video> 는 영상을 영영 열지 못한다 — 스피너만 돌고
    #   readyState 0, networkState 2(LOADING), duration NaN 에서 멈춘다.
    #   파일은 멀쩡했다 (ffprobe 720프레임, yuv420p, faststart, 디코드 정상).
    #   파이썬 기본 http.server 로 따로 띄워도 똑같이 멈추는 걸로 서버 쪽임을 확인했다.
    #   -> 206 Partial Content 를 직접 만든다. 나중에 외부 공개할 때도 필요하다.
    def send_head(self):
        rng = (self.headers.get("Range") or "").strip()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng)
        if not m: return super().send_head()
        path = self.translate_path(self.path)
        if not os.path.isfile(path): return super().send_head()
        size = os.path.getsize(path)
        a, b = m.group(1), m.group(2)
        if a == "":
            if b == "": return super().send_head()
            start, end = max(0, size - int(b)), size - 1
        else:
            start = int(a); end = int(b) if b else size - 1
        end = min(end, size - 1)
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb"); f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)


# 🔴 [실측] IPv4 로만 열면 "localhost" 접속이 2,050 ms 걸린다.
#   Windows 는 localhost 를 ::1 로 먼저 풀고, 그 시도가 약 2초 뒤에야 포기한다.
#   (ws://localhost 2077/2053/2050 ms vs ws://127.0.0.1 1/1/1 ms)
#   -> IPv6 듀얼스택으로 연다. Windows 는 V6ONLY 가 기본 1이라 꺼줘야 한다.
class Dual(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    def server_bind(self):
        try: self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError: pass
        super().server_bind()
