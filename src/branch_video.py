"""분기 녹화를 나란히 붙인 영상으로 만든다 (02문서 §10.4).

⚠️ 확대는 반드시 최근접(neighbor). bilinear 로 키우면 아타리 스프라이트가 뭉갠다.
   여기서는 numpy 로 정수배 확대하므로 보간이 아예 없다.
⚠️ MediaRecorder 를 안 쓰는 이유는 05문서 §3.7 — 실시간 사양이라 프레임을 버린다.
   두 분기가 서로 다르게 버려지면 '같은 시작점' 주장이 깨진다.
"""
import sys, subprocess, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT/"out"/"rec"
W, H, Z = 160, 210, 3
PAD, HDR, GAP = 14, 74, 10
FONT = "C:/Windows/Fonts/malgun.ttf"
FONTB = "C:/Windows/Fonts/malgunbd.ttf"
BG, FG, DIM, ACC, ALM = (14, 17, 19), (230, 237, 240), (139, 152, 159), (79, 209, 197), (255, 107, 94)
def say(*a): print(*a, flush=True)


def load(name):
    p = REC/f"{name.replace(' ', '_')}.rgb"
    a = np.fromfile(p, dtype=np.uint8)
    return a.reshape(-1, H, W, 3)


def load_ship(name):
    return np.load(REC/f"{name.replace(' ', '_')}_ship.npy")


def compose(panels, out_name, fps=60, split_frame=None, trails=None, tail_len=90):
    """panels: [(라벨, 부제, rgb배열)]  — 전부 같은 프레임 수여야 한다.
    trails: 패널별 (x, y, present) 배열. **배의 궤적을 겹쳐 그린다.**
    화면만 보면 분기가 잘 안 보인다 — 운석은 탄도 운동이라 총에 맞기 전까진 같은 길을
    간다. 실제로 갈라지는 건 배의 궤적이라, 그걸 안 그리면 세 화면이 똑같아 보인다."""
    n = min(len(p[2]) for p in panels)
    k = len(panels)
    pw, ph = W*Z, H*Z
    cw = PAD + k*(pw + GAP) - GAP + PAD
    ch = HDR + ph + PAD + 34
    f_big = ImageFont.truetype(FONTB, 21)
    f_sm = ImageFont.truetype(FONT, 14)
    f_tm = ImageFont.truetype(FONT, 15)

    base = Image.new("RGB", (cw, ch), BG)
    d = ImageDraw.Draw(base)
    for i, (lab, sub, _) in enumerate(panels):
        x = PAD + i*(pw + GAP)
        # 색 규칙 (06문서 §2): 경고색은 **병변에만**. 음성 대조는 강조색.
        col = FG if lab.startswith("온전") else (ACC if "무작위" in lab else ALM)
        d.text((x, 16), lab, font=f_big, fill=col)
        d.text((x, 45), sub, font=f_sm, fill=DIM)
        d.rectangle([x-1, HDR-1, x+pw, HDR+ph], outline=(35, 42, 46))
    tmpl = np.asarray(base)

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{cw}x{ch}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(REC/out_name)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for t in range(n):
        fr = tmpl.copy()
        for i, (_, _, arr) in enumerate(panels):
            x = PAD + i*(pw + GAP)
            big = np.repeat(np.repeat(arr[t], Z, axis=0), Z, axis=1)
            fr[HDR:HDR+ph, x:x+pw] = big
        im = Image.fromarray(fr); dd = ImageDraw.Draw(im)
        if trails is not None:
            for i, tr in enumerate(trails):
                x0 = PAD + i*(pw + GAP)
                col = FG if i == 0 else (ACC if "무작위" in panels[i][0] else ALM)
                lo = max(0, t - tail_len)
                pts = [(x0 + tr[j, 0]*Z + 3*Z, HDR + tr[j, 1]*Z + 5*Z)
                       for j in range(lo, min(t+1, len(tr))) if tr[j, 2] > 0.5]
                for j in range(1, len(pts)):
                    # 랩어라운드(화면 가장자리 순간이동) 구간은 잇지 않는다
                    if abs(pts[j][0]-pts[j-1][0]) > 60*Z or abs(pts[j][1]-pts[j-1][1]) > 60*Z:
                        continue
                    a_ = j/len(pts)
                    dd.line([pts[j-1], pts[j]], width=2,
                            fill=tuple(int(c*(0.25+0.75*a_)) for c in col))
                if pts:
                    x, y = pts[-1]
                    dd.ellipse([x-13, y-13, x+13, y+13], outline=col, width=2)
        y = HDR + ph + 9
        if split_frame is not None and t < split_frame:
            dd.text((PAD, y), f"같은 시작점 · 공통 구간  {t/fps:5.2f}s", font=f_tm, fill=DIM)
        else:
            e = (t - (split_frame or 0))/fps
            dd.text((PAD, y), f"분기 후  +{e:5.2f}s", font=f_tm, fill=ACC)
            dd.text((PAD+170, y), "여기서부터 입력은 같고 회로만 다르다", font=f_tm, fill=DIM)
        proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close(); proc.wait()
    say(f"  -> out/rec/{out_name}  ({n}프레임 / {n/fps:.1f}s / {cw}x{ch})")


pre = load("_prefix")
say(f"공통 구간 {len(pre)}프레임")
CONDS = {
    "온전": ("온전한 뇌", "166,700 뉴런 전부"),
    "무작위 2세포": ("무작위 2세포 끔", "166,700개 중 아무 2개"),
    "DNp11": ("DNp11 끔", "지목한 2세포"),
    "DNp02": ("DNp02 끔", "지목한 2세포"),
    "LC4 배선 섞기": ("LC4 배선 섞기", "세포는 전부 살아 있다"),
}
tail = {k: load(k) for k in CONDS}
full = {k: np.concatenate([pre, v]) for k, v in tail.items()}
sp = len(pre)

say("영상 합성")
pre_s = load_ship("_prefix")
shp = {k: np.concatenate([pre_s, load_ship(k)]) for k in CONDS}

def go(keys, name):
    compose([(CONDS[k][0], CONDS[k][1], full[k]) for k in keys], name,
            split_frame=sp, trails=[shp[k] for k in keys])

go(("온전", "무작위 2세포", "DNp11"), "분기_무작위2세포_vs_DNp11.mp4")
go(("온전", "DNp11", "DNp02", "LC4 배선 섞기"), "분기_4분할.mp4")
go(("온전", "무작위 2세포"), "분기_음성대조.mp4")
say("완료")
