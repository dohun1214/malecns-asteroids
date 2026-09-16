"""화면 -> LC4 자극. 09문서에서 측정한 상수만 쓴다.

  아타리 픽셀 종횡비  세로/2.0      (probe_aspect.py, 보정 없으면 최대 21도 오차)
  배 heading          90 + 22.5*orient  수학각, 반시계 (probe_heading.py)
  h<=0 물체 버리기     7.09% 가 쓰레기 (probe_game.py)
  Player 부재 처리     프레임의 50% (probe_game.py)

[!] OCAtari 의 dx/dy 를 쓰지 않는다.
   운석 위치는 2프레임 주기로만 갱신돼서 dx/dy 가 '절반의 프레임에서 정확히 0'이다
   (probe_game 에서 비영 비율이 딱 50% 로 나온 게 그 신호였다).
   4프레임마다 결정하는 우리 루프는 표본 위상이 고정되므로 항상 0 위상에 걸릴 수 있고,
   그러면 팽창률이 전부 0 이 되어 뇌에 자극이 하나도 안 간다. 실제로 그렇게 됐다.
   -> 운석을 직접 추적해서 결정 간격(4프레임) 동안의 각크기 변화로 dtheta 를 낸다.
"""
import numpy as np

ASPECT = 2.0          # 세로 팽창 계수 (실측)
DEG_PER_ORIENT = 22.5
ORIENT0_DEG = 90.0    # orientation 0 = 화면 위쪽


def ship_heading_deg(orientation):
    return (ORIENT0_DEG + DEG_PER_ORIENT*float(orientation)) % 360.0


def _rel(ax, ay, sx, sy):
    """종횡비 보정된 상대 위치 (수학 좌표: y 위쪽)."""
    return (ax - sx), -(ay - sy)/ASPECT


class Vision:
    """프레임마다 운석의 방위각·각크기·팽창률을 낸다.
    OCAtari 의 Player.dx/dy 는 항상 0 이므로 배 속도는 직접 차분한다."""

    def __init__(self, dt_frames=4):
        self.dt_frames = dt_frames
        self.prev_ship = None
        self.prev = []            # 이전 결정 시점의 운석 [(x, y, w, h, theta, id)]
        self._next_id = 0

    def reset(self):
        self.prev_ship = None; self.prev = []; self._next_id = 0

    def parse(self, objs):
        ship = None; asts = []
        for o in objs:
            if o is None: continue
            name = type(o).__name__
            w, h = o.wh
            if w <= 0 or h <= 0:          # 7.09% 가 쓰레기
                continue
            if name == "Player": ship = o
            elif name == "Asteroid": asts.append(o)
        return ship, asts

    def looming(self, objs):
        """-> (ship_xy, heading_deg, [dict(phi_rel, theta, dtheta, dist)])
        ship 이 없으면 (None, None, [])"""
        ship, asts = self.parse(objs)
        if ship is None:
            self.prev_ship = None
            return None, None, []
        sx, sy = float(ship.xy[0]), float(ship.xy[1])
        head = ship_heading_deg(getattr(ship, "orientation", 0))
        # 배 속도 (직접 차분, 랩어라운드 보정)
        if self.prev_ship is None:
            svx = svy = 0.0
        else:
            dx = sx - self.prev_ship[0]; dy = sy - self.prev_ship[1]
            dx -= 160.0*round(dx/160.0); dy -= 210.0*round(dy/210.0)
            svx, svy = dx, -dy/ASPECT
        self.prev_ship = (sx, sy)

        out = []
        cur = []
        for a in asts:
            ax, ay = float(a.xy[0]), float(a.xy[1])
            w, h = a.wh
            rx, ry = _rel(ax, ay, sx, sy)
            d = float(np.hypot(rx, ry))
            if d < 1e-6: d = 1e-6
            r = max(float(w), float(h)/ASPECT)/2.0        # 종횡비 보정된 반지름
            theta = 2.0*np.arctan2(r, d)
            cur.append([ax, ay, w, h, float(np.degrees(theta)), -1])
            # phi_rel: 배의 정면이 0, **오른쪽이 양수**(시계 방향).
            # [버그 이력] 수학 규약(world - head, 반시계=양수=왼쪽)으로 두고
            # cells_for 가 양수를 오른쪽 눈으로 보냈다. 그러면 배가 dΔ 만큼 돌 때
            # 목표 방위가 2dΔ 만큼 움직여 **영원히 정렬이 안 된다** (추진 0%).
            phi = (head - np.degrees(np.arctan2(ry, rx)) + 180.0) % 360.0 - 180.0
            out.append(dict(phi_rel=float(phi), theta=float(np.degrees(theta)),
                            dtheta=0.0, dist=d, x=ax, y=ay, w=w, h=h))

        # 직전 결정 시점의 같은 운석을 찾아 각크기 변화로 팽창률을 낸다.
        # 같은 매칭으로 지속 ID 도 물려준다 (사건 기반 지표용, 이슈 #6).
        used = set()
        for o, c in zip(out, cur):
            ax, ay, w, h, th, _ = c
            best, bd, bi = None, 1e9, None
            for j, (px, py, pw, ph, pth, pid) in enumerate(self.prev):
                if pw != w or ph != h or j in used:
                    continue
                ddx = ax - px; ddy = ay - py
                ddx -= 160.0*round(ddx/160.0); ddy -= 210.0*round(ddy/210.0)
                dd = ddx*ddx + ddy*ddy
                if dd < bd: bd, best, bi = dd, pth, j
            if best is not None and bd <= (16.0*self.dt_frames)**2:
                o["dtheta"] = (th - best)/self.dt_frames
                c[5] = self.prev[bi][5]
                used.add(bi)
            else:
                c[5] = self._next_id; self._next_id += 1
            o["id"] = c[5]
        self.prev = cur
        return (sx, sy), head, out


class LC4Map:
    """방위각 -> LC4 자극. 게이트 1(a)에서 검증된 '위치로 K개 고르기'를 그대로 쓴다."""

    def __init__(self, lc4_idx, pos, side, valid, theta, k=16):
        """lc4_idx: LC4 세포의 **전역 뉴런 인덱스** (lc4_position.npz 의 'idx').

        [버그 이력] 여기서 np.flatnonzero(m) (= LC4 배열 안의 위치 0~125) 을 썼더니
        전역 인덱스 0~125 번 뉴런을 자극했고, 그 안에 DNp01(0,6) / DNp11(92) / DNp02(103)
        이 들어 있어서 **LC4 가 아니라 하행뉴런을 직접 찌르고 있었다.**
        증상: 좌반구 DN 이 400 결정 내내 정확히 0, 방향 상관 r = -0.08.
        """
        ax = pos[:, 0]*np.cos(theta) + pos[:, 1]*np.sin(theta)
        self.k = k
        self.idx = {}; self.ax = {}; self.lo = {}; self.hi = {}
        for h in ("L", "R"):
            m = valid & (side == h)
            a = ax[m]; o = np.argsort(a)
            self.idx[h] = np.asarray(lc4_idx)[m][o]      # 전역 인덱스, 앞쪽 -> 뒤쪽 정렬
            self.ax[h] = a[o]
            self.lo[h], self.hi[h] = float(a[o][0]), float(a[o][-1])

    def cells_for(self, phi_rel_deg):
        """|phi| 0도(정면) ~ 180도(정후방) 를 LC4 전후축에 선형 대응.
        ⚠️ 선형 가정은 부과값이다 (겹눈은 전방이 더 조밀)."""
        h = "R" if phi_rel_deg >= 0 else "L"       # 오른쪽 위협 -> 오른쪽 눈
        t = min(abs(phi_rel_deg), 180.0)/180.0
        target = self.lo[h] + t*(self.hi[h] - self.lo[h])
        a = self.ax[h]
        j = int(np.searchsorted(a, target))
        lo = max(0, min(len(a) - self.k, j - self.k//2))
        return self.idx[h][lo:lo + self.k]

    def rates(self, looms, n_neurons, th50, cap_hz, min_dtheta=0.0):
        """팽창(접근) 중인 운석마다 자극을 더한다. -> (indices, rates_hz)

        구동량으로 '팽창률'이 아니라 '각크기 theta'를 쓴다.
        이유(실측): 아타리 운석의 프레임당 각크기 변화는 배가 정지해 있으면 1e-6 수준까지
        떨어지고 전체 분포가 5자릿수에 흩어진다. 어떤 단일 상수로도 등급이 안 나온다.
        각크기는 중앙값 7.2도 / 90%ile 22도 / 최대 152도로 범위가 안정적이다.

        포화형:  rate = cap * theta/(theta + th50),  th50 = 실전 각크기 중앙값
        접근 여부(dtheta > 0)는 게이트로만 쓴다 — looming 검출기는 확대 방향에 선택적이다.
        [부과값] 이 부호화 방식 자체는 선택이다. 회계에 적는다.
        """
        acc = {}
        for L in looms:
            if L["dtheta"] <= min_dtheta: continue
            r = cap_hz*L["theta"]/(L["theta"] + th50)
            for c in self.cells_for(L["phi_rel"]):
                acc[int(c)] = acc.get(int(c), 0.0) + r
        if not acc: return np.empty(0, np.int64), np.empty(0, np.float32)
        idx = np.fromiter(acc.keys(), np.int64, len(acc))
        rate = np.fromiter(acc.values(), np.float32, len(acc))
        return idx, np.minimum(rate, cap_hz)
