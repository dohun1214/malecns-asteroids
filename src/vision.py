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

# 🔴 [이슈 #56] 화면 랩어라운드. **화면을 본 사람이 물어서 찾았다** (네 번째다).
#   예전엔 _rel() 에 감싸기 보정이 아예 없었다. 배 속도와 운석 추적에는 있었는데,
#   정작 뇌에 들어가는 방위각을 만드는 곳에만 빠져 있었다.
#   실측(뇌 정책, 12,499쌍): x 축만 봐도 21.8% 가 감싸기가 더 가깝고,
#   **가장 가까운 운석의 정체가 바뀌는 결정이 14.4%** 였다.
#
#   x = 160: 배를 옆으로 밀어 직접 관측 (x=4 -> 164, d=160).
#   y = 178: **운석의 감싸기 사건을 개루프로 잡아 확정** (probe_wrap12.py).
#     배를 정지시키고 운석 트랙의 끝/시작을 잇는다. x(=160 확정)가 맞아떨어지는 짝만 쓰면
#     가짜가 걸러진다.  P = (끝 y + 속도x경과) - (시작 y).
#     **86개 사건 전부 정확히 178, 표준편차 0.0.**
#   독립 확인 2건이 맞아떨어진다:
#     ① 렌더 화면의 비배경 픽셀 (probe_wrap9): 행 5~14 점수 HUD / 15~16 빈 줄 /
#        **18~194 놀이터(177행)**.  좌표는 2칸씩이고 194 다음이 18 로 감기므로 주기는 **178**.
#     ② 자가시험으로 열을 재면 0~159 = 160 (직접 관측값과 일치).
#   예전 코드는 y 주기를 **210**(화면 높이)으로 쓰고 있었다. 놀이터가 아니다.
#   🔴 여기까지 오는 데 측정 방법 **5개가 실패**했다. 특히 폐루프 행동 스캔
#      (probe_wrapfit)은 원리적으로 불가능하다 — 12문서의 "폐루프 조건 비교에서는
#      궤적 발산 자체가 교란 변수다"에 그대로 걸린다. 개루프여야 한다.
WRAP_X = 160.0
WRAP_Y = 178.0
# 놀이터 세로 범위. 이 밖의 Player 좌표는 **쓰레기다** (아래 PLAYFIELD 주석).
FIELD_Y0, FIELD_Y1 = 18.0, 194.0


def ship_heading_deg(orientation):
    return (ORIENT0_DEG + DEG_PER_ORIENT*float(orientation)) % 360.0


def _wrap(d, p):
    """토러스에서의 최단 변위."""
    return d - p*np.round(d/p)


def _rel(ax, ay, sx, sy):
    """종횡비 보정된 상대 위치 (수학 좌표: y 위쪽). **화면 감싸기 보정 포함.**"""
    return _wrap(ax - sx, WRAP_X), -_wrap(ay - sy, WRAP_Y)/ASPECT


class Vision:
    """프레임마다 운석의 방위각·각크기·팽창률을 낸다.
    OCAtari 의 Player.dx/dy 는 항상 0 이므로 배 속도는 직접 차분한다."""

    def __init__(self, dt_frames=4):
        self.dt_frames = dt_frames
        self.prev_ship = None
        # 배 속도 (수학 좌표, 종횡비 보정, 결정 간격당 픽셀).
        # [실측] 이 게임은 관성이 있다 — 추진을 끊어도 3초 뒤까지 속도의 67% 가 남고
        #   회전만 해도 유지된다 (probe_inertia.py). 예전에는 이 값을 계산만 하고
        #   **반환도 사용도 안 했다.** 제어에 속도항이 없어서 가려는 방향과 실제 가는
        #   방향이 중앙값 90도 어긋났다 (이슈 #36).
        self.ship_v = (0.0, 0.0)
        self.prev = []            # 이전 결정 시점의 운석 [(x, y, w, h, theta, id)]
        self._next_id = 0

    def reset(self):
        self.prev_ship = None; self.prev = []; self._next_id = 0
        self.ship_v = (0.0, 0.0)

    def parse(self, objs):
        ship = None; asts = []
        for o in objs:
            if o is None: continue
            name = type(o).__name__
            w, h = o.wh
            if w <= 0 or h <= 0:          # 7.09% 가 쓰레기
                continue
            if name == "Player":
                # 🔴 [이슈 #56] 배가 화면 위로 나가는 순간 OCAtari 가 y 를 **520~528** 로
                #   보고한다 (화면 높이는 210). 예전엔 그걸 그대로 배 위치로 썼다 —
                #   **전체 뇌 정책에서 프레임의 18.9%.** 그 프레임은 모든 방위각이 틀리고
                #   배 속도(관성 보정 입력)도 튄다. 놀이터 밖이면 '배 없음'으로 처리한다.
                #   (배 부재는 이미 09문서대로 다루고 있다 — 자극만 비우고 뇌는 계속 돌린다.)
                y = float(o.xy[1])
                if FIELD_Y0 - 4.0 <= y <= FIELD_Y1 + 4.0:
                    ship = o
            elif name == "Asteroid": asts.append(o)
        return ship, asts

    def looming(self, objs):
        """-> (ship_xy, heading_deg, [dict(phi_rel, theta, dtheta, dist)])
        ship 이 없으면 (None, None, [])"""
        ship, asts = self.parse(objs)
        if ship is None:
            self.prev_ship = None; self.ship_v = (0.0, 0.0)
            return None, None, []
        sx, sy = float(ship.xy[0]), float(ship.xy[1])
        head = ship_heading_deg(getattr(ship, "orientation", 0))
        # 배 속도 (직접 차분, 랩어라운드 보정)
        if self.prev_ship is None:
            svx = svy = 0.0
        else:
            dx = float(_wrap(sx - self.prev_ship[0], WRAP_X))
            dy = float(_wrap(sy - self.prev_ship[1], WRAP_Y))
            svx, svy = dx, -dy/ASPECT
        self.prev_ship = (sx, sy)
        self.ship_v = (float(svx), float(svy))

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
                ddx = float(_wrap(ax - px, WRAP_X))
                ddy = float(_wrap(ay - py, WRAP_Y))
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


class LC10aMap(LC4Map):
    """방위각 -> LC10a 자극. **LC4Map 과 같은 규칙을 그대로 쓴다** (게이트 4, 20문서).

    새 상수를 하나도 안 만드는 것이 요점이다:
      - 축       LC4 가 찾은 theta_L 을 그대로 (같은 육각 좌표 틀)
      - 위치     2단 전파 (`lc10a_position.py`)
      - 부호화   LC4 와 같은 포화형 rate = cap * theta/(theta + th50), 같은 th50/cap
      - k        LC4 와 같은 값

    LC4 와 **다른 점은 하나뿐**이고, 그건 상수를 더하는 게 아니라 **빼는** 것이다:
      🔴 `dtheta > 0` (접근 중) 게이트를 **쓰지 않는다.**
         LC4 는 looming 검출기라 확대 방향에 선택적이지만, LC10a 는 문헌상
         **움직이는 표적**에 반응한다 ("enhancing their sensitivity to moving targets",
         Hindmarsh Sten 2021). 멀어지는 표적도 쫓는다. 게이트를 빼면 선택이 하나 줄어든다.

    ⚠️ 08문서 §8.4 의 부과값 4건은 그대로 진다 (각성 게이팅 없음이 제일 무겁다).
    """

    def rates(self, looms, n_neurons, th50, cap_hz, min_dtheta=None):
        acc = {}
        for L in looms:
            r = cap_hz*L["theta"]/(L["theta"] + th50)
            for c in self.cells_for(L["phi_rel"]):
                acc[int(c)] = acc.get(int(c), 0.0) + r
        if not acc:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        idx = np.fromiter(acc.keys(), np.int64, len(acc))
        rate = np.fromiter(acc.values(), np.float32, len(acc))
        return idx, np.minimum(rate, cap_hz)
