"""하행뉴런 발화 -> 회피 벡터 -> 아타리 액션.

🔴 부호 (08문서 5장, Dombrovski 2023 원문 확인):
     DNp02 = 전방 수용장 -> **후진** 도피
     DNp11 = 후방 수용장 -> **전진** 도피
   즉 채널이 가리키는 건 '위협 방향'이고, 가야 할 곳은 그 반대다.
   안 뒤집으면 초파리가 운석으로 돌진한다.

몸 좌표계: x = 배의 오른쪽, y = 배의 정면.
"""
import numpy as np

DEG_PER_ORIENT = 22.5
ORIENT0_DEG = 90.0


class Decoder:
    """회피 벡터를 **하행뉴런의 하류 운동 집단**에서 읽는다 (이슈 #1).

    왜 DN 에서 직접 안 읽는가:
      1. DN 은 종류당 2세포뿐이라 66.6ms 창에서 스파이크가 1~2개 -> 방향이 2비트로 양자화
      2. 더 나쁜 건, `alive` 마스크가 출력 전파만 막는데 스파이크 판독은 마스크 적용 후
         목록에서 읽는다. DNp02 를 끄고 DNp02 출력을 읽으면 **정의상 0** 이다.
         네트워크를 통한 인과가 아니라 동어반복이다.

    판독 집단은 vnc_targets.py 가 연결성에서 고른다 (이름이 아니라 특이도 기준):
      DNp02 전용 하류 160개 / DNp11 전용 하류 347개. 공통은 88개뿐이라 거의 분리돼 있다.
      판독 세포 중 lesion 대상은 하나도 없으므로 동어반복이 원천 차단된다.

    실측 (vnc_sweep.py):
      방위각 스윕 r = -0.994(좌) / -0.985(우), 단조
      앞쪽 위협에서 DNp02 를 끄면 전후 채널 +0.551 -> +0.050 (91% 붕괴)
      뒤쪽 위협에서 DNp11 을 끄면 -0.646 -> +0.238 (**부호 반전**)
      음성 대조(무작위 2개)는 소수점까지 동일
    """

    def __init__(self, idx, nside, readout, align_slop=1, min_intensity=0.0):
        self.a02, self.a11 = readout["dn02_only"], readout["dn11_only"]
        s02, s11 = readout["side_dn02"], readout["side_dn11"]
        self.right = np.concatenate([self.a02[s02 == "R"], self.a11[s11 == "R"]])
        self.left  = np.concatenate([self.a02[s02 == "L"], self.a11[s11 == "L"]])
        self.all   = np.concatenate([self.a02, self.a11])
        # 화면 표시용으로 DN 자체도 계속 읽는다
        d02, d11, d04 = idx["DNp02"], idx["DNp11"], idx["DNp04"]
        self.dn = dict(p02_L=d02[nside[d02] == "L"], p02_R=d02[nside[d02] == "R"],
                       p11_L=d11[nside[d11] == "L"], p11_R=d11[nside[d11] == "R"],
                       p04=d04)
        self.align_slop = align_slop
        self.min_intensity = min_intensity
        self.tau = 1.0
        self._s = None

    def smooth(self, r):
        if self._s is None:
            self._s = dict(r)
        else:
            a = 1.0/self.tau
            for k in r: self._s[k] = (1-a)*self._s[k] + a*r[k]
        return self._s

    def reset(self):
        self._s = None

    def channels(self, rate_of):
        raw = dict(a02=rate_of(self.a02), a11=rate_of(self.a11),
                   r=rate_of(self.right), l=rate_of(self.left),
                   inten=rate_of(self.all))
        raw.update({k: rate_of(v) for k, v in self.dn.items()})
        r = self.smooth(raw)
        fore    = r["a02"] - r["a11"]        # + = 앞쪽 위협 (DNp02 하류 우세)
        # [실측] VNC 하류는 교차 투사다. DN 은 동측이지만(좌 LC4 -> DNp02_L 19.7 / _R -2.1)
        # 그 하류 운동 집단은 반대쪽이 더 크다(좌 LC4 자극 -> 좌우채널 +0.16/+0.42).
        # 하행뉴런이 정중선을 교차하는 건 해부학적으로 정상이다. 부호를 뒤집는다.
        # 안 뒤집으면 좌우가 거울처럼 반전돼 위협 쪽으로 조종한다 (생존 326 -> 이 버그로).
        lateral = r["l"]   - r["r"]          # + = 오른쪽 위협 (교차 보정)
        n = np.hypot(lateral, fore)
        return dict(lateral=lateral, fore=fore, intensity=r["inten"], norm=n,
                    unit=(lateral/n, fore/n) if n > 1e-9 else (0.0, 0.0),
                    # 집단 투표를 화면에 그리려면 **합친 채널이 아니라 집단별 값**이 필요하다.
                    # l/r 은 해부학적 좌/우 판독 집단이고, 교차 투사 때문에
                    # 좌측 집단이 '오른쪽 위협'을 뜻한다 (10문서 §3).
                    l=r["l"], r=r["r"],
                    **{k: r[k] for k in ("p02_L","p02_R","p11_L","p11_R","p04","a02","a11")})

    def action(self, ch, orientation, actions):
        """-> (action_index, 목표 orientation 또는 None, 상태 문자열)"""
        NOOP = actions.index("NOOP")
        if ch["intensity"] < self.min_intensity or ch["norm"] < 1e-9:
            return NOOP, None, "자극 없음"
        # 위협의 몸 좌표 방위각 (정면 0, 오른쪽 +)
        psi_threat = np.degrees(np.arctan2(ch["lateral"], ch["fore"]))
        psi_escape = psi_threat + 180.0                      # ← 반대로 간다
        head = (ORIENT0_DEG + DEG_PER_ORIENT*orientation) % 360.0
        world = (head - psi_escape) % 360.0                  # 몸 오른쪽 = 세계각 감소
        # [버그 이력] 예전엔 목표를 16방위로 먼저 반올림하고 정수 차이를 썼다.
        #   diff = (tgt - ori + 8) % 16 - 8  은 범위가 **-8..+7** 이라 칸이 하나 비대칭이다.
        #   diff = -8 은 '정반대 방향'이라 좌우 어디로 돌든 같은데 **항상 RIGHT** 로 갔다.
        #   순수 산술만으로 우회전이 53.8%(slop=1) 가 된다 (probe_bias.py D).
        # -> 반올림 전의 연속 각도로 판단한다. 정확히 ±180 일 때만 진짜 동점이고,
        #    연속값이라 사실상 안 나온다. tgt 는 표시용으로만 남긴다.
        ddeg = ((world - ORIENT0_DEG) - DEG_PER_ORIENT*orientation + 180.0) % 360.0 - 180.0
        tgt = int(round((world - ORIENT0_DEG)/DEG_PER_ORIENT)) % 16
        if abs(ddeg) <= (self.align_slop + 0.5)*DEG_PER_ORIENT:
            return actions.index("UP"), tgt, "추진"
        # probe_heading: LEFT 가 orientation 을 증가시킨다 (반시계)
        return (actions.index("LEFT") if ddeg > 0 else actions.index("RIGHT")), tgt, "회전"
