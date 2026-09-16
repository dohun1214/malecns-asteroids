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
    def __init__(self, idx, nside, align_slop=1, min_intensity=0.0):
        """min_intensity: DNp04(강도 채널) 하한.
        [실측] 66.6ms 결정 창에서 약하고 공간적으로 퍼진 자극은 DNp04 를 못 깨운다
        (LC4 전체 150Hz 면 345Hz 로 정상 반응하지만, 게임의 16~43세포 자극에선 0).
        그래서 강도 채널을 '행동 게이트'로 쓰지 않는다. 방향 채널(norm)만으로 판단한다.
        DNp04 는 화면 표시용으로만 남긴다."""
        d02, d11, d04 = idx["DNp02"], idx["DNp11"], idx["DNp04"]
        self.g = dict(
            p02_L=d02[nside[d02] == "L"], p02_R=d02[nside[d02] == "R"],
            p11_L=d11[nside[d11] == "L"], p11_R=d11[nside[d11] == "R"],
            p04=d04)
        self.align_slop = align_slop
        self.min_intensity = min_intensity
        # [실측] 66.6ms 결정 창에서 DN 한 종류(2세포)의 스파이크는 1~2개뿐이다.
        # 게이트 1(a)의 동작점은 300ms 창에서 검증한 값이라 4.5배 짧은 창에서는
        # 방향이 스파이크 양자화에 묻힌다. 지수 평활로 유효 적분 시간을 늘린다.
        # tau=4 결정 = 약 267ms 로 게이트 1(a) 창과 같은 자릿수가 된다.
        self.tau = 1.0   # 등급 판독이라 평활 불필요 (지연 0)
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
        """rate_of(indices) -> 평균 Hz"""
        r = self.smooth({k: float(rate_of(v)) for k, v in self.g.items()})
        lateral = (r["p02_R"] + r["p11_R"]) - (r["p02_L"] + r["p11_L"])   # + = 오른쪽
        fore    = (r["p02_L"] + r["p02_R"]) - (r["p11_L"] + r["p11_R"])   # + = 앞쪽
        inten   = r["p04"]
        n = np.hypot(lateral, fore)
        return dict(lateral=lateral, fore=fore, intensity=inten, norm=n,
                    unit=(lateral/n, fore/n) if n > 1e-9 else (0.0, 0.0), **r)

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
        tgt = int(round((world - ORIENT0_DEG)/DEG_PER_ORIENT)) % 16
        diff = (tgt - orientation + 8) % 16 - 8              # -8..7
        if abs(diff) <= self.align_slop:
            return actions.index("UP"), tgt, "추진"
        # probe_heading: LEFT 가 orientation 을 증가시킨다 (반시계)
        return (actions.index("LEFT") if diff > 0 else actions.index("RIGHT")), tgt, "회전"
