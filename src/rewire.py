"""이슈 #4 / 게이트 3: LC4 -> DN 차수·부호 보존 rewire 대조군 (01문서 1장 무기 ②).

lesion 은 '세포를 껐더니 무너진다'인데, rewire 는 **세포를 하나도 안 끄고 연결만 섞어서**
무너뜨린다. 배선 자체가 원인이라는 더 강한 증거다.

방법 — packed int32 표현의 이점:
  한 워드에 (가중치 상위 14bit | post 하위 18bit) 가 같이 들어 있다.
  선택한 위치들끼리 **packed 워드를 치환**하면 그게 곧 (post, weight) 동시 치환이고,
  crow 를 안 건드리므로 출차수가 자동 보존된다.
    - 출차수: 각 LC4 의 선택된 엣지 '개수'가 그대로 -> 보존
    - 입차수/입력총량: (post, weight) 다중집합이 그대로 -> 보존
    - 부호: LC4 는 전부 콜린성(+126 / -0) 이므로 자동 보존
  바뀌는 건 '누가 누구에게 연결되는가' 뿐이다.

⚠️ 01문서 1장: 전체 뇌 rewire 는 증거가 못 된다 (뇌가 침묵해 잴 것이 없어진다).
   세포 타입으로 고른 국소 rewire 만 의미가 있다.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
POST_BITS = 18
POST_MASK = (1 << POST_BITS) - 1


def unpack(packed):
    p = packed.astype(np.int64)
    return (p & POST_MASK), (p >> POST_BITS)      # post, signed weight


def select_lc4_dn(crow, packed, lc4, dn_all):
    """LC4 를 소스로 하고 DN 을 타깃으로 하는 엣지 위치."""
    N = crow.size - 1
    pre = np.repeat(np.arange(N, dtype=np.int64), np.diff(crow))
    post, _ = unpack(packed)
    is_lc4 = np.zeros(N, bool); is_lc4[lc4] = True
    is_dn = np.zeros(N, bool); is_dn[dn_all] = True
    return np.flatnonzero(is_lc4[pre] & is_dn[post]), pre


def rewire(packed, sel, seed, frac=1.0):
    """선택된 위치 중 frac 비율을 골라 packed 워드를 치환한다."""
    rng = np.random.default_rng(seed)
    out = packed.copy()
    n = int(round(len(sel)*frac))
    if n < 2: return out, 0
    take = rng.choice(len(sel), size=n, replace=False)
    pos = sel[take]
    out[pos] = out[pos][rng.permutation(n)]
    return out, n


def check_preserved(crow, orig, new, pre):
    """출차수 / 입차수 / 입력총량 / 부호가 정확히 보존됐는가."""
    N = crow.size - 1
    p0, w0 = unpack(orig); p1, w1 = unpack(new)
    r = {}
    r["출차수"] = "보존 (crow 불변)"
    ind0 = np.bincount(p0, minlength=N); ind1 = np.bincount(p1, minlength=N)
    r["입차수 최대차"] = int(np.abs(ind0 - ind1).max())
    s0 = np.bincount(p0, weights=np.abs(w0), minlength=N)
    s1 = np.bincount(p1, weights=np.abs(w1), minlength=N)
    r["입력총량 최대차"] = float(np.abs(s0 - s1).max())
    r["부호 분포 동일"] = bool((np.bincount((np.sign(w0)+1).astype(np.int64), minlength=3)
                            == np.bincount((np.sign(w1)+1).astype(np.int64), minlength=3)).all())
    r["가중치 다중집합 동일"] = bool(np.array_equal(np.sort(w0), np.sort(w1)))
    changed = int((orig != new).sum())
    r["바뀐 엣지"] = changed
    r["전체 대비"] = f"{changed/len(orig)*100:.4f}%"
    return r


if __name__ == "__main__":
    crow = np.load(G/"out_crow.npy").astype(np.int64)
    packed = np.load(G/"out_packed.npy").astype(np.int32)
    C = dict(np.load(G/"circuit_idx.npz"))
    lc4 = C["LC4"]
    dn_all = np.concatenate([C[k] for k in
                             ("DNp01","DNp02","DNp03","DNp04","DNp05","DNp06",
                              "DNp09","DNp10","DNp11")])
    sel, pre = select_lc4_dn(crow, packed, lc4, dn_all)
    post, w = unpack(packed)
    print(f"전체 엣지 {len(packed):,}")
    print(f"LC4 -> DN 엣지 {len(sel):,}개, 시냅스 {int(np.abs(w[sel]).sum()):,}")
    print(f"  LC4 부호 분포: +{int((w[sel] > 0).sum()):,} / -{int((w[sel] < 0).sum()):,}"
          f"   <- 전부 양수여야 부호 보존이 자동으로 된다")
    for frac in (0.25, 0.5, 1.0):
        new, n = rewire(packed, sel, seed=0, frac=frac)
        chk = check_preserved(crow, packed, new, pre)
        print(f"\nfrac={frac:.2f}  치환 {n:,}개")
        for k, v in chk.items():
            print(f"    {k:<16} {v}")
        np.save(G/f"out_packed_rewire{int(frac*100)}.npy", new)
    # 음성 대조: 같은 개수의 무관한 엣지 (LC4 가 아닌 소스)
    rng = np.random.default_rng(99)
    N = crow.size - 1
    is_lc4 = np.zeros(N, bool); is_lc4[lc4] = True
    other = np.flatnonzero(~is_lc4[pre])
    ctrl_sel = rng.choice(other, size=len(sel), replace=False)
    # 부호가 섞이면 안 되므로 같은 부호끼리만
    wc = w[ctrl_sel]
    pos_only = ctrl_sel[wc > 0]
    newc, nc = rewire(packed, pos_only, seed=0, frac=1.0)
    chk = check_preserved(crow, packed, newc, pre)
    print(f"\n음성 대조 (LC4 아닌 흥분성 엣지 {nc:,}개 치환)")
    for k, v in chk.items():
        print(f"    {k:<16} {v}")
    np.save(G/"out_packed_ctrl.npy", newc)
    print(f"\n-> graph/out_packed_rewire{{25,50,100}}.npy, out_packed_ctrl.npy")
