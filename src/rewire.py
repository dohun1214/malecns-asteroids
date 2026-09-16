"""이슈 #4 / 게이트 3: 차수·부호 보존 rewire 대조군 (01문서 1장 무기 ②).

lesion 은 '세포를 껐더니 무너진다'인데, rewire 는 **세포를 하나도 안 끄고,
차수도 하나도 안 바꾸고, 연결 상대만 섞어서** 무너뜨린다.
"배선 자체가 원인"이라는 더 강한 증거다.

방법 — packed int32 표현의 이점:
  한 워드에 (가중치 상위 14bit | post 하위 18bit) 가 같이 들어 있다.
  선택한 위치들끼리 **packed 워드를 치환**하면 그게 곧 (post, weight) 동시 치환이고,
  crow 를 안 건드리므로 출차수가 자동 보존된다.
    - 출차수      : 각 소스의 선택된 엣지 '개수'가 그대로            -> 보존
    - 입차수/입력총량: (post, weight) 다중집합이 그대로               -> 보존
    - 부호        : 같은 부호 집합 안에서만 섞으면 자동 보존
  바뀌는 건 '누가 누구에게 연결되는가' 뿐이다.

⚠️ 01문서 1장: 전체 뇌 rewire 는 증거가 못 된다 (뇌가 침묵해 잴 것이 없어진다).
   세포 타입으로 고른 국소 rewire 만 의미가 있다. 여기서는 전체의 0.008% 만 건드린다.

⚠️ 남는 자유도: 치환이 같은 (pre, post) 쌍을 두 번 만들 수 있다. scatter 커널에서는
   두 번 더해지므로 '가중치가 합쳐진 엣지 1개'와 같다. 차수 회계상으로는 보존이지만
   유효 엣지 수는 조금 줄 수 있다 -> `check_preserved` 가 중복 개수를 같이 보고한다.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
POST_BITS = 18
POST_MASK = (1 << POST_BITS) - 1
DN_ALL = ("DNp01", "DNp02", "DNp03", "DNp04", "DNp05", "DNp06",
          "DNp09", "DNp10", "DNp11")


def unpack(packed):
    p = packed.astype(np.int64)
    return (p & POST_MASK), (p >> POST_BITS)      # post, signed weight


def load():
    crow = np.load(G/"out_crow.npy").astype(np.int64)
    packed = np.load(G/"out_packed.npy").astype(np.int32)
    C = dict(np.load(G/"circuit_idx.npz"))
    pre = np.repeat(np.arange(crow.size-1, dtype=np.int64), np.diff(crow))
    return crow, packed, C, pre


def _mask(N, idx):
    m = np.zeros(N, bool); m[np.asarray(idx)] = True
    return m


def select(N, pre, packed, src_idx, dst_idx):
    """src_idx 를 소스로 하고 dst_idx 를 타깃으로 하는 엣지의 위치."""
    post, _ = unpack(packed)
    return np.flatnonzero(_mask(N, src_idx)[pre] & _mask(N, dst_idx)[post])


def rewire(packed, sel, seed, frac=1.0):
    """선택된 위치 중 frac 비율을 골라 packed 워드를 치환한다.
    부호가 섞이면 안 되므로 **같은 부호끼리만** 섞는다."""
    rng = np.random.default_rng(seed)
    out = packed.copy()
    n = int(round(len(sel)*frac))
    if n < 2: return out, 0
    pos = sel[rng.choice(len(sel), size=n, replace=False)]
    _, w = unpack(packed)
    moved = 0
    for s in (+1, -1):                       # 흥분성/억제성 각각 따로 섞는다
        grp = pos[np.sign(w[pos]) == s]
        if len(grp) < 2: continue
        out[grp] = out[grp][rng.permutation(len(grp))]
        moved += len(grp)
    return out, moved


def check_preserved(orig, new, N):
    """출차수 / 입차수 / 입력총량 / 부호 / 가중치 다중집합이 정확히 보존됐는가."""
    p0, w0 = unpack(orig); p1, w1 = unpack(new)
    ind0 = np.bincount(p0, minlength=N); ind1 = np.bincount(p1, minlength=N)
    s0 = np.bincount(p0, weights=np.abs(w0).astype(np.float64), minlength=N)
    s1 = np.bincount(p1, weights=np.abs(w1).astype(np.float64), minlength=N)
    ch = np.flatnonzero(orig != new)
    return dict(
        outdeg="보존 (crow 불변)",
        indeg_maxdiff=int(np.abs(ind0-ind1).max()),
        instr_maxdiff=float(np.abs(s0-s1).max()),
        sign_same=bool(np.array_equal(np.bincount((np.sign(w0)+1).astype(np.int64), minlength=3),
                                      np.bincount((np.sign(w1)+1).astype(np.int64), minlength=3))),
        wmultiset_same=bool(np.array_equal(np.sort(w0), np.sort(w1))),
        changed=int(ch.size), pct=f"{ch.size/len(orig)*100:.4f}%")


def conditions(C, N, pre, packed):
    """실험에 쓸 rewire 조건 정의. gate3_rewire.py 가 이걸 import 한다.

    lc4_dn      LC4 -> DN 직접 엣지만            (가장 국소)
    lc4_p0211   LC4 -> DNp02/DNp11 만            (경사를 이루는 바로 그 엣지)
    lc4_all     LC4 의 **모든** 출력 엣지         (LC4 투사 전체를 무작위화)
    lplc2_dn    LPLC2 -> DN                      특이성 대조 (같은 종류, 방향 미기여)
    ctrl_random LC4 아닌 흥분성 엣지, 엣지 수 일치   크기 대조
    ctrl_matched 위와 같되 **시냅스 총량까지 일치**  강도 대조
    ctrl_all    LC4 아닌 뉴런들의 출력 엣지,
                lc4_all 과 엣지 수 일치            lc4_all 의 짝 대조
    """
    _, w = unpack(packed)
    dn = np.concatenate([C[k] for k in DN_ALL])
    lc4_dn = select(N, pre, packed, C["LC4"], dn)
    lc4_p0211 = select(N, pre, packed, C["LC4"],
                       np.concatenate([C["DNp02"], C["DNp11"]]))
    is_lc4_pre = _mask(N, C["LC4"])[pre]
    lc4_all = np.flatnonzero(is_lc4_pre)
    lplc2_dn = select(N, pre, packed, C["LPLC2"], dn)

    rng = np.random.default_rng(99)
    other = np.flatnonzero(~is_lc4_pre & (w > 0))
    ctrl_random = rng.choice(other, size=len(lc4_dn), replace=False)
    # 시냅스 총량까지 맞춘 대조: 가중치가 큰 엣지 위주로 뽑아 총합을 맞춘다
    target = int(np.abs(w[lc4_dn]).sum())
    cand = rng.permutation(other)
    cw = np.abs(w[cand]).cumsum()
    ctrl_matched = cand[:int(np.searchsorted(cw, target)) + 1]
    ctrl_all = rng.choice(np.flatnonzero(~is_lc4_pre), size=len(lc4_all), replace=False)
    return dict(lc4_dn=lc4_dn, lc4_p0211=lc4_p0211, lc4_all=lc4_all,
                lplc2_dn=lplc2_dn, ctrl_random=ctrl_random,
                ctrl_matched=ctrl_matched, ctrl_all=ctrl_all)


if __name__ == "__main__":
    crow, packed, C, pre = load()
    N = crow.size - 1
    post, w = unpack(packed)
    S = conditions(C, N, pre, packed)
    print(f"전체 엣지 {len(packed):,}  뉴런 {N:,}")
    for name, sel in S.items():
        print(f"  {name:<12} 엣지 {len(sel):>6,}  시냅스 {int(np.abs(w[sel]).sum()):>8,}"
              f"  부호 +{int((w[sel]>0).sum()):,} / -{int((w[sel]<0).sum()):,}")
    print()
    rows = {}
    for name, sel in S.items():
        for frac in ((0.25, 0.5, 1.0) if name in ("lc4_dn", "lc4_all") else (1.0,)):
            for seed in (0, 1, 2):
                new, n = rewire(packed, sel, seed=seed, frac=frac)
                chk = check_preserved(packed, new, N)
                # 중복 엣지(같은 pre->post 가 두 번)가 몇 개 생겼나
                key0 = pre*(N+1) + post
                p1, _ = unpack(new)
                key1 = pre*(N+1) + p1
                dup = int(len(key1) - len(np.unique(key1)))
                dup0 = int(len(key0) - len(np.unique(key0)))
                tag = f"{name} frac={frac:.2f} seed={seed}"
                rows[tag] = dict(moved=n, dup_new=dup, dup_orig=dup0, **chk)
                print(f"{tag:<34} 치환 {n:>5,}  변경 {chk['changed']:>5,} ({chk['pct']})  "
                      f"입차수차 {chk['indeg_maxdiff']}  입력총량차 {chk['instr_maxdiff']:.1f}  "
                      f"부호 {chk['sign_same']}  가중치집합 {chk['wmultiset_same']}  "
                      f"중복 {dup0}->{dup}")
    bad = [k for k, v in rows.items()
           if v["indeg_maxdiff"] or v["instr_maxdiff"] or not v["sign_same"]
           or not v["wmultiset_same"]]
    print(f"\n보존 위반 조건: {bad if bad else '없음 — 전부 통과'}")
    (ROOT/"out").mkdir(exist_ok=True)
    (ROOT/"out"/"rewire_preserve.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-> out/rewire_preserve.json")
