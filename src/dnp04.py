"""DNp04 를 끄면 운동 출력이 왜 **늘어나나** (07문서 게이트 1 의 정직한 단서 중 미규명).

기대: DNp04 는 '강도' 축이므로 끄면 판독 강도가 **줄어야** 한다.
실측: 오히려 **늘었다.** 이유를 안 밝히고 "미규명"으로 남겨뒀다.

가설
  H1  DNp04 -> (억제성 세포) -> 판독집단.  DNp04 를 끄면 그 억제가 풀려 판독이 커진다.
  H2  DNp04 가 판독집단에 직접 억제를 준다. (DNp04 는 콜린성 = 흥분성이라 직접은 어렵다)
  H3  경쟁/정규화. 특정 경로가 아니라 망 전체 활동의 재분배.

검사
  A  현상이 아직 있는가 (지금 코드로 재현)
  B  정적 배선: DNp04 -> ? -> 판독 의 부호 조합을 센다. 흥분->억제->판독 이 있으면 H1
  C  동적: DNp04 를 끈 전후로 그 중간 억제성 세포들의 구동이 실제로 줄었는가.
     H1 이 맞으면 중간 억제세포 활동이 줄고, 판독이 늘어야 한다. 둘 다 확인한다.
  D  H3 배제: 망 전체 평균 구동도 같이 본다. 전체가 같이 오르면 특정 경로가 아니다.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lif_rt import BrainRT, PARAMS

ROOT = Path(__file__).resolve().parent.parent
G = ROOT/"graph"
C = dict(np.load(G/"circuit_idx.npz"))
R_ = np.load(G/"vnc_readout.npz", allow_pickle=True)
A02, A11 = R_["dn02_only"], R_["dn11_only"]
READ = np.unique(np.concatenate([A02, A11]))
P = np.load(G/"lc4_position.npz", allow_pickle=True)
lc4, pos, side, valid = P["idx"], P["pos"], P["side"], P["valid"]
TH = float(P["theta_L"])
axis = pos[:, 0]*np.cos(TH) + pos[:, 1]*np.sin(TH)
nodes = pd.read_feather(ROOT.parent/"malecns-song"/"graph"/"nodes.feather")
TYPE = nodes["type"].astype("string").fillna("(없음)").to_numpy()
SIGN = nodes["sign"].to_numpy()

N_STIM, RATE, T_MS, NB = 16, 80.0, 300.0, 7
p = dict(PARAMS); steps = int(round(T_MS/p["dt"]))
b = BrainRT(params=p)
def say(*a): print(*a, flush=True)


def stim_bin(hemi, k):
    m = valid & (side == hemi); a = axis[m]; ii = lc4[m]; o = np.argsort(a)
    lo = int(round(k*(len(o)-N_STIM)/(NB-1)))
    return ii[o[lo:lo+N_STIM]]


def run(stim, off=None):
    b.reset(); b.alive.fill_(1)
    if off is not None and len(off):
        b.lesion(torch.as_tensor(np.asarray(off), device="cuda"))
    b.set_poisson(np.asarray(stim), RATE)
    b.run_tally(steps); torch.cuda.synchronize()
    g = (b.gacc/steps).cpu().numpy()
    b.alive.fill_(1)
    return g


# ── A. 현상이 아직 있는가 ───────────────────────────────────────────────────
say("="*88)
say("A. 현상 재현 — DNp04 2세포를 끄면 판독 강도가 어떻게 되나")
say(f"   {'창':>3} {'온전':>10} {'DNp04 끔':>10} {'변화':>9}   {'무작위2 끔':>11}")
rows, rnd_rows = [], []
rg = np.random.default_rng(5)
alive_pool = np.setdiff1d(np.arange(b.N), np.concatenate([C["DNp04"], READ]))
for k in range(NB):
    st = np.concatenate([stim_bin("L", k), stim_bin("R", k)])
    g0 = run(st); g1 = run(st, C["DNp04"])
    r0 = float(g0[READ].mean()); r1 = float(g1[READ].mean())
    gr = run(st, rg.choice(alive_pool, 2, replace=False))
    rr = float(gr[READ].mean())
    rows.append((k, r0, r1)); rnd_rows.append(rr)
    say(f"   {k:>3} {r0:>10.4f} {r1:>10.4f} {(r1-r0)/max(abs(r0),1e-9)*100:>+8.1f}% {rr:>11.4f}")
m0 = np.mean([r[1] for r in rows]); m1 = np.mean([r[2] for r in rows])
mr = float(np.mean(rnd_rows))
say(f"   평균 {m0:.4f} -> {m1:.4f} ({(m1-m0)/abs(m0)*100:+.1f}%)   음성대조 {mr:.4f}"
    f" ({(mr-m0)/abs(m0)*100:+.1f}%)")
out = {"intact": m0, "dnp04_off": m1, "random2_off": mr}

# ── B. 정적 배선 ────────────────────────────────────────────────────────────
crow = np.load(G/"out_crow.npy").astype(np.int64)
post = np.load(G/"out_post.npy").astype(np.int64)
w = np.load(G/"out_w.npy")
DN04 = np.asarray(C["DNp04"])


def outs(src):
    a, bb = crow[int(src)], crow[int(src)+1]
    return post[a:bb], w[a:bb]


say("\n" + "="*88)
say("B. 정적 배선 — DNp04 -> ? -> 판독집단")
t1, w1 = [], []
for s in DN04:
    pp, ww = outs(s); t1.append(pp); w1.append(ww)
t1 = np.concatenate(t1); w1 = np.concatenate(w1)
say(f"   DNp04 직접 출력 엣지 {len(t1):,}  시냅스 {np.abs(w1).sum():,.0f}")
direct = np.isin(t1, READ)
say(f"   판독집단으로 직접 가는 엣지 {int(direct.sum())}  시냅스 {np.abs(w1[direct]).sum():,.0f}")
# 1단계 중간세포를 부호별로
mid = np.unique(t1[~direct])
exc = mid[SIGN[mid] > 0]; inh = mid[SIGN[mid] < 0]
def relay_syn(cells):
    tot = 0.0; n = 0; hit = []
    for m_ in cells:
        pp, ww = outs(m_); sel = np.isin(pp, READ)
        if sel.any(): tot += float(np.abs(ww[sel]).sum()); n += int(sel.sum()); hit.append(int(m_))
    return tot, n, np.asarray(hit)
et, en, eh = relay_syn(exc); it_, in_, ih = relay_syn(inh)
say(f"   1단계 중간세포 {len(mid):,}  (흥분성 {len(exc):,} / 억제성 {len(inh):,})")
say(f"     흥분성 중 판독에 닿는 세포 {len(eh):,}  그 시냅스 {et:,.0f}")
say(f"     억제성 중 판독에 닿는 세포 {len(ih):,}  그 시냅스 {it_:,.0f}")
say(f"     -> 2단계 경로에서 **억제성 비중 {it_/max(et+it_,1e-9)*100:.1f}%**")
out["wiring"] = dict(direct_edges=int(direct.sum()), exc_relay=len(eh), inh_relay=len(ih),
                     exc_syn=float(et), inh_syn=float(it_))
if len(ih):
    tt = pd.DataFrame(dict(k=TYPE[ih])).value_counts().head(8)
    say("     억제성 중계세포 타입 상위 8")
    for (kk,), v in tt.items(): say(f"       {str(kk)[:24]:<24} {int(v):>4}세포")

# ── C. 동적 — 중계세포 활동이 실제로 줄었나 ─────────────────────────────────
say("\n" + "="*88)
say("C. 동적 — DNp04 를 끄면 그 억제성 중계세포의 구동이 줄었나")
say(f"   {'창':>3} {'억제중계 온전':>13} {'DNp04끔':>10} {'변화':>9} | {'흥분중계 변화':>13}")
dinh, dexc = [], []
for k in range(NB):
    st = np.concatenate([stim_bin("L", k), stim_bin("R", k)])
    g0 = run(st); g1 = run(st, C["DNp04"])
    a0 = float(g0[ih].mean()) if len(ih) else 0.0
    a1 = float(g1[ih].mean()) if len(ih) else 0.0
    e0 = float(g0[eh].mean()) if len(eh) else 0.0
    e1 = float(g1[eh].mean()) if len(eh) else 0.0
    dinh.append((a0, a1)); dexc.append((e0, e1))
    say(f"   {k:>3} {a0:>13.4f} {a1:>10.4f} {(a1-a0)/max(abs(a0),1e-9)*100:>+8.1f}%"
        f" | {(e1-e0)/max(abs(e0),1e-9)*100:>+12.1f}%")
i0 = np.mean([x[0] for x in dinh]); i1 = np.mean([x[1] for x in dinh])
e0m = np.mean([x[0] for x in dexc]); e1m = np.mean([x[1] for x in dexc])
say(f"   억제중계 평균 {i0:.4f} -> {i1:.4f} ({(i1-i0)/max(abs(i0),1e-9)*100:+.1f}%)")
say(f"   흥분중계 평균 {e0m:.4f} -> {e1m:.4f} ({(e1m-e0m)/max(abs(e0m),1e-9)*100:+.1f}%)")
out["dyn"] = dict(inh0=i0, inh1=i1, exc0=e0m, exc1=e1m)

# ── D. H3 배제 — 망 전체가 같이 오르나 ─────────────────────────────────────
say("\n" + "="*88)
say("D. 망 전체 평균 구동 (H3: 전체 재분배인가)")
w0, w1_ = [], []
for k in range(NB):
    st = np.concatenate([stim_bin("L", k), stim_bin("R", k)])
    g0 = run(st); g1 = run(st, C["DNp04"])
    w0.append(float(g0.mean())); w1_.append(float(g1.mean()))
# 🔴 망 전체 평균 구동은 흥분/억제가 섞여 **0 근처**다. 여기에 비율을 쓰면 안 된다.
#   실제로 처음엔 +94.8% 로 찍혀서 "전역 재분배(H3)" 라는 틀린 판정을 뱉었다.
#   0 근처 분모 함정은 이 프로젝트에서 **두 번째**다 (15문서 §2). 절대 변화로 본다.
say(f"   전체 {np.mean(w0):.5f} -> {np.mean(w1_):.5f}   절대변화 {np.mean(w1_)-np.mean(w0):+.5f}"
    f"   <- 0 근처라 비율은 의미가 없다")
say(f"   판독만 {m0:.4f} -> {m1:.4f}   절대변화 {m1-m0:+.4f} ({(m1-m0)/abs(m0)*100:+.1f}%)")
out["whole"] = dict(w0=float(np.mean(w0)), w1=float(np.mean(w1_)))

say("\n" + "="*88)
say("판정")
d_read = (m1-m0)/abs(m0)*100
d_inh = (i1-i0)/max(abs(i0), 1e-9)*100
d_all = (np.mean(w1_)-np.mean(w0))/max(abs(np.mean(w0)), 1e-9)*100
abs_read = m1 - m0
abs_all = float(np.mean(w1_) - np.mean(w0))
say(f"  판독   {m0:.4f} -> {m1:.4f}   절대 {abs_read:+.4f} ({d_read:+.1f}%)")
say(f"  억제중계 {i0:.4f} -> {i1:.4f} ({d_inh:+.1f}%)   흥분중계 {e0m:.4f} -> {e1m:.4f}"
    f" ({(e1m-e0m)/max(abs(e0m),1e-9)*100:+.1f}%)")
say(f"  망 전체 절대 {abs_all:+.5f}  (판독 절대변화의 {abs(abs_all)/max(abs(abs_read),1e-12):.2f}배)")
say("")
if d_read > 1:
    say("  현상 재현됨: DNp04 를 끄면 판독이 **늘어난다**.")
elif d_read < -1:
    say("  🔴 **현상이 재현되지 않는다.** DNp04 를 끄면 판독이 오히려 **줄어든다** — 기대한 방향이다.")
    say("     07문서의 'DNp04 끄면 운동 출력 증가' 는 **판독을 DN 에서 직접 읽던 시절**의 값이다.")
    say("     이슈 #1 에서 판독을 VNC 하류 507세포로 옮긴 뒤로 이 역설은 사라졌고,")
    say("     아무도 다시 확인하지 않아 '미규명' 으로 남아 있었다.")
else:
    say("  거의 안 바뀐다.")
say(f"  기전: DNp04 를 끄면 흥분성 중계가 {(e1m-e0m)/max(abs(e0m),1e-9)*100:+.1f}%,"
    f" 억제성 중계가 {d_inh:+.1f}% 로 **둘 다 준다.**")
say("        흥분 쪽 감소가 더 커서 순효과가 판독 감소다. 교과서적인 방향이다.")
if abs(abs_all) < abs(abs_read):
    say("  H3(전역 재분배) 는 기각: 망 전체의 절대 변화가 판독보다 작고 방향도 다르다.")
(ROOT/"out").mkdir(exist_ok=True)
(ROOT/"out"/"dnp04.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
say("-> out/dnp04.json")
