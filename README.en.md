<div align="center">

# malecns-asteroids

**All 166,700 neurons of a fruit-fly connectome, simulated in real time, playing Atari Asteroids.**<br>
Silence **two cells** from the browser and watch the escape behaviour collapse on the spot.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Triton-76B900.svg)](https://triton-lang.org/)
[![Realtime 7.3x](https://img.shields.io/badge/realtime-7.3%C3%97-brightgreen.svg)](#performance)

[한국어](README.md) · [Quick start](#quick-start) · [Evidence](#evidence) · [Honest limitations](#honest-limitations)

</div>

---

## In one paragraph

This runs the **MaleCNS v1.0 connectome** released by Google/Janelia
(166,700 neurons, 10,520,431 synapses at w ≥ 3) as a whole-brain LIF spiking model.
On-screen asteroid positions are injected into **LC4 visual projection neurons**; the escape
direction is read out from the **507 VNC motor cells downstream of DNp02 / DNp11** and turned
into joystick actions. Everything runs inside a **real-time 60 fps** budget.

## Why this exists

Plenty of projects run a connectome. Most cannot answer *"how is this different from an `if`
statement?"* If 166,700 neurons produce "enemy on the left, turn right", that is three lines of code.

> **The answer is not the title. It is the toggle button. A rule-based program has no neuron to switch off.**

So this repository is built to demonstrate **"the named circuit is the cause"**, not "it plays well".

---

## Evidence

### ① Silencing 2 neurons changes half the behaviour — silencing a random 2,000 changes nothing

The stimulus sequence an intact brain actually saw is recorded to a **tape** and replayed
**literally identically** to every condition (paired replay), so trajectory divergence cannot leak in.

| Manipulation | Cells | Action agreement | Thrust | Fore/aft channel |
|---|---:|---:|---:|---:|
| Intact | — | 100.0 % | 29.0 % | −0.139 |
| **Random 2 / 20 cells** (5 seeds) | 2 / 20 | **100.00 ± 0.00 %** | 29.0 % | −0.139 |
| Random 200 / 2,000 cells | 200 / 2,000 | 92.8 % / 86.7 % | — | — |
| **DNp11** | **2** | **57.4 %** | **0.0 %** | **+0.596** ← sign flip |
| **DNp02** | **2** | **73.2 %** | 46.5 % | −0.685 |
| DNp04 / DNp01 (GF) | 2 | 81.6 % / 89.0 % | 30.3 % / 26.8 % | ~unchanged |

**Two DNp11 cells are 3,216× more effective per cell than 2,000 random ones.**
The dose response is monotonic (100 → 84.6 → 78.8 → 67.7 → 0 %) and survives three
different controller parameter sets.

### ② Nothing is silenced — only the *wiring* is shuffled — and it still breaks

In-degree, sign and weight multiset are **all preserved**; only the partners are permuted.
The uninformed baseline is 90°.

| Condition | Escape-azimuth error | Fraction of the distance to chance |
|---|---:|---:|
| Intact | **43.6°** | — |
| Shuffle all LC4 projections | 105.0° | **132 %** |
| LC4 → DN only | 80.9° | 80 % |
| **Same size, shuffled elsewhere** | 43.5° | **−0 %** |
| Synapse-count-matched control | 43.4° | −0 % |

### ③ Branch recordings from a byte-identical starting point

ALE system state + six OCAtari internal buffers + RNG + 16 brain tensors are all restored, so
the runs start from **exactly the same point**. Running `intact` twice is **byte-identical**.

| Branch | Thrust | Diverges at |
|---|---:|---:|
| Intact / **random 2 cells** | 18.5 % | **never** |
| **DNp11 (2 cells) off** | **1.5 %** | frame 25 (0.42 s) |
| DNp02 (2 cells) off | 44.4 % | frame 5 |
| **LC4 wiring shuffled** (every cell alive) | 21.5 % | frame 9 |

### ④ "Isn't it just parameter count?" — no

An MLP with the same input and a matched parameter count, **randomly initialised**, sits at chance.

| | Mean error | within ±45° |
|---|---:|---:|
| Random MLP (hidden 11 / 96 / 256) | 88.5 / **81.9** / 88.0° | 26–28 % |
| **Fly brain** | **47.6°** | **61.3 %** |

> A **supervised** MLP trained on ground truth reaches 43.6° on the same held-out set, beating the
> fly's 56.2°. That is not a refutation — **the fly never saw a label.** We report it anyway.

---

## Performance

| Configuration | Compute per frame (66.7 ms) | vs real time |
|---|---:|---:|
| post-major sparse CSR, dt = 0.1 | 498 ms | 0.13× |
| dt = 0.2 + int32 + unrolled CUDA Graph | 118.5 ms | 0.56× |
| **pre-major event-driven Triton + CUDA Graph** | **9.2 ms** | **7.3×** |
| ↑ plus game, pacing and streaming (production loop) | 11–26 ms | 2.6–5.9× |

The bottleneck was **memory bandwidth**, not kernel launches. A post-major CSR reads the entire
nnz every step regardless of who fired (measured 236 GB/s = 82 % of spec). A pre-major scatter
reads **only the out-edges of neurons that actually fired**: 84.2 MB → about 1 MB, a 13× win on
the same GPU.

**Passed a 30-minute uninterrupted soak**: 60.0 fps average, 23.5 ms per decision, +66 MB VRAM,
100 % lesion-state sync.

---

## Quick start

### Requirements

- NVIDIA GPU (developed and measured on an RTX 4060 Ti 8 GB), CUDA
- Python 3.11
- `ffmpeg` (only for compositing the branch videos)
- MaleCNS v1.0 connectome data — obtained separately (see `graph/`)

### Install

```bash
pip install triton-windows websockets psutil numpy torch pandas pyarrow
pip install "gymnasium[atari]"            # gymnasium 1.3.0 + ale-py 0.12.1
pip install setuptools==69.5.1            # before ocatari
pip install --no-build-isolation ocatari  # 2.2.1
```

### Run

```bash
python src/server.py          # dashboard at http://localhost:8766/index.html
python src/sanity.py 3000     # 9 automated checks — run after any large change
python src/regate2.py         # game-skill comparison
python src/gate2_assay.py     # gate 2 (lesion causality)
python src/gate3_replay.py    # gate 3 (rewire)
```

---

## Dashboard

Open it in a browser and **switch neurons off yourself.**

```
┌─ header ───────────────────────────────────────────────┐
│ fps │ ms/decision │ step │ spiking neurons │ score      │
├────────────────────────┬───────────────────────────────┤
│ Asteroids              │ whole brain, live (139,662 pts)│
│  + fly overlay         │  only silenced cells go dark   │
├──────────┬─────────────┴─┬───────────┬─────────────────┤
│ LC4 looming │ spike raster │ pop. vector │ decoder x4   │
├────────────────────────────────────────────────────────┤
│ DNp11 off · DNp02 off · GF off · half LC4 · shuffle LC4 │
└────────────────────────────────────────────────────────┘
```

- 3D brain: **139,662** soma coordinates as a three.js `Points` cloud, spikes decaying in a shader
- Toggles are applied **on decision boundaries**. A CUDA Graph bakes pointers, not values, so no
  recapture is needed — 29.5 µs
- Silenced cells go dark **per cell, not per group** (switch off 2, exactly 2 turn dark)
- The `분기 영상` button plays the three branch videos from ③

---

## How it works

```
screen ─▶ Vision ─▶ LC4Map ─▶ [ 166,700-neuron LIF ] ─▶ Decoder ─▶ action
        azimuth,    hex-coord     Triton scatter          507 VNC
        angular     stimulation   + 333-step CUDA Graph   cells
        size
```

| Piece | What | Key idea |
|---|---|---|
| `lif_rt.py` | whole-brain LIF | event-driven Triton kernels, packed int32 edges, 333-step unrolled CUDA Graph |
| `vision.py` | screen → stimulus | azimuth, angular size, expansion — **pure geometry, zero fitted constants** |
| `decode.py` | spikes → action | readout from 507 VNC cells (reading DNs directly is tautological) |
| `play.py` | game loop | 60 Hz pacing, one decision every 4 frames |
| `server.py` | dashboard server | 60 Hz sim thread + single-slot mailbox + WebSocket |
| `sanity.py` | automated checks | 9 assertions **plus a self-test** |

### Why it is bit-reproducible

Weights are **integers** (synapse count × sign), so every partial sum stays inside the exactly
representable fp32 range (|sum| ≤ 120k ≪ 2²⁴). Addition order therefore cannot change the result.
That is why `int32` atomic accumulation matches a completely different reference implementation
**spike for spike** (51,644 / 51,644 cumulative, never diverging).

Without this, claim ③ ("same starting point") would not hold.

---

## Honest limitations

This repository also records **what does not work**. That is part of the evidence.

| | Status |
|---|---|
| DNp02 fires above the literature's subthreshold report | ❌ Intrinsic limit of whole-brain LIF. **Never quote absolute Hz** |
| GF azimuth dependence is the opposite of the literature | ❌ Same limit. No design impact — GF is not used for direction |
| Left-turn bias of +22.4 %p | ⏸ Diagnosed: not LC4 cell count, but **per-cell synaptic strength (right 10 % stronger)** and a **39 % mismatch in DN→readout ipsi/contra ratio**. Deliberately **not corrected**, to avoid adding a fitted constant |
| Escape direction error of 82.8° | ⏸ Still large after inertia compensation. The dominant cause is that **the target moves faster than the ship can turn** (55 % of target changes exceed the 22.5° per-decision turn). Structural, given the real-time budget |
| A supervised MLP does better | ⏸ 43.6° vs 56.2°. Not a refutation, but we do not claim the connectome is optimal |
| Sign of the fore/aft axis | ⚠️ The one imposed value. Confirmed indirectly by behaviour (it flees threats) |

### 🔴 The same class of trap caught us four times

All four came from **"4 frames per decision" being misaligned with Atari's frame granularity**,
and all four produced **plausible numbers with no error**.

| # | What | How it surfaced |
|---|---|---|
| 1 | Asteroid `dx/dy` is exactly 0 on half the frames | expansion rate was always 0 |
| 2 | The ship is absent 20 % of the time, in ~265-frame blocks | the 3D brain froze |
| 3 | Sprites are drawn on alternating frames | asteroids rendered as empty boxes |
| 4 | **Fire needs ≥2 frames held and then released** | **not a single shot was being fired** |

**The first three were caught by measurement. The fourth was caught by a human looking at the screen.**
That is why `sanity.py` exists — it does not check whether results are good, it checks whether
things **actually happen**.

> `FIRE_PRESS` (how many of a decision's 4 frames hold the button) **only works at 2.**
> At 1 the ROM misses it entirely depending on phase — zero shots across every seed tested.
> At 3 or more the release window is too short to re-fire. It is boxed in on both sides —
> **do not change this value.**

---

## Repository layout

```
src/
  lif_rt.py        whole-brain LIF + Triton kernels + CUDA Graph
  vision.py        screen -> azimuth / angular size / expansion, LC4Map
  decode.py        VNC readout -> escape vector -> action
  play.py          game loop and policies (brain / rule-based / random)
  server.py        dashboard server (WebSocket + static)
  static_server.py static server with Range (206) support
  sanity.py        9 automated checks + self-test
  gate_1a/b/c.py   gate 1 — monotonicity, laterality, cancellation
  gate2_assay.py   gate 2 — lesion causality assay
  gate3_*.py       gate 3 — degree- and sign-preserving rewire
  branch_record.py branch recording (incl. byte-identical verification)
  mlp_control.py   size-matched MLP control
  verify.py        silence ladder + reference equivalence
web/               dashboard (three.js, single HTML file)
graph/             preprocessed connectome (packed int32 edges)
out/               measurement results as JSON
```

## References

- Dombrovski et al. 2023, *Nature* — [Synaptic gradients transform object location to action](https://www.nature.com/articles/s41586-022-05562-8) ← primary citation
- Dombrovski et al. 2025, *Nature* — [LC4 vs LPLC2](https://www.nature.com/articles/s41586-025-09037-4)
- Jang & von Reyn 2023, *JEB* — [Azimuthal invariance in the GF escape circuit](https://journals.biologists.com/jeb/article/226/8/jeb244790/307120)
- Shiu et al. 2024, *Nature* — [Whole-brain LIF model](https://www.nature.com/articles/s41586-024-07763-9)
- Nern et al. 2025, *Nature* — [Connectome-driven neural inventory of a complete visual system](https://www.nature.com/articles/s41586-025-08746-0)
- Tools: [OC_Atari](https://github.com/k4ntz/OC_Atari) · [ALE](https://ale.farama.org/environments/asteroids/) · [Triton](https://triton-lang.org/) · [three.js](https://threejs.org/)

## License

[MIT](LICENSE)
