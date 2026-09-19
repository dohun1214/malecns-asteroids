<div align="center">

# malecns-asteroids

**All 166,700 neurons of a fruit-fly connectome, simulated in real time, playing Atari Asteroids.**<br>
Put the fly's **escape circuit** or its **pursuit circuit** on the joystick,<br>
then silence **two cells** from the browser and watch that behaviour collapse on the spot.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Triton-76B900.svg)](https://triton-lang.org/)
[![Realtime 7.3x](https://img.shields.io/badge/realtime-7.3%C3%97-brightgreen.svg)](#performance)
[![Double dissociation](https://img.shields.io/badge/double_dissociation-0%25_vs_−100%25-ff6b5e.svg)](#-same-cells-off-0--in-one-mode-100--in-the-other--a-double-dissociation)

[한국어](README.md) · [Quick start](#quick-start) · [Evidence](#evidence) · [Honest limitations](#honest-limitations)

</div>

---

## In one paragraph

This runs the **MaleCNS v1.0 connectome** released by Google/Janelia
(166,700 neurons, 10,520,431 synapses at w ≥ 3) as a whole-brain LIF spiking model.
On-screen asteroid positions are injected into **LC4 visual projection neurons**; the escape
direction is read out from the **507 VNC motor cells downstream of DNp02 / DNp11** and turned
into joystick actions. Everything runs inside a **real-time 60 fps** budget.

The same brain has **two modes**. Both run all 166,700 neurons — a mode selects *which readout
is wired to the joystick*, it does not switch any neuron off.

| Mode | Circuit | Behaviour |
|---|---|---|
| **Flee** | `LC4 → DNp02/DNp11 → 507 VNC cells` | turns **away** from the threat |
| **Chase** | `LC10a → AOTU019/025 → DNa02·13·15 → 1,076 cells` | turns **toward** the target |

We did not choose the sign of the chase steering signal — it falls out of the wiring:
**AOTU025 is excitatory and ipsilateral, AOTU019 is inhibitory and contralateral.**

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
| Intact | — | 100.0 % | 28.6 % | −0.075 |
| **Random 2 cells** (5 seeds) | 2 | **100.00 ± 0.00 %** | 28.6 % | −0.075 |
| Random 200 cells | 200 | 86.9 % | 28.6 % | −0.077 |
| **DNp11** | **2** | **56.6 %** | **0.0 %** | **+0.729** ← sign flip |
| **DNp02** | **2** | **66.3 %** | 53.0 % | −0.756 |
| DNp04 / DNp01 (GF) | 2 | 82.0 % / 86.0 % | 30.4 % / 26.2 % | ~unchanged |

Silencing two DNp11 cells changes far more than silencing 200 random ones.
The dose response is monotonic (100 → 83.4 → 76.1 → 65.2 → 0 %) and survives three
different controller parameter sets.

### ② Nothing is silenced — only the *wiring* is shuffled — and it still breaks

In-degree, sign and weight multiset are **all preserved**; only the partners are permuted.
The uninformed baseline is 90°.

| Condition | Escape-azimuth error | Fraction of the distance to chance |
|---|---:|---:|
| Intact | **63.4°** | — |
| Shuffle all LC4 projections | 99.6° | **136 %** |
| LC4 → DN only | 87.2° | 90 % |
| LC4 → DNp02/11 only | 74.6° | 42 % |
| **Same size, shuffled elsewhere** | 63.1° | **−1 %** |
| Synapse-matched control / LPLC2→DN | 63.4° | **0 %** |

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

### ⑤ Same cells off: 0 % in one mode, −100 % in the other — a double dissociation

A one-way dissociation ("switch it off and it breaks") cannot rule out *"anything you switch off
breaks it."* Switching off **the same 275 cells** and getting opposite results per mode can.

| | Aiming (on-target fraction) | Avoidance (frames per life) |
|---|---:|---:|
| **Flee** mode, LC10a off | **+0.0 %** (identical to the decimal) | **+0.0 %** |
| Flee mode, DNp11 off | +25.6 % | **−60.9 %** |
| **Chase** mode, LC10a off | **−100.0 %** | −19.5 % |
| Chase mode, DNp11 off | +41.5 % | −0.2 % |

The nose really does turn: angular distance to the nearest asteroid goes **124.7° → 67.9°**.
As a negative control, silencing **the same number (275) of random cells** costs aiming
**nothing at all** (8.5 ± 2.3 % vs 5.5 % intact, 5 seeds). Only when those 275 cells are
LC10a does it go to 0 %.

**And the direction disappears without switching off a single cell.**
Shuffling only the **335 edges** of `LC10a → AOTU019/025` (0.0032 % of the graph), 10 seeds:

| Fraction rewired | 10 % | 25 % | 50 % | 75 % | 90 % | **100 %** |
|---|---:|---:|---:|---:|---:|---:|
| Directional information lost | 1.5 % | 4.2 % | 12.5 % | 27.4 % | 32.0 % | **60.1 %** |
| Signal amplitude retained | 106 % | 98 % | 87 % | 72 % | 67 % | **51.9 %** |

| Control (100 %, 10 seeds) | Information lost |
|---|---:|
| Same edge count, **other LC10a outputs** | 1.5 % |
| **Escape circuit** (LC4→DN) shuffled | 0.0 % |
| Size-matched random excitatory edges | −0.1 % |

**A lesion left 0 % of the signal** — which invites "you just cut the input". Rewiring leaves the
signal intact and removes only its direction. In the actual game the angular distance to the
nearest asteroid widens from **67.9° to 83.0°** (a same-size shuffle of other LC10a outputs
stays at 63.9°), while **100 % of decisions still carry a steering signal.**

> 🔴 The narrow on-target fraction shows nothing in-game (5.5 % intact vs 6.1 ± 1.5 % shuffled) —
> with an intact value that small the metric drowns in noise, so **read the angle instead.**
> Before issue #56 was fixed the same metric read −52.9 %; that number was an artifact of the
> screen-wrap bug.

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
│ [flee|chase] │ per-mode lesion buttons · shuffle · restore  │
└────────────────────────────────────────────────────────┘
```

- 3D brain: **139,662** soma coordinates as a three.js `Points` cloud, spikes decaying in a shader
- Toggles are applied **on decision boundaries**. A CUDA Graph bakes pointers, not values, so no
  recapture is needed — 29.5 µs
- Silenced cells go dark **per cell, not per group** (switch off 2, exactly 2 turn dark)
- The `분기 영상` button plays the three branch videos from ③
- **`Flee / Chase` toggle** — both brains stay resident and only the readout is swapped
  (fps 60.0 holds). Buttons, raster bands, population vector and decoder bars all switch per mode
  - Flee: `DNp11` `DNp02` `Giant Fiber` `half LC4` `shuffle LC4 wiring`
  - Chase: `LC10a` `shuffle LC10a wiring` `DNp11`
  - Lesions are applied to **both brains**, so state never desyncs when you switch modes

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
| `lc10a_position.py` | LC10a receptive fields | **two-stage hex-coordinate propagation** — only 0.9 % of LC10a input carries coordinates |
| `pursuit_targets.py` | pursuit readout | 550 left / 526 right VNC cells; cross-projection under 0.1 % |
| `play.py` | game loop | 60 Hz pacing, one decision every 4 frames. `BrainPolicy` (flee) / `PursuitPolicy` (chase) |
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
| Left-turn bias of +15.1 % | ⏸ Diagnosed: not LC4 cell count, but **per-cell synaptic strength (right 10 % stronger)** and a **39 % mismatch in DN→readout ipsi/contra ratio**. Deliberately **not corrected**, to avoid adding a fitted constant. *A rule-based controller on the same geometry leans the other way (−13.8 %)* |
| Accuracy of the y wrap period (177) | ⚠️ Measured as the playfield's vertical extent (rows 18–194) in rendered frames. **±2 px uncertain.** A behavioural check had no discriminating power (`probe_wrapval.py`). It is 33 px closer than the previous value of 210 |
| The `frames per life` metric | ⚠️ Counted as runs where the ship is visible. Fixing issue #56 changed ship-absence detection (85.5 % → 74.7 %), so runs break more often. **Do not compare policies on it** |
| Escape direction error of 82.8° | ⏸ Still large after inertia compensation. The dominant cause is that **the target moves faster than the ship can turn** (55 % of target changes exceed the 22.5° per-decision turn). Structural, given the real-time budget |
| A supervised MLP does better | ⏸ 43.6° vs 56.2°. Not a refutation, but we do not claim the connectome is optimal |
| Sign of the fore/aft axis | ⚠️ An imposed value. Confirmed indirectly by behaviour (it flees threats) |
| **No arousal gating in chase mode** | ⚠️ The literature reports *"almost no response"* from LC10a without arousal. Our model has no P1 and no internal state, so it assumes the fly is **always aroused**. The heaviest imposed assumption in gate 4 |
| LC10a is a **courtship pursuit** circuit, not a gun | ⚠️ The same class of analogy as using an escape circuit to dodge asteroids, but a stronger one. No paper says LC10a aims anything |
| The medial/lateral **label** for AOTU019 | ❓ The literature puts AOTU019 on the central visual field; our coordinates say the opposite. The *subset separation* replicates in both hemispheres (p = 0.0034), but we make **no claim about which end is central** |
| **Seed spread at 100 % rewire** | ⚠️ r = −0.381 ± 0.374 (5 of 10 seeds above −0.5). The dose response is monotonic, but 100 % is noisy. *Our first estimate used 3 seeds and **overstated** the loss as 85.8 %; 10 seeds corrected it to 60.1 %* |
| Chase mode does **not** thrust | ⏸ Steering only. Adding thrust would need a new readout channel and a new alignment criterion — two new choices |

### 🔴 The same class of trap caught us four times

All four came from **"4 frames per decision" being misaligned with Atari's frame granularity**,
and all four produced **plausible numbers with no error**.

| # | What | How it surfaced |
|---|---|---|
| 1 | Asteroid `dx/dy` is exactly 0 on half the frames | expansion rate was always 0 |
| 2 | The ship is absent 20 % of the time, in ~265-frame blocks | the 3D brain froze |
| 3 | Sprites are drawn on alternating frames | asteroids rendered as empty boxes |
| 4 | **Fire needs ≥2 frames held and then released** | **not a single shot was being fired** |

And **a human watching the screen caught four more** — zero shots, ignored inertia, no aiming,
and **screen wraparound** (issue #56).

| # | What | Size |
|---|---|---|
| 5a | **No wrap correction in the bearing computation** (ship velocity and asteroid tracking had it) | 21.8 % of pairs on the x axis alone; **the nearest asteroid is a different one on 14.4 % of decisions** |
| 5b | The y wrap period used **screen height 210** (the playfield is 177) | — |
| 5c | `Player.y` is reported as **520–528** as the ship exits the top, and we used it as the ship | **18.9 % of frames** under the whole-brain policy |

After the fix **the brain beats the rule-based controller for the first time**, and over half of
the left-turn bias disappeared (+37.5 % → +15.1 %). Every gate was re-measured and all still pass.

**The first three were caught by measurement. The fourth was caught by a human looking at the screen.**
That is why `sanity.py` exists — it does not check whether results are good, it checks whether
things **actually happen**.

> `FIRE_PRESS` (how many of a decision's 4 frames hold the button) **only works at 2.**
> At 1 the ROM misses it entirely depending on phase — zero shots across every seed tested.
> At 3 or more the release window is too short to re-fire. It is boxed in on both sides —
> **do not change this value.**

> **Scores** (8 episodes, 6,000-frame cap, random no-op start):
> whole brain **1,815 ± 617** · rule-based 1,445 ± 405 · do-nothing 399 ± 298.
> The brain **wins** — before issue #56 was fixed the two were tied within 1σ (2,332 vs 2,480).

---

## Repository layout

```
src/
  lif_rt.py        whole-brain LIF + Triton kernels + CUDA Graph
  gate4_sweep.py   gate 4 - pursuit azimuth sweep
  gate4_dissoc.py  gate 4 - double dissociation (2 modes x 3 lesions)
  gate4_control.py gate 4 - random-275-cell negative control
  gate4_rewire*.py gate 4 - LC10a->AOTU rewire (sweep / in-game)
  lc10a_position.py  LC10a two-stage coordinate propagation
  pursuit_targets.py pursuit readout (connectivity specificity)
  fire_aim.py      aiming measurement (with random-heading control)
  fire_ceiling.py  fire cap + perfect-aim ceiling
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
- Wilson Lab 2026, *Neuron* — [Specialized parallel pathways for adaptive control of visual object pursuit](https://www.cell.com/neuron/fulltext/S0896-6273(26)00001-2) ← key citation for chase mode
- Hindmarsh Sten et al. 2021, *Nature* — [Sexual arousal gates visual processing during courtship](https://www.nature.com/articles/s41586-021-03714-w)
- Ribeiro et al. 2018, *Cell* — [Visual projection neurons mediating directed courtship](https://www.cell.com/cell/fulltext/S0092-8674(18)30788-8)
- Tools: [OC_Atari](https://github.com/k4ntz/OC_Atari) · [ALE](https://ale.farama.org/environments/asteroids/) · [Triton](https://triton-lang.org/) · [three.js](https://threejs.org/)

## License

[MIT](LICENSE)
