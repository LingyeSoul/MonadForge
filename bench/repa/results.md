# REPA layer × σ probe — findings

Where does Anima's **base** DiT hold a PE-Spatial-alignable representation, and
does that depth move with the noise level σ? The shipped relational REPA term
anchors **block 8 of 28** (≈29% depth) — a number transplanted from Yu et al.'s
SiT-XL/2 ablation (arXiv:2410.06940), never measured on Anima's backbone +
PE-Spatial target + flow schedule. This probe measures it analytically, with
**zero training compute**.

Probe: `probe_layer_sigma_cka.py`. Targets discriminated from the related
`probe_dog_target.py` (which evaluates the *target* side only — PE features, no
DiT forward).

## The ruler + the three confound controls

For each cached real latent we renoise `x_σ = (1−σ)·x0 + σ·ε` at each σ and run
**one** feature-tap forward capturing every block
(`forward_mini_train_dit(..., return_block_features={0..27},
return_features_early=True)`). Each block's tokens are pooled to the encoder
grid exactly as the training term does (`pool_dit_tokens_to_grid`), then scored
by **centered linear CKA** vs the image's own PE-Spatial patch tokens. CKA is a
normalized Gram-of-Grams overlap — the *no-training analog of the relational
Gram loss* (`library/training/repa.py::relational_gram_loss`); its centering is
the CKA analog of `spatial_norm`'s DC removal.

Raw matched CKA is **badly confounded** — three layers had to be peeled off
before the number meant anything:

| confound | what inflates CKA | control |
|---|---|---|
| shared spatial layout | any two feature maps of the same image, on the same grid, correlate token-to-token regardless of content | **mismatched CKA** — image *i*'s DiT tokens vs a *different same-grid image*'s PE; `gap = matched − mismatched` is the content-*specific* alignment |
| caption-driven output reconstruction | the deep blocks reconstruct the caption's output field, registered to the target → high gap even at σ=1 (pure noise) | **σ→1 floor subtraction** — `useful_gap = gap − gap(σ=1)` isolates the part driven by *processing the noisy input* (what REPA regularizes; the diffusion loss already trains the reconstruction) |
| low-frequency global composition | smooth global layout dominates the Gram and persists into high σ | **DoG target band-pass** (`--target_dog`, the shipped `repa_target_dog` lever) — strips the low band from the PE target |

`useful_gap` is the headline metric. The early "representation regime" peak
(layers ≤ `--repr_regime_max`, default 11) is reported separately from the
global argmax as the **REPA-defensible** target (least redundant with the
flow-matching objective).

## Runs

| run | result dir | N | target |
|---|---|---|---|
| non-DoG | `results/20260615-1434-full-sweep-v2/` | 96 | spatial_norm (centered CKA) |
| DoG | `results/20260615-1443-full-sweep-dog/` | 96 | DoG band-pass σ₁=min/16 |

Reproduce (each ≈10 min on 16 GB, base DiT only):

```bash
uv run python bench/repa/probe_layer_sigma_cka.py --num_samples 96 --label full-sweep-v2
uv run python bench/repa/probe_layer_sigma_cka.py --num_samples 96 --target_dog --label full-sweep-dog
```

## Result 1 — the landscape is bimodal (robust to DoG)

Band-averaged `useful_gap` (σ∈[0.45, 0.9]):

| layer | non-DoG | DoG | regime |
|---|---|---|---|
| 7–11 | 0.175–0.181 | 0.057–0.059 | **early plateau** (peak ~L9/L11) |
| 12–20 | 0.021–0.025 | 0.006–0.008 | **dead zone** (rep. unlike PE) |
| 22 | 0.179 | 0.061 | re-emergence |
| 24 | 0.225 | 0.087 | late regime |
| **26** | **0.298** | **0.119** | **global peak** |
| 27 | 0.305 | 0.111 | global peak |

Three regimes, separated by a hard bottleneck at blocks 12–20: **build** (0–11)
→ **heavy mixing** (12–20, geometry unlike PE) → **re-emerge into a PE-aligned
semantic space** (21–27) before the final decode.

The early regime is a **flat plateau** — layers 7–11 are within noise of each
other, so **shipped layer 8 is already optimal within the representation-forming
zone**. No static-layer win there, DoG or not.

## Result 2 — DoG refuted the "deep peak = low-freq reconstruction" hypothesis

Hypothesis: the deep peak is low-frequency global-composition reconstruction, so
band-passing the target should crush it relative to the early regime.

**Refuted.** DoG shrank everything ~3× (it removes the dominant low band) but the
deep/early ratio *grew*:

| | deep (L26) / early (L8) |
|---|---|
| non-DoG | 1.68× |
| DoG | **2.05×** |

The high-frequency band that survives band-passing is *more* concentrated in the
deep layers. The deep-regime alignment is **genuine high-freq content**, not a
low-freq artifact.

What DoG *did* fix: the **σ-tail confound**. Non-DoG, deep layers held alignment
into high noise (low-freq global layout is reconstructable even from near-noise).
Under DoG every layer's `useful_gap` decays cleanly to **0 by σ=1** — e.g. L26:
`0.136 (σ.45) → 0.085 (σ.85) → 0.024 (σ.95) → 0.000 (σ1)`. So the persistent
high-σ component *was* a confound, and it was the **low-frequency band** — DoG is
the right tool to strip it.

## Result 3 — "dynamic against timestep" is a **weight** lever, not a **layer** lever

- **Layer ridge is σ-stable.** The argmax-layer per σ-column barely moves (span
  0–1 block across the band). A per-σ *layer* schedule is **refuted** — one layer
  sits on the ridge at every σ.
- **The target decays monotonically with σ.** Content-specific alignment peaks at
  the **clean end** and falls to ~0 by **σ≈0.85** (DoG: early layers die by
  ~σ0.85, deep layers hold to ~σ0.95). Uniform `repa_weight` across σ wastes
  supervision above σ≈0.8 — there is no recoverable clean-PE target there
  (the input is noise; nothing to align to).

So the dynamic lever is an **σ-tapered `repa_weight`** (peak at σ≲0.6, ~0 above
0.85), sharper still under `target_dog`. The `sigma_weight` array in each run's
`heatmap.npz` is that schedule, peak-normalized.

## Verdicts

1. **Layer 8 stays** for the representation regime — it's on a flat plateau, robust
   to DoG. The original "is 8 wrong?" → no, within the defensible zone.
2. **The deep peak (24–27) is real and ~1.7–2× stronger**, and DoG confirms it's
   genuine high-freq content. **But** it feeds the velocity head, so its alignment
   may be **redundant with the flow-matching loss** — a base-model probe cannot
   distinguish "rich representation" from "reconstruction the diffusion loss
   already trains." This motivated a **layer-vs-8 training A/B** → now run and
   **settled (layer 8 wins, layer 26 is worse than no REPA)** — see **Result 4**.
3. **σ-tapered `repa_weight`** is the high-confidence, low-risk win and the real
   answer to "dynamic against timestep."

## Artifacts (per run)

- `summary.md` — verdicts + full layer×σ `useful_gap` table + decomposition.
- `cka_by_layer_sigma.csv` — matched / mismatched / gap per (layer, σ).
- `heatmap.npz` — `cka_mean`, `cka_mismatch`, `gap_mean`, `useful_gap`, `floor`,
  `gram_mean` (the actual `relational_align_loss` cross-check), `ridge`,
  `sigma_weight`, and the derived `l_star_*`.

## Open threads

- **σ-weight wiring** into `REPAMethodAdapter` (the adapter already sees σ per
  step; `repa_anneal_steps` is a per-*training-step* cutoff, this is a per-*σ*
  taper — a new lever).
- ~~Layer-24-vs-8 A/B~~ **DONE — see Result 4.** Layer 8 wins; deep is net-negative.
- ~~The probe measures where alignment *exists*, not where *injecting* it helps~~
  — the two training-side tools in Result 4 (`probe_layer_grad_conflict.py`,
  `compare_repa_ckpts.py`) now supply that complement.

# Result 4 — the layer-8-vs-26 A/B, settled (training + two new tools)

The probe (Results 1–3) measures *where alignment exists*, not *where injecting
it helps*. The A/B settled the latter, and two new tools explain **why**.

## The A/B

`anima_repa4_tenth` (repa_layer **8**) vs `anima_repa5_tenth` (repa_layer **26**)
— identical runs (dim 16/α 16, 3 epochs, relational + target_dog, `tenth`
preset), differing only in the anchor block. Training loss already favored 8
(0.085 vs 0.091 final); the deep peak the CKA probe pointed at (26) **lost**.

## Two new tools

- `probe_layer_grad_conflict.py` — base-model gradient interaction. For each
  block output hₗ: FM-stress-weighted per-token cos(∂L_fm/∂hₗ, ∂L_repa/∂hₗ) and
  the orthogonal-push ratio mag = ‖g_repa‖/‖g_fm‖. Needs gradient checkpointing
  (forced on) to backprop the full stack on 16 GB.
- `compare_repa_ckpts.py` — head-to-head on the **trained artifacts** over a
  held-out probe set: (1) held-out flow-matching loss by σ, (2) **DoG-Gram align
  loss** by layer = the exact trained objective, Δ vs base, (3) ΔW per-block
  Frobenius norm + stable rank. Layer-vectorized (batched `bmm`) — GPU-resident,
  ~3 min for base + 2 adapters × 64 images × 11 σ × 28 layers.

```bash
uv run python bench/repa/probe_layer_grad_conflict.py --num_samples 32 --target_dog --label full-sweep
uv run python bench/repa/compare_repa_ckpts.py --num_samples 64   # defaults to the repa4/repa5 pair
```

Result dirs: grad sweep `results/20260615-1549-full-sweep/`; comparison
`results/20260615-1631-compare-ckpts/`.

## Finding A — held-out denoising: 8 > base > 26

FM loss on 64 held-out images, band σ∈[0.45,0.9], **best at every single σ**:

| | band FM | vs base |
|---|---|---|
| **layer 8** | **0.0768** | −0.0002 (best) |
| base | 0.0770 | — |
| **layer 26** | **0.0781** | **+0.0011 (worse than no REPA)** |

Margins are small (3-epoch `tenth` run) but the *ranking* is perfectly consistent
across all 11 σ. **Deep REPA is net-negative for the actual objective.**

## Finding B — the DoG-Gram delta resolves the CKA paradox

Raw matched CKA *dropped* at both trained layers (a ruler mismatch — REPA trains
the DoG/high-freq target, raw CKA is dominated by the low band DoG strips). On the
**trained objective** (DoG-Gram align loss, lower = aligned), both injected
alignment — and layer 26 injected **~8× more**, yet denoises worse:

| | align loss @ its layer | Δ vs base |
|---|---|---|
| layer 8 (base 0.152) | 0.061 | **−0.091** |
| layer 26 (base 0.914) | 0.106 | **−0.808** |

Two structural reasons deep is damaging:
1. **Base alignment is depth-dependent** — block 8 is already near the DoG target
   (base 0.15); block 26 is far (base 0.91). Closing that gap = a large rewrite.
2. **Deep anchoring spills backward** — layer 26 drove align loss down across
   blocks **9–27** (−0.26 @ L10, −0.24 @ L14, −0.11 @ L20), restructuring the
   denoising-critical back half. Layer 8's effect stays shallow (blocks 2–11).
   The ΔW table confirms it: layer 26's update is a big **near-rank-1** rewrite of
   blocks 25–27 (‖ΔW‖ ~1.2, srank ~1.1) vs layer 8's gentle ~0.35.

## Finding C — the grad probe predicted it (with a caveat)

Base-model gradients: REPA is **~orthogonal to FM at every depth** (max|cos| =
0.002 over 28 layers) — so the "deep = directionally redundant" reading is *not*
the mechanism. The depth signal is magnitude: ‖g_fm‖ **collapses ~4 orders**
toward the output (2.9e-2 @ L0 → 2.6e-6 @ L26), so FM barely constrains deep
outputs and REPA's push lands unopposed there → mag = ‖g_repa‖/‖g_fm‖ rises 31×
(0.010 @ L8 → 0.318 @ L26, peaking in the σ band). Caveat: mag is
activation-scale-confounded; it's *suggestive*, and the held-out FM + DoG-Gram
deltas above are the unconfounded confirmation.

## Verdict 4

- **Keep layer 8.** Confirmed safe and marginally positive; deep anchors
  (≥~16, and 26 specifically) are **net-negative** — over-optimize the aux loss by
  rewriting the output pathway. **Do not chase the CKA deep peak.**
- **Alignment is not the goal** — *cheap* alignment is. The right layer is one
  with real headroom (base align loss not ~0) that is shallow enough that reaching
  it doesn't rewrite the denoising path. Layer 8 sits there; layers 1–5 are
  already aligned (base loss ~0.06–0.10 → little to inject); ≥16 is the damage
  zone.
- **Within the repr regime (8–11) the optimum is untested.** Layers 9–11 have more
  DoG headroom (base 0.25–0.43) but ‖g_fm‖ is already ~3× weaker by L11 (more
  spillover risk). If a single cheap A/B is worth it, **layer 10** is the only
  candidate worth trying against 8 — but expect second-order gains: REPA is ~4% of
  total loss, so the anchor choice within 8–11 is a small lever. Higher-ROI levers
  are the σ-tapered weight (Result 3) and possibly **multi-layer** alignment
  (a few shallow-mid anchors at small weights) over any single deeper anchor.
