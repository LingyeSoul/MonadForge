# bench/spd — Spectral Progressive Diffusion precondition tests

[Spectral Progressive Diffusion](https://howardxiao.ca/speed/) (SPD, Xiao et al.,
arXiv:2605.18736) accelerates DiT/flow-matching inference by growing spatial
resolution along the denoising trajectory — running early (noise-dominated)
steps at low resolution and injecting high-frequency detail via spectral noise
expansion only when finer frequencies emerge from noise.

The whole δ-optimal resolution schedule is derived from one empirical claim
(Eq. 4): the latent **power spectrum decays as a power law**,
`P_ω ∝ |ω|^{-β}` with `β ∈ [2,3]` (FLUX latents: β≈1.92). SPD only pays off if
high frequencies genuinely carry far less signal than low frequencies. Anima
trains on anime/illustration data — flat color fills + hard line art — whose
spectral statistics are not obviously natural-image-like, so this needed
checking on *our* VAE + *our* data before any integration work.

## `measure_latent_spectrum.py` — does Eq. (4) hold for Anima latents?

Encodes `image_dataset/` **originals** (not the resized cache — resizing is a
low-pass that would bias the HF tail) through the Anima VAE via the exact
training transform, computes the radially-averaged 2D-FFT power spectrum of the
per-channel-standardized latents, and fits β over a mid-frequency band.

```bash
uv run python -m bench.spd.measure_latent_spectrum --num_images 200 --max_side 1536
```

Drops `radial_profile.csv`, `spectrum.png`, and a `result.json` envelope with
β / R² / per-channel spread + a verdict into `results/<ts>-<label>/`.

### Result (2026-05-20, 200 images @ 1536px, 16 latent channels)

**β = 2.26, R² = 0.9994** over k∈[0.06, 0.5]·Nyquist; per-channel
2.26 ± 0.23 (range [1.95, 2.71]). A near-perfect power law across the whole
spectrum, squarely inside the paper's [2,3] range. **The SPD spectral premise
holds for Anima** — the flat-fill mass dominates the low-frequency end enough to
keep the decay clean despite the line-art HF energy. The very-high-k tail
(k>0.7) flattens to a noise floor ~3 decades below DC, consistent with the
paper's "high frequencies are noise-dominated" picture.

## `per_artist_spectrum.py` — is beta robust across styles?

Fits beta *within each artist folder* (the aggregate pools all styles, and
cel-shaded-flat vs painterly-detailed could behave differently).

```bash
uv run python -m bench.spd.per_artist_spectrum --n_artists 30 --per_artist 12
```

### Result (2026-05-20, 30 artists × 12 images @ 1280px)

**beta = 2.26 ± 0.08 across 30 artists**, range [2.06, 2.47], **30/30 inside
[2,3]**, every R² > 0.99. The exponent is essentially a property of the VAE
latent space, not of any particular style — per-artist spectra nearly overlap.
SPD's spectral premise is robust across the dataset; no per-style schedule
adaptation needed. Flattest/least-HF: `wagashi`, `fizz`, `ootomo_takuji`
(β≈2.06–2.14); most-detailed: `ama_mitsuki` (β≈2.47).

## `probe_lowres_denoise.py` — does the bare DiT denoise at low res? (Phase 2, the real go/no-go)

The static spectrum (above) only says HF *carries less signal*. Training-free SPD
additionally needs the **bare DiT to denoise a lower-resolution latent** for the
early steps, then accept a spectral-noise-expansion handoff to full resolution.
Anima trains only at the ~4096-token bucket and pins one static inference shape,
so this was the real integration risk — if the model mushes at low res,
training-free SPD is dead.

The probe loads the bare DiT (no LoRA) eager/dynamic-shape (no `torch.compile`),
encodes one prompt, and runs from the same seed noise a full-res Euler baseline
vs an SPD variant (DCT low-pass init → low-res steps → spectral expansion Eq. i–iii
+ timestep alignment Eq. 5–6 → full-res finish). The SPD math mirrors the
community `SamplerSPEED` node (`../comfy/custom_nodes/comfyui-speed/`), so
`--community` reproduces the exact `GJ5Rt3Xz` workflow schedule.

```bash
uv run python -m bench.spd.probe_lowres_denoise                  # single-stage 0.5×, handoff at 30% of steps
uv run python -m bench.spd.probe_lowres_denoise --community      # exact GJ5Rt3Xz schedule (0.5→0.75→1.0 @ σ 0.8/0.6)
uv run python -m bench.spd.probe_lowres_denoise --stages 0.5 0.75 1.0 --transition_sigmas 0.8 0.6
```

Verdict is **visual** (saves `baseline_*.png` / `spd_*.png` / `compare_*.png`
montages); auto-metrics in `result.json` flag only *hard* divergence (NaN/Inf,
latent-std blow-up, sharpness collapse/grain) — they cannot certify coherence.

### Result (2026-05-20, 2 seeds × 2 schedules @ 1024², CFG=4, 28 steps)

**COHERENT — PASS.** Both schedules denoise with no instability (std ×0.95, no
NaN); all montages show intact subjects/anatomy and **no smear or double-image at
the handoff**. SPD output runs sharper/higher-contrast and diverges
compositionally from baseline (fresh HF noise at expansion). The single-stage
2-stage schedule rendered sign text more legibly than the community 3-stage —
the `GJ5Rt3Xz` schedule is **viable but not obviously optimal**. Training-free SPD
is alive on Anima. → Phase 3 (integration + speed/quality bench vs Spectrum/Turbo).

### Schedule sweep — tuning the `GJ5Rt3Xz` workflow (2026-05-20, 832×1216, 2 seeds, with wall-clock)

Swept 5 schedules to revise the community `SamplerSPEED [0.5, 0.75, 0.8, 0.6]` for a
better speed/quality knee. All coherent (text legible both seeds); no hard divergence.

| schedule (stages @ σ) | speedup | lowfreq_mse | notes |
|---|---|---|---|
| `0.5/0.75/1.0 @ 0.8/0.6` (community) | ×1.43 | 0.530 | transitions too early — wastes low-res budget |
| `0.5→1.0 @ 0.7` (single-early)       | ×1.32 | 0.525 | slowest; one handoff, high σ |
| **`0.5→1.0 @ 0.5` (single-late)**    | **×1.65** | **0.523** | **recommended knee** — one handoff, simplest, cleanest |
| `0.5/0.75/1.0 @ 0.7/0.5` (2stage-late) | ×1.61 | 0.524 | minimal-change drop-in (keeps community shape) |
| `0.4/0.7/1.0 @ 0.7/0.5` (aggressive) | ×1.73 | 0.563 | fastest; more divergence/variance |

**Finding:** the community `0.8/0.6` thresholds expand too early (most steps end up
full-res). Moving the handoff later (σ≈0.5) buys ~15–20% more speed at equal quality.
A *single* handoff (single-late) matches a 2-stage ramp in quality while being simpler
and faster — fewer spectral-noise injections. SPD-on-Anima also runs consistently
**sharper/higher-contrast** than baseline (sharpness ×1.5–2.6) — a behavioral signature,
not a transparent speedup. Recommended `SamplerSPEED` widgets (start, mid, t1, t2):
**`[0.5, 0.5, 0.99, 0.5]`** (single handoff 0.5→1.0 @ σ0.5). Caveat: even tuned (×1.73
max), standalone SPD is slower than the Spectrum node we already ship (~3.75×) — the
real question is SPD∘Spectrum (Phase 3), untested.

## What this does NOT yet test (separate preconditions for SPD)

1. **Autoregression *dynamics* (Fig 2b):** that low frequencies actually resolve
   early in *Anima's* denoising trajectory. Needs DiT forwards — a per-step
   `x_t` spectral-emergence probe. (Not yet built — Phase 1.) This is what would
   let the δ-optimal schedule be *derived* (Prop 1/2) instead of hand-tuned like
   the `GJ5Rt3Xz` knobs.
2. ~~Resolution generalization~~ — **tested 2026-05-20, PASSES** (see above).
