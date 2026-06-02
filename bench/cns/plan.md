# CNS — Colored Noise Sampling (arXiv 2605.30332) on Anima

Training-free SDE sampler. Replaces the **white** noise that `ERSDESampler` injects
each step (`library/inference/sampling.py:163-171`) with **frequency-colored**,
RMS-normalized noise that dumps the fixed stochastic-energy budget into the
frequency bands the network has **not yet resolved** at that step.

Source paper: "Colored Noise Diffusion Sampling", Davidson/Issachar/Benaim
(Hebrew U.). Project page https://hadardavidson.github.io/CNS/ . Local PDF at
repo root `2605.30332v1.pdf`.

## Why it might work on Anima — precondition already half-verified

CNS feeds on **spectral bias**: low-freq structure resolves early in sampling,
high-freq detail late. We have *independently measured this* on Anima:
`project_sigma_signal_resolves_by_045` (base resolves x0 by σ≈0.45; e_low
triples in the σ<0.45 tail). That finding **is** the CNS γ-matrix viewed from
the σ axis. So the premise survives contact with Anima — unlike Spectral
Guidance (no low-rank subspace → NO-GO). This probe confirms the staircase is
sharp enough *as a frequency-resolved matrix* before we touch the sampler.

## The mechanism (Algorithm 1, paper p.7)

Precompute a `[T, F]` **completion matrix** γ(f,t) once per model (Alg 2):
γ(f,t) = 1 − |X₀(f) − X_pred(f,t)|² / |X₀(f)|², from ODE trajectories. Then per
SDE step recolor the injected noise:

    scale = sqrt(1 - γ[t])              # Eq.11 numerator: energy → unresolved bands
    W     = fft2(white) * scale[freq_bins]
    w_c   = ifft2(W);  w_c /= std(w_c)  # RMS-normalize → conserve total variance
    x += drift*dt + sqrt(2*D)*w_c*sqrt(dt)

RMS/variance conservation (their whole §A) is load-bearing: over-inject → off
manifold; under-inject → mode collapse. It is a **zero-sum reallocation** of a
fixed budget, not a global noise scale-up.

## Precondition that IS met here: we run er_sde

CNS only acts on the **stochastic** path. Anima's CLI default is `--sampler
euler` (deterministic ODE → `er_sde is None` → no injected noise → CNS is a
literal no-op). But our production inference uses `--sampler er_sde` (and it
already composes with Spectrum), so CNS has a real surface. The fair A/B
baseline for any CNS result is **er_sde white-noise**, never the euler default.

## Compatibility with the other training-free plugins

Seams in the per-step loop (`generation.py:708-826`):

| Plugin       | Operates on                       | Conflict with CNS noise-recolor? |
|--------------|-----------------------------------|----------------------------------|
| **mod-guid** | model-forward AdaLN (t-embedding) | none — orthogonal seam. *But* it changes the v-trajectory, so γ should be recalibrated with mod-guidance ON if co-deployed (2nd-order). |
| **SMC-CFG**  | velocity/drift CFG combine (pre-`denoised`) | none — drift seam, CNS is the noise term. Genuinely independent. |
| **Spectrum** | model-forward (skip blocks on cached steps) | structurally clean (different seam) BUT real tension: Spectrum forecasts a *smooth* feature trajectory; recolored noise concentrates energy in **high-freq late**, exactly where the Chebyshev forecaster is weakest. We already tolerate er_sde white noise under Spectrum, and CNS conserves total variance, so expect ≤ that perturbation — but **A/B Spectrum cached-step error must not grow**. cf. `project_spectrum_er_sde_forecastability` (er_sde forecasts 1.6× worse than euler). |
| **DCW**      | post-step LL-band x-space correction | watch, not block: CNS already injects *less* energy into resolved LL; DCW then bias-corrects LL. Could partially double-count. |

## Phases

- **Phase 0 — DONE, GO (2026-05-31).** Base 1024², 8 trajectories, cfg=1:
  staircase σ50 spread **+0.410** (low-freq σ≈0.79 → high-freq σ≈0.38),
  linear-target MAE 0.101, aggregate σ50 **0.486** (independently re-derives
  `project_sigma_signal` ≈0.45). γ heatmap = paper Fig. 3 on Anima. Result:
  `bench/cns/results/20260531-1625-base-1024/`. Full sweep (4 configs, n=8):
  | config | spread | agg σ50 | linear MAE |
  |---|---|---|---|
  | base 1024 cfg1 | +0.410 | 0.486 | 0.101 |
  | LoRA 1024 cfg1 | +0.409 | 0.490 | 0.100 |
  | base portrait cfg4 | +0.521 | 0.394 | 0.144 |
  | LoRA portrait cfg4 | +0.521 | 0.398 | 0.142 |
  → **LoRA transparent** (γ reusable across adapter on/off); **CFG×aspect sharpens**
  the staircase (γ needs per-(CFG×aspect) calibration, but deploy config has MORE
  bias). cfg+aspect varied together — disentangle in Phase 1.
- **Phase 0 (this dir) — `gamma_probe.py`, read-only.** Capture ODE (euler)
  trajectories on the base model via a wrapped Euler step, compute γ(f,t),
  radial-bin, and report whether the low-freq-early/high-freq-late staircase is
  sharp. GO gate: clear t50 spread across frequency + strong deviation from the
  linear γ=1−σ target + aggregate-γ crossing near σ≈0.45 (consistency with
  `project_sigma_signal_resolves_by_045`). Also emits a `beta_preview` matrix =
  the exact per-step colored scale CNS would apply, so Phase 1 wiring is drop-in.
  - Run once on **base**, once on the **LoRA** (does the adapter flatten the
    bias?), and at **cfg=1 and cfg=4** (DCW taught us CFG shifts the spectrum).
  - Per-aspect: γ likely needs per-bucket calibration (mirror DCW_ASPECT_BUCKETS).
- **Phase 1 — WIRED (2026-05-31).** Recolors `ERSDESampler._sample_noise` behind
  `--cns <path|auto>` (+ `--cns_strength` safety blend). Pieces:
  - **Completion matrix** = `bench/cns/calibrate.py`: drives the Phase-0 euler
    capture across the **deploy config** — `--cfg 4.0`, top-`--n_aspects 3`
    `DCW_ASPECT_BUCKETS` ((1200,896),(1344,800),(896,1200), all (H,W)) — and
    bundles per-aspect γ[A,T,F] into `networks/calibration/cns_gamma.npz` (the
    artifact `--cns auto` loads). γ measured from the deterministic euler ODE at
    cfg=4.0 (Alg. 2); recoloring then applies on er_sde. `gamma_probe.py` stays
    the single-config Phase-0 staircase check; calibrate.py reuses its helpers.
    Prompts are **real captions** (richest/most-tags across distinct artists,
    sampled from preprocessed stems → image_dataset .txt), not synthetic — they
    matter: real vs synthetic β MAD **0.036** (agg σ50 0.50 vs 0.41), ~3–5× the
    cross-aspect MAD.
  - **Single averaged γ, shipped under `networks/calibration/cns_gamma.npz`**
    (not models/). Cross-aspect variation is **cosmetic** (β MAD ~0.01, mean
    dev-from-avg 0.008 — same conclusion as `project_dcw_bucket_prior_cosmetic`),
    so `--average_aspects` (default on) writes one `(1,T,F)` γ; the recolorer's
    nearest-aspect select degrades to index 0, serving any resolution. Radial
    bins are normalized [0,1] → resolution-agnostic by construction. Caveat: all
    calibrated aspects are 4200-token AR 0.6–1.34; extreme shapes are unmeasured
    extrapolation. Per-aspect γ kept in the results/ bench record for audit.
  - **Runtime** = `library/inference/corrections/cns.py::CNSRecolorer`: locks
    onto the calibrated aspect nearest in AR to the latent shape, **σ-interpolates**
    the γ row each step (robust to step-count / flow_shift mismatch vs calib),
    `scale=sqrt(1−γ)[bin]`, fft2 × scale, ifft2, per-channel RMS-renorm. Verified
    unit-variance + HF-energy gain at mid-σ on synthetic γ.
  - **A/B TODO**: **CMMD** (live val — ImageNet FID deltas don't transfer), read
    grids not just the number (pose-blind metric lesson). `--sampler er_sde --cns
    auto` vs er_sde white. Compose-test with er_sde∘Spectrum (cached-step error
    must not grow — see Spectrum row above).
  - **Calibration is GPU-heavy** (generates 3 aspects × prompts × seeds at cfg=4);
    not run yet — `python bench/cns/calibrate.py --cfg 4.0 --n_aspects 3`. It
    **loads the DiT once + precomputes all text + frees the TE** (TE→free→DiT
    invariant) and **compiles by default** (`--compile`/`--no-compile`): the 3
    default aspects are all token-count 4200, so `compile_blocks` builds a single
    native-flatten graph reused across them. The `_StepCapture` monkeypatch wraps
    the eager *sampler* step, outside the compiled DiT region — orthogonal.
- **Shelve cheaply** if Phase 0 staircase is flat (e.g. LoRA flattens it).

## Artifacts (per run, under `bench/cns/results/<ts>[-label]/`)
- `gamma.npz` — `gamma[T,F]`, `beta_preview[T,F]`, `sigmas[T+1]`, `timesteps[T]`,
  `radial_centers[F]`, `gamma_linear_target[T,F]`.
- `gamma_matrix.png` — γ + β heatmaps + t50-vs-freq staircase.
- `result.json` — standard bench envelope (metrics: t50 spread, linear-target MAE,
  aggregate-σ50).
