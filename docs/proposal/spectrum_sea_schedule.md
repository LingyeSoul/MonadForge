# Spectrum scheduler: SEA-distance trigger (SeaCache metric, Spectrum reuse)

Status: **P1 WIRED + prompt-generalization MEASURED (2026-06-22). Predictive
validation DONE (`docs/findings/seacache_sea_decision_metric.md`: SEA-filtered
input distance ranks the true counterfactual skip-cost at ρ +0.51 vs raw −0.36
and lagging-residual +0.16). End-to-end CMMD A/B NOT yet run — the ship gate
below.**

**P1 measurement (`bench/spectrum_sea/prompt_generalization.py`, 16 real
post_image_dataset captions, 28 steps, cfg 4, matched compute):**
- **δ is highly portable.** Calibrating δ on one prompt lands all 16 within ±1
  of the target forward count; the per-prompt *ideal* δ has cv 0.04. The
  single-key disk cache (keyed on steps/warmup/stop/refresh_ratio/cfg/sampler/
  H/W — **not** prompt) is sound.
- **Content-adaptivity is real but modest:** ~15% of decision steps differ by
  prompt at matched compute, concentrated in mid-low σ where content forms. The
  σ-trend dominates the schedule; SEA is mostly a fixed reschedule with a ~15%
  per-prompt wiggle.
- **⚠ Reallocation vs the window is negligible at matched compute** (σ<0.45 tail
  60% vs 60% actual; mid+ 60% vs 57%). The "spends tail forwards on mid-σ" pillar
  below was an artifact of an unequal `stop_caching_step` between arms — at equal
  stop, SEA does not move compute out of the tail. The tail/mid split is governed
  by `stop_caching_step`, **orthogonal to the SEA metric.** Expect P2 to be a
  near-tie; the real question is whether the modest mid-σ reshuffle helps CMMD.
- **Bug fixed:** the hard-coded `refresh_ratio = 0.62` (the proposal's *24-step
  total* fraction) over-computed by +22% at 28 steps when applied to the decision
  region. Auto-δ now defaults its target to the window schedule's *own* decision
  refresh fraction at the live geometry (`_window_decision_fraction`), so the SEA
  arm is matched-compute by construction at any step count.

## What changes (and what doesn't)

Replace only Spectrum's **when-to-skip** decision — the content-blind growing
window — with SeaCache's **accumulate-SEA-distance-until-δ** rule. Keep everything
else: Chebyshev block-feature forecasting, the head reconstruction
(`_spectrum_fast_forward`), warmup/stop forcing, and — critically — the per-step
`noise_pred` reconstruction that keeps SMC-CFG / mod-guidance / DCW composing
(the reuse path is untouched, so the plugin boundary is unaffected).

Today (`networks/spectrum.py`, `spectrum_denoise`):

```python
# decision — lines ~305-309
if i < warmup_steps or i >= stop_at:
    actual = True
else:
    actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0
# advance — lines ~384-389
if i >= warmup_steps:
    curr_ws = round(curr_ws + flex_window, 3)
```

Proposed (SeaCache Eq. 4/8 — accumulate the SEA-filtered relative-L1 distance of
the *input* latent across steps, refresh when it crosses δ):

```python
# once per step, before the decision — x_t == `latents` is available pre-forward
sea_now = _sea_filter(latents[:, :, 0], float(sigmas[i]))      # (B,C,H,W)
if sea_prev is not None:
    sea_accum += _l1rel(sea_now, sea_prev)                     # Eq. 3 distance
sea_prev = sea_now

if i < warmup_steps or i >= stop_at:
    actual = True
elif schedule == "sea":
    actual = sea_accum >= delta
else:                                                          # legacy window
    actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0

# on an actual forward: sea_accum = 0.0   (reset the accumulator)
```

`_sea_filter` / `_l1rel` are the already-written + math-verified helpers in
`bench/spectrum_sea/sea.py` (σ-dependent Wiener gain, unit-mean normalized). Ship
step promotes them to a library home (e.g. `networks/spectrum_sea.py`) so both the
runner and the ComfyUI node import them; ~40 lines, no new deps. Cost is one
FFT/iFFT on a 16×128×128 latent per step — negligible vs a block forward, **zero**
extra DiT forwards. CFG is irrelevant to the decision: x_t is shared across
cond/uncond, so one accumulator drives both branches.

## The δ knob (the one real design question)

SeaCache exposes δ as the latency/quality dial. Accumulated relative-L1 distance
is roughly scale-stable across resolution (the L1rel normalization) but still
depends on step count, so δ wants per-config calibration. Two surfaces:

- `--spectrum_delta <float>` — explicit, for sweeps.
- `--spectrum_delta auto --spectrum_refresh_ratio 0.62` — **default**: on the
  first generate, dry-run the accumulator over the schedule and binary-search δ so
  the post-warmup refresh fraction matches the target (0.62 ≈ what the current
  window spends at 24 steps, from the bench). This makes the SEA arm a
  *like-for-like* swap at matched compute — exactly what the A/B needs — instead of
  a free speed/quality re-pick.

Keep `schedule="window"` as the default until the A/B clears; `"sea"` is opt-in
(`--spectrum_schedule sea`). `warmup_steps` / `stop_caching_step` still force the
early and final steps under both. `window_size` / `flex_window` go unused in SEA
mode (leave them for the legacy path).

## Why this is the right cut

- **Validated against the real objective**, not a proxy: the SEA distance predicts
  true injected x̂₀ error; raw distance is *actively misleading* (anti-correlated),
  so this isn't a no-op rename.
- **Orthogonal to the plugin boundary.** The findings doc shows SeaCache-as-reuse
  would desync SMC-CFG/mod-guidance; SeaCache-as-*decision* touches neither — the
  forecast+head reconstruct still runs every step.
- ~~**A concrete reallocation exists.**~~ **(REFUTED at matched compute,
  2026-06-22.)** The hoped-for reallocation — SEA stops triggering in the σ<0.45
  tail and spends those forwards on mid-σ — does **not** materialize when both
  arms use the same `stop_caching_step`: tail actual-rate is 60% for both. The
  apparent reallocation was a `stop_at` mismatch artifact. Skipping the tail is
  governed by `stop_caching_step`, not the SEA metric. What remains is a ~15%
  per-prompt mid-σ reshuffle — the only thing P2 can actually test.

## Phases

- **P1 — wire it (small).** Promote `sea.py` → `networks/spectrum_sea.py`; add
  `schedule` / `delta` / `refresh_ratio` to `spectrum_denoise` + the `--spectrum_*`
  CLI flags; mirror into the ComfyUI node (`ComfyUI-Spectrum-KSampler/spectrum.py`,
  `SpectrumState.should_cache`). Off by default.
- **P2 — ship gate (A/B).** Same prompts/seeds, `schedule=window` vs `sea` with δ
  auto-tuned to matched forward count, render and score by **CMMD**
  (`project_cmmd_val_signal`) vs the full-compute reference — **not** FM-MSE
  (`project_fm_val_loss_uninformative`). Flip the default only on a CMMD win or
  tie-at-lower-variance. Reuse `bench/spectrum_sea/` for the harness.
- **P3 — (optional) β tune.** All validation used the natural-image power-law
  β = 2, untuned; a quick β sweep at fixed δ if P2 is borderline.

## Risks / watch

- δ portability across step-count / resolution — the `auto` refresh-ratio mode is
  the mitigation; never ship a hard-coded δ.
- Variable per-prompt refresh count (a feature for latency, a confound for
  benchmarking) — the A/B must match forward count per arm, hence δ-per-arm tuning.
- If P2 shows no CMMD win at matched compute, this stays an off-by-default option
  and the finding is "predictive-but-not-end-to-end" — still worth keeping the
  validated metric documented.

Refs: `docs/findings/seacache_sea_decision_metric.md` (validation + the composition
argument), `bench/spectrum_sea/` (`sea.py`, `run_bench.py`, `phase1_counterfactual.py`),
SeaCache arXiv:2602.18993v2 §4 (Eq. 3/4/8).
