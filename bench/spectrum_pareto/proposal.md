# Spectrum Benchmark Redesign Proposal

## Context

The archived Spectrum benchmark under `_archive/bench/spectrum` produced a
misleading Pareto ranking: `dpmpp_sde + linear_quadratic` looked best, Heun was
ranked highly in some slices, and `linear_quadratic` looked broadly safe. Real
image checks contradicted that: `linear_quadratic` degraded output quality,
`dpmpp_sde` is not useful for Spectrum's speed goal because it needs extra
denoiser evaluations, and Heun was a poor practical choice.

The core problem is not that the drift math was useless. It answered the wrong
question. It measured how robust a sampler/scheduler is to injected forecast
error, but Spectrum needs the best image quality per actual DiT cost.

## Diagnosis

The old benchmark conflated four different effects:

1. Base sampler/scheduler quality.
   A scheduler can be robust to Spectrum error but still produce worse images
   when run without Spectrum. That appears to be the `linear_quadratic` failure.

2. Error cleanup bought by extra evaluations.
   Heun and DPM++ SDE style samplers can suppress injected error because their
   corrector or midpoint calls re-query the model. That should be charged as
   compute, not treated as a free robustness advantage.

3. Step count versus effective model cost.
   Spectrum optimizes actual full DiT forwards. A candidate with the same
   denoising step count but more internal denoiser calls is not on the same
   speed frontier.

4. Local forecast residual versus final image quality.
   Relative L2 residual and toy drift are useful for shortlisting, but lower
   residual does not guarantee better Anima samples. They need a final paired
   image gate.

## Correct Objective

The benchmark should optimize a two-axis Pareto frontier:

```text
maximize: image quality and baseline faithfulness
minimize: effective DiT forward cost and wall time
```

A candidate is a tuple:

```text
sampler, scheduler, infer_steps, Spectrum schedule, Spectrum predictor params
```

Where Spectrum schedule includes:

```text
warmup_steps, window_size, flex_window, stop_caching_step, optional adaptive gate
```

And predictor params include:

```text
m, lam, w, calibration_strength
```

The score must separate:

```text
Q_base(candidate) = quality(all_actual_candidate vs current default baseline)
Q_spec(candidate) = quality(Spectrum_candidate vs all_actual_candidate)
C(candidate)      = effective full-DiT-equivalent cost and measured wall time
```

Do not call a candidate Pareto optimal unless `Q_base` passes first. This would
have caught the `linear_quadratic` issue.

## Effective Cost Model

Use effective NFE instead of denoising steps.

```text
effective_cost =
  full_forward_count * full_forward_cost
  + cached_forward_count * fast_path_cost
  + internal_denoiser_calls * full_forward_cost
```

For in-tree `networks/spectrum.py`, cached steps still run:

```text
t_embedder + final_layer + unpatchify
```

So cached steps are not free. Estimate their cost once by timing forced cached
fast-path calls and normalize by full DiT forward time.

For ComfyUI sampler sweeps, every corrector, midpoint, or substep denoiser call
must be counted. If the call runs the full model, it counts as a full forward.
If Spectrum wraps that call and caches it, record whether it was full or cached.

## Proposed Analytical Bench

### Phase 1: Base Sampler Gate

Run all-actual generation with Spectrum disabled.

Compare each sampler/scheduler candidate against the current production default
using fixed prompt/seed pairs.

Record:

- final image grid
- CLIP/DINO/LPIPS-style paired distance where available
- latent endpoint distance
- manual pass/fail notes for visible artifacts
- wall time and full model calls

Reject any scheduler/sampler whose all-actual images are worse before running
Spectrum tuning. This keeps robustness metrics from blessing a bad base path.

### Phase 2: Shadow-Forward Spectrum Bench

Run actual Spectrum generation, but on cached steps also compute the hidden
actual DiT output for measurement only.

Do not feed the hidden actual output back into the sampler. The sampled latent
trajectory must remain the real Spectrum trajectory.

At each cached step, record:

```text
noise_pred_actual_shadow
noise_pred_spectrum_used
denoised_actual_shadow
denoised_spectrum_used
feature_residual = final_layer_input_actual - final_layer_input_predicted
latent_before_step
latent_after_step
sigma
step_index
```

This measures the true forecast error on the off-trajectory latents that
Spectrum actually created, not only on an all-actual replay trajectory.

### Phase 3: Step Sensitivity Weighting

Raw residual mean is not enough. Estimate how damaging each step's residual is.

For a small calibration subset:

1. Run the all-actual trajectory.
2. Inject a normalized residual at one step.
3. Finish denoising.
4. Measure final latent and image drift.

This gives a per-step sensitivity curve:

```text
sensitivity[i] = final_damage_per_unit_error_at_step_i
```

Use it to score Spectrum settings:

```text
weighted_error = sum_i sensitivity[i] * observed_forecast_error[i]
```

This should replace the old mean residual and endpoint-only toy drift as the
main analytical shortlist metric.

### Phase 4: Pareto Selection

For each accepted base sampler/scheduler and Spectrum setting, report:

- effective NFE
- wall time
- full forward count
- cached fast-path count
- internal denoiser call count
- weighted forecast error
- final latent drift versus all-actual candidate
- image metrics versus all-actual candidate
- image metrics versus current production baseline

Then compute the nondominated set:

```text
candidate A dominates B if:
  A has no worse quality,
  A has no higher effective cost,
  and A is strictly better on at least one axis.
```

## Recommended Search Space

Start narrow. Broad sampler sweeps created false confidence.

Primary in-tree candidates:

- `euler`: main Spectrum target because it has one model call per denoising step.
- `er_sde`: compare only if real images justify its stochastic behavior and
  effective cost.

Low priority:

- `lcm`: only for distilled few-step models, not normal Anima Spectrum tuning.

ComfyUI-only broad sampler tests should be treated as diagnostic, not default
selection. Any sampler with extra denoiser calls must prove that its quality
gain beats the lost Spectrum speedup.

Scheduler search should be gated by all-actual image quality before robustness:

1. official/current flow-shift schedule
2. simple or production-equivalent schedules
3. experimental schedules only if they pass the base quality gate

Do not promote `linear_quadratic` based only on drift robustness.

## Better Direction: Adaptive Cache Policy

A fixed `window_size` and `flex_window` is weaker than an online confidence
policy.

On each actual step:

```text
pred_before_update = forecaster.predict(i)
actual = captured_feature
residual = actual - pred_before_update
update running error model
update forecaster with actual
```

On each candidate cached step:

```text
expected_damage = predicted_error * sensitivity[i]
if expected_damage < threshold:
    cache
else:
    run actual forward
```

This uses the analytical bench to learn where Spectrum is safe rather than
forcing one global cache rhythm across all prompts and all sigma regions.

Expose this as an optional mode:

```text
--spectrum_policy fixed      # current behavior
--spectrum_policy adaptive   # confidence-gated caching
--spectrum_error_budget 0.0x # target weighted-error budget
```

The fixed policy should remain for reproducibility and comparison.

## Implementation Tracks

### Track 1: Measurement Harness

Create a new live benchmark outside `_archive`, for example:

```text
bench/spectrum_pareto/
```

Minimum scripts:

```text
bench/spectrum_pareto/base_gate.py
bench/spectrum_pareto/shadow_forward.py
bench/spectrum_pareto/analyze_pareto.py
```

The harness should output the standard `result.json` envelope used by other
bench scripts, plus CSV summaries and image grids.

### Track 2: Instrument Spectrum Cost

Add optional counters around `spectrum_denoise()`:

```text
full_forward_count
cached_forward_count
fast_path_time_ms
full_forward_time_ms
effective_nfe
```

Keep this instrumentation disabled by default or behind a benchmark flag.

### Track 3: Shadow Actuals

Add a benchmark-only mode that computes hidden actual outputs on cached steps.
This should live in the bench harness or a private runner, not in normal
inference defaults.

Important invariant:

```text
shadow actuals are recorded only; they never update latents or forecasters
except through explicitly measured diagnostics.
```

### Track 4: Sensitivity Calibration

Build a small step-sensitivity estimator over a fixed prompt/seed set. Cache the
result by:

```text
model sha, image size, infer_steps, sampler, scheduler, guidance_scale
```

This lets future sweeps reuse the expensive sensitivity curve.

### Track 5: Adaptive Policy Prototype

Implement adaptive caching after the bench can prove where fixed schedules fail.
The first version can use a simple exponential moving average of observed
forecast residuals and a precomputed step-sensitivity curve.

## Acceptance Criteria

The redesigned benchmark is useful only if it prevents the previous false
ranking.

It should be able to state:

1. Whether a scheduler is good without Spectrum.
2. Whether Spectrum damages that scheduler beyond an acceptable threshold.
3. How many full-DiT-equivalent forwards the candidate really costs.
4. Whether a sampler's robustness is real or bought by extra model calls.
5. Which candidates remain nondominated after quality gates and cost accounting.

Do not update production defaults from analytical metrics alone. The final
default change still needs paired real-image confirmation on the accepted
frontier.

## Validation Notes

This proposal is based on static review of:

- `_archive/bench/spectrum/README.md`
- `_archive/bench/spectrum/analyze_drift.py`
- `_archive/bench/spectrum/replay_forecaster.py`
- `_archive/bench/spectrum/results/20260524-2031-faithful-samplers/`
- `docs/methods/spectrum.md`
- `networks/spectrum.py`
- `library/inference/generation.py`
- `library/inference/sampling.py`
- `inference.py`

No new runtime benchmark was executed while writing this proposal.
