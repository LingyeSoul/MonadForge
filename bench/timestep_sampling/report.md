# Timestep-sampling σ-signal probe — report

**Question.** Anima trains LoRA with `timestep_sampling = "sigmoid"` and
`discrete_flow_shift = 1.0` (`configs/base.toml`). Are these pointed at the σ
region where an adapter actually has something to learn, or inherited defaults?

**TL;DR.** `discrete_flow_shift` is a *no-op* under sigmoid sampling, so the
only live lever is *where the sampling density sits on the σ axis*. On a
full-res **latent-MSE** basis the bare base DiT already reconstructs the target
well for **σ ≲ 0.55**, and the default sigmoid schedule spends **~58% of its
training draws below σ 0.55**. (An earlier draft put the crossover at σ0.75 /
86% — that used a 96px-downsampled *pixel*-MSE whose extra low-pass inflated the
dead zone; latent-MSE is the honest lens and roughly halves the "wasted"
fraction.) On a *content-reconstruction* basis the schedule still over-samples
where the frozen base needs little help — but this probe cannot see
*style/identity* signal (see caveat 2), so it is a hypothesis generator for a
CMMD sweep, not a verdict. Don't change `base.toml` off this alone.

---

## What `sigmoid` + `flow_shift=1.0` actually are

- `timestep_sampling="sigmoid"` with `sigmoid_scale=1.0` draws
  `σ = sigmoid(𝒩(0,1))` — a **logit-normal(0,1) over σ, bell centered at σ=0.5**
  (`library/runtime/noise.py:102`). This is precisely the SD3-recommended
  mid-noise emphasis; a sensible inherited default.
- `discrete_flow_shift` is read **only** by the `"shift"` branch
  (`noise.py:110`) and the inference scheduler. Under `"sigmoid"` it never
  touches the training draw. `1.0` is also the identity transform. So as
  configured it is inert at train time — neither optimal nor suboptimal, just
  unused. (To bias the schedule you must change `sigmoid_scale`, switch to the
  `logit_normal` weighting path so `logit_mean` goes live, or add a
  `t_min`/`t_max` window — `noise.py:137`.)

## Method (`probe_sigma_signal.py`, no training)

For 16 real cached dataset latents (`post_image_dataset/lora/**`, each paired
with its cached `crossattn_emb`), at each σ on a 10-point grid — over
`--num_seeds` independent noise draws, averaged (default 3; the table below is
the original single-seed run, see caveat 1) — do a bare-DiT forward and
reconstruct the model's x0 estimate:

```
x_σ      = (1-σ)·x0 + σ·ε            # Anima FM noising (noise.py:164)
v        = DiT(x_σ, σ, crossattn)    # σ∈[0,1] is the time arg (generation.py:296)
x0_pred  = x_σ − σ·v                 # target velocity = ε − x0 (train.py:922)
```

Decode `x0_pred` and measure how far it is from the true x0. Low error ⇒ the
base already knows the answer at that σ (nothing to learn); high error ⇒ the
base is uncertain (structure being decided — where capacity could help).
Overlay each candidate schedule's sampling density on that error curve.

## Results (n=16, RTX 5060 Ti, `results/20260526-1538-base/`)

| σ | FM-MSE (v) | x0 latent-MSE | x0 pixel-MSE |
|------|-----------|---------------|--------------|
| 0.05 | 0.293 | 0.0007 | 0.00002 |
| 0.15 | 0.123 | 0.0028 | 0.00008 |
| 0.25 | 0.080 | 0.0050 | 0.00019 |
| 0.35 | 0.064 | 0.0079 | 0.00037 |
| 0.45 | 0.058 | 0.0117 | 0.00069 |
| 0.55 | 0.056 | 0.0169 | 0.00125 |
| 0.65 | 0.058 | 0.0244 | 0.00223 |
| 0.75 | 0.066 | 0.0370 | 0.00428 |
| 0.85 | 0.081 | 0.0582 | 0.00868 |
| 0.95 | 0.128 | 0.1157 | 0.02589 |

**x0-recoverability degrades monotonically with σ.** The headline metric is the
**latent-MSE** column (full VAE-latent resolution). Normalized to its max (at
σ0.95) it stays under 0.2 through σ0.55 (0.006→0.146), crosses ~0.21 at σ0.65,
then climbs to 0.32 / 0.50 / 1.0 at σ0.75 / 0.85 / 0.95. The pixel-MSE column
is shown for context but is **not** the headline: it's computed on a 96px
downsample, an extra low-pass that flattens the curve and pushes its <0.2
crossover up to σ0.75 — overstating the low-σ dead zone (see caveat 3).

**FM-MSE (velocity error) is U-shaped**, min ≈0.056 at σ0.55, high at both ends.
Velocity is hardest to predict at the extremes (near-clean: tiny target, large
relative error; near-noise: little conditioning signal). This is *not* the same
as x0-recoverability — don't conflate "velocity is hard to predict" with "the
image isn't determined yet." The x0 view is the relevant one for "where can an
adapter add information."

**Schedule vs signal mismatch** — fraction of each schedule's draws spent below
σ0.55 (where the base reconstructs x0 within <0.2 of max latent-MSE ≈ wasted):

| schedule | mass below σ0.55 |
|---|---|
| `sigmoid ∘ t_max=0.95` | **62.5%** |
| **`sigmoid` (default)** | **57.8%** |
| `uniform` | 55.0% |
| `logit_normal μ=+0.5` (high-σ skew) | 38.3% |

The default sigmoid (blue in `density_overlay.png`) peaks at σ0.5 — inside the
dead zone — and only its right slope reaches the high-error tail. A
positive-mean logit-normal (green) shifts the peak to ~σ0.7, onto the rising
error, and is the only arm spending a minority of draws in the dead zone.
`t_max=0.95` is **worse**: it compresses everything below σ0.95 (a rescale, not
a clip), pushing yet more mass down into the dead zone — counterproductive. The
direction is unchanged from the earlier px-MSE numbers; only the magnitude
shrinks (a majority, not the ~86% the low-passed metric implied).

## Interpretation

On a content-reconstruction basis the default schedule spends the large
majority of training draws at σ where the frozen base already reproduces the
target. This is consistent with T-LoRA's prior (full rank at high noise, rank→1
near clean — `docs/methods/timestep_mask.md`) and with the SPD σ_resolve table
(coarse structure locks in early; `bench/spd/plan.md`). A high-σ-skewed
schedule puts proportionally more capacity where the base is uncertain.

## Caveats — read before acting

1. **n=16 fixes the shape, not the exact crossover; this run was single-seed.**
   The flat-then-steep curve is stable. The latent-MSE `<0.2·max` crossover is
   σ0.55 here; treat it as a soft boundary (the threshold is a heuristic to
   quantify the eyeball, not a hard line). The numbers above were recomputed
   from this run's latent-MSE column — the probe now defaults to `--num_seeds 3`
   (per-prompt seed variance dominates single-draw labels,
   `project_dcw_seed_variance_dominates`), so the *next* refresh will be
   seed-averaged and may nudge the crossover.
2. **This sees *content*, not *style* — the load-bearing caveat.** The probe
   uses dataset images whose content the base can already reconstruct. But the
   reason to train a style/identity LoRA is to change **low-σ rendering**
   (texture, line, palette) even where content latent-MSE is ≈0. So "low σ =
   nothing to learn" holds for content, **not** for style adaptation —
   skewing draws away from low σ could *hurt* a style LoRA. This is why the
   probe cannot decide the schedule.
3. **Even latent-MSE under-weights fine detail.** It's full-res in VAE-latent
   space (not the 96px pixel downsample), so it's the honest headline — but VAE
   latents are themselves compressed, so it still understates high-frequency /
   mid-σ signal. The 96px pixel-MSE compounds this and is kept only as a strip
   annotation.
4. **FM-MSE / recon-error ≠ quality on Anima** (`project_fm_val_loss_uninformative`).
   This is a "where is the base uncertain" diagnostic, not a quality metric.

## Verdict / next step

The default `sigmoid` is a defensible, SD3-aligned choice and `flow_shift=1.0`
is inert — there is no bug here. The probe raises a *testable hypothesis* that
the schedule over-samples a low-σ region the base already handles, with
`logit_normal μ≈+0.5` (or a `t_min` floor cutting the dead zone) as the natural
alternative arm. Because of caveat 2 this genuinely could land either way, so
the next step is a **CMMD-scored training sweep** (sigmoid baseline vs high-σ
skew vs `t_min` floor), not a config change. `t_max=0.95` is already ruled
*out* by this probe.

---
*Probe: `bench/timestep_sampling/probe_sigma_signal.py` · run:
`results/20260526-1538-base/` (n=16, single-seed) · headline recomputed on
latent-MSE · `git 2439e0d`*
