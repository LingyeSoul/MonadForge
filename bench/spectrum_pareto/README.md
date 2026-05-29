# bench/spectrum_pareto

Live successor to the retired `_archive/bench/spectrum/` drift bench. Motivation
and full design: `proposal.md` ("Spectrum Benchmark Redesign Proposal") at the
repo root.

The old drift bench answered the wrong question — it ranked *robustness to
injected forecast error* and blessed `dpmpp_sde + linear_quadratic`, which real
images contradicted. This bench instead optimizes **image quality per actual DiT
cost**, gated on base-sampler quality first.

## Status: Phase 0 — forecastability probe (this is what's built)

Per the build decision (probe-first, bench-only), only the de-risking probe is
implemented so far. It answers one question before any harness is built:

> The mod-guidance distillation pool (`post_image_dataset/distill_mod_synth/`,
> ~1.8k clean latents) is a large, aspect-balanced **er_sde @ cfg 2.5** corpus we
> already paid for. Can we reuse it as the Spectrum bench baseline? Only if
> er_sde's block-feature trajectory is as Chebyshev-forecastable as euler's.

er_sde injects fresh stochastic noise into the latent each step, which may make
the per-step feature sequence jittery and raise the forecast residual — shrinking
Spectrum's safe block-skips. (er_sde is still **1 model eval/step**, so it's on
the speed frontier; the risk is forecastability, not cost.) The proposal makes
euler the primary target for exactly this reason. This probe measures the gap.

### Files

| file | role |
|---|---|
| `capture_features.py` | All-actual per-step `final_layer`-input feature capture (faithful to `spectrum_denoise`). `--from_synth_pool N` draws captions + native buckets from the distill pool. Exposes `capture_one()` / `pool_samples()`. |
| `replay_forecaster.py` | Offline forecaster sweep over captures, using the *shipped* `SpectrumPredictor` and production cache schedule. Exposes `replay_at_combo()` for single-combo scoring. |
| `compare_samplers.py` | **The probe.** Paired capture of er_sde vs euler over pool prompts, replayed at the production-default forecaster; emits a PASS/FAIL verdict. |

### Run the probe

```bash
# ~N×len(seeds)×len(samplers) captures; DiT loaded once (shared_models).
uv run python -m bench.spectrum_pareto.compare_samplers \
    --n 16 --seeds 0 1 --buckets 128x128 150x112 \
    --steps 28 --guidance_scale 2.5 --tol 1.15
# → bench/spectrum_pareto/results/<ts>-cmp-samplers/{verdict.md,pairs.csv,result.json}
```

Pass `--save_captures` to persist the per-trajectory npz (reusable by
`replay_forecaster.py` for the full m/lam/w sweep later).

### Verdict gate

`compare_samplers.py` compares the **candidate** sampler (first in `--samplers`,
default `er_sde`) against the **baseline** (last, default `euler`) on mean
relative-L2 forecast residual at the production-default combo
(m3/lam0.1/w0.3, ws2/fx0.25/wu7):

- **PASS** — candidate mean rel-L2 ≤ `--tol` × baseline (default 1.15×). er_sde is
  ~as forecastable → reuse the pool: build the bench on er_sde and reuse the
  stored `x0` as **free Phase-1/4 endpoints** (the stored latents *are* the
  all-actual er_sde endpoints; with matching seed/steps/cfg, Phase-4's
  Spectrum-vs-all-actual diff costs only the Spectrum side).
- **FAIL** — er_sde residual materially worse → target euler with fresh
  generation; the `x0` pool can't be repurposed.

Both `mean_ratio` and a **paired** median ratio are reported; the paired figure
guards against prompt-mix imbalance.

> Caveat: lower forecast residual is a *shortlist* signal, not a quality oracle
> on Anima (cf. project memory `fm_val_loss_uninformative` / `cmmd_val_signal`).
> This probe gates **effort**, not production defaults.

## Not yet built (the expand path, if the probe PASSes)

Tracks from `proposal.md`, deferred until the probe validates the target sampler:

1. **Base sampler gate** (`base_gate.py`) — all-actual quality vs production
   default; reject bad base paths before Spectrum tuning.
2. **Shadow-forward bench** (`shadow_forward.py`) — true forecast error on the
   *off-trajectory* latents Spectrum actually creates (record-only shadow
   actuals; never fed back).
3. **Step-sensitivity calibration** — per-step `final_damage_per_unit_error`,
   cached by (model sha, size, steps, sampler, scheduler, cfg).
4. **Pareto selection** (`analyze_pareto.py`) — nondominated set over effective
   NFE, wall time, weighted forecast error, and paired image metrics.
5. **Adaptive cache policy** prototype — confidence-gated caching
   (`expected_damage = predicted_error × sensitivity[i]`). The hard part is
   predicting error on cached steps without ground truth; the shadow-forward
   bench is the tool to *falsify* an EMA gate before building it. Project memory
   on DCW (`dcw_v4_g_obs_load_bearing`, `dcw_seed_variance_dominates`) argues for
   an online/observed estimator over a learned prompt-feature predictor.

Production `networks/spectrum.py` is intentionally **untouched** in this phase
(Track 2 cost counters deferred); all capture uses a bench-only runner override.
