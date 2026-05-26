# bench/timestep_sampling

Is Anima's default training-time timestep schedule pointed at the right σ?

`configs/base.toml` trains LoRA with:

```toml
timestep_sampling = "sigmoid"     # σ = sigmoid(N(0,1)) — logit-normal, bell at σ=0.5
discrete_flow_shift = 1.0
```

Two things worth knowing before benching:

- **`discrete_flow_shift` is inert here.** It is only consumed by the `"shift"`
  branch (`library/runtime/noise.py:110`) and the inference scheduler. Under
  `"sigmoid"` sampling it does nothing at train time. The only live lever is
  *where the sampling density sits on the σ axis*.
- **`"sigmoid"` with `sigmoid_scale=1.0` is exactly logit-normal(0,1) over σ** —
  the SD3-recommended mid-noise emphasis. It's a sensible inherited default;
  it has just never been A/B'd on Anima.

## `probe_sigma_signal.py` — cheap, no training

For real cached dataset latents it reconstructs the bare DiT's x0 estimate at
each σ (`x0_pred = x_σ − σ·v`) and decodes it, so you can **eyeball where the
base stops being able to recover the image** — i.e. where an adapter has
something to learn. It overlays each candidate schedule's sampling density on
the measured per-σ reconstruction error.

```bash
uv run python -m bench.timestep_sampling.probe_sigma_signal
uv run python -m bench.timestep_sampling.probe_sigma_signal --num_samples 4
uv run python -m bench.timestep_sampling.probe_sigma_signal --adapter output/ckpt/foo.safetensors
```

Artifacts in `results/<ts>-<label>/`:

- `x0_vs_sigma_s{i}.png` — **the deliverable**, one file per sample (legible at
  a glance — open them side by side). Strip of decoded x0_pred across σ next to
  the true x0, with each column's pixel-MSE annotated. Read left→right: where
  does the prediction diverge from the target?
- `density_overlay.png` — schedule densities (sigmoid / uniform /
  logit-normal μ=+0.5 / sigmoid∘t_max=0.95) over the per-σ recon-error curve.
- `sigma_signal.csv`, `result.json` — numbers + the standard envelope.
  `result.json:metrics.low_signal_mass_fraction_per_schedule` = the fraction of
  each schedule's draws spent at σ where the base already reconstructs x0 well
  (≈ wasted training samples).

**Caveat (load-bearing):** FM-MSE / recon-error is a *where-is-the-base-uncertain*
diagnostic, **not** a quality metric — lower FM val loss has never tracked
better samples on Anima (`project_fm_val_loss_uninformative`). This probe
motivates which arms to put in a CMMD-scored training sweep; it does not, on
its own, prove a schedule trains a better adapter.
