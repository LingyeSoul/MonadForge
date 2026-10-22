# bench/moss_sigma_staircase

At which σ does a bare MOSS-Image forward's `x0_pred` reach the final picture? Anima's σ
staircase (`docs/findings/sigma_signal_where_anima_resolves.md`) re-measured on MOSS's own
sampler grid; Tier 3 §3 of `docs/proposal/moss_anima_negatives_revisit.md`. No gate: it sets
the σ knee that the σ-gated levers read. Summary, readings and hypotheses:
`docs/moss/findings/sigma_staircase.md`. This file is the run log.

```bash
make daemon-run ARGS="bench/moss_sigma_staircase/probe_x0.py --label base-sft"
```

## Probe

`probe_x0.py` (the docstring has the details). 16 samples from
`post_image_dataset/moss/_calib_mixed` (evenly spaced by stem, native size, ~1 MP). Each is
noised as `(1 − σ)·x0 + σ·ε` (one ε per sample, the same at every σ) at all 50 σ of
`snapped_sigmas(50, 5)`. One reference forward runs at true caption length (CFG 1, no
LoRA), then `x0_pred = x_σ − σ·v`. Base and SFT run in one job: 1600 forwards at
0.8 s/fwd, no swap, ~22 min.

- `rel` = latent-MSE / `E[x0²]`, the error against predicting 0. Latents are
  BN-normalised.
- `norm95` = latent-MSE / latent-MSE at σ 0.952 (the grid point nearest 0.95), Anima's
  normalisation.
- `_low` = the same on a 4×4 average-pooled latent (1/64 of the image side): layout and
  colour blocks without the detail.
- Knees are the σ where `norm95` crosses Anima's three levels: 0.331 ("recognisable", σ 0.75
  on Anima), 0.20, 0.092 ("essentially final", σ 0.45 on Anima).

## Runs

**2026-10-21, base-sft** (`results/20261021-2204-base-sft`; smoke at
`results/20261021-2143-smoke`, 1 sample, 8 steps).

| level (`norm95`) | Anima σ | MOSS Base σ (step / 50) | MOSS SFT σ | Base, 4×4 pool σ (step / 50) |
|---|---|---|---|---|
| 0.331 "recognisable" | 0.75 | 0.737 (32) | 0.739 | **0.870 (22)** |
| 0.20 | 0.63 | 0.572 (39) | 0.575 | **0.802 (28)** |
| 0.092 "final" | 0.45 | 0.337 (45) | 0.339 | **0.634 (37)** |

The step column counts the 50-step grid's steps above that σ. SFT's `rel` is 1–3.5 % higher
than Base's at every σ. Base `rel_low` is 0.32 at σ 0.95.
