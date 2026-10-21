# bench/moss_sigma_staircase

At which σ does a bare MOSS-Image forward's `x0_pred` reach the final picture? This is
Anima's σ staircase (`docs/findings/sigma_signal_where_anima_resolves.md`) re-measured on
MOSS's own sampler grid. Tier 3 §3 of `docs/proposal/moss_anima_negatives_revisit.md`; run
2026-10-21. A measurement with no gate: it sets the σ knee that the σ-gated levers read.

Summary of what the run found: `docs/moss/findings/sigma_staircase.md`. This file is
the run log.

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
- `norm95` = latent-MSE / latent-MSE at σ 0.952 (the grid point nearest 0.95). This is
  Anima's normalisation, so the curve shapes compare directly.
- `_low` = the same on a 4×4 average-pooled latent (1/64 of the image side). It measures
  layout and colour blocks without the detail.
- Knees are the σ where `norm95` crosses the three Anima levels:
  - 0.331, where Anima's `x0_pred` is "recognisable" at σ 0.75;
  - 0.20;
  - 0.092, where Anima is "essentially final" at σ 0.45.

## Results

**2026-10-21 — base-sft** (`results/20261021-2204-base-sft`; smoke at
`results/20261021-2143-smoke`, 1 sample, 8 steps).

| level (`norm95`) | Anima σ | MOSS Base σ (step / 50) | MOSS SFT σ | Base, 4×4 pool σ (step / 50) |
|---|---|---|---|---|
| 0.331 "recognisable" | 0.75 | 0.737 (32) | 0.739 | **0.870 (22)** |
| 0.20 | 0.63 | 0.572 (39) | 0.575 | **0.802 (28)** |
| 0.092 "final" | 0.45 | 0.337 (45) | 0.339 | **0.634 (37)** |

The step column counts the 50-step grid's steps above that σ.

- **Full latent: the same staircase as Anima, a little slower at the low end.** On
  Anima's normalisation, MOSS's curve lies on Anima's down to σ ≈ 0.75 (0.341 at 0.746
  against 0.331). Below that it sits above Anima's: 0.163 against ~0.117 at σ 0.5. The
  "final" level arrives at σ 0.34, not 0.45.
- **Layout resolves early.** The 4×4-pooled error reaches Anima's "recognisable" level
  at σ 0.87 and its 20 % level at σ 0.80. On the decoded grids, composition and pose are
  fixed by σ 0.90, and σ 0.80 matches the final image apart from detail: eyes,
  accessories, line weight. σ 0.95 is still a blur.
- **SFT resolves where Base does.** Every knee is within 0.012 σ, and `rel` is 1–3.5 %
  higher at every σ. SFT's glossy pull is not a change in where the picture is decided.
- **Per-sample spread is small**: std of `rel` at σ 0.5 is 0.037 against a mean of
  0.106.

### What the sampler and training spend on each side

| | above 0.87 (layout recognisable) | above 0.80 (layout 20 %) | above 0.75 | above 0.45 |
|---|---|---|---|---|
| MOSS 50-step shift-5 grid | 22 steps | 28 | 31 | 43 |
| Anima 28-step shift-3 grid | — | — | 14 | 22 |
| MOSS training σ draws (`TrainRequest` defaults) | 29 % | 45 % | 56 % | 92 % |

Reading: the σ knee itself carries over from Anima. What changes is the step index it
falls on. On MOSS, σ 0.75 is step 31 of 50 (Anima: step 14 of 28), and the layout is
formed within the first ~22–28 steps. Everything after step ~39 is the detail tail
(< 0.2 normalised error).

### Consequences for the levers that read the knee

- **DAVE τ** is a step fraction (`dave.py`: steps `0 … round(τ·steps) − 1`). The
  shipped τ 0.10 covers steps 0–4, which is σ 1.0 → 0.983. All of that is before the
  layout knee: the pooled error goes from 0.65 to 0.58 `rel_low` over those steps, and is 0.32 at σ 0.95. The
  window is defined in steps, not σ, so nothing in it reads an Anima σ. Whether it
  should reach further toward σ 0.87 is a question for the DAVE line
  (`project/moss_lora/dave.md`), not something this probe answers.
- **Spectrum `flex_window`** grows the forecast window per full step. It has no σ
  boundary, so there is nothing to move.
- **REPA σ taper** (`project/moss_lora/repa.md` § Open: "no target above σ ≈ 0.85"):
  this matches MOSS's own shape. At σ 0.85, `x0_pred` is still at 0.52 of the σ 0.95
  error, and the layout only becomes recognisable at 0.87. A taper at σ 0.85 cuts ~33 % of
  training draws.
- **§4 text-drive window**: the plan is written above σ ≈ 0.80 (steps 0–28) and is
  layout-recognisable by 0.87. §4's front-loaded window should be read over σ ≥ 0.80,
  not Anima's σ ≥ 0.85.

The Anima finding's text puts its 20 % crossover at σ ≈ 0.55, but its own table
(0.142 at 0.55, 0.214 at 0.65) puts it at σ ≈ 0.63. The comparison above uses the table.
