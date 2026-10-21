# Where MOSS-Image resolves the picture: Anima's σ knee, later steps

Measured 2026-10-21 on MOSS-Image Base and SFT. Probe, run and full tables:
`bench/moss_sigma_staircase/` (`probe_x0.py`, `README.md`, run
`results/20261021-2204-base-sft`). This re-measures the Anima finding
`docs/findings/sigma_signal_where_anima_resolves.md` on MOSS
(`docs/proposal/moss_anima_negatives_revisit.md` Tier 3 §3).

## Setting

On Anima, one bare forward's `x0_pred = x_σ − σ·v` is a recognisable picture at
σ ≈ 0.75 and essentially final at σ ≈ 0.45. Several MOSS levers were set against that
shape. MOSS differs in three places that could move it:

- Training σ is noise-biased (median 0.78; 34 % of draws above 0.85).
- The sampler runs shift 5 snapped to the training grid (`model.snapped_sigmas`).
- The DiT is single-stream: the text rows run all 42 blocks on every step.

The probe noises 16 cached latents (`_calib_mixed`, native size, ~1 MP) as
`(1 − σ)·x0 + σ·ε`, with one ε per sample shared over σ. It runs this at each of the 50 σ
of the default sampler grid: one forward at true caption length, CFG 1, no LoRA. Error is
latent-MSE against the clean latent, in two forms:

- normalised by its value at σ 0.952, Anima's normalisation;
- on a 4×4 average-pooled latent, which keeps layout and colour blocks and drops
  detail.

## Observations

| level (normalised error) | Anima σ | MOSS full latent σ (steps above / 50) | MOSS 4×4-pooled σ (steps above) |
|---|---|---|---|
| 0.331, Anima's "recognisable" | 0.75 | 0.737 (32) | 0.870 (22) |
| 0.20 | 0.63 | 0.572 (39) | 0.802 (28) |
| 0.092, Anima's "final" | 0.45 | 0.337 (45) | 0.634 (37) |

Base; SFT's knees are within 0.012 σ of these.

- **The full-latent curve is Anima's down to σ ≈ 0.75** (0.341 at σ 0.746 against
  Anima's 0.331 at 0.75). Below that, MOSS converges more slowly: 0.163 at σ 0.5 against
  Anima's ~0.117, and the "final" level arrives at σ 0.34, not 0.45.
- **Layout resolves well above that.** On the pooled latent the error reaches the
  "recognisable" level at σ 0.87 and 20 % at σ 0.80. On the decoded grids:
  - σ 0.95 is still a blur;
  - composition and pose are fixed by σ 0.90;
  - σ 0.80 matches the final image except for detail: eyes, accessories, line weight.
- **SFT resolves where Base does.** Its error is 1–3.5 % higher at every σ and the knees
  do not move. SFT's glossy pull is not a change in where the picture is decided.
- **The samples agree**: the std of the error at σ 0.5 is 0.037 against a mean of 0.106.

Where the steps go, for the same σ boundaries:

| | above 0.87 | above 0.80 | above 0.75 | above 0.45 |
|---|---|---|---|---|
| MOSS 50-step shift-5 grid | 22 steps | 28 | 31 | 43 |
| Anima 28-step shift-3 grid | — | — | 14 | 22 |
| MOSS training σ draws (`TrainRequest` defaults) | 29 % | 45 % | 56 % | 92 % |

## Reading

The σ knee carries over from Anima. The step index it falls on does not: σ 0.75 is step
31 of 50 on MOSS against step 14 of 28 on Anima.

- On MOSS the layout is formed within the first 22–28 steps.
- Everything after step ~39 is the detail tail, under 0.2 normalised error.
- Training spends 45 % of its draws on the layout-forming band above σ 0.80, and 8 % on
  the detail tail below σ 0.45.

For the levers set against Anima's shape:

- **DAVE τ** and **Spectrum `flex_window`** are defined in steps, not σ, so no Anima σ
  number sits in them. The shipped DAVE τ 0.10 covers steps 0–4, which is σ 1.0 → 0.983.
  Over those steps the pooled error goes from 0.65 to 0.58; the layout knee at σ 0.87 is
  step 22.
- **The REPA σ 0.85 taper** (`project/moss_lora/repa.md`) fits MOSS's own shape: at
  σ 0.85, `x0_pred` is still at 0.52 of the σ 0.95 error, and the layout becomes
  recognisable only at 0.87.
- **Tier 3 §4 (text drive across σ)** should read its front-loaded window over
  σ ≥ 0.80 (steps 0–28), not Anima's σ ≥ 0.85.

The Anima finding's text puts its 20 % crossover at σ ≈ 0.55, but its own table (0.142
at 0.55, 0.214 at 0.65) puts it at σ ≈ 0.63. The comparison here uses the table.

## Hypotheses (not tested)

- **Lowering the sampler shift would move steps from layout formation to the detail
  tail.** Shift 5 spends 22 of 50 steps above σ 0.87. Whether a lower shift helps at
  fixed step count is open. On Anima, reshaping the σ schedule at fixed NFE was refuted
  (`_archive/bench/sigma_reshape/`), so expectations are low. Shift is a `GenerateRequest`
  knob, so this is a `make gen`-scale A/B, not a training question.
- **A detail- or style-oriented LoRA may be under-served by the default training σ.**
  Only 8 % of draws fall below σ 0.45, where MOSS is still converging (it reaches the
  "final" level at 0.34, later than Anima). An arm with a less noise-heavy `sigma_mu`
  would test it. The earlier MOSS LoRA round found style at high σ, so this cuts the
  other way and needs its own arm before anyone acts on it.
- **Where the picture resolves is a property of the denoiser, not the sampler.** A
  schedule change moves which steps land on each side of the knee; it cannot move the
  knee. Only training can, and the knee is largely the SNR at which `x0` shows through
  `x_σ`, so a σ reweighting would likely move it little.
