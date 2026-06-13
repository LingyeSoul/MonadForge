# MOD × DAVE "why better" — best-of-N order statistics with correlated draws

Status: **PROPOSED — not started.**

- Planned bench: `bench/dave/bench_bestofn.py` (new; generation arms reuse the
  Phase-4 probe flags)
- Premise sources: `bench/dave/README.md` § Phase 4 (MOD is common-mode +
  lock-invariant, DAVE de-correlates, interaction ≈ 0; `probe_mod_presence.py`
  excludes the no-op reading), `docs/inference/dave.md` (compose row),
  field observation 2026-06-11 (MOD-alone layout failures — every-seed
  back-view / letterbox — rescued by composing DAVE τ0.10·s0.5).
- Metric-trap priors this design must obey:
  `docs/findings/spectral_fraction_metric_inverts.md` (validate any score
  against eyeball before trusting it), FM-val-loss uninformative (CMMD is the
  only distribution-level signal, and it is *not* per-image), real captions
  mandatory (generic-prompt probes have produced artifacts before — the
  dynamic-spectrum ‖v‖-gate lesson).

## The claim being made precise

"MOD + DAVE makes better images" is not a per-image property — it is a
property of the **batch you pick from**. The Phase-4 mechanism (MOD shifts all
seeds' DC together; DAVE de-correlates seeds across layout basins) has a
direct image-level consequence in order statistics:

Under an equicorrelated-Gaussian model of per-image quality `q` across seeds
(mean `μ`, between-seed std `σ`, pairwise correlation `ρ`):

```
E[best of n seeds] = μ + σ·√(1−ρ)·e_n ,   e_n := E[max of n iid N(0,1)]
N_eff := N / (1 + (N−1)ρ)                 (intuitive readout: "effective tries")
```

Each arm moves exactly one parameter, per the Phase-4 mechanism:

| arm | μ (mean quality) | ρ (seed correlation) | predicted consequence |
|---|---|---|---|
| vanilla | baseline | lock-level (high-ish) | baseline curve |
| MOD | **↑** (quality steer) — but failure events hit the *whole batch* (common-mode) | unchanged | curve shifts up, doesn't steepen; batch-failure rate ≈ per-image failure rate |
| DAVE | ~unchanged | **↓** (AC-sim 0.58→0.27 is a correlation measurement) | curve steepens with n; batch-failure rate ≈ pᴺᵉᶠᶠ |
| MOD+DAVE | ↑ | ↓ | **multiplicative**: up-shifted *and* steepened — the eyeballed "much better image" |

Two falsifiable channels:

1. **Best-of-n curves.** Per arm, fit `(μ, σ, ρ)` from the seed scores, predict
   `E[best of n]` for `n = 1..N`, overlay the *observed* curve (subsampled max
   over seed subsets). Match → the mechanism explains the eyeball, fully.
   MOD+DAVE *above* its prediction → a residual per-seed interaction exists
   (Phase 4 says it shouldn't — that would be a real discovery).
2. **Batch-coherent failure rate.** Define failure semantically (tagger emits a
   `from_behind`-class tag the prompt didn't ask for; analytic black-bar
   detector for letterbox). Prediction: under MOD the failure events are
   seed-coherent (all-N-fail probability ≈ p, not pᴺ); DAVE collapses
   all-N-fail toward pᴺᵉᶠᶠ. This is the quantitative form of "re-rolling
   doesn't fix MOD failures, DAVE does".

Note what channel 2 also establishes: the cross-seed AC-sim diagnostic is
structurally blind to MOD-style failures (the batch agrees because the *aim*
moved, not because the lock tightened) — any auto-detector for this failure
class must be semantic, not spectral.

## Phase 0 — estimator validation (gate, cheap, mostly CPU)

No generation. Score an existing eyeball-labeled set (the 2026-06-11
`comfy/output/hydratest/` MOD failures + their DAVE-rescued counterparts, plus
any `bench/dave/eyeball.py` output) with the candidate per-image `q`:

- `q_tag`: Anima Tagger tag-recall F1 vs the prompt's tags (adherence).
- `q_pe`: PE-Core image↔prompt similarity (the CMMD embedding path, per-image).
- failure flags: tagger back-view-class tags + black-bar edge detector.

**Gate (pre-registered): each scorer must rank the eyeballed-bad images below
the eyeballed-good ones (AUC ≥ 0.8 on the labeled pairs), and the two scorers
must agree directionally.** A scorer that fails stays out (spectral-fraction
lesson: a metric that can't reproduce the eyeball on *known* cases predicts
nothing). If both fail → stop; the proposal has no instrument.

## Phase 1 — the 4-arm bench

`bench/dave/bench_bestofn.py`: 4 arms × P×N images, **real dataset captions**
(sample via `caption_index.json`, not hand prompts), same seed set across arms,
ortho LoRA + `pooled_text_proj-0611` + stock step_i8_skip27 w=3, DAVE τ0.10·s0.5
(the field operating point). Suggested P=6 prompts × N=12 seeds = 288 images
(24 steps, compile_blocks on — DAVE hooks are compile-safe); ~1–2 GPU-h.

Per (arm, prompt): estimate `μ, σ, ρ` (ρ from the score vector; report the
PE-Core embedding pairwise-cosine ρ as the mechanism-side mirror — it should
reproduce the Phase-4 ordering vanilla≈mod ≫ dave≈mod+dave, a built-in sanity
check tying image space back to the lock probe). Then:

- verdict plot: predicted vs observed best-of-n per arm (the headline artifact);
- `N_eff` per arm ("DAVE turns 12 seeds into ~11 tries; MOD alone leaves ~2");
- failure table: per-image rate p, all-N-fail rate, vs the pᴺᵉᶠᶠ prediction;
- sanity row: per-arm CMMD (distribution-level corroboration only — not a gate).

## What would make this MORE than a write-up of the obvious

- If the equicorrelated model *fits*, the result is a transferable formula: any
  common-mode lever (CFG direction, prompt boilerplate, future steering heads)
  can be assigned a (Δμ, Δρ) signature with this bench, and "should I stack a
  de-correlator?" becomes arithmetic in `e_n·√(1−ρ)` instead of eyeballing.
- If MOD+DAVE beats its predicted curve, Phase 4's zero-interaction is
  incomplete at the image level — hunt the residual (most likely candidate:
  DAVE escaping a *below-average* locked basin raises μ itself; measurable as
  P(bad basin)×severity from the failure table).

## Out of scope

- Training-signal use of any of this (the turbo `val/div_ac_sim` pass is
  separate and already wired; this proposal is inference-side).
- Spectrum/SPD compose (DAVE v0 explicitly doesn't support it).
- Tuning MOD's w or DAVE's dose — arms run the shipped/field operating points.
