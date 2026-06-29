# Spectrum × SeaCache — is a spectral-evolution-aware decision metric better? (Phase 0)

**Question.** Spectrum (shipped: Chebyshev block-feature forecasting) schedules
skips with a content-blind growing-window heuristic. SeaCache (arXiv:2602.18993v2)
proposes a smarter *decision metric* — measure cache distance in a
Spectral-Evolution-Aware (SEA) filtered space that downweights high-freq noise.
Would that make better skip decisions **on Anima**, where spectral-evolution
priors have inverted before (CTCal, σ-reshape; x̂₀ resolves by σ≈0.45)?

**Method.** One real euler `generate()` per prompt (12 prompts, 24 steps, CFG 4,
1024², compiled). Capture per step: the pre-`final_layer` block feature (the
tensor Spectrum forecasts) + guided x̂₀ = x_t − σ·v. Compare three decision
metrics against a ground-truth skip-cost (‖Δx̂₀‖, the content the step actually
moves):

| metric | what | computable before the forward? |
|---|---|---|
| `raw_input`  | ‖Δx_t‖ relative-L1 (TeaCache-style) | yes |
| `sea_input`  | ‖Δ SEA_σ(x_t)‖ (SeaCache) | yes |
| `resid_feat` | ‖feat − Cheb(feat)‖, Spectrum's own forecast residual | no (lagging) |

**The trap (and the fix).** The pooled correlation is dominated by the monotone
σ-trend — which the blind window *already* exploits, so it says nothing about
decision quality. Pooled numbers (`raw −0.99, sea +0.51`) are an artifact. The
verdict is the **step-stratified** correlation: at a fixed step σ is identical
across prompts, so ranking metric-vs-GT *across prompts within each step* removes
the trend and isolates the only thing a content-adaptive metric can add. (A first
σ-quantile-bin detrend was too coarse — it left the trend in; caught by a
monotone-only sanity check returning −0.998 instead of ~0. Step-stratification
passes: monotone→0.04, injected signal→1.0.)

## Verdict (step-detrended, vs envelope-free GT ‖Δx̂₀‖)

| metric | detrended ρ |
|---|---|
| `raw_input`  | **−0.438** |
| `sea_input`  | **+0.569** |
| `resid_feat` | **+0.298** |

- **A — SeaCache's core claim CONFIRMED on Anima.** SEA-filtering flips a metric
  that *anti*-predicts skip-cost (raw −0.44) into one that predicts it (+0.57).
  Mechanism is exactly the paper's thesis: raw distance entangles high-freq noise
  that doesn't move the low-freq content; the filter strips it. Despite the
  prior-inversion track record, the SEA filter is worth grafting onto Spectrum's
  scheduler. `dA = +1.01`.

- **B — "use Spectrum's own residual instead" loses AS TESTED.** The
  feature-space forecast residual carries real adaptive signal (+0.30) but *less*
  than SEA-input (+0.57); `dB = −0.27`. So naively repurposing Spectrum's
  existing calibration residual as the schedule trigger is **not** better than
  SeaCache's metric. **Caveat:** the tested residual is in *feature* space
  (cond-only) while the GT is *latent*-space post-CFG content motion — the actual
  recommendation (SEA-filter the *output* residual) was **not** tested, so B
  refutes the naive-feature-residual idea, not the SEA-output-residual idea.

- **C — schedule mismatch.** SEA-weighted skip-cost in the σ<0.45 tail ≈ 0.0%
  (consistent with x̂₀ resolving by σ≈0.45), yet the blind schedule force-computes
  the last 3 steps and computes 62% overall. The content-adaptive opportunity is
  in σ≥0.45; the blind window mis-spends compute on the cheap tail.

## Phase 1 — true counterfactual GT (`phase1_counterfactual.py`)

Phase 0's GT was an x̂₀-motion *proxy*. Phase 1 replaces it with the true cost:
at each step, actually cache it — Chebyshev-forecast both CFG branches, run only
the head (`_spectrum_fast_forward`), CFG-combine, measure how far the resulting
x̂₀ lands from the real full-compute x̂₀. Non-circular: the metrics under test are
the ones a scheduler has *before* the forward (`sea_input` leading; `lag_resid` =
the last actual-step residual Spectrum carries, lagging).

| metric | ρ vs true counterfactual | vs unfiltered GT |
|---|---|---|
| `sea_input` (SeaCache, leading) | **+0.51** | +0.65 |
| `raw_input` (baseline) | −0.36 | −0.46 |
| `lag_resid` (Spectrum has, lagging) | +0.16 | — |
| **proxy(x̂₀) vs true GT** | **+0.82** | — |

- SEA helps on the **real** objective (flips −0.36 → +0.51). `dA = +0.88`.
- The Phase-0 proxy was **faithful** (+0.82) — cheap proxy is valid for iteration.
- Spectrum's lagging residual loses to leading SEA-input by +0.35 — refutes
  "Spectrum's own residual beats the input proxy" (true in principle, but the
  residual is only available lagging).

**Verdict: graft SeaCache's SEA-filtered input distance as Spectrum's scheduler
metric.** Keep Chebyshev forecasting for reuse. Full write-up:
`docs/findings/seacache_sea_decision_metric.md`. Open: end-to-end A/B (wire the
SEA-distance trigger into `spectrum_denoise`, render samples, score by CMMD).

Artifacts: `results/20260622-1431-phase0-detrend/` (Phase 0) +
`results/20260622-1445-phase1/` (Phase 1) — `result.json`, `per_step.csv`
(re-analyze without a GPU run). Re-run: `python bench/spectrum_sea/run_bench.py`
/ `python bench/spectrum_sea/phase1_counterfactual.py`.
