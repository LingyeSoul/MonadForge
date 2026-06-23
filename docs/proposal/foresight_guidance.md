# Foresight Guidance (FSG) for Anima — golden-path CFG calibration

**Status:** Phase-0 + Phase-1 (eyeball) PASSED → green-lit. **Stage 2 plugin BUILT**
(`library/inference/corrections/fsg.py`, `--fsg` / `FSG=1`, invariant test) — Stage 3
rigor (matched-NFE A/B, er_sde render, saturation-confound) **still owed** (§8–§9).
**Paper:** "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations" (NeurIPS 2025, arXiv 23177). Code: github.com/Ka1b0/Foresight-Guidance.
**Benches:** `bench/fsg/probe_golden_path.py` (premise), `bench/fsg/render_compare.py` (eyeball A/B).
**Memory:** [[project_fsg_golden_path_phase0]].

## TL;DR

FSG is a training-free, checkpoint-agnostic inference stack that reframes CFG as
*fixed-point calibration toward a golden path* — a latent state where the
conditional and unconditional predictions agree. At scheduled timesteps it runs
K forward(conditional)–backward(unconditional) iterations over a long interval Δσ
to pull `x_t → x̂_t` onto that path, then denoises from `x̂_t`.

On Anima the premise **holds** — but **not where the paper says**. The fixed-point
operator contracts and the conditional/unconditional gap shrinks in **σ∈[0.45, 0.85]**;
at the noisy top (σ≈0.94, the paper's primary target region) it *diverges*. A
VAE-decode A/B in the working band is **visibly, clearly better** than baseline —
more saturated, more detailed, more complete — and does **not** drift to mush.

This proposal records the groundings and lays out the build, with three open
questions answered: composition with existing stacks, inference cost, and
hyperparameter calibration.

## 1. Background and flow-matching translation

The paper is ε-prediction + DDIM. Anima is **velocity-prediction flow-matching**,
sampled at production **CFG=4 / er_sde**. The FSG forward-backward operator maps
cleanly onto Anima's reversible Euler ODE (`library/inference/sampling.py::step`),
so no DDIM machinery is needed. At a scheduled σ with latent `x`, interval `Δσ`,
calibration guidance `γ`:

```
v^γ(x,σ) = v^u(x,σ) + γ·(v^c(x,σ) − v^u(x,σ))     # CFG-guided velocity
forward :  x' = x − Δσ · v^γ(x, σ)                  # denoise σ → σ−Δσ (conditional)
invert  :  x'' = x' + Δσ · v^u(x', σ−Δσ)            # re-noise back (unconditional)
F(x) = x'' ; iterate x ← F(x), K times ; then denoise from x̂ = x^(K)
```

The golden path is the fixed point where conditional-forward and
unconditional-backward agree, i.e. where `‖v^c − v^u‖` is minimized.

## 2. Phase-0 findings — premise holds, but mid-σ only

`bench/fsg/probe_golden_path.py` runs a deterministic Euler-CFG=4 trajectory from
noise on **real training captions** (`post_image_dataset/resized/**/*.txt`), and
at early-σ steps branches a clone to iterate F, measuring the prediction gap
`‖v^c−v^u‖/‖v^u‖` (the flow analogue of paper Fig 2b) and contraction
`ρ = ‖Δx_k‖/‖Δx_{k−1}‖`. Result (6 prompts × 2 seeds, K=4, 1024px):

| σ | gap drop (k=0→4) | contraction ρ̃ | verdict |
|---|---|---|---|
| **0.94** | **−20% (gap GREW)** | **1.04 (diverges)** | ✗ |
| 0.85 | +13.9% | 0.93 | ✓ (registered) |
| 0.75 | +12.3% | 0.93 | ✓ (registered) |
| 0.62 | +14.5% | 0.94 | ✓ (exploratory) |
| 0.43 | +11.0% | 0.95 | ✓ (exploratory) |

**Key Anima-specific finding:** the paper prescribes concentrating iterations in
the *earliest / noisiest* stages ([2/3 T, T], 3:2:1 early-weighted). On Anima
that is exactly the **dead zone** — at σ≈0.94 there is barely any conditional
structure yet, so cond≈uncond and iterating amplifies noise (ρ>1). The working
band is **mid-σ**, consistent with: x̂₀ resolves by σ≈0.45 ([[project_sigma_signal_resolves_by_045]])
and FEI/cross-attn discrimination peaks at the clean end, not the noisy end
([[project_cbs_monitor_vs_fei_routing]]). The pre-registered gate (σ≥0.75) is
cleared at σ=0.85/0.75; σ=0.62/0.43 are below-gate exploratory.

## 3. Phase-1 findings — visibly better, not mush

`bench/fsg/render_compare.py` VAE-decodes baseline vs FSG (deterministic Euler
CFG=4, 4 real captions, K=4). The FSG arms are **clearly better** across all
prompts: more saturated, more detailed, more finished (added backgrounds,
clothing detail, cleaner faces/hands), same composition. This refutes the
"contracting latent ≠ same image" confound: a falling gap *does* track real
refinement here, not drift to a flat low-velocity region.

The **narrow registered band (0.75–0.85) does ~all the work** — its latent drift
is nearly identical to the wide band (0.45–0.85): 0.38 vs 0.40, 0.22 vs 0.23,
0.53 vs 0.57, 0.44 vs 0.46 — and the images are near-indistinguishable. So the
win rests on the defensible registered band.

Caveat (kept honest): part of the visible "better" is saturation/contrast, and
Anima has a history of levers that turn out to be global-tone-only
([[project_mod_guidance_text_derivative_orthogonal]], null-TTA). It is *more*
than tone (added structure), but ruling out a pure saturation confound is on the
checklist (§7).

## 4. Where it hooks — the calibration seam

FSG occupies a **new seam — pre-step latent calibration** — that no existing
plugin uses. In `generation.py::generate_body`, immediately before the per-step
velocity forwards (current line ~781), insert: *if this step is scheduled, run
the FSG calibration loop on `latents`, then proceed.* Plugin home:
`library/inference/corrections/fsg.py` (mirrors DCW/SMC/DAVE), threaded via
`SamplerSideChannels`. Enable via CLI (`--fsg`, `--fsg_band`, `--fsg_k`,
`--fsg_d_sigma`, `--fsg_gamma`) + `GenerationRequest` fields, composed into the
`test-*` flag family.

## 5. Composition with existing inference stacks (Q1)

FSG sits at a seam distinct from every existing stack, so most compose for free:

| Stack | Seam | Composition |
|---|---|---|
| **mod-guidance** | AdaLN embedding (every forward) | ✅ FSG's calibration forwards inherit mod's steering — mod steers the field, FSG calibrates the latent onto that field's golden path. Complementary jobs. |
| **DCW** | post-step x-space | ✅ Serial: FSG calibrate → denoise → DCW correct. |
| **DAVE** | block-forward hooks | ✅ FSG forwards see DAVE-attenuated features; orthogonal like DAVE×mod. |
| **CNS** | noise injection (er_sde) | ✅ Orthogonal — FSG calibration is deterministic. |
| **SMC-CFG** | CFG-combine (replaces CFG) | ⚠️ Only overlap. Rule: FSG calibration uses plain γ-combine (paper's operator); the outer denoise step uses the configured CFG variant. |
| **Spectrum / SPD** | alternate denoise runners | 🔜 v2 — they replace the loop and Spectrum caches block feats that FSG's extra forwards perturb. |

## 6. Inference cost (Q2)

Extra forwards = `3 · K · M` (per scheduled step, per iteration: v^c+v^u at σ,
v^u at σ−Δσ). Baseline 20-step CFG = 40 NFE.

| Config | Extra NFE | Total | vs baseline |
|---|---|---|---|
| Baseline (20 steps) | 0 | 40 | — |
| Render-A/B setting (M≈7, K=4) | +84 | 124 | ~3.1× |
| Conservative (M=3, K=3) | +27 | 67 | +68% |
| + reuse last-iter v^c/v^u for the step | −2M | ~60 | +50% |

The visible win used the ~3× setting. **The matched-NFE test is therefore the
load-bearing experiment** (§7): does FSG-at-67-NFE beat a plain 33-step baseline?
The paper's efficiency claim is precisely "fewer, longer subproblems beat more
short ones" — must be confirmed on Anima. Cost levers: fewer scheduled steps
(FSG's design point), per-σ K taper (ρ≈0.93 ⇒ gap-drop plateaus by K≈3), and
velocity reuse.

## 7. Hyperparameter calibration (Q3)

All inference-time, no retraining, checkpoint-agnostic. The Phase-0 probe **is**
the calibration instrument:

- **Band [σ_lo, σ_hi] + per-σ K_i:** read off the probe's ρ and gap-drop curves —
  calibrate where ρ<0.95, spend more K where gap-drop is largest. Current:
  [0.45, 0.85], K≈3.
- **K:** bounded by contraction (error ~ρ^K, ρ≈0.93 ⇒ K=3–4 captures ~all gain;
  probe traces plateau by k≈2–3).
- **Δσ (interval), γ (calibration guidance):** cheap `render_compare` sweep
  ({0.05, 0.1, 0.15, 0.2} × γ∈{1, 4, …}). Too-large Δσ is what makes σ=0.94
  diverge — there is a stability ceiling to map.
- **λ (outer CFG++):** the paper denoises with CFG++ after calibration; A/B
  whether Anima benefits vs plain CFG.
- **Portability:** likely portable across checkpoints like DCW's λ / SEA's δ
  were ([[reference_spectrum_node_dcw_defaults]], [[project_spectrum_sea_schedule_prompt_gen]]);
  verify once on 2 checkpoints rather than per-LoRA.

## 8. Staged build plan

1. **Stage 1 — CFG×K (cheapest).** Increasing fixed-point iterations on the
   *linear* operator (no forward-backward) restricted to mid-σ; the paper shows
   this alone already helps. Smallest diff, fastest sanity check.
2. **Stage 2 — full FSG plugin.** `library/inference/corrections/fsg.py` with the
   forward-backward operator + scheduled band + CLI/GenerationRequest surface +
   velocity reuse.
3. **Stage 3 — rigor (CONTRIBUTING Tier-2: bench + invariant test).**
   - **Matched-NFE A/B** (the decisive one): FSG-N vs plain-baseline-N.
   - **Production er_sde CFG=4** render — the Euler-only signal is not production;
     er_sde flipped-then-reclosed the σ-reshape result ([[project_sigma_reshape_no_win]]).
   - **Saturation confound** check vs a contrast/saturation bump.
   - Invariant test: FSG with K=0 / empty band == baseline bit-exact.

## 9. Open risks

- **Matched-NFE** may erase the win (the render was ~3× compute). Decisive.
- **er_sde** is the real sampler and reversed σ-reshape once; Euler-only is not proof.
- The payoff lands in the already-near-resolved σ band where σ-reshape found no
  fixed-NFE headroom. FSG's *mechanism* is distinct (a consistency operator, not
  a schedule reshape — verifier agreed), but the overlap of *payoff zone* is why
  matched-NFE is load-bearing.
- Part of the visible win may be global tone/saturation (Anima lever history).
