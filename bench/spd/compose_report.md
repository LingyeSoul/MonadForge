# SPD ∘ Spectrum composition — Phase-0 report

Bench record for `docs/proposal/spd_spectrum_compose.md` Phase 0. All runs
2026-05-21 on a single RTX 5060 Ti (16 GB), bare Anima DiT unless noted, via
`bench/spd/probe_compose.py`. Builds on the SPD precondition work in `plan.md`
(Phase 0/1) — Anima latents are power-law (β≈2.26), bands resolve in frequency
order, σ_resolve monotone.

**TL;DR.** The arith gate and the 0(a) eyeball both clear; the *real*
precondition — 0(b) feature LL-DCT continuity across the resolution handoff —
**FAILS**. The block-output feature's low-frequency *pattern* reorients at the
seam (its norm is preserved, its direction is not), so the band-aligned
forecaster has nothing stable to carry across. Loading the Case-B SPD
trajectory adapter (`anima_spd.safetensors`) does **not** fix this — it makes
the reorientation slightly worse. **Do not build the band-aligned forecaster
(Phase 1).** If SPD∘Spectrum ships at all, it is the 0(a) naive-reset compose.

---

## Phase-0(0) — arithmetic gate (`--mode arith`, R2)

Block-FLOP proxy over the σ schedule (no GPU). Can the principled "skip blocks
only where every band is settled" compose beat Spectrum at all?

| denoiser | attn ×speedup | forwards |
|---|---|---|
| baseline | 1.00 | 28/28 |
| spectrum_alone (lib defaults) | 1.75 | 16/28 |
| spd_alone | 1.43 | 28/28 |
| naive_compose | 2.25 | 19/28 |
| **banded_compose** | **1.92** | 23/28 |
| spectrum_node (the real competitor) | 1.75 | 16/28 |

**Verdict: PROCEED** — banded ×1.92 > Spectrum-node ×1.75 (+0.17, thin, within
proxy slop). The R2 fact is sharper than the proposal states it: refreshing the
HF coefficients of the block-output feature *requires running the blocks* (the
feature is the blocks' output; there is no partial forward that yields only HF),
so a band-aligned step can skip blocks **only when σ < σ_resolve(top band) =
0.29**. The arith gate says the SPD token-saving still buys back more than that
costs — so the question moves to whether the band-aligned forecast has *signal*.

> Correction carried from the precondition work: Spectrum on Anima is **≈×1.75**,
> not the ×3.75 the proposal originally cited (wall-clock: 7.7 s vs 13.3 s
> baseline = ×1.73 @ 768²). So "SPD competes with Spectrum" softens to "SPD ≈
> Spectrum," which is why stacking — not standalone speed — is the whole case.

## Phase-0(a) — naive-compose eyeball (`--mode gpu`)

Reset the Spectrum forecaster at the SPD handoff and re-warm; eyeball the seam.
3 seeds, 768², single-late knee 0.5→1.0 @ σ0.7, CFG=4.

- naive-reset compose is **coherent in all seeds, no seam smear**, sharper than
  baseline (the SPD signature), and **fastest at ×1.98 wall** (vs Spectrum
  ×1.73). The proposal's central fear — re-warm at the handoff crumbling quality
  — did **not** materialize.
- **Kill-up tentatively fires:** if a naive reset already composes cleanly, the
  band-aligned forecaster isn't needed *for quality*. (Caveats: σ0.7 puts the
  re-warm early where it overlaps natural warmup, so it's cheap; a later/
  aggressive knee or HF-detail prompts would stress it more — untested.)

## Phase-0(b) — feature LL-DCT continuity (`--mode continuity`) — **the killer**

The real precondition. SPD's spectral continuity is proven for the *latent*; the
`final_layer`-input feature `x_B_T_H_W_D` is a *nonlinear* function of it, so its
LL band being continuous across the handoff is a hypothesis, not a theorem.

**Method.** Capture the cond feature each Euler step over a real `--spd`
trajectory; bring its LL band onto the common stage-0 patch grid via SPD's own
`dct_lowpass_init` (the operator `spectral_expand` inverts); compare the **seam**
residual (stage-0 Chebyshev fit, keyed to the re-spaced σ̃, extrapolated to the
first stage-1 step) against the **within-stage** one-step forecast residual.
Reported raw (carries the κ scale step) and standardized (per-channel unit-var =
shape only). 5 real prompts sampled from `image_dataset/` captions (seed 40),
1024²/CFG=4. PASS gate: standardized ratio ≤ 2.5.

**Results (corrected metric — see erratum):**

| config | knee | standardized ratio | within | raw ratio | verdict |
|---|---|---|---|---|---|
| bare DiT | σ0.7 | **×4.06 ± 0.68** | ~0.20 | 17.1 | FAIL |
| bare DiT | σ0.5 | **×4.49 ± 0.64** | ~0.20 | 15.6 | FAIL |
| **+ `anima_spd.safetensors`** (Case-B, its trained knee) | σ0.5 | **×5.64 ± 0.78** | ~0.20 | 18.6 | FAIL |

**What FAILs, precisely.** The LL *norm* is continuous across the seam (the
standardized LL-norm trajectory is flat ~1447.4 with no jump). What breaks is the
LL *vector*: within a stage consecutive LL patterns are nearly collinear
(residual ~0.20), but across the seam the pattern **reorients hard** (residual
~0.88) at preserved norm. Mechanistically — at the transition SPD overwrites the
latent's HF slots with fresh σ-scaled noise; the feature is the block *output*
and attention is global, so the feature's LL-scale spatial pattern gets rewritten
even though its energy is conserved. The band-aligned forecaster carries that
pattern, so a reorientation breaks its forecast right after the seam.

This is **R1**, the proposal's flagged "most likely killer." Because
standardization already grants more scale freedom than the single κ scalar the
forecaster could actually apply, no scale transform (κ or otherwise) repairs a
direction change — the FAIL is robust to the normalization choice.

**Does the SPD fine-tune narrow the gap?** No — it *widens* it (×4.49 → ×5.64 at
the matched σ0.5 knee; within unchanged ~0.20). The Case-B adapter is trained to
produce correct velocity at the κ-expanded stage-entry state, i.e. to *actively
reveal the deferred HF* at the handoff — which is exactly what perturbs the LL
pattern. The trajectory adapter and the band-aligned-forecast premise are pulling
in opposite directions. So the proposal's one escape hatch — its listed "later
stack: band-aligned Spectrum on an SPD-LoRA trajectory" — is also falsified by
direct measurement, not just the training-free path.

### Erratum — the first 0(b) FAIL was contaminated (caught 2026-05-21)

The initial runs reported ×3.65 / ×4.02 / ×4.82 and I concluded FAIL. The plotted
standardized LL-norm jumped 1450→2350 at the seam, which a sharp eye flagged as
suspicious. Root cause: `feat_ll_vector` standardized each feature over its
*native* patch grid (32² stage-0 vs 64² stage-1) **before** the DCT/LL extraction;
the grid doubling makes per-channel std differ ~2×, injecting a spurious scale
step exactly at the seam. Proven on a synthetic that is continuous *by
construction* (stage-1 = stage-0 embedded via SPD's convention):
native-grid-standardize → false seam jump ≈ 1.0; the fix → ≈ 0.0.

**Fix:** low-pass every feature to the common stage-0 grid via `dct_lowpass_init`
*before* standardizing, so the normalization is over the same grid at every step.
After the fix the LL norm is continuous (the artifact jump is gone) **but the
seam ratio stays ×4–5.6** — the genuine pattern reorientation survives. So the
verdict is unchanged; the *reason* is refined from "the LL magnitude jumps" to
"the LL pattern reorients at preserved norm."

**Lesson:** a metric that changes basis across the very seam it measures must be
unit-tested on a continuous-by-construction synthetic before any PASS/FAIL is
trusted. (The buggy result dirs were deleted; the table above is the corrected
metric.)

## Phase-0(b) follow-on — the "opposite LoRA" frontier (`--mode frontier`) — also FAILS

The 0(b) finding ("the SPD-LoRA *widens* the seam because it's trained to reveal
HF") prompted a natural counter-idea (`proposal.md` discussion 2026-05-21): instead
of training the trajectory adapter to reveal HF, train an **opposite** adapter that
*minimizes* the seam reorientation — a `loss_seam` regularizer. Before training
anything, the cheap gate is: does the **frozen** DiT have *any* operating point where
the seam is forecastable with detail intact? If yes, training has an anchor to pull
toward; if not, a LoRA would have to manufacture a regime the base never exhibits.

`--mode frontier` sweeps (transition σ × HF-injection γ) on the bare DiT — γ added as
`spectral_expand(hf_scale=…)`, the same knob a `loss_seam` would use — and reports the
standardized seam ratio (continuity) vs HF latent energy relative to the full-res
baseline (detail). 1024²/CFG=4/seed 40, 5 prompts, gate ≤2.5.

| σ \ γ | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| **0.7** seam | ×4.07 | ×4.32 | ×4.82 | ×4.48 | **×4.06** |
| **0.7** detail | 0.39 | 0.42 | 0.48 | 0.71 | 1.16 |
| **0.5** seam | ×4.94 | ×5.16 | ×5.50 | ×5.21 | **×4.49** |
| **0.5** detail | 0.17 | 0.17 | 0.18 | 0.31 | 0.93 |

**Verdict: INTRINSIC FAIL.** Two things kill the opposite-LoRA bet:

1. **No frozen cell clears the gate** — every cell is ×4.06–5.50, all ≫2.5. There is
   no nearby operating point for a `loss_seam` to anchor to.
2. **HF magnitude is the wrong knob, and points the wrong way.** Reducing γ does *not*
   improve continuity — the seam ratio is non-monotone, peaks mid-γ, and is *lowest at
   γ=1.0* (the full paper fill) at both knees. Meanwhile detail collapses monotonically
   as γ→0 (0.93→0.17 at σ0.5). So γ→0 is strictly dominated: worst continuity *and*
   worst detail. Mechanistically, γ<1 is an *off-manifold* (under-noised) state, and the
   DiT's feature response to a malformed input is *less* continuous, not more — which
   confirms 0(b)'s reading that the reorientation is the global-attention feature re-mix
   at the grid change, not a function of how much fresh HF is injected.

The σ0.5/γ1.0 cell reads **×4.49**, reproducing the 0(b) continuity table exactly — a
built-in cross-check that the frontier metric and the continuity metric agree.

**Consequence:** do **not** proceed to a `loss_seam` overfit. The "opposite LoRA" has
no frozen frontier to move toward; it would have to *create* a continuity regime the
base model never shows at any injection strength — a far bigger bet than "shift the
Pareto frontier," and unmotivated by any positive signal. The SPD trajectory adapter's
forward path stays the two surviving ideas (projected-teacher low-res target +
on-policy handoff tail), neither of which touches the Spectrum-carry premise.

## Decision

- **Phase 1 (band-aligned forecaster): do NOT build.** 0(b) shows the feature's
  LL band does not flow through the handoff (pattern reorients), bare *and* with
  the SPD-LoRA. The only thing that survives the seam is the LL norm, which is
  not enough to forecast from.
- **If SPD∘Spectrum ships:** the 0(a) naive-reset compose (coherent, ×1.98). SPD's
  standalone case remains its quality signature + the Case-B fine-tune, not
  speed-stacking with Spectrum.
- **"Opposite LoRA" (train for seam continuity): do NOT build.** The frozen-frontier
  follow-on (above) shows no operating point clears the seam gate at any HF-injection
  strength, and less injection makes continuity *worse* — the regularizer has no anchor
  and the wrong knob. Falsified before training, at the cost of one offline sweep.
- **Open / hardening:** all 0(b) runs are single-seed (40); a 2nd seed would
  harden it, though the 5-prompt spreads are tight and consistent across 3
  configs. A κ-corrected-raw variant (apply the known κ, no per-channel std) is
  the most *faithful* forecaster test, but it is bounded below by the
  standardized number, which already fails.

## Reproduce

```bash
uv run python -m bench.spd.probe_compose --mode arith
uv run python -m bench.spd.probe_compose --mode gpu --n_prompts 5
uv run python -m bench.spd.probe_compose --mode continuity --n_prompts 5            # bare σ0.7
uv run python -m bench.spd.probe_compose --mode continuity --n_prompts 5 \
    --spd_transition_sigmas 0.5                                                     # bare σ0.5
uv run python -m bench.spd.probe_compose --mode continuity --n_prompts 5 \
    --spd_transition_sigmas 0.5 --lora output/ckpt/anima_spd.safetensors           # + Case-B adapter
uv run python -m bench.spd.probe_compose --mode frontier --n_prompts 5 \
    --frontier_transition_sigmas 0.7 0.5 --frontier_hf_scales 0.0 0.25 0.5 0.75 1.0  # opposite-LoRA gate
```

Result envelopes (corrected metric): `bench/spd/results/20260521-1405-phase0b-bare-s07-fix/`,
`…-1407-phase0b-bare-s05-fix/`, `…-1410-phase0b-spdlora-s05-fix/` (each with
`ll_continuity.png`); frontier follow-on: `…-1507-seam-frontier/` (`frontier.png`).
