# SPD ∘ Spectrum — band-aligned feature forecasting across the resolution handoff

A proposal to compose **SPD** (Spectral Progressive Diffusion, token reduction via
progressive resolution — `networks/spd.py`) with **Spectrum** (Chebyshev feature
forecasting, block-skipping on cached steps — `networks/spectrum.py`) so the two
*multiply* their speedups instead of fighting over the same part of the trajectory.

The two are currently **mutually exclusive** — both replace the denoise loop, and
`docs/experimental/spd.md` warns-and-ignores Spectrum under `--spd`. That exclusion
is correct as shipped (the naive compose breaks; see below), not merely conservative.
This proposal is the design for making them compose, plus the cheap bench that can
kill it first.

## TL;DR of the problem

- Spectrum forecasts the **block-output feature** captured by the `final_layer`
  pre-hook (`spectrum.py:299`) in **spatial-token space**, and
  `ChebyshevForecaster` hard-asserts the feature shape never changes
  (`spectrum.py:113`, *"Feature shape must remain constant"*).
- At an SPD resolution transition the feature is on a different token grid
  (low-res → full-res). The stage-0 forecaster buffer is unusable in stage-1.
  The only naive option is to **reset the forecaster at the handoff and re-warm
  from scratch**.
- That reset lands in the worst possible place. Spectrum forces `warmup_steps`
  (default 6) actual forwards + 3 forced at the tail (`spectrum.py:286, 318`). On a
  28-step run with a σ≈0.5 knee, stage-1 is ~14 steps — so ~9 of 14 **full-res**
  steps become forced forwards, and the speedup evaporates in exactly the expensive
  phase. Worse, the re-warm happens where Chebyshev is *least* accurate: late,
  full-res, HF emerging → features changing fast.

The deeper issue is **redundancy**: Spectrum (block-skip) and SPD (token-shrink)
both harvest savings from the *smooth early region* of the trajectory, by different
mechanisms. SPD already made the early steps cheap by lowering resolution, so there
is little left for Spectrum to skip there; and Spectrum is forced to re-warm in the
late full-res phase where it is both most needed and worst-behaved. The `bench/spd/plan.md`
Phase-3 framing — *"orthogonal: token-reduction vs block-skipping"* — is true on the
**FLOP axis** but false on the **trajectory-region axis**. This is why standalone SPD
(×1.73 max, `docs/experimental/spd.md`) is roughly on par with the Spectrum node
(~×1.75): they compete for the same budget. A naive compose will likely land *below*
Spectrum alone.

## The idea: forecast in the band SPD preserves, forward the band it adds

SPD already keeps the latent in **DCT space** at the transition (`dct_lowpass_init` /
`spectral_expand`, paper Eq. i–iii + 5–6). Its whole premise (spectral autoregression,
ref [9]; `bench/spd/` Phase 0/1) is that **low-frequency content is resolved early and
preserved across the resolution handoff** — stage-1's LL block *is* stage-0's spectrum,
embedded into a larger grid with fresh HF slots appended.

So instead of forecasting the raw spatial-token feature with a hard shape lock,
forecast the **2D-DCT coefficients of the feature** along the patch grid:

- The **LL coefficient band is (hypothesised) continuous across the transition** —
  the stage-0 trajectory of those coefficients continues straight into stage-1, no
  shape break, no re-warm. The Chebyshev window *survives the handoff* for the LL band.
- **Actual forwards in stage-1 are spent only on the newly-added HF slots** — which is
  exactly the band SPD declares unresolved and the whole reason resolution was raised.

This dissolves both failure modes at once: the shape break (forecast the shared band,
not the changing grid) and the misplaced re-warm (the LL forecast carries through;
only HF needs warming, and HF is cheap to warm because it's a small slice). It is also
strictly *more principled* than vanilla Spectrum: forward the band that's genuinely
changing, cache the band that's settled — the same logic SPD uses to pick its
resolution schedule.

This is the only version of the composition that can multiply rather than re-warm.
If you reset at the transition you get SPD's stage-0 saving + a crippled Spectrum in
stage-1, which is plausibly worse than either alone.

### Two seams that must be handled or the forecast biases at the boundary

Same family of gotchas that bites DCW/SMC against SPD (the σ-reindex), here for the
forecaster:

1. **Fit in aligned σ̃, not step-index.** Spectrum's Chebyshev basis is keyed to
   **step index** (`_taus`, `spectrum.py:87`: `τ = 2(t/total_steps) − 1`). SPD
   re-spaces the σ schedule at the handoff (Eq. 5–6), so the step→σ map has a
   discontinuity there. Re-key the basis to σ̃ so the polynomial doesn't straddle the
   reshape.
2. **Carry stage-0 coefficient observations through SPD's κ rescale.** `spectral_expand`
   scales the latent by `κ = r/(1+(r−1)σ)` and realigns the timestep. The feature is a
   (nonlinear) function of that rescaled input, so the LL-coefficient buffer can't be
   concatenated raw — push the stage-0 observations through the same κ/σ̃ transform
   before continuing the fit, or the seam is biased.

### The cache schedule itself can come from stage-0

A bonus the structure hands us. Spectrum's cache *schedule* (which steps to skip) is
**not signal-derived today** — it's a deterministic growing-window heuristic on step
index (`curr_ws`, `spectrum.py:321`). The *principled* "which steps are forecastable"
map = feature smoothness = the **per-band `σ_resolve` curve** measured in `bench/spd/`
Phase 1 (`measure_autoregression.py`: low bands lock by σ≈0.75, top band by σ≈0.29).
That curve is a property of the latent spectral autoregression — **resolution-independent,
derivable from stage-0 / offline**. Under the band-aligned forecaster it maps directly:
a band stays cacheable until its `σ_resolve`, then flips to actual-forward. This replaces
the window heuristic with a derived per-band schedule and ties the composition back to
the Phase-1 bench we already have.

## Why this is worth doing

1. **It's the open question the SPD bench already flagged.** `bench/spd/plan.md`
   Phase 3 lists "SPD∘Spectrum" as the real value question (not SPD vs a naive
   50-step baseline) but gives it no design — just "Bench both." This is that design.
2. **Standalone SPD doesn't justify itself on speed alone** (×1.73 ≈ Spectrum's
   ×1.75). SPD's standalone case is its *quality signature* (sharper/higher-contrast)
   and the fine-tune (Case B). For SPD to earn a place on the **speed** axis it has to
   *stack* with Spectrum. This proposal is that stacking story.
3. **The pieces already exist.** DCT primitives in `networks/spd.py`
   (`dct_lowpass_init` / `spectral_expand`), the forecaster in `networks/spectrum.py`,
   and the σ_resolve curve in `bench/spd/`. The new work is a band-aware forecaster and
   the σ̃/κ bookkeeping at the seam — not new math.

## Phasing & gates

Gated like `bench/spd/plan.md` — each phase can kill the idea before the next.

### Phase 0 — preconditions (cheap, ~1 day, two measurements that can both kill it)

Both run on a real `--spd` trajectory; no pipeline changes (a throwaway probe like
`probe_lowres_denoise.py`).

- **(a) Naive-compose floor.** Reset the Spectrum forecaster at the SPD transition and
  measure end-to-end speedup + CMMD vs Spectrum-alone and SPD-alone, single-late knee.
  - **Kill-up:** if naive compose already multiplies cleanly (≥ Spectrum-alone speedup
    at matched quality), ship *that* — the band-aligned forecaster isn't needed. (Low
    probability given the re-warm arithmetic above, but it's the cheapest possible win.)
  - **Proceed:** naive compose ≤ Spectrum-alone (expected) → the band-aligned version
    is the only path; go to (b).
- **(b) Feature LL-DCT continuity probe — the real precondition.** SPD's spectral
  continuity is proven for the **latent** (`bench/spd/` Phase 0/1); the feature
  (`final_layer` input) is a *nonlinear* function of the latent, so its LL-DCT band
  being continuous across the handoff is a **hypothesis, not a theorem**. Measure it:
  on one seed, 2D-DCT the captured feature along the patch grid at the steps bracketing
  the transition, embed the stage-0 LL block into the stage-1 grid (mirroring
  `spectral_expand`), apply the κ/σ̃ transform, and report the LL-coefficient
  trajectory's smoothness across the seam (e.g. residual of a Chebyshev fit through the
  transition vs within a single stage).
  - **Pass:** LL coefficients are smooth across the seam (fit residual at the boundary
    ≈ within-stage residual) → the band-aligned forecast has signal. → Phase 1.
  - **Fail:** LL coefficients jump at the seam → the feature doesn't inherit the
    latent's spectral continuity; band-aligned forecasting is dead. Ship Case-A SPD or
    Spectrum standalone, and record the negative result. **This is the most likely
    killer and the reason Phase 0 exists.**

### Phase 1 — band-aligned forecaster + speed/quality bench (~3 days, conditional on 0b PASS)

- Implement a banded forecaster variant: 2D-DCT the captured feature, maintain the
  Chebyshev window on the **LL coefficient block** keyed to σ̃, carry it through the κ
  transform at the transition; actual-forward to refresh **HF slots** in stage-1.
- Bench (`bench/spd/bench_compose.py`, standard `bench/_common.py` envelope) on the
  production env (CFG=4, the DCW aspect buckets): speedup + ImageReward / CLIP-IQA /
  CMMD for **{ full-res baseline, Spectrum-alone, SPD-alone, naive-compose,
  band-aligned-compose }**.
- **Pass:** band-aligned compose beats Spectrum-alone speedup at matched (within-noise)
  quality — i.e. the SPD token saving genuinely *adds on top of* Spectrum's block
  saving.
- **Weak:** band-aligned ≈ Spectrum-alone → the HF-slot forwards in stage-1 eat the
  SPD saving; the composition is a wash. Ship Spectrum-alone; document.

### Phase 2 — schedule derivation from σ_resolve (~1 day, optional polish)

- Replace Spectrum's window heuristic with the per-band `σ_resolve` cache schedule from
  `bench/spd/measure_autoregression.py`. Re-run the Phase-1 bench with the derived
  schedule.
- **Pass:** derived schedule matches or beats the heuristic → single-knob (the same δ
  that SPD uses) drives both the resolution transitions *and* the cache schedule. Clean
  story.

**Skip rule:** if Phase 0(b) FAILs, stop — the whole approach rests on feature LL-DCT
continuity. If Phase 1 is WEAK, ship Spectrum-alone; SPD's case remains quality + the
Case-B fine-tune, not speed-stacking.

## Risks and failure modes

### R1: feature LL-band is not continuous across the handoff
The headline risk — gated explicitly by Phase 0(b). The latent's spectral continuity
doesn't automatically transfer to the nonlinear block-output feature. If it fails, the
band-aligned forecast has no signal and the proposal is dead.

### R2: HF-slot forwards in stage-1 cost as much as full forwards
Spectrum's `_spectrum_fast_forward` (`spectrum.py:208`) skips *all blocks* on cached
steps — it only runs `t_embedder + final_layer + unpatchify`. But the band-aligned
variant needs the **HF coefficients of the block output**, which requires *running the
blocks*. So "forward only HF slots" is not free the way a cached step is — you still
pay a full block forward, you just discard/forecast the LL part of its output.
*Mitigant / reframing:* the saving then is **not** "skip blocks for HF steps" but
"skip blocks for LL-dominated steps and use the SPD low-res grid for the rest." The
honest win may be narrower than the framing suggests — Phase 1's bench is what
adjudicates whether the combined saving beats Spectrum alone. If R2 bites hard, the
composition reduces to "Spectrum on the low-res stage only," which is close to the
naive floor. **This is the second-most-likely killer and Phase 1 must measure it
directly, not assume the multiply.**

### R3: σ̃/κ bookkeeping is fiddly and silently wrong
A biased seam transform degrades quality subtly (not a crash). *Mitigant:* unit-test
that the band-aligned forecaster, run with a single stage (no transition), reproduces
vanilla Spectrum bit-for-bit — the transition machinery must be a no-op when there's
no transition.

### R4: Euler-only constraint inherited from SPD
SPD v0 is Euler-only (ER-SDE/LCM precompute coefficients incompatible with mid-loop σ
re-spacing — `docs/experimental/spd.md`). The composition inherits this. Not a new
risk, just a scope note: the bench runs Euler.

## Out of scope

- **DCW / SMC-CFG in the same run.** Those have their own SPD-composition story
  (the σ-reindex; see the discussion that spawned this proposal). One axis at a time —
  this proposal is SPD ∘ Spectrum only.
- **Turbo.** SPD-vs-Turbo at matched quality is a separate `bench/spd/plan.md` Phase-3
  line. Turbo is few-step (different payload), not block-skip.
- **The SPD fine-tune (Case B).** Orthogonal — Case B is a trained adapter; this is a
  training-free inference composition. They could later stack (band-aligned Spectrum on
  an SPD-LoRA trajectory), but not in v0.
- **Video / temporal frequencies.** Anima is image-only.

## File-level plan

New files:
- `bench/spd/probe_compose.py` — Phase 0 throwaway: (a) naive-reset compose timing +
  CMMD, (b) the feature LL-DCT continuity measurement. No pipeline changes.
- `bench/spd/bench_compose.py` — Phase 1 five-way speed/quality bench. Standard
  `bench/_common.py` envelope.

Touched files (only after Phase 0 PASS):
- `networks/spectrum.py` — banded forecaster variant (DCT the captured feature; window
  on LL coefficients keyed to σ̃; κ-transform carry at the transition). Gated behind a
  flag so vanilla Spectrum is untouched by default.
- `networks/spd.py` — expose `spectral_expand`'s coefficient-embedding + κ/σ̃ transform
  as reusable helpers the forecaster can call (the bookkeeping must match the sampler
  bit-for-bit — same lesson as the Case-B train/infer geometry alignment).
- `library/inference/generation.py` — allow `--spd` + `--spectrum` together (lift the
  mutual-exclusion raise) once the composed runner exists.
- `inference.py` — CLI: a single composed path, no new user-facing flag beyond passing
  both `--spd` and `--spectrum`.

## Open questions before kicking off

1. **Does R2 sink it before R1?** If running blocks to get HF coefficients costs a full
   forward, the only block-skip saving is on LL-dominated steps inside the low-res
   stage — which SPD already made cheap. Worth a back-of-envelope FLOP estimate *before*
   even Phase 0(b): count, for the single-late knee, how many steps are LL-dominated
   (cacheable) vs HF-active (must forward) per the σ_resolve curve, and whether that
   leaves a multiply. If the arithmetic says no, this proposal is a paper exercise and
   we stop here.
2. **Band granularity.** Single LL/HF split (cheapest) vs the K-band radial decomposition
   already in `measure_autoregression.py`. Start with LL/HF; the K-band version is a
   Phase-2 refinement only if Phase 1 shows a clean per-band cacheability gradient.
3. **Flag surface.** Composed-only path (pass both flags) vs an explicit `--spd_spectrum`
   mode. Default to the former unless the runner divergence is large enough to warrant
   its own entry point.
