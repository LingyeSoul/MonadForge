# FSG follow-up plan — CFG++ λ sweep & production (er_sde, ~30-step) calibration

**Status:** mechanism settled; tuning + Tier-2 gates open. This plan sequences the
two remaining calibration jobs (λ for the CFG++ substrate, band/K for the real
step count + sampler) and the confound read that decides whether FSG ships.

Read first: `docs/proposal/foresight_guidance.md`, memory
`project_fsg_golden_path_phase0`. Tools: `probe_golden_path.py` (gap/ρ mechanism),
`render_compare.py` (eyeball A/B). Production point: **er_sde, CFG=4, 28 steps,
flow_shift 3.0, 1024 tier**.

## What's settled — do NOT redo

- **Band/operator:** FSG contracts only in **mid-σ**; working band [0.45, 0.85],
  narrow [0.75, 0.85], K=3, Δσ=0.1. σ≈0.94 diverges — confirmed against THREE
  confounds: fixed-Δσ, long Δσ=0.5·σ (worse), and CFG-vs-CFG++ substrate (CFG++
  mitigates ~5× but doesn't cure). The band does not move with substrate/interval.
- **CFG++ substrate:** implemented as a σ-scheduled guidance reweight
  (`sampling.cfgpp_guidance_weight`, `w_eff = λ(1−σ')σ/(σ−σ')`), composes with
  er_sde/Euler/lcm. λ is a **flow-space** coeff (NOT the paper's DDIM λ=0.6).
  er_sde+cfg++ λ2 already renders cleanly. Bit-identical to the Euler
  calibrate-then-step form (invariant test).
- **The shipped plugin is `fsg/cfg`** (foresight on plain CFG) — an off-paper
  variant; real FSG = `fsg/cfg++`. Both land in the same mid-σ band on Anima.
- **No trustworthy Anima quality reward** (null-TTA negative; CMMD/PE are
  global-tone only) → final calls are eyeball A/B + saturation/contrast stats.

## Schedule facts (flow_shift 3.0) — drive the NFE budget

| steps | #steps in [0.75,0.85] | in [0.45,0.85] |
|---|---|---|
| 20 (bench so far) | 4 | 9 |
| 28 (production) | 5 | 13 |
| 30 | 5 | 13 |

`w_eff(λ=2)` over 28 steps ramps **2.0 → peak ~15 (mid-σ) → 2.0**: CFG++ guidance is
**mid-σ-loaded**, peak ≈ 7.5·λ (bounded because Δσ is small where w_eff is large).

**NFE cost of FSG** = base `2N` (cond+uncond) + `3·K·n_band` extra forwards.
At 28 steps, narrow band (n_band=5), K=3 → 56 + 45 = **101 forwards ≈ 1.8× plain CFG**.
This sets the matched-NFE baseline (Plan C): plain CFG at ~50 steps.

---

## Plan A — CFG++ λ sweep (pick λ*)

**Goal:** find λ where `cfg++` matches-or-beats plain CFG=4 quality. Anima was tuned
for CFG=4; λ is a free flow-space coeff. Estimate: λ≈1.5–2 ≈ CFG=4 total guidance.

**Hypothesis:** too-low λ → washed-out/under-guided; too-high → over-saturated
(the mid-σ peak amplifies). There is a λ* near 1.5–2 that tracks CFG=4.

**Grid:** λ ∈ {1.0, 1.5, 2.0, 3.0} (add 4.0 only if 3.0 still looks under-guided).

**Method:** render `cfg++(λ)` vs CFG=4 baseline, **same prompts/seeds, er_sde**,
4–6 real captions × 2 seeds, 28 steps. For each arm log:
- **mean HSV saturation** and **RMS contrast** (the saturation-confound metric —
  the whole FSG "win" question is "is it just a tone bump"; quantify it here, not
  just eyeball). λ* = closest sat/contrast to baseline with equal-or-better detail.
- eyeball: clean lines, no blow-out, no wash-out.

**Success:** one λ* where `cfg++(λ*)` ≈ baseline saturation/contrast (±~10%) and is
visually as clean or cleaner. Record λ* → feeds Plan B & C.

**Tooling gap (prereq):** `render_compare.py` is Euler-only and renders a fixed
4-arm set. Need either (a) extend it: `--sampler {euler,er_sde}` + `--cfgpp_lambdas`
(one arm per λ) + per-arm sat/contrast in the result.json, **or** (b) a thin
`lambda_sweep.py` looping `inference.py --cfgpp --cfgpp_lambda <λ> --sampler er_sde`
into one contact sheet. Prefer (a) — keep one calibration instrument.

---

## Plan B — production calibration (er_sde, 28–30 steps)

**Why:** band/K were tuned at **Euler, 20 steps**. Production is **er_sde, 28 steps**.
The σ-band is resolution-of-step-count-independent (it's a σ-level property, already
robust), so [0.75,0.85] should still be the contracting band — but two things change:

1. **#steps-in-band 4→5** ⇒ more foresight applications ⇒ more cumulative effect +
   more NFE. K=3 may now **over-calibrate**; test K∈{2,3}.
2. **er_sde stochasticity** ≠ deterministic Euler — the operator was only ever
   measured on Euler. er_sde injects noise each step; re-confirm the gap still
   shrinks / the win survives the stochastic sampler (this is the standing
   "er_sde flipped σ-reshape once" risk, `project_sigma_reshape_no_win`).

**Steps:**
1. **Re-probe mechanism at 28 & 30 steps:**
   `probe_golden_path.py --infer_steps 28 --cfgpp --cfgpp_lambda <λ*>` and `--infer_steps 30`.
   Confirm σ=0.85 still sweet spot, σ=0.94 still diverges, band contracts on the
   denser grid. (Probe trajectory is deterministic; this checks the *operator*, not
   the sampler.)
2. **Re-tune K for the new band-step count:** render `fsg/cfg++` at 28 steps with
   K∈{2,3}, band [0.75,0.85]; pick the smallest K that holds the win (error ~ρ^K,
   ρ≈0.9 ⇒ K=2 may suffice once n_band=5). Lower K = lower NFE.
3. **er_sde sampler check:** render the chosen config on **er_sde** (production) vs
   the Euler render — confirm the win isn't a deterministic-only artifact.
4. **Tier-table note:** band is 1024-tier; 768 shifts down to ~[0.62,0.75]
   (`foresight_guidance.md §1a`). 28/30-step calibration here is 1024 only; the
   per-tier band table is a separate task.

**Success:** a production config `(band, K, λ*, sampler=er_sde, steps=28)` whose
`fsg/cfg++` render holds the visible win at the lowest defensible NFE.

---

## Plan C — the confound read + Tier-2 gates (depends on A, B)

The payoff question. With λ* and the production config fixed, render one sheet,
same prompts/seeds, er_sde, 28 steps:

| arm | tests |
|---|---|
| baseline (CFG=4) | reference |
| cfg++ (λ*) | substrate alone |
| fsg/cfg | shipped variant (foresight on CFG) |
| **fsg/cfg++ (λ*)** | **faithful FSG** |

**Decisive read — `fsg/cfg++` vs `cfg++`:** if foresight still helps on a substrate
where it **can't** masquerade as extra CFG (the first-order foresight≈+CFG argument
only bites on the CFG base), the win is a real golden-path effect. If not, the
original `fsg/cfg` win was the effective-CFG-boost confound — close the line.

**Remaining Tier-2 gates (from the proposal doc):**
- **Matched-NFE** (decisive): `fsg/cfg++` @28 (≈101 fwd) vs plain CFG @~50 steps
  (≈100 fwd). If the longer plain run matches it, the knob is NFE-for-nothing.
- **Saturation confound:** already quantified in Plan A (sat/contrast stats) — carry
  the metric into this sheet so "better" isn't just a global tone bump.

**Decision gate:** ship to the Spectrum node (per `foresight_guidance.md`) **only if**
`fsg/cfg++` beats both `cfg++` AND matched-NFE plain CFG, on the eyeball + sat/contrast.
Otherwise write the negative finding and close.

---

## Sequencing

1. **Tooling:** extend `render_compare.py` (`--sampler`, `--cfgpp_lambdas`,
   sat/contrast in result.json). One small PR.
2. **Plan A** → λ*.
3. **Plan B** → production `(band, K)` at 28/30 steps on er_sde.
4. **Plan C** → confound read + matched-NFE. Ship-or-close decision.

Each stage is one bench run + eyeball; record results under
`bench/fsg/results/<ts>-<label>/` and update `project_fsg_golden_path_phase0`.
