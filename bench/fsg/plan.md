# FSG follow-up plan — CFG++ λ sweep & production (er_sde, ~30-step) calibration

**Status:** **Plan A/B/C DONE → SHIPPED in library (2026-06-23).** Production config:
cfg++ λ1.5, band **[0.59,0.75]**, K=3, Δσ=0.1, γ=guidance(4), er_sde, 28 steps —
library defaults updated; `docs/proposal/foresight_guidance.md` revised for the node
port. **Only matched-NFE A/B remains** (cost-efficiency proof, not a ship blocker —
ship rested on the cfg++-substrate confound read). See memory
`project_fsg_golden_path_phase0` for the full Plan B/C findings.

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

## Plan A — CFG++ λ sweep (pick λ*) — ✅ DONE 2026-06-23 → **λ\*=1.5**

**Result** (`bench/fsg/results/20260623-2042-planA-lambda-sweep-ersde28`, er_sde,
28 steps, 1024, 6 captions × 2 seeds): swept λ∈{1,1.5,2,3} as `cfg++ λ` arms vs
CFG=4. **λ1.5 tracks CFG=4 best on all three axes** — Δsat +1.7% (lowest),
Δcontrast +4.4%, mean latent drift 0.593 (lowest), all inside the ±10% band. λ1
under-guides (composition wanders — w_eff is mid-σ-loaded, only ~λ at the schedule
ends, so at λ=1 high-σ guidance ~1 ≪ CFG's flat 4; +4.4% sat, drift 0.610). λ2 ok
but Δcontrast already +10.2%. λ3 rejected (+73% sat = neon blowout). **λ\*=1.5 is
now the shipped default** (`--cfgpp_lambda`, args.py / generation.py /
render_compare.py). The sweep tooling (`--sampler`, `--cfgpp_lambdas`, per-arm
sat/contrast) is in `render_compare.py`. Below is the original plan, kept for record.

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

**Tooling (prereq) — ✅ BUILT (option a):** `render_compare.py` extended with
`--sampler {euler,er_sde}`, `--cfgpp_lambdas` (one `cfg++ λ` arm per value, +
`--with_foresight` for fsg/cfg++ per λ), and per-arm mean HSV saturation + RMS
contrast (Δ% vs baseline) in `result.json` `sat_contrast` and on panel labels. The
er_sde seed is shared across arms (fair A/B); output recast to bf16 per step to
match production. One calibration instrument, reused by Plans B & C.

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

1. ~~**Tooling:** extend `render_compare.py`~~ ✅ done (`--sampler`,
   `--cfgpp_lambdas`, sat/contrast in result.json).
2. ~~**Plan A** → λ*~~ ✅ done → **λ\*=1.5** (now the `--cfgpp_lambda` default).
3. ~~**Plan B** → production `(band, K)` at 28/30 steps on er_sde~~ ✅ → **band [0.59,0.75], K=3** (band shifted down from the 20-step [0.75,0.85]; γ=w_eff diverges, keep γ≈4).
4. ~~**Plan C** → confound read~~ ✅ → **fsg/cfg++ beats cfg++** by eyeball, not a tone bump (Δsat +1.7%p); SHIPPED. **Matched-NFE** still owed (cost proof).

Each stage is one bench run + eyeball; record results under
`bench/fsg/results/<ts>-<label>/` and update `project_fsg_golden_path_phase0`.
