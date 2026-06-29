# Foresight Guidance (FSG) — shipping to the Spectrum ComfyUI node

**Status:** library plugin BUILT + **shipped with the production config** (2026-06-23 Plan
A/B/C). er_sde/28-step calibration done, eyeball A/B done, saturation confound quantified;
**library defaults now = the production point** (see §0). **CFG++ is now wired into the
spectrum runner too** (2026-06-23) — `--cfgpp` threads through `SamplerSideChannels` and the
spectrum CFG-combine applies the σ-scheduled reweight, so faithful `fsg/cfg++` composes under
`--spectrum` (previously cfgpp was dropped under spectrum). **Node port DONE** (2026-06-23):
`fsg.py` + CFG++ ported to `~/ComfyUI-Spectrum-KSampler`, exposed on the Advanced node —
**pending in-ComfyUI verification** (ported against ComfyUI internals, not yet run live) and
the still-unrun **Matched-NFE A/B** (the one rigor item; the ship decision rested on the
cfg++-substrate read instead, see §6). Earlier build-decision content lives in git history.
**Paper:** "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations"
(NeurIPS 2025, arXiv 23177). **Library plugin:** `library/inference/corrections/fsg.py`
(`--fsg` / `FSG=1`). **Benches:** `bench/fsg/probe_golden_path.py`, `bench/fsg/render_compare.py`.
**Docs:** `docs/inference/fsg.md`. **Memory:** [[project_fsg_golden_path_phase0]].

## 0. The shipped production config (what the node must reproduce)

The library now defaults to the point validated on the production sampler. **The node must
expose / default to exactly this** — anything else is off-calibration.

| knob | value | source |
|---|---|---|
| substrate | **CFG++** (`--cfgpp`), σ-scheduled reweight | faithful Algorithm 1; composes with er_sde |
| `cfgpp_lambda` (λ) | **1.5** | Plan A — tracks CFG=4 saturation/contrast/composition |
| `fsg_band` | **[0.59, 0.75]** | Plan B — 28-step er_sde contracting band (1024 tier) |
| `fsg_k` (K) | **3** | Plan B — K=2 drift-saturates but K=3 adds visible detail (eyeball) |
| `fsg_d_sigma` (Δσ) | 0.1 | unchanged; stability is governed by γ·Δσ |
| `fsg_gamma` (γ) | guidance_scale (=4) | **must stay ≈4**, NOT matched to λ's w_eff — see §1b |
| sampler / steps | er_sde, 28 | production point |

**Two findings that reshape the old plan (don't undo):**
- **λ=1.5 ≠ "CFG=4 per-step".** The CFG++ substrate delivers a *bell-shaped* w_eff that peaks
  ≈11 in-band (≈2.7× CFG=4) but is applied over tiny Δσ, so the *integrated* tone matches
  CFG=4 (that's what λ=1.5 buys). λ is the height of a mid-σ-loaded schedule, not a flat scale.
- **Band moved down with step count.** [0.75, 0.85] was a *20-step Euler* artifact; at 28-step
  er_sde σ≈0.84 stops contracting and the sweet spot is σ≈0.75, so the band is **[0.59, 0.75]**.

## 1. What's already true (don't redo)

The operator (flow-matching translation of the paper's ε-pred/DDIM forward-backward) — at a
scheduled σ with latent `x`, interval `Δσ`, calibration guidance `γ`:

```
v^γ(x,σ) = v^u(x,σ) + γ·(v^c(x,σ) − v^u(x,σ))     # CFG-guided velocity
forward :  x' = x − Δσ · v^γ(x, σ)                  # denoise σ → σ−Δσ (conditional)
invert  :  x'' = x' + Δσ · v^u(x', σ−Δσ)            # re-noise back (unconditional)
F(x) = x'' ; iterate x ← F(x), K times ; then denoise from x̂ = x^(K)
```

Settled facts the node port inherits:

- **Library plugin exists and works** — `FSGCalibrator` (~145 lines, `corrections/fsg.py`),
  CLI surface, `GenerationRequest` fields, `test-*` flag composition, K=0/empty-band ==
  baseline invariant test (`tests/test_fsg_invariant.py`).
- **The library spectrum runner already honors FSG** — `networks/spectrum.py` computes
  `fsg_steps`, forces them to actual forwards, calls `fsg.calibrate()`, excludes them from
  the SEA decision denominator, and folds `(fsg_steps, k, d_sigma, gamma)` into the δ cache
  key (≈ lines 381–454, 495–536, 727). **The node port mirrors this diff.**
- **Validated band / K (1024 tier, 28-step er_sde):** band **[0.59, 0.75]**, **K=3**, Δσ=0.1,
  γ=guidance (=4). σ≈0.84+ no longer contracts at 28 steps and σ≈0.94 *diverges* (ρ>1) — the
  paper's "iterate in the noisiest stage" prescription is wrong on Anima. The 20-step Euler band
  was [0.75, 0.85]; the band is **step-count-dependent** (§1a) — re-probe if `infer_steps` moves.

### 1b. γ must NOT be matched to the CFG++ substrate (it diverges)

Naively, "faithful Algorithm 1 on CFG++" wants the foresight forward to push at the substrate's
own guidance (w_eff≈11 in-band). It **diverges**: at γ=11/Δσ=0.1 the in-band operator blows up
(gap *grows* 47–90%, ρ>1, drift 3× larger). The fixed-point operator's stability is set by the
product **γ·Δσ** — γ=4·0.1=0.4 contracts (ρ≈0.93), γ=11·0.1=1.1 runs away. So the only stable
regime is a **modest γ≈4** (plain-CFG-strength foresight) on top of the CFG++ substrate; faithful
γ=w_eff is a clean negative. Keep `fsg_gamma`=guidance. (To use a stronger γ you'd have to shrink
Δσ to hold γ·Δσ≈0.4 — untested, shortens the lookahead horizon.)

### 1a. Band is token-tier-dependent (resolution sweep, 2026-06-23)

The band is **not** globally resolution-invariant — this directly drives the node's default UX:

- **Robust across the dominant 1024 tier.** The four most-used real shapes
  (864×1216, 848×1232, 896×1200, 768×1360 — all ~4080–4200 tok) reproduce [0.75,0.85]/K=3:
  σ=0.85 is the sweet spot, frac_shrunk=1.0 in-band, σ=0.94 inverts. So [0.75,0.85] is sound
  for ~the whole head of the dataset.
- **Shifts DOWN at the 768 tier.** At 768² (~2.3k tok) the band slides ~one schedule notch:
  **σ=0.85 actively diverges** (gap +13%, ρ=0.97, both samples grew), the peak moves to
  **0.75**, and the clean band is **≈[0.62, 0.75]**. A fixed [0.75,0.85] default would fire
  half its steps in a diverging zone at low-token renders — i.e. *hurt* output, not just waste
  NFE. (Direction is opposite the naive guess — fewer tokens pushed it down. 1536²/512² unprobed.)

**Second axis — step count (2026-06-23, Plan B).** The band also slides down as `infer_steps`
rises, independent of tier. At the **1024 tier** it went [0.75, 0.85] @ 20-step Euler →
**[0.59, 0.75] @ 28-step er_sde** (σ≈0.84 stops contracting on the denser grid). So the shipped
[0.59, 0.75] default is the *28-step* 1024 band — coincidentally near the 768-tier/20-step band.
Both axes (fewer tokens, more steps) push the band down. **Caveat:** the 28-step resolution
sweep (768²/896×1200/1024²) ran at n=3 and the PRESENT/ABSENT label is *noise-dominated near the
gate* (identical config flipped ABSENT@n12 → PRESENT@n3); ρ-contraction is the robust signal, the
label is not. Treat the per-tier×per-step band table as approximate until re-probed at larger n.

**Implication for the node:** band must be a user knob (or auto-derived from token count **and**
step count), defaulting to the shipped 28-step/1024 values with a tooltip; never a silent fixed constant.

## 2. Why the Spectrum node — and why not a model patch

FSG is a **sampler-loop** operation, not a model modification:

- It runs a K-iteration fixed-point loop **between** sampler steps, **changes the NFE count**
  (+3·K per scheduled step), and **mutates the latent before the step**.
- A ComfyUI `MODEL` patch (`set_model_unet_function_wrapper` / attention patch) only
  intercepts a *single* forward call — it can't own inter-step integration or change NFE.
  This is why `AnimaModGuidance` legitimately *is* a model patch (per-forward AdaLN steering)
  but FSG can't be: different seam.
- A standalone competing KSampler can't compose with Spectrum (two nodes can't both own the
  loop) and would duplicate the whole denoise loop in a second hand-maintained file — strictly
  worse, and it throws away the FSG×SEA interaction.

The Spectrum node is already the consolidation point (`SpectrumKSampler` bundles mod-guidance,
SMC-CFG, spectrum accel; DCW lives on the Advanced node), and the library already encodes the
FSG×Spectrum interaction. So FSG ships as **another scalar-config stack on that sampler**,
mirroring DCW.

## 3. Integration reality — hand-mirror, NO vendor-sync

The Spectrum repo (`~/ComfyUI-Spectrum-KSampler`, symlinked at
`../comfy/custom_nodes/comfyui-spectrum-ksampler`) is **not** part of `scripts/release/sync_vendor.py`
(which targets only tagger / directedit / trainer / the hydralora-adapter repo) and has **no
`_vendor/` tree**. Its `spectrum.py` is **994 lines vs the library's 750** — a hand-maintained
reimplementation against **ComfyUI's sampler internals** (`comfy` `calc_cond_batch`, model
hooks, `get_executing_context()`), **not** the library's `generate_body` / `SamplerSideChannels`.
`dcw.py`, `cns.py`, `mod_guidance.py`, `smc_cfg.py` are siblings ported the same way.

So porting FSG is **manual**, mirroring how DCW was, with one real subtlety: the calibrator's
velocity calls must be rewritten against ComfyUI's model-call surface (cond/uncond via
`calc_cond_batch` / `apply_model`), **not** the library's `anima(x, t, embed)` + hydra setters.
The operator *math* is unchanged; only how `v^c`/`v^u` are obtained differs.

(Separate, deferred decision: bring the Spectrum repo into `sync_vendor` to kill the drift
permanently. Bigger one-time refactor — not now; port FSG by hand like its siblings.)

## 4. Port plan — three pieces, mirror DCW  ✅ DONE (pending in-ComfyUI verify)

**Landed 2026-06-23.** `fsg.py` (FSGCalibrator + CFG++) ported to the node;
FSG + CFG++ knobs added to `SpectrumKSamplerAdvanced`; `spectrum_sample` installs
both and forces FSG steps actual. One change vs the original plan below: the
velocity source is ComfyUI's `comfy.samplers.calc_cond_batch` (cond/uncond
denoised → v-space), and FSG runs as a **CALC_COND_BATCH wrapper** (mutates the
sampler's `x` in place, like DCW) using a *cleaned* `model_options` for its own
K-loop forwards (strips the Spectrum `model_function_wrapper` + the CALC_COND_BATCH
wrappers so its forwards hit the real DiT and never re-enter FSG/DCW/cache, while
keeping mod-guidance). CFG++ rides a `sampler_cfg_function` (the SMC seam). The
σ schedule for CFG++'s σ_next and FSG's forced-step set is recomputed via a
throwaway `KSampler`. Invariant-tested (constant-field fixed point; cfgpp weight)
against monkeypatched `calc_cond_batch`. **Not yet run inside a live ComfyUI.**

Original three-piece plan (for reference):

1. **`fsg.py` in the repo.** Port `FSGCalibrator` verbatim *except* `_velocity`: replace the
   `anima(...)` + `set_hydra_*` calls with the node's cond/uncond velocity path
   (`calc_cond_batch` over positive/negative conditioning at arbitrary (x, σ)). Keep `band`,
   `k`, `d_sigma`, `gamma`, `scheduled()`, and the K-loop math identical.
2. **Node surface.** Add a `_FSG_INPUTS` flat-scalar dict (mirror `_DCW_INPUTS`,
   `nodes.py:420`): `fsg` (bool / "off"|"on"), `fsg_band_lo`, `fsg_band_hi`, `fsg_k`,
   `fsg_d_sigma`, `fsg_gamma`. Merge into `SpectrumKSamplerAdvanced.INPUT_TYPES`; thread as
   `sample()` kwargs into `_run_spectrum`. (Optionally a single `fsg` toggle on the basic node.)
3. **Runner (`_run_spectrum`).** Mirror the library diff: compute `fsg_steps` from the
   schedule (`{i : fsg.scheduled(σ_i)}`); inside the denoise loop, on a scheduled step
   **calibrate `latents` first**, force that step **actual** (exclude from the SEA/window
   decision denominator — third forced-actual class alongside warmup + tail), and add
   `(tuple(fsg_steps), k, d_sigma, gamma)` to the spectrum cache key so δ recalibrates when
   they change. The node loop already has the forced-actual concept for warmup/tail; FSG slots
   in identically.

Cost surfaced in the node log line, same as the library (`+3·K·M` extra forwards).

## 5. Band default UX (the §1a finding)

**Node surface as shipped:** a single `fsg` BOOLEAN on the **simple** `SpectrumKSampler`
(one switch = the validated stack: CFG++ λ=1.5 + FSG band [0.59,0.75]/K=3, and it
auto-disables SMC-CFG since CFG++ owns the combine), with the full `cfgpp_lambda` /
`fsg_band_lo/hi` / `fsg_k` / `fsg_d_sigma` / `fsg_gamma` knobs on the **Advanced** node.

- Default `fsg_band_lo=0.59`, `fsg_band_hi=0.75`, `fsg_k=3`, `cfgpp_lambda=1.5`, `fsg_gamma`
  =guidance — the shipped 28-step/1024 point (§0). Tooltip: *"σ-band where calibration fires.
  Calibrated for the 1024 token tier at ~28 steps; the band moves DOWN for more steps and for
  low-token (~768px) renders, and UP for fewer steps. Re-tune if you change steps/resolution."*
- Long-term: **auto-derive the band from token count AND step count** via a 2-axis table — but
  that needs 1536²/512² and 20/30/40-step probed first (we have 768+1024 × 20+28). Ship manual
  knobs now, auto-table later. Never hardcode a silent fixed band.

## 6. Gate — what must pass before the node ships

Tier-2 status as of 2026-06-23 (library shipped on this basis):

- ✅ **Production er_sde CFG=4 render** — DONE. 4-arm A/B (baseline / cfg++ / fsg/cfg / fsg/cfg++)
  on er_sde, 28 steps, band [0.59,0.75], K∈{2,3}, 6 captions. fsg/cfg++ **clearly beats cfg++**
  by eyeball (K=3; K=2 close); cfg++ family preferred over plain CFG.
- ✅ **Saturation confound** — quantified. fsg/cfg++ vs cfg++ is Δsat **+1.7%p**, Δcontrast
  **−1%p** — the win is *not* a global tone bump (sat/contrast in `render_compare` result.json).
- ✅ **CFG++-substrate anti-confound read** (substituted for matched-NFE as the decisive test):
  on plain CFG, foresight ≈ extra effective CFG (first-order), so a fsg/cfg "win" is ambiguous.
  On the CFG++ substrate it can't masquerade as more CFG — and fsg/cfg++ still beats cfg++. That
  is the real golden-path signal, and it's what the ship decision rests on.
- ⚠️ **Matched-NFE A/B** — **NOT run.** FSG-at-28 (≈101 fwd) vs plain CFG @~50 steps (≈100 fwd):
  if the longer plain run matches it, the knob is NFE-for-nothing. The cfg++ read above is
  stronger evidence of a *real* effect, but matched-NFE is still the clean cost-efficiency proof
  and should be run before the node knob is advertised as "free quality."

Shipping order: library shipped (defaults = §0) → port to the node (DONE, §4) → **verify in a
live ComfyUI** + **run matched-NFE** before publishing/advertising the node. A node that exposes a
wrong-tier band or an NFE-for-nothing knob is worse than no node — so the node code has landed but
the knob is not yet advertised as "free quality" pending those two.

## 7. Open risks

- **Matched-NFE may erase the win** — still owed; the one decisive cost test not yet run (§6).
- **er_sde** — now the validated sampler (the render A/B ran on er_sde, not just Euler), so this
  risk is largely retired; the remaining hedge is matched-NFE.
- **Payoff-zone overlap** with the already-near-resolved σ band where σ-reshape found no
  fixed-NFE headroom — FSG's mechanism is distinct (a consistency operator) but the overlap is
  why matched-NFE is load-bearing.
- **Band-axis hazard** — the band depends on BOTH token tier AND step count (§1a); a fixed
  default mis-fires off the 28-step/1024 point. The node must expose / auto-derive the band.
- **Noisy band labels** — the per-tier×per-step band table is n=3 in places and the gate flips
  with prompt sample (§1a); ρ-contraction is the trustworthy signal, not PRESENT/ABSENT.
- **Hand-mirror drift** — the node's `spectrum.py` diverges from the library by hand; any later
  FSG change in the library must be re-ported until/unless the repo joins `sync_vendor`.
