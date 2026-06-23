# Foresight Guidance (FSG) — shipping to the Spectrum ComfyUI node

**Status:** library plugin BUILT + validated (Phase-0 premise, Phase-1 eyeball, resolution
sweep); **Tier-2 rigor still owed** (matched-NFE A/B, er_sde render, saturation confound);
**node port NOT started.** This doc is the node-shipping plan — the "should we build FSG"
question is settled (yes, in the library). Earlier build-decision content lives in git history.
**Paper:** "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations"
(NeurIPS 2025, arXiv 23177). **Library plugin:** `library/inference/corrections/fsg.py`
(`--fsg` / `FSG=1`). **Benches:** `bench/fsg/probe_golden_path.py`, `bench/fsg/render_compare.py`.
**Docs:** `docs/inference/fsg.md`. **Memory:** [[project_fsg_golden_path_phase0]].

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
- **Validated band / K (1024 token tier):** working band **σ∈[0.45, 0.85]**, narrow default
  **[0.75, 0.85]** (carries ~all the Phase-1 win at ~half the NFE), **K=3** (error ~ρ^K,
  ρ≈0.93 ⇒ K=3–4 captures ~all gain), Δσ=0.1, γ=guidance. σ≈0.94 *diverges* (ρ>1) — the
  paper's "iterate in the noisiest stage" prescription is wrong on Anima.

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

**Implication for the node:** band must be a user knob (or auto-derived from token count),
defaulting to the 1024-tier values with a tooltip; never a silent fixed constant.

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

## 4. Port plan — three pieces, mirror DCW

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

- Default `fsg_band_lo=0.75`, `fsg_band_hi=0.85` (1024-tier) with a tooltip: *"σ-band where
  calibration fires. Calibrated for the 1024 token tier; for ~768px / low-token renders drop
  ~0.1 (0.85 diverges there)."*
- Long-term: **auto-derive the band from the latent's token count** via a per-tier table — but
  that needs 1536² and 512² probed first (we only have 768 + 1024). Ship manual knobs now,
  auto-table later. Never hardcode a silent fixed band.

## 6. Gate — what must pass before the node ships

FSG is still **pre-Tier-2**; do not expose a node knob until:

- **Matched-NFE A/B** (decisive): FSG-at-N vs a plain longer baseline at the same N. The
  Phase-1 win used ~3× compute; if a longer plain run matches it, the knob is pointless.
- **Production er_sde CFG=4 render** — the eyeball used deterministic Euler; er_sde
  flipped-then-reclosed the σ-reshape line ([[project_sigma_reshape_no_win]]).
- **Saturation confound** — rule out that "better" is a global tone/contrast bump (Anima lever
  history: [[project_mod_guidance_text_derivative_orthogonal]], null-TTA).

Shipping order: land Tier-2 in the library → then port to the node with a defensible band
default. A node that exposes a wrong-tier band or an NFE-for-nothing knob is worse than no node.

## 7. Open risks

- **Matched-NFE may erase the win** (decisive, still owed).
- **er_sde** is the real sampler and reversed σ-reshape once; Euler-only isn't proof.
- **Payoff-zone overlap** with the already-near-resolved σ band where σ-reshape found no
  fixed-NFE headroom — FSG's mechanism is distinct (a consistency operator) but the overlap is
  why matched-NFE is load-bearing.
- **Tier-band hazard** — a fixed default mis-fires off the 1024 tier (§1a); the node must
  expose / auto-derive the band.
- **Hand-mirror drift** — the node's `spectrum.py` diverges from the library by hand; any later
  FSG change in the library must be re-ported until/unless the repo joins `sync_vendor`.
