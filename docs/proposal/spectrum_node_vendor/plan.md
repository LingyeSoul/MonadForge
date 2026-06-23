# Spectrum node vendor migration — plan

Stop hand-mirroring anima_lora code into `ComfyUI-Spectrum-KSampler`. Each
sampler-boundary technique splits into a **pure-compute core** (torch/numpy only,
single source of truth, vendored verbatim) and a **ComfyUI seam** (the
`calc_cond_batch` / `sampler_cfg_function` / `model_function_wrapper` plumbing,
which genuinely differs and stays node-side).

Infra is in place: `scripts/release/sync_vendor.py` has a `build_spectrum_vendor()`
target writing the cores into `ComfyUI-Spectrum-KSampler/_vendor/`, and the node's
`__init__.py` bootstraps `sys.path` (live repo first, `_vendor/` fallback) so node
files use plain absolute `from library…` / `from networks…` imports.

## Done (this pass)

| Method | Core (single source) | Node keeps (seam) |
|---|---|---|
| FSG | `library/inference/corrections/fsg_core.py` (config gate + fixed-point loop vs velocity callbacks + `cfgpp_guidance_weight`) | CFG++ install, CALC_COND_BATCH wrapper, `calc_cond_batch` velocity source |
| SMC | `library/inference/corrections/smc_cfg.py` (already pure — `combine`) | `_make_smc_cfg_function`, denoised↔v-space convert |
| SPD | `networks/spd_core.py` (DCT helpers + `spectral_expand`) | SPEED sampler, schedule resolution |
| CNS | `library/inference/corrections/cns_core.py` (`radial_bins` + recolor) | `from_path` download, ER-SDE loop |
| DCW | `networks/dcw.py` + `library/inference/corrections/dcw_calibrator.py` (+ `library/runtime/fei.py`) — **drift fixed**: node now gets `fei_k`/v6 schemas + `record_latent_pre_forward` wired into the step hook | DCWState, hooks, `install_dcw`, download, `dm.preprocess_text_embeds` setup |

All bit-exact to prior node behavior (v5/v4 DCW unchanged); standalone vendor
import verified; `make test-unit` green (the one `test_doc_refs` failure is
pre-existing and unrelated).

## Next: `spectrum.py` and `mod_guidance.py`

The two largest remaining hand-mirrors. Both more ComfyUI-entangled than the five
above — map the core/seam boundary first (read both sides), then extract.

### `mod_guidance.py` (do first — smaller, cleaner)
- Library home: `library/inference/corrections/mod_guidance.py` (~6.5KB).
- Node copy: `~/ComfyUI-Spectrum-KSampler/mod_guidance.py` (~36KB — much of the
  bulk is ComfyUI block-patching + `_extract_raw_and_t5` + `pooled_text_proj`
  plumbing, i.e. seam).
- Hypothesis: the **AdaLN modulation-vector math** (the σ-profile → per-block
  shift/scale delta) is the shareable core; the `Block.forward` splice and the
  cond/uncond plumbing are seam.
- Watch: `pooled_text_proj` loads in `load_dit_model`, not `setup_mod_guidance`
  (see [[project_sea_delta_generalizes_guidance]]); `_extract_raw_and_t5` is
  node-local and is also imported by the DCW calibrator seam — keep it node-side.
- Steps: confirm the library mod-vector fn is pure (no anima-model dep) → if so
  it's a direct vendor like SMC; else extract a `mod_guidance_core.py`. Add to
  `SPECTRUM_VERBATIM`, rewire node to import it, delete the duplicated math.

### `spectrum.py` (do last — largest, most state)
- Library home: `networks/spectrum.py` (the CLI Spectrum runner + `_combine_guided`
  / `_window_decision_fraction`, already exercised by `tests/test_fsg_invariant.py`).
- Node copy: `~/ComfyUI-Spectrum-KSampler/spectrum.py` (~46KB).
- Core candidates: the **Chebyshev ridge-regression fit + feature-forecast**
  numerics, the SEA window-decision fraction (`_window_decision_fraction`), and
  `_combine_guided` (CFG / CFG++ / SMC merge). Seam: the `model_function_wrapper`,
  block-skip orchestration, per-step cache state machine, `SpectrumState`.
- Risk: the forecasting state is tightly coupled to the ComfyUI forward wrapper;
  the extractable core is likely just the polynomial fit/predict + the combine +
  the decision-fraction helper. Map carefully before committing — don't force a
  split that drags ComfyUI state into the core.
- `spectrum_sea.py` / `forecaster.py` are node-side SEA helpers; check whether any
  of their math duplicates `networks/spectrum.py` and fold those too.

## Convention reminders
- Edit the math in the **core**; never hand-edit `_vendor/`. Re-run
  `python scripts/release/sync_vendor.py` (a.k.a. `make vendor-sync`) before any
  node publish — see [[feedback_vendor_sync]].
- Cores must stay import-pure: torch / numpy / stdlib only, no `comfy`, no
  anima-model imports. If a candidate needs the DiT, it's seam, not core.
- Add each new vendored file to `SPECTRUM_VERBATIM` (+ its package dir to
  `SPECTRUM_PACKAGE_DIRS` if a new namespace level appears).
