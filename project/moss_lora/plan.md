# moss_lora — MOSS-Image on a 16 GB card

**Not Anima.** MOSS-Image (OpenMOSS, Apache-2.0, released 2026-10-11): 5B single-stream
DiT, frozen Qwen3.5-9B TE, frozen FLUX.2 VAE (`OpenMOSS-Team/MOSS-Image-Base`, `-SFT`).
Code map `library/moss/CLAUDE.md`; user doc `docs/moss/README.md`.

## Active

| Sub-line | State | Log / code |
|---|---|---|
| i2i / colorize (VAE reference on RoPE `r`) | `moss_colorize_3k_ocr` is the kept run; `colorize_run.py` reproduces it | `docs/proposal/moss_i2i.md`, `colorize_ocr.md`, `src/colorize_*.py` |
| Correction training (arXiv 2610.14136), dual LoRA | joint D + C holds to K 8; CFG 1.5 + K 2 ≈ CFG 3 at 2× forwards (one seed) | `correction.md`, `src/correction*.py` |
| SWIFT target extrapolation (arXiv 2610.25844) | first arm in run-to-run noise | `swift.md`, `src/swift_train.py` |
| Anima negatives revisit | Tier 1 measured, Tier 2 unmeasured | `docs/proposal/moss_anima_negatives_revisit.md` |
| TE quantization | INT8 W8A8 (SmoothQuant + act-aware GPTQ) is the default TE | `src/te_*.py`, `bench/moss_te_fp8` |
| Technical-report digest | reference | `report_notes.md` |

## Done (shipped or settled)

`done/`: `channel_scaling.md` (shipped, on by default), `dave.md` (shipped),
`sampler.md` (ER-SDE default; CNS bench-only), `soup.md` (3-seed soup keeps the style).
Archived: REPA (parked), caption dropout, template mask → `_archive/moss_lora/`; the
template-mask code is on `moss-template-mask` in the private `preserve` mirror.

## src/

| Script | Role |
|---|---|
| `_render.py` | shared render loop (DiT + LoRA + compile, ER-SDE + Spectrum, latent cache, grid) |
| `lora_ab.py`, `ab_cols.py` | LoRA A/B grids; one sheet per cell |
| `vision_te.py` | TE call with an image in the user turn |
| `colorize_{cache,eval,run,wild}.py` | colorize pipeline: cache → `scripts/moss/train.py` → eval, chained by `colorize_run.py`; `wild` for real pages |
| `correction.py`, `correction_train.py`, `correction_ab.py` | corrector LoRA, its trainer, CFG × K grids |
| `swift_train.py` | SWIFT trainer |
| `te_gptq.py`, `te_calib.py`, `te_calib_eval.py` | calibrated TE rounding (fp8 / INT8), its calibration manifest, per-category eval |
| `te_int8_compare.py`, `te_int8_act_compare.py` | the INT8 comparison jobs behind the default TE |

`space/`: the two Space prompts every A/B renders.

## Open questions

1. Character knowledge: does a name-only prompt (`akita neru`) work without the trait list?
2. Resolution ceiling: the report caps Base at ~1 MP, the Space presets run ~2.1 MP
   (`report_notes.md` § (a)). Keep LoRA buckets ≤ ~1 MP until Base is compared at 1 vs 2 MP.
