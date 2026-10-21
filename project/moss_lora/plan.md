# moss_lora — MOSS-Image on a 16 GB card

**Not Anima.** MOSS-Image (OpenMOSS, Apache-2.0, released 2026-10-11) is a 5B single-stream
DiT trained from scratch on a frozen Qwen3.5-9B text encoder and the frozen FLUX.2 VAE
(`OpenMOSS-Team/MOSS-Image-Base`, `-SFT`; SFT is a fine-tune of Base). The line ported
Spectrum and a light LoRA onto the 5070 Ti and grew from there. Code map:
`library/moss/CLAUDE.md`; user doc: `docs/moss/README.md`.

## Status

| Sub-line | State | Where |
|---|---|---|
| Relational REPA | parked: seed variance swamps any effect | `repa.md` |
| Anima negatives revisit (sinks, 2:4 sparsity) | Tier 1 measured, Tier 2 unmeasured | `docs/proposal/moss_anima_negatives_revisit.md` |
| i2i / colorize (VAE reference on RoPE `r`) | **active** | `docs/proposal/moss_i2i.md`, `src/colorize_{cache,eval,wild,scale}.py` |
| Colorize OCR caption (layout + bbox, fp8 TE) | **read**: OCR-read lines letter right; colour even by eye, 64 px luma PSNR −1.7 dB on illus. (uneven, cause open) | `colorize_ocr.md`, `src/colorize_ocr.py` |
| Correction training (arXiv 2610.14136) as a dual LoRA | **active**: joint D + C holds to K 8; CFG 1.5 + K 2 ≈ CFG 3 at 2× the forwards (one seed) | `correction.md`, `src/correction*.py` |
| Technical-report digest | reference | `report_notes.md` |

`src/`: the colorize probes, `lora_ab.py` (LoRA A/B grids), `vision_te.py` (TE call with
an image in the user turn), `_render.py` (their shared render loop), `colorize_te_fp8.py`
(colorize holdout through an fp8 TE) and `te_gptq.py` (GPTQ-rounded fp8 TE, written in
`library/moss/fp8.py`'s format; bench `bench/moss_te_fp8`), `correction.py` /
`correction_train.py` / `correction_ab.py` (corrector LoRA, its trainer over
`library/moss/train.py`, CFG × K grids). `space/`: the two
Space prompts every A/B renders.

## Open questions

Per sub-line open items live in each log's `## Open`. Line-wide:

1. Character knowledge: does a name-only prompt (`akita neru`) work without the trait
   list?
2. Resolution ceiling: the report's text caps Base at ~1 MP, the Space presets run
   ~2.1 MP (`report_notes.md` § (a)). Keep LoRA buckets ≤ ~1 MP until Base is compared
   at 1 MP vs 2 MP.
