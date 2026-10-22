# MOSS-Image LoRA (alpha)

**Not Anima.** [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
(OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a frozen Qwen3.5-9B text encoder and
the FLUX.2 VAE. This repo trains LoRAs for it and samples it with Spectrum on a 16 GB
card. It is a separate line (`library/moss/`): Anima configs, presets, `make lora`,
caches and LoRAs do not apply to it, and a MOSS LoRA does not load on Anima.

Status: **alpha**. The measurements below come from one style set (caststation, 30 images
of one artist). Flags and defaults may still move. Step-by-step GUI walkthrough, field
tables and troubleshooting: `docs/guidelines/moss-guidebook.md`.

## Quickstart

```bash
make download-model ARGS=moss-base   # DiT + text encoder + VAE + tokenizer, ~27 GB
make download-model ARGS=moss-sft    # optional: the SFT DiT alone, ~10 GB (reuses Base's TE/VAE)
make gui-moss                        # Preprocess → Train → Test, each a daemon job
```

Weights land in `models/moss_image/MOSS-Image-{Base,SFT}/` (elsewhere:
`ANIMA_MOSS_MODEL_DIR` in `.env`, or `model_dir` in the GUI). The GUI caches text
embeddings and latents for a folder of images with `.txt` captions (default
`post_image_dataset/resized`, the Anima resized tree; point `src` at one subfolder to train
on one set), trains a LoRA on the cache, and renders a prompt with and without the LoRA at
the same seed. It shares its code with `make gui-qwen` (`gui/lines/`).

The same jobs from the CLI, through the daemon:

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'post_image_dataset/moss/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors --spectrum"
```

`--help` on each script lists every flag (one definition: `library/moss/requests.py`).
`--config <file.toml>` sets any flag from a flat TOML of field names
(`configs/moss/cache.toml` and `configs/moss/lora.toml` hold the shipped defaults; copy one
for a recipe). Order: field default → `--config` → CLI flags. Unknown keys, wrong types and
values outside a flag's choices are an error.

## Text encoder precision

`--text_encoder` (cache and generate; GUI: Advanced) picks the Qwen3.5-9B weights. The
tokenizer always comes from Base. A quantized file downloads on first use, or via
`make download-model ARGS=moss-te-int8` (`moss-te-fp8`).

| Value | File | Size | Notes |
|---|---|---|---|
| `int8` (default) | [`sorryhyun/MOSS-Image-Qwen3.5-9B-int8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-int8) | 9.2 GB | W8A8; all 32 layers on a 16 GB card, peak 6.5 GiB |
| `fp8` | [`sorryhyun/MOSS-Image-Qwen3.5-9B-fp8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-fp8) | 9.2 GB | float8 weights, bf16 matmul; ~1.7× slower to encode |
| `bf16` | Base's shards | 16.8 GB | the reference; peak 13.0 GiB |

Both quantized files are GPTQ-calibrated on MOSS prompts and pages. Their embeddings differ
from bf16 by ~4–5 % (relative L2) on captions and ~10 % on long JSON prompts and on pages
through the vision tower; fp8 images on the same seed keep composition and text and change
in small details (`_archive/bench/moss_te_fp8`). **Cache and generate with the same setting**: the TE
cache records which encoder wrote it and switching re-encodes it, so a cache made before
int8 became the default re-encodes on the next `cache` run unless it passes
`--text_encoder bf16` (or `text_encoder = "bf16"` in a copy of `cache.toml`).

## Training

- **Defaults**: `--rank 32`, alpha 128 for `lora` (scale 4; `lokr` / `loha` use
  alpha = rank), `--lr 5e-5`, `--channel_scaling_alpha 0.5`, `--dc_isolate`,
  `--down_init weight_svd`. On caststation (20 epochs) this stayed clean and on-prompt at
  the style strength of lr 1e-4, where a plain LoRA at lr 1e-4 added things the prompt
  never asked for. Mechanism and the lr / alpha history: `docs/moss/channel_scaling.md`.
- **Time**: 30 images × 20 epochs (600 steps) takes about 18–21 min on a 5070 Ti. The
  LoRA moves line work and palette toward the set; character identity holds, including
  characters the set does not contain.
- **Memory**: the default plan is no checkpointing, `--blocks_to_swap 12`,
  `--activation_memory_budget 0.1`: 1.85 s/step, ~13.1 GB peak reserved, ~1.7 GB free at
  ~4.5k joint tokens. `--blocks_to_swap 24` leaves room for longer captions at the same
  speed; `--grad_checkpointing --blocks_to_swap 0` fits without swap, slower. The best swap
  count depends on the PCIe link and GPU; `tune_swap.py` times each count on the cache's
  largest sample (about 4 min) and prints the flags to train with. Full plan table:
  guidebook §4.2.

  ```bash
  make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
  ```

- **`--down_init`**: `weight_svd` starts `down` on the base weight's top-r right singular
  vectors (it kept the artist's style better by eye than `kaiming`; the two land samples
  as far apart as a seed change). `--svd_slice k` seeds from singular vectors
  `[k·r, (k+1)·r)` and `--svd_slice_count N` interleaves `{k, k+N, …}`, as in Anima's
  `svd_slice` (`docs/methods/svd-down-lora.md`), so LoRAs on different slices start in
  orthogonal input subspaces. MOSS's targets are all ≥ 3072 wide, so rank 32 fits 96
  slices. Under `--dc_isolate` row 0 is the same DC direction in every slice.
- **LyCORIS** (`--algo lokr` / `--algo loha`, from `lycoris-lora`): LoKr is the
  Kronecker product (`--factor` caps the small factor; a `--rank` at or above half the
  larger factor's side makes `w2` full and the scale 1, hence `--rank 100000 --factor 8`),
  LoHa the Hadamard product of two rank-r pairs. Files keep LyCORIS tensor names and load
  through the same `--lora`. Both need `--blocks_to_swap 24 --activation_memory_budget 0.1`
  at 1024² (swap 12 / budget 0.15 ran out of memory); `tune_swap.py` measures a LoRA only.
  `--down_init` applies to `lora` only. At lr 1e-4, LoKr f8 came close to the LoRA and LoHa
  r16 overshot (palette shift, costume leakage); LoHa at lr 5e-5 is unmeasured.
- **Checkpoints**: `--save_every_epochs N` also saves `<name>/<name>_eN.safetensors`
  beside the final file (the last epoch is the final file). A `.json` run report
  (per-step loss) lands beside the LoRA.
- **`--fit_report`** (off): trains the largest sample (longest caption) at step 1 and
  reports its peak against free VRAM, with a `--blocks_to_swap` suggestion.
- **Caption dropout** (`--caption_dropout`, default 0): keep 0. On caststation 0.05 and 0.1
  weakened the style under CFG, because the uncond prediction picks up the style and CFG
  cancels part of it (`_archive/moss_lora/caption_dropout.md`).
- **A LoRA trained on Base also works on SFT**: it removes most of SFT's glossy look and
  lands close to Base + LoRA.

## Captions

Anima-style tag captions train and prompt fine (MOSS was trained on natural language; long
or character-heavy captions are untested). By default (`caption_format` on the cache,
`prompt_format` on generate, both `layout_json`) the Anima caption is mapped onto MOSS's
layout-JSON prompt (`anime_tools.captions.layout_json`):

- rating / meta, characters, series and artists (with their `@`) get their own `summary`
  keys; the rating collapses to `sfw` (safe, sensitive) or `nsfw` (nsfw, explicit);
- pixel-describing meta tags (`cropped`, `scan`, …) and the remaining tags become
  `summary.caption`;
- each position clause becomes a `character` item with its relative position (never a box),
  each OCR line a `text` item. Only a `{stem}.layout.txt` from anime_tools' Export
  (`caption_format layout_json` with `combine_ocr`) gives text items an OCR `bbox` (MOSS's
  `abs_centralized` / `yxyx`) and the top-level `bbox_coord_space` / `bbox_format` keys; the
  cache's own conversion of a `.txt` has no boxes.

On caststation this kept the style strength of plain tags and leaked fewer of the set's
attributes onto other characters. A `layout_json` cache also stores the plain caption, used
on `plain_caption_rate` (default 0.2) of the steps. **Train and prompt in the same form**
(`text` on both for plain captions); switching re-encodes the TE cache. In ComfyUI the
**Anima Caption → MOSS Layout JSON** node ([MOSS Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit),
needs `anime_tools`) does the same conversion.

## Sampling

- **Sampler defaults** come from the reference pipeline: CFG 3, the uncond branch skips
  block 22, shift 5. Don't carry Anima habits over.
- **`--sampler er_sde`** (default; `euler` is the pipeline's ODE): Anima's ER-SDE-Solver-3
  on the same σ grid, same speed, composes with Spectrum. Sharper line work and denser
  detail than Euler, but drifts from some prompt details (sunset light, hair colour); a
  different sample at the same seed.
- **`--spectrum`**: 2.6× at 50 steps with the default `flex_window 0.5`, hard to tell
  apart from the full 50 steps at grid scale.
- **`--compile`** (on; `--no-compile` for eager): about 1.37× on top, with or without
  Spectrum (Spectrum + compile 20 s vs plain eager 72 s at 1024²), for a few seconds of
  compile per job. Fine detail drifts from the eager image; composition does not.
- **DAVE** (`--dave auto`, off): more same-prompt variety across seeds; `docs/moss/dave.md`.
- **CNS** has no `generate.py` flag. The six MOSS γ files
  (`networks/calibration/moss_cns_gamma*.npz`, `docs/inference/cns.md`) are used by the
  bench and by the ComfyUI Spectrum node's `er_sde_cns`, which picks one by variant and cfg.

## ComfyUI

LoRA files use the kohya `lora_unet_` key layout and load through ComfyUI's LoRA loader on a
model from the `moss-image` loader nodes (files saved before 2026-10-17 used another layout
and do not). The Spectrum node pack's **KSampler (Spectrum for MOSS)** carries the sampler
defaults and Spectrum; `ComfyUI-Anima-DAVE` 0.2.0 and later runs DAVE on MOSS. The fp8 text
encoder loads in **MOSS-Image Load Qwen3.5** as-is (from `models/text_encoders/`); the INT8
file is in ComfyUI's `int8_tensorwise` + `convrot` format for the same loader.

## i2i (image to image)

An i2i LoRA learns a reference → target edit; the reference rides every forward as clean
latent tokens on RoPE's `r` axis. Two tasks have a recipe in `configs/moss/i2i/<task>/`
(`cache.toml`, `train.toml`, `generate.toml`):

| Task | Reference | Text encoder input |
|---|---|---|
| `colorize` | the B&W page at half size | the page through Qwen3.5's vision tower, then a caption with its OCR'd lines |
| `sanitize` | the texted page at full size | the clean twin's caption |

```bash
make i2i-stage TASK=colorize                    # pair tree under post_image_dataset/i2i/
make i2i-preprocess TASK=colorize MODEL=moss    # cache (scripts/moss/colorize_cache.py; sanitize: cache.py)
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/i2i/colorize/train.toml"
make daemon-run ARGS="scripts/moss/generate.py --config configs/moss/i2i/colorize/generate.toml \
    --lora output/ckpt_moss/moss_i2i_colorize.safetensors --ref_image <page or folder>"
```

`--ref_image` is one image or a folder (one job per image and prompt); the output keeps its
aspect with the area capped at `width × height`. `--ref_scale` and `--ref_vision` must match
the cache the LoRA trained on; the LoRA's own `ref_timestep` / `ref_isolate` apply. The GUI's
**i2i** tab (`make gui-moss`, or MOSS mode in `make gui`) runs the same four steps.

## Research flags

Advanced, not recommended: `--repa_weight` / `--repa_layer` / `--repa_pe_root` (relational
REPA, parked, `_archive/moss_lora/repa.md`); `--ref_timestep` / `--ref_isolate` /
`--caption_variant_rate` (set by the i2i recipes above; probes in `docs/proposal/moss_i2i.md`).

## More

- Channel scaling and `--dc_isolate`: `docs/moss/channel_scaling.md`. DAVE: `docs/moss/dave.md`.
- Findings (attention sink, vision-tower reference, σ staircase, Spectrum schedule):
  `docs/moss/findings/`.
- Proposals: `docs/proposal/moss_i2i.md`, `docs/proposal/moss_anima_negatives_revisit.md`.
- Run logs: `project/moss_lora/` (`plan.md`, open sub-lines, technical-report notes);
  finished logs (sampler, channel scaling, DAVE, seed soup) in `project/moss_lora/done/`;
  parked lines (caption dropout, REPA) in `_archive/moss_lora/`.
- Code map: `library/moss/CLAUDE.md`.
