# MOSS-Image LoRA (alpha)

**Not Anima.** [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
(OpenMOSS, Apache-2.0) is a 5B single-stream DiT on a frozen Qwen3.5-9B text encoder and
the FLUX.2 VAE. This repo trains LoRAs for it and samples it with Spectrum on a 16 GB
card. It is a separate line (`library/moss/`): Anima configs, presets, `make lora`,
caches and LoRAs do not apply to it, and a MOSS LoRA does not load on Anima.

Status: **alpha**. One real style run so far (below). Expect flags and defaults to move.

## Setup

```bash
make download-model ARGS=moss-base   # DiT + text encoder + VAE + tokenizer, ~27 GB
make download-model ARGS=moss-sft    # optional: the SFT DiT alone, ~10 GB (reuses Base's TE/VAE)
```

Weights land in `models/moss_image/MOSS-Image-{Base,SFT}/`. Another location:
`ANIMA_MOSS_MODEL_DIR` in `.env`, or `model_dir` in the GUI.

## GUI

```bash
make gui-moss
```

Three steps, each a daemon job:

1. **Preprocess**: caches text embeddings and latents for a folder of images with `.txt`
   captions. The default source is `post_image_dataset/resized`, the Anima resized tree
   with its revised captions. Point `src` at one subfolder to train on one set.
2. **Train**: trains a LoRA on the cache. **Preprocess, then train** runs both in order.
3. **Test**: renders the prompt with the LoRA and without it (same seed), side by side.

The window shares its code with `make gui-qwen` (`gui/lines/`).

## CLI

The GUI submits these. Run them through the daemon:

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'output/moss/cache/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'output/moss/cache/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/moss/lora/moss_lora.safetensors --spectrum"
```

`--help` on each script lists every flag (one definition: `library/moss/requests.py`).

## What to expect

- **Training** (5070 Ti, Base, rank 16, compiled, checkpointed, no block swap): 30
  images × 20 epochs took 21 min at about 2.1 s/step, with an 11.8 GiB peak. The LoRA
  moved line work and palette toward the set. Character identity held, including
  characters the set does not contain.
- **Memory plan**: the default is no checkpointing, 24 blocks swapped to CPU and
  `--activation_memory_budget 0.2`, which ran at 1.78 s/step (−15 %) on a synthetic
  1024² fit. The best swap count depends on the PCIe link and the GPU: more swap lets
  the budget rise (less recompute) until the transfers stop hiding behind compute.
  `--grad_checkpointing --blocks_to_swap 0` is the old plan. To measure your machine's
  pair (about 4 min, on the cache's largest sample):

  ```bash
  make daemon-run ARGS="scripts/moss/tune_swap.py --cache output/moss/cache/<set>"
  ```

  It prints one line per swap count and the flags to train with.
- **A LoRA trained on Base also works on SFT.** It removes most of SFT's glossy look and
  lands close to Base + LoRA.
- **Captions**: Anima-style tag captions trained and prompted fine. MOSS itself was
  trained on natural language, and long or character-heavy captions are untested.
  `caption_format=layout_json` (cache) / `prompt_format=layout_json` (generate) map the
  Anima caption onto MOSS's layout-JSON prompt (`anime_tools.captions.layout_json`):
  rating / meta, characters, series and artists get their own `summary` keys, the
  remaining tags become `summary.caption`, each position clause a `subject` item with
  its relative position, each OCR line a `text` item. Train and prompt in the same
  form; switching it re-encodes the TE cache. Untested on a LoRA so far.
- **Spectrum** (`--spectrum`, Test panel): 2.6× at 50 steps with the default
  `flex_window 0.5`. At grid scale it is hard to tell apart from the full 50 steps.
- **`--compile`** (generate, on by default; `--no-compile` for eager): about 1.37× more
  on top, with or without Spectrum (Spectrum + compile 20 s vs 72 s for plain eager at
  1024²), for a few seconds of compile once per job. Fine detail drifts from the eager
  image; composition does not.
- **Sampler defaults** come from the reference pipeline: CFG 3, the uncond branch skips
  block 22, shift 5. Don't carry Anima habits over.
- **`--sampler er_sde`**: Anima's ER-SDE-Solver-3 on the same σ grid, same speed, and it
  composes with Spectrum. Base, 3 prompts, one seed: sharper line work and denser
  detail, but it drifts from some prompt details (sunset light, hair colour). It is
  a different sample from Euler's at the same seed.

## More

Measured numbers, A/B notes and open questions are in `project/moss_lora/`: `lora.md`
(training), `spectrum.md`, `cache.md`, `phase0.md` (parity with the reference pipeline).
Code map: `library/moss/CLAUDE.md`.
