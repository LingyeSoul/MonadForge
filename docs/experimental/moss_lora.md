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
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors --spectrum"
```

`--help` on each script lists every flag (one definition: `library/moss/requests.py`).

`--config <file.toml>` sets any flag from a flat TOML of field names (`configs/moss/lora.toml`
holds the shipped defaults; copy it for a recipe). Order: field default → `--config` → CLI flags.
Unknown keys, wrong types and values outside a flag's choices are an error.

## What to expect

- **Defaults**: `--rank 32`, alpha 128 for `lora` (scale 4; `lokr` / `loha` fall back
  to alpha = rank), `--lr 2e-5`. On the caststation set (20 epochs, `weight_svd`,
  layout_json, artist tag injected), rank 32 / alpha 128 at lr 1e-4 pushed the style
  hard (white backgrounds, flat pale fills) but lost details on two complex gallery
  prompts (wrong bag holder, missing plush); at lr 2e-5 the details held with a
  milder style shift, judged much better.
- **Training** (earlier default: 5070 Ti, Base, rank 16, alpha 16, lr 1e-4, compiled, checkpointed, no block swap): 30
  images × 20 epochs took 21 min at about 2.1 s/step, with an 11.8 GiB peak. The LoRA
  moved line work and palette toward the set. Character identity held, including
  characters the set does not contain.
- **Memory plan**: the default is no checkpointing, 12 blocks swapped to CPU and
  `--activation_memory_budget 0.1` (rank 32): 1.85 s/step with ~1.7 GB free and a
  ~13.1 GB peak reserved at ~4.5k joint tokens (image 4096 + text 455). The earlier
  default (rank 16, budget 0.15) measured 1.77 s/step (−15 % against checkpointing)
  with ~0.7 GiB free on a synthetic 1024² fit at 4.4k joint tokens; rank 16 at
  budget 0.1 measured 1.82 s/step with ~2.0 GB free. `--blocks_to_swap 24`
  leaves more room for longer captions at the same speed; budgets 0.1–0.3 barely move
  the step. The best swap count depends on the PCIe link and the GPU: more swap lets
  the budget rise (less recompute) until the transfers stop hiding behind compute.
  `--grad_checkpointing --blocks_to_swap 0` is the old plan. To measure your machine's
  pair (about 4 min, on the cache's largest sample):

  ```bash
  make daemon-run ARGS="scripts/moss/tune_swap.py --cache output/moss/cache/<set>"
  ```

  It prints one line per swap count and the flags to train with.
- **`--down_init weight_svd`** (the default; `kaiming` is the random init): the LoRA's
  down projection starts on the base weight's top-r right singular vectors. On the
  caststation set it kept the artist's style better by eye than Kaiming; the two
  inits land different samples at the same seed, as far apart as a seed change.
- **LyCORIS** (`--algo lokr` / `--algo loha`, from `lycoris-lora`): LoKr is the
  Kronecker product (`--factor` caps the small factor; a `--rank` at or above half the
  larger factor's side makes `w2` a full matrix and the scale 1, the usual
  `--rank 100000 --factor 8` setup), LoHa the Hadamard product of two rank-r pairs.
  Files keep LyCORIS's tensor names and load through the same `--lora`. Both need more
  room than a LoRA: at 1024² the earlier default swap 12 / budget 0.15 ran out of memory, so
  train them with `--blocks_to_swap 24 --activation_memory_budget 0.1` (LoKr f8
  2.26 s/step with 3.6 GB free, LoHa r16 2.49 s/step with 1.7 GB free, against
  LoRA's 1.77). `tune_swap.py` measures a LoRA only. `--down_init` applies to `lora`.
  On the caststation set at lr 1e-4, LoKr f8 came close to the LoRA and LoHa r16
  overshot (palette shift, costume leakage onto other characters); try LoHa lower.
  In the GUI, `algo` and `factor` sit on the Train step; on the CLI:

  ```bash
  make daemon-run ARGS="scripts/moss/train.py --cache 'output/moss/cache/<set>' --epochs 20 --algo lokr --rank 100000 --factor 8 --blocks_to_swap 24 --activation_memory_budget 0.1"
  make daemon-run ARGS="scripts/moss/train.py --cache 'output/moss/cache/<set>' --epochs 20 --algo loha --rank 16 --lr 5e-5 --blocks_to_swap 24 --activation_memory_budget 0.1"
  make daemon-run ARGS="scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors --spectrum"
  ```

  The LoHa lr above is the next arm to try, not a measured setting.
- **A LoRA trained on Base also works on SFT.** It removes most of SFT's glossy look and
  lands close to Base + LoRA.
- **Captions**: Anima-style tag captions trained and prompted fine. MOSS itself was
  trained on natural language, and long or character-heavy captions are untested.
  By default (`caption_format` on the cache, `prompt_format` on generate, both
  `layout_json`) the Anima caption is mapped onto MOSS's layout-JSON prompt
  (`anime_tools.captions.layout_json`): rating / meta, characters, series and artists
  (with their `@`) get their own `summary` keys; meta tags that describe the pixels
  (`cropped`, `scan`, …) and the remaining tags become `summary.caption`, each
  position clause a `subject` item with its relative position, each OCR line a `text`
  item. On a 30-image style set this kept the style strength of plain tags and leaked
  fewer of the set's attributes onto other characters. Train and prompt in the same
  form (`text` on both for plain captions); switching it re-encodes the TE cache. In
  ComfyUI, the **Anima Caption → MOSS Layout JSON** node (in the Anima Tagger pack,
  `anime_tools`) does the same conversion in front of the text encoder.
- **Spectrum** (`--spectrum`, Test panel): 2.6× at 50 steps with the default
  `flex_window 0.5`. At grid scale it is hard to tell apart from the full 50 steps.
- **`--compile`** (generate, on by default; `--no-compile` for eager): about 1.37× more
  on top, with or without Spectrum (Spectrum + compile 20 s vs 72 s for plain eager at
  1024²), for a few seconds of compile once per job. Fine detail drifts from the eager
  image; composition does not.
- **Sampler defaults** come from the reference pipeline: CFG 3, the uncond branch skips
  block 22, shift 5. Don't carry Anima habits over.
- **`--sampler er_sde`** (the default; `--sampler euler` is the pipeline's ODE):
  Anima's ER-SDE-Solver-3 on the same σ grid, same speed, and it composes with Spectrum. Base, 3 prompts, one seed: sharper line work and denser
  detail, but it drifts from some prompt details (sunset light, hair colour). It is
  a different sample from Euler's at the same seed.

## More

Measured numbers, A/B notes and open questions are in `project/moss_lora/`: `lora.md`
(training), `spectrum.md`, `cache.md`, `phase0.md` (parity with the reference pipeline).
Code map: `library/moss/CLAUDE.md`.
