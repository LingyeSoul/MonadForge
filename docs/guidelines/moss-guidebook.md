# MOSS-Image LoRA Guidebook

A guide to training and testing LoRAs for MOSS-Image in the GUI. Open it with `make gui` and set the top-bar **Model** dropdown to **MOSS-Image** (`make gui-moss` opens the same panel in its own window). The panel is in English, or in Chinese when the GUI language is Chinese.

The MOSS line is **alpha**: one real style run has been measured so far (a 30-image artist set), and flags and defaults may still move. Every number below comes from that run and its benchmarks on a 16 GB RTX 5070 Ti.

---

## Table of Contents

1. [What MOSS-Image Is](#1-what-moss-image-is)
2. [Setup](#2-setup)
3. [Preprocess, Train, Test](#3-preprocess-train-test)
4. [Recommended Settings](#4-recommended-settings)
5. [Testing](#5-testing)
6. [Troubleshooting](#6-troubleshooting)

- [Appendix. CLI](#appendix-cli)

---

## 1. What MOSS-Image Is

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) (OpenMOSS, Apache-2.0) is a separate model from Anima, trained by this repo as its own line.

| | Anima | MOSS-Image |
|---|---|---|
| Diffusion model | Anima DiT | 5B single-stream DiT (42 blocks) |
| Text encoder | Qwen3 0.6B | Qwen3.5-9B |
| VAE | QwenImage VAE | FLUX.2 VAE |
| Caches | `post_image_dataset/lora/` | `output/moss/cache/` |
| LoRAs | `output/ckpt/` | `output/ckpt_moss/` |
| Settings | variant TOML + Hardware preset | the panel's fields |

- Caches, settings and LoRAs do not cross: a MOSS LoRA does not load on Anima, and an Anima LoRA does not load on MOSS. Anima's variants, presets and `make lora` do not apply here.
- MOSS ships two DiTs. **Base** is the pretrained model; **SFT** is a fine-tune of it with a glossy, gacha-like look. Train on Base. A LoRA trained on Base also works on SFT: it removes most of SFT's gloss and lands close to Base + LoRA.

---

## 2. Setup

| Pack | Contents | Size |
|---|---|---|
| `moss-base` | DiT, text encoder, VAE, tokenizer | ~27 GB |
| `moss-sft` (optional) | the SFT DiT alone; reuses Base's text encoder and VAE | ~10 GB |

Download them in the GUI's Models dialog, or:

```bash
make download-model ARGS=moss-base
make download-model ARGS=moss-sft
```

Weights land in `models/moss_image/MOSS-Image-Base/` and `models/moss_image/MOSS-Image-SFT/`. To keep them elsewhere, set `ANIMA_MOSS_MODEL_DIR` in `.env`, or `model_dir` under **Advanced** in the panel (the folder that holds both `MOSS-Image-*` folders). The panel shows `Model not found: …` until the folder is in place.

The default caption format (`layout_json`, §3.1) also needs the danbooru tag CSV, which `make download-models` already fetches (`make download-model ARGS=danbooru_tags` on its own).

---

## 3. Preprocess, Train, Test

The panel has two tabs, **Preprocess** and **Train**; the **Test** box sits at the bottom of the Train tab. Less-used fields are folded under **Advanced**. Each button submits a job to the training daemon: switching the Model dropdown back to Anima or closing the GUI leaves it running, reopening re-attaches to it (`Re-attached to running job …`), and **Stop** cancels it.

### 3.1 Dataset and captions

Images with same-name `.txt` captions, subfolders included. The default source is `post_image_dataset/resized`, the Anima resized tree with its revised captions, so a dataset already preprocessed for Anima works as is. Point `src` at one subfolder to train on one set.

- **Anima-style tag captions work** for training and prompting. MOSS itself was trained on natural language; long or character-heavy captions are untested.
- **`layout_json` (default)**: the Anima caption is mapped onto MOSS's layout-JSON prompt. Rating/meta, characters, series and artists get their own keys, the other tags become the caption, each position clause becomes a subject with its position, and each OCR text line becomes a text item. On the 30-image set this kept the style as strong as plain tags and leaked fewer of the set's attributes onto other characters.
- **Train and prompt in the same format**: `caption_format` (Preprocess) and `prompt_format` (Test) must match — both `layout_json`, or both `text` for plain captions. Prompts are typed in Anima tag format either way; `layout_json` converts them.
- **The trigger is the artist tag as it appears in the captions**, e.g. `@channel (caststation)`. Put it in the prompt the same way.
- A natural-language prompt passes through `layout_json` as one caption string and behaves like the plain-text prompt.
- In ComfyUI, the **Anima Caption → MOSS Layout JSON** node (Anima Tagger pack) does the same conversion in front of the text encoder.

### 3.2 Preprocess

Caches the Qwen3.5-9B text embedding and the FLUX.2 VAE latent of every image, so neither model loads during training.

| Field | Default | What it does |
|---|---|---|
| `src` | `post_image_dataset/resized` | Image + caption folder |
| `out` | `output/moss/cache` | Cache folder: two files per image, flat |
| `resolution` | `1024` | Pixel-area cap: at most ~1024² pixels at the image's own aspect, both sides multiples of 16, never upscaled |
| `caption_format` | `layout_json` | `layout_json` or `text` (§3.1) |

Press **Run preprocessing**. The status line counts images, captions and cached files. Re-running is incremental: new images are added, captions edited since their cache are re-encoded (the status line says how many), and a `caption_format` change re-encodes the text cache. Images that share a file name across subfolders get the folder name as a cache prefix.

### 3.3 Train

| Field | Default | What it does |
|---|---|---|
| `cache` | `output/moss/cache` | The Preprocess output |
| `output` | `output/ckpt_moss/moss_lora.safetensors` | LoRA path; a `.json` run report (per-step loss) lands beside it |
| `model` | `base` | Which DiT to train on |
| `epochs` | `8` | Passes over the cache (one image per step) |
| `save_every_epochs` | `0` | Also save every N epochs, as `<name>_e10.safetensors` etc. |

Rank, alpha, lr and the adapter type are in §4. **Train** reads `cache`; **Preprocess, then train** runs Preprocess first and points `cache` at its `out`.

### 3.4 Test

Renders the prompt twice with the same seed: once with the LoRA at `lora_multiplier`, once without it. The two images appear side by side in the panel and differ by the LoRA only.

- `lora` empty = the Train tab's `output`. Set it to test another file, e.g. an `_e10` checkpoint.
- `model` = `sft` tests the Base LoRA on SFT.
- Output: `output/moss/test/<date-time>/lora/` and `…/base/`, each with a `manifest.json`. Every Test gets a new folder, so earlier runs stay.

---

## 4. Recommended Settings

The measured run: 30 images of one artist (`@channel (caststation)`), Base, LoRA rank 16, alpha 16, lr 1e-4 (the earlier defaults), `weight_svd`, 20 epochs = 600 steps, about 18–21 minutes. Of the variants tried, this one carried the most of the artist's cues with no leakage. The current defaults (rank 32, alpha 128, lr 2e-5) came from a later comparison, below.

| Field | Default | Notes |
|---|---|---|
| `algo` | `lora` | `lokr` / `loha`: §4.1 |
| `rank` | `32` | 37.2 M parameters at rank 16 (the earlier default) |
| `alpha` | `128` for `lora` (scale 4 at rank 32); = rank for `lokr` / `loha` | scale = alpha / rank |
| `lr` | `2e-5` | AdamW, 5 % linear warmup |
| `down_init` | `weight_svd` | `weight_svd` or `kaiming` |
| `epochs` | `8` | the measured run used 20 |
| `caption_dropout` | `0` | the measured run used 0 |

- **alpha and lr**: under AdamW, raising the scale is the same as raising the lr of the LoRA's up matrix by that factor. Rank 16 with alpha 90 (scale 5.6) at lr 1e-4 puts the up matrix at ~5.6e-4. That arm matched the default's style but started copying the set's signature: one sample carried the artist's watermark and handle. Alpha 90 at lr 5e-5 gave a softer, partial transfer. That comparison ranked alpha = rank at lr 1e-4 first, and it was the default until the later comparison below.
- **Current defaults (rank 32, alpha 128, lr 2e-5)**: on the caststation set (20 epochs, `weight_svd`, layout_json) with the artist tag injected, the two complex gallery prompts (a ten-character ensemble, a 4:1 banner) behaved differently. Rank 32 / alpha 128 at lr 1e-4 pushed the style hard (white backgrounds, flat pale fills) but lost prompt details (the wrong character held the bag, a plush was missing). At lr 2e-5 the details held (right bag holder, plush present) with a milder style shift, judged much better, so it became the default. Under AdamW the effective lr at scale 4 and lr 2e-5 is 8e-5 for the up matrix and 2e-5 for the down matrix.
- **down_init**: `weight_svd` starts the LoRA's down projection on the base weight's top singular vectors; `kaiming` is the random init. `weight_svd` kept the artist's style better by eye. The two inits give different samples at the same seed, about as different as a seed change.
- **epochs**: at 10 of 20 epochs the results were looser and more varied; at 20 they were cleaner and more uniform, with a mild drift toward frontal, centred upper bodies. Set `save_every_epochs` to compare both from one run.
- **sigma_\*** (Advanced): the defaults are MOSS's final pretraining noise schedule. Anima's schedule (`sigma_mu 0`, `sigma_s 1`, `sigma_cutoff 1.0`, `sigma_uniform 0`) gave the weakest transfer of every arm.

### 4.1 LoKr and LoHa

`algo` = `lokr` (Kronecker product) or `loha` (Hadamard product of two rank-r pairs), from LyCORIS. Files keep LyCORIS's tensor names and load through the same `lora` field. `down_init` applies to `lora` only.

| Setup | Fields | Speed (16 GB, 1024²) | Result on the artist set at lr 1e-4 |
|---|---|---|---|
| LoKr f8 | `rank 100000`, `factor 8` | 2.26 s/step, 3.6 GB free | Close to the LoRA, slightly less character in the faces |
| LoHa r16 | `rank 16` | 2.49 s/step, 1.7 GB free | Overshot: grey-teal palette shift, the set's costume on other characters |

- Both need more memory than a LoRA: set `blocks_to_swap 24` (Basic) and `activation_memory_budget 0.1` (Advanced). The earlier defaults (swap 12, budget 0.15) ran out of memory at 1024².
- LoKr with `rank` at or above half the larger factor's side (the `rank 100000` above) uses a full matrix for that factor and ignores alpha (scale 1).
- LoHa: lr 5e-5 is the next setting to try; it has not been measured.

### 4.2 Memory plan (16 GB)

Defaults: `grad_checkpointing` off, `compile` on, `blocks_to_swap 12` (Basic; DiT blocks kept on the CPU and streamed in) and `activation_memory_budget 0.1` (Advanced; how much of the forward pass is kept for the backward, the rest is recomputed).

| Plan | Fields | s/step | Free VRAM |
|---|---|---|---|
| Default | swap 12, budget 0.1 (rank 32) | 1.85 | ~1.7 GiB |
| Earlier default | swap 12, budget 0.15 (rank 16) | 1.77 | ~0.7 GiB |
| More room | swap 24, budget 0.2 | 1.78 | 2.26 GiB |
| Checkpointing | `grad_checkpointing` on, swap 0 | 2.09 | 1.60 GiB |

Synthetic 1024² image + 300 text tokens (~4.4k tokens in all), LoRA rank 16, RTX 5070 Ti. The current default (rank 32, budget 0.1, swap 12) measured 1.85 s/step with ~1.7 GB free and a peak reserved of ~13.1 GB at ~4.5k joint tokens (image 4096 + text 455); rank 16 at budget 0.1 measured 1.82 s/step with ~2.0 GB free. Real samples ran up to ~4.9k tokens, and longer captions use more memory, which `swap 24` leaves room for at the same speed.

- Between swap 8 and 36 the speed barely moves (1.75–1.84 s/step), and budgets 0.1–0.3 barely change it either; swap 40 is slower (2.34 s/step) because the PCIe transfers stop hiding behind compute. Below that point, the swap count trades VRAM, not speed.
- The best pair depends on the PCIe link and the GPU. `scripts/moss/tune_swap.py` (CLI, [Appendix](#appendix-cli), about 4 minutes) times each swap count on the cache's largest sample and prints the fields to train with.
- `blocks_to_swap -1` sizes the swap from free VRAM; it is calibrated for LoRA and under-reserves for LoKr/LoHa.
- `compile` off needs `grad_checkpointing` on. Compiled training ran 2.0 against 3.1 s/step, for ~30 s of compile at the start.

---

## 5. Testing

The sampler defaults come from MOSS's reference pipeline; Anima's sampler habits do not carry over.

| Field | Default | What it does |
|---|---|---|
| `cfg` | `3.0` | CFG scale; `1.0` = no uncond pass |
| `uncond_skip` | `22` | The uncond pass is the model with block 22 skipped (still two passes per step); `''` = plain CFG |
| `shift` | `5.0` (Advanced) | Scheduler shift |
| `steps` | `50` | |
| `sampler` | `er_sde` | `er_sde` or `euler` |
| `width` / `height` | `1024` | Multiples of 16 |
| `negative_prompt` | empty | The pipeline default |

- **er_sde vs euler**: `euler` is the reference pipeline's sampler. `er_sde` costs the same and gives sharper line work and denser detail; with the LoRA, hands came out cleaner. It also drifts from some prompt details (sunset light, hair colour, "sitting on a bench") and gave up a little of the set's look. The two give different images at the same seed, so compare LoRA against base within one sampler.
- **Spectrum** (checkbox): most steps forecast the model's output instead of running all 42 blocks. At 50 steps with the default `spectrum_flex_window 0.5` (Advanced) it is 2.6× faster for +0.85 GiB, and hard to tell apart from the full 50 steps at grid scale. `0.25` is 2.2× and more conservative; `1.0` is 3.1× and loses fine detail.
- **compile** (Advanced, on): about 1.37× on top, with or without Spectrum, for a few seconds of compile per job. Composition is the same as uncompiled; fine detail (hair strands, patterns) differs, so compare compiled images with compiled images.

| Base, 1024², 50 steps | Time per image |
|---|---|
| Plain, uncompiled | 72 s |
| Plain, compiled | 52 s |
| Spectrum, compiled | 20 s |

**Comparing**: the Test button already renders LoRA and no-LoRA at the same seed. To compare two LoRAs or epochs, keep the prompt, seed, sampler and compile setting the same and change only `lora`.

---

## 6. Troubleshooting

| Symptom | What to do |
|---|---|
| Out of memory in training | Basic: `blocks_to_swap 24`, then Advanced: `activation_memory_budget 0.1`; or run `tune_swap.py`. LoKr/LoHa need 24 / 0.1. `grad_checkpointing` on with `blocks_to_swap 0` is the slower plan that fits without swap |
| Style too weak | More epochs (the measured run used 20). Check the prompt carries the artist tag spelled as in the captions, and that `prompt_format` matches `caption_format` |
| Style too strong, the set's signature or watermark shows up, every image has the same composition | Test an earlier `_e<N>` checkpoint or lower `lora_multiplier`. A higher effective lr is what started copying the signature (the default lr is now 2e-5) |
| The set's costume or colours leak onto other characters | Use `caption_format layout_json` (the default; it leaked less than plain tags). Switch LoHa back to LoRA or LoKr |
| Switched `caption_format` | Run Preprocess again: the text cache is re-encoded, latents are reused. The status line counts do not show the switch. Set `prompt_format` to match, then retrain |
| `layout_json needs the danbooru tag CSV` | `make download-model ARGS=danbooru_tags` |
| `No LoRA at …` on Test | Train first, or set `lora` in the Test box |

---

## Appendix. CLI

The GUI submits these jobs; they can also be run through the daemon directly. `--help` on each script lists every flag. `train.py --config configs/moss/lora.toml` reads flags from a TOML file (the shipped defaults; copy it to keep a recipe); flags given on the command line still win.

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'output/moss/cache/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'output/moss/cache/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors --spectrum"
make daemon-run ARGS="scripts/moss/tune_swap.py --cache output/moss/cache/<set>"
```

LoKr / LoHa:

```bash
make daemon-run ARGS="scripts/moss/train.py --cache 'output/moss/cache/<set>' --epochs 20 --algo lokr --rank 100000 --factor 8 --blocks_to_swap 24 --activation_memory_budget 0.1"
make daemon-run ARGS="scripts/moss/train.py --cache 'output/moss/cache/<set>' --epochs 20 --algo loha --rank 16 --lr 5e-5 --blocks_to_swap 24 --activation_memory_budget 0.1"
```

More measurements: [`docs/experimental/moss_lora.md`](../experimental/moss_lora.md).
