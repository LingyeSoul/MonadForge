# Colorization (EasyControl)

Manga / illustration **colorization** as an EasyControl control task: feed a
black-and-white screentoned page (line art + halftone tone), get a colorized
image. This is the first project under `easycontrol_adapters/` — a workspace for
per-task EasyControl control adapters that share the shipped EasyControl network
(`networks/methods/easycontrol.py`) but differ in how the *condition* is built.

## The idea

Real B&W manga has **no color ground truth**, so we can't make training pairs by
desaturating. We invert the direction:

- **target** = the color illustrations we already have (`post_image_dataset/lora/`
  latents + captions — reused as-is, nothing re-cached).
- **condition** = a synthetic *mangafied* version of the same image (XDoG lineart
  + algorithmic screentone — the toned value range is split into a few luminance
  bands and each gets its own *pattern*: clustered dots, parallel-line/hatch tone,
  or cross-hatch, so darks/mids/lights carry distinct tones the way a real page picks
  tone by value), cached to `cond_cache_dir`.

Synthesizing a manga-like condition matches the inference distribution (real
screentoned pages) far better than naive grayscale. EasyControl's extended
self-attention cond stream does the conditioning; only the cond source changes
(cond ≠ target), wired via the new `cond_cache_dir` subset knob.

## Caption policy — caption-free by default

Trained with **`caption_dropout_rate = 0.8`** so the model colorizes from manga
*structure* alone most of the time and never becomes caption-dependent. At
inference:

- **default: empty prompt** → auto-colorization, no typing, no tagger.
- **optional steer**: a short prompt (`pink hair`) nudges specific colors.

Note the information-theoretic floor: B&W manga doesn't contain hair/eye/costume
color, so the caption-free default guesses *modal* colors — correct that with the
optional prompt when it matters. This is expected, not a bug.

## v0 scope (this build)

- **Pure `cv2`/`numpy` mangafication** — no model downloads. See `mangafy.py`.
- Training targets (danbooru illustrations) have **no speech bubbles**; inference
  manga pages do. v0 colorizes the art and you composite the original B&W text
  back over the result at inference. (Bubble pass-through is Phase B.)

## Files

| File | Role |
|------|------|
| `mangafy.py` | color RGB → B&W manga (XDoG lineart + dot/line/cross screentone), per-stem jitter |
| `prep.py` | mangafy `resized/` images → staging dir → VAE-encode into `colorize_cond/` |

Configs: `configs/datasets/colorize.toml`, `configs/methods/colorize.toml`.

## Run

The colorization project rides the existing `exp-easycontrol*` targets via the
`EASYADAPTER=colorize` selector (no separate targets):

```bash
# 1. Build the condition latents (mangafy + VAE-encode). Idempotent.
make exp-easycontrol-preprocess EASYADAPTER=colorize
#    or directly:  python easycontrol_adapters/colorization/prep.py
#    QA a few first:  python easycontrol_adapters/colorization/prep.py --limit 8
#    Inspect staged manga PNGs under post_image_dataset/colorize_staging/

# 2. Train (frozen DiT, adapter-only, caption-free default).
make exp-easycontrol EASYADAPTER=colorize

# 3. Inference — feed a real B&W manga page as the control image (empty prompt).
REF_IMAGE=post_image_dataset/resized/takaman_\(gaffe\)/7645571.png \
    make exp-test-easycontrol EASYADAPTER=colorize
#    Optional steer:  ... ARGS='--prompt "pink hair, red tie"'
```

`EASYADAPTER=colorize` ⇒ `configs/methods/colorize.toml` (train),
`easycontrol_adapters/colorization/prep.py` (preprocess), and the `anima_colorize`
checkpoint + `output/tests/colorize/` (inference). Unset ⇒ default ref==target
EasyControl.

## Phase B (deferred)

- Learned lineart (Anime2Sketch / sketchKeras) + **ScreenVAE** screentone synthesis
  for on-manifold tones.
- **Speech-bubble pass-through:** paste synthetic bubbles onto the manga cond and
  mask those regions from the loss, so the model leaves text untouched.
- Gradient/special screentones; difficulty curriculum (single clean figure →
  multi-figure dense pages).
