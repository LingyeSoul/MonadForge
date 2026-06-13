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

## Caption policy — color-only captions

The condition (mangafied lineart + screentone) already encodes *everything
spatial*: composition, poses, objects, layout. The one variable B&W manga can't
carry is **hue/chroma** (and **which series** the page is from). So the text
channel for colorization is reduced to **color tags + the copyright/series tag** —
hair/eye/skin color, `<color> <garment>`, background color, plus the bare
copyright tag (`color_caption.filter_to_colors_and_copyright`; pass
`--no-text_keep_copyright` for the color-only `filter_to_colors`). Every surviving
token is then a fact the model can't get from structure, which gives a strong
**prompt→color** binding instead of the weak, arbitrary steering you get when
color tags are buried in a full caption. Copyright tags are recognized against
`groups.copyright` in `post_image_dataset/captions/caption_index.json`.

The TE cache lives in its own `text_cache_dir`
(`post_image_dataset/easycontrol/colorize/text`) — a TE-only redirect, so the **latents still
come from the shared `lora/` cache** (nothing re-encoded there). Built by
`prep.py`'s text stage as a multi-variant cache (shipped defaults:
`--text_shuffle_variants 2`, `--text_tag_dropout_rate 0.5`): v0 = the full color
set, v1 = smart-shuffled with each color tag independently dropped at p≈0.5, so
the model learns to colorize from a *partial* color spec, not just a complete
palette. The **copyright tag is protected from tag-dropout** — it rides every
variant. With `use_shuffled_caption_variants_only=true` (colorize default) the
loader never draws v0, so every non-empty step trains a partial spec.

Two independent knobs shape the per-step caption, and it's worth not conflating
them:

- **`caption_dropout_rate`** — the *auto-color floor*. That fraction of steps
  drop the caption entirely (→ uncond `T5("")`), training the empty-prompt
  default. This knob does **not** balance full-vs-partial color, and a high rate
  (the old `0.9`) over-trains the unconditional path into weak, arbitrary steering.
- **`use_shuffled_caption_variants = true`** — the *full-vs-partial balance*. On
  the captioned steps, the loader draws **20% v0 (full colors) / 80% v1+
  (partial)** (`strategy.py`), so partial prompts ("pink hair" alone) work.
- **`use_shuffled_caption_variants_only = true`** (colorize default) — drop v0
  from the pool entirely: captioned steps draw **only** the partial v1+ variants
  (uniform). Every non-empty step then trains a partial spec; the copyright tag
  still rides every variant (it's dropout-protected, not a color tag).

At inference:

- **empty prompt** → auto-colorization (guesses *modal* colors).
- **color prompt** (`pink hair, blue eyes, white dress`) → reliably steers, and
  partial specs work because of the tag dropout above.

Note the information-theoretic floor: B&W manga doesn't contain hair/eye/costume
color, so the empty-prompt default guesses — correct it with a color prompt when
it matters. This is expected, not a bug.

## v0 scope (this build)

- **Pure `cv2`/`numpy` mangafication** — no model downloads. See `mangafy.py`.
- Training targets (danbooru illustrations) have **no speech bubbles**; inference
  manga pages do. v0 colorizes the art and you composite the original B&W text
  back over the result at inference. (Bubble pass-through is Phase B.)

## Files

| File | Role |
|------|------|
| `mangafy.py` | color RGB → B&W manga (XDoG lineart + dot/line/cross screentone), per-stem jitter |
| `color_caption.py` | reduce a full Anima caption to color tags only (`filter_to_colors`) |
| `prep.py` | mangafy `resized/` → VAE-encode into `easycontrol/colorize/cond/`; re-encode color-only captions into `easycontrol/colorize/text/` |

Config: `configs/easycontrol/colorize.toml` — a single self-contained descriptor
(top-level `name` + `[staging]` / `[preprocess]` / `[training]` knob tables + a
`[general]`/`[[datasets]]` blueprint, same shape as `near_twins.toml`). Colorize
trains the base `easycontrol` method with the descriptor's `[training]` table
folded in as CLI overrides; the staging/preprocess knobs drive `prep.py`. Edit
this one file to retune colorize.

## Run

The colorization project rides the existing `easycontrol*` targets via the
`EASYADAPTER=colorize` selector (no separate targets):

```bash
# 1a. Staging — synthesize the synthetic B&W manga condition tree (mangafy only).
make easycontrol-staging EASYADAPTER=colorize
#    or directly:  python easycontrol_adapters/colorization/prep.py --skip_encode --skip_text
#    QA a few first:  make easycontrol-staging EASYADAPTER=colorize ARGS="--limit 8"
#    Inspect staged manga PNGs under post_image_dataset/easycontrol/colorize/staging/
# 1b. Preprocess — VAE-encode the cond latents + cache color-only text. Idempotent.
make easycontrol-preprocess EASYADAPTER=colorize
#    (re-stage inline in one shot:  make easycontrol-preprocess EASYADAPTER=colorize ARGS="--no-skip_mangafy")

# 2. Train (frozen DiT, adapter-only, caption-free default).
make easycontrol EASYADAPTER=colorize

# 3. Inference — feed a real B&W manga page as the control image (empty prompt).
REF_IMAGE=post_image_dataset/resized/takaman_\(gaffe\)/7645571.png \
    make test-easycontrol EASYADAPTER=colorize
#    Color steer:  ... ARGS='--prompt "pink hair, blue eyes, white dress"'
```

### Inference settings

- `--easycontrol_image_match_size` — always on (the task sets it); picks the
  token bucket matching the page aspect ratio so tall manga pages don't squash.
- `--easycontrol_scale` (cond structure adherence) — **1.0** (trained default);
  1.1–1.2 if color bleeds past lines, 0.7–0.8 for looser/creative coloring.
- `--guidance_scale` (cfg) — **empty prompt → 1.0–1.5** (nothing for high cfg to
  push toward → oversaturates); **color prompt → 3.0–4.5** (this is what makes a
  color prompt bite).
- `--infer_steps` 20–28, `--sampler euler` — colorization is "easy" (structure
  given); more steps buy little.
- Feed a **real screentoned B&W page** as the cond (it's VAE-encoded as-is, no
  XDoG at inference). A flat grayscale photo / clean lineart is out-of-distribution.

`EASYADAPTER=colorize` ⇒ `configs/easycontrol/colorize.toml` (descriptor: staging
+ preprocess + train knobs, folded onto the base `easycontrol` method),
`easycontrol_adapters/colorization/prep.py` (staging + preprocess), and the
`anima_colorize` checkpoint + `output/tests/colorize/` (inference). Unset ⇒
default ref==target EasyControl.

## Phase B (deferred)

- Learned lineart (Anime2Sketch / sketchKeras) + **ScreenVAE** screentone synthesis
  for on-manifold tones.
- **Speech-bubble pass-through:** paste synthetic bubbles onto the manga cond and
  mask those regions from the loss, so the model leaves text untouched.
- Gradient/special screentones; difficulty curriculum (single clean figure →
  multi-figure dense pages).
