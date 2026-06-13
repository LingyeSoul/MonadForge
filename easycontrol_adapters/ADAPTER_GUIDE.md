# Building your own EasyControl adapter

This guide shows you how to add a **new EasyControl control task** to Anima for
your own use. It's a local adapter — it lives under
`easycontrol_adapters/<your_task>/`, not in git. We follow the one worked
example, **colorize** (`easycontrol_adapters/colorization/`), step by step, and
point out exactly what to change for your own task.

The one thing to take away:

> **You are not writing model code.** The network, the forward pass, the
> `b_cond` gate, the inference cache — all of that already ships and is shared by
> every control task. The *only* thing your task changes is **how the condition
> image is built.** Everything else in this guide is wiring around that.

---

## 0. The idea — what an EasyControl adapter is

EasyControl guides generation using a **reference image**. That reference is run
through the VAE into *cond tokens*, which flow alongside the image you're
generating. At each step, the model gets to look at both. (Full architecture:
`docs/experimental/easycontrol.md`.)

Plain EasyControl uses the **same image** as both the reference and the target —
so it just learns to copy. A *control task* breaks that: it pairs each target
with a **different** reference, so the model learns `reference → target` instead
of `copy`.

| | plain EasyControl | a control task (e.g. colorize) |
|---|---|---|
| target (what you want) | image X | the color image X |
| reference (the hint) | image X (identical) | a *changed* version of X (B&W manga of X) |
| what it learns | copy | manga → color |
| text | full caption | (optional) shorter caption |

Here's the colorize trick, which you can reuse: **real B&W manga has no color
version to learn from**, so you can't just collect `(B&W, color)` pairs. So you
flip it around — start with color images you already have (those are the
targets) and **make** the B&W reference from each one (line art + screentone, by
algorithm). The key is that the B&W you generate has to look like the real B&W
you'll feed in at inference time. If you were building `depth → image` or
`pose → image` instead, your "make the reference" step would be a depth estimator
or a pose detector run over your training images.

So your whole job is: **write a function `target_image → reference_image`, cache
its output, point a dataset at it, and add a config plus a one-line name
registration.**

---

## 1. The four things you touch

To add an adapter named `<task>`, you create or edit exactly these four:

| # | Thing | colorize version | what it's for |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` project | `colorization/` (`mangafy*.py`, `color_caption.py`, `prep.py`) | builds and caches the reference image (and maybe a shorter text cache) |
| 2 | `configs/datasets/<task>.toml` | `configs/easycontrol/colorize.toml` | a dataset that pairs each target with its reference via **cond_cache_dir** |
| 3 | `configs/methods/<task>.toml` (+ `configs/gui-methods/<task>.toml`) | `configs/easycontrol/colorize.toml` | the config — points at the dataset, sets LR / epochs / `network_args` |
| 4 | `scripts/tasks/{training,inference}.py` | `_EASYADAPTERS = {"colorize"}` + branches | makes `EASYADAPTER=<task>` work with the `make easycontrol*` commands |

We go through them in order. None of this touches `networks/`.

---

## 2. Thing 1 — the adapter project (`easycontrol_adapters/<task>/`)

This is where the real work is. It does two jobs: **make the reference image**
and **cache it**. (Optionally, a third: build a task-specific text cache.)

### 2a. The function that builds the reference

It's a plain function: take a color image as RGB `uint8 (H,W,3)` plus a seed,
return a reference image as RGB `uint8 (H,W,3)` of the **same size**. (Same size
matters — see §3.) It must give the same output for the same seed, so re-runs and
parallel workers all agree.

In colorize this is `mangafy.py::mangafy_array` (with a GPU twin
`mangafy_gpu.py::mangafy_array_gpu`):

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

Four things worth copying:

- **Seed each image from its name, deterministically.** colorize uses
  `zlib.crc32(stem)` — **not** Python's `hash()`, which changes per process and
  would make parallel workers disagree. You can still add variety (colorize
  jitters the screentone angle per page) — just derive it from the seed, so it
  stays reproducible.
- **Import heavy stuff lazily.** colorize has three engines (`cv2` / `gpu` /
  `sd`) and only loads the 3.5 GB SD model if a page actually needs it. If your
  builder needs a model (depth net, line extractor), import it only when used.
- **A no-download fallback is great to have.** colorize's `cv2`/`gpu` engines
  need zero downloads, so you can prep and train on a fresh checkout with no
  extra download step. Give yours one if you can.
- **Write files atomically.** colorize's `_save_png_atomic` writes to a temp file
  then renames it. Without this, an interrupted run can leave a half-written PNG
  that the "skip if it exists" check will trust forever. This is a real bug —
  copy the pattern.

If your reference is something you **already have on disk** (real depth maps,
real sketches), you can skip the build step entirely and just cache those
directly. You only need to build the reference when you have to derive it from
the targets.

### 2b. (Optional) a shorter text cache

colorize doesn't only change the reference — it also **trims the caption down to
color words** (`color_caption.py::filter_to_colors`). The reason is worth
understanding: the reference (line art + screentone) already encodes *everything
about shape and layout*, so the only thing text still needs to say is the one
thing B&W can't — **color**. Trimming the caption to color words makes every
remaining word something the model genuinely can't get from the reference. That
gives a strong `prompt → color` link instead of weak nudging by a few color
words buried in a long caption.

Ask the same question for your task: **what does the reference already lock down,
and what's left for text to decide?** A `pose → image` reference fixes the pose
but not the clothing or setting — so you'd probably keep the *full* caption. A
`depth → image` reference fixes layout but not identity or color. colorize's
"trim the caption to whatever's still ambiguous" is a *pattern* to consider, not
a rule — many adapters keep captions as-is and skip the text cache completely
(just leave `text_cache_dir` out of the dataset — see §4).

If you do trim captions, note colorize's two separate knobs (don't mix them up):

- **`caption_dropout_rate`** — the auto-color *floor*. About 5% of training steps
  drop the caption entirely, which teaches the model what to do when you give it
  no prompt. Keep this **low** (`0.05`); a high value over-trains the no-prompt
  path and makes prompts weak.
- **`use_shuffled_caption_variants`** — the full-vs-partial *balance*. The text
  cache holds several versions (v0 = the full color set, v1+ = shuffled with each
  word dropped about half the time), and the loader picks v0 20% of the time and
  v1+ 80% of the time. This is what makes partial prompts like "pink hair" on
  their own still work.

### 2c. `prep.py` — the cache builder

Three stages, each **idempotent** (it skips work that's already done, so it's
safe to re-run):

1. **Build** — walk every color image under `--src`
   (`post_image_dataset/resized`), run your builder function, and write the
   reference PNG to a `--staging` folder that mirrors the source layout.
2. **Encode** — VAE-encode those staged references into `--cond_cache_dir` using
   `library.preprocess.cache_latents`, at each image's **native size** so the
   reference latent ends up the same shape as its target latent. Same
   `{stem}_{WxH}_anima.npz` format as the normal cache.
3. **(Optional) Text** — re-encode captions through your filter into a
   `--text_cache_dir`, using `library.preprocess.cache_text_embeddings` with a
   `caption_transform=` (and `caption_shuffle_variants` /
   `caption_tag_dropout_rate`).

Use the existing library helpers — `library.preprocess.{cache_latents,
cache_text_embeddings, tqdm_progress}` and
`library.preprocess._dataset.walk_images`. Don't write your own encode loop;
`prep.py` is just a thin shell over these, exactly like the scripts in
`scripts/preprocess/`.

Two correctness traps colorize handles for you, that you get for free by copying
its structure:

- **Stems must match.** The text stage reads `.txt` captions from the caption
  master (`image_dataset/`, laid out the same as `resized/`) so the resulting
  text cache file names line up with what the loader looks for (it keys off
  `image_dir=post_image_dataset/resized`). If your cache file names don't match
  the target stems, the loader just won't pair them — silently.
- **The uncond sidecar.** colorize's text stage re-creates the shared `T5("")`
  empty-prompt sidecar if it's missing. If you build a text cache and use caption
  dropout, do the same
  (`library.inference.uncond.stage_uncond_sidecar_with_models`).

---

## 3. The one rule you can't break — reference and target must have the same token count

The DiT runs on Anima's **native-shape bucketing** (two token-count families,
4032 and 4200; see CLAUDE.md and `docs/experimental/easycontrol.md` under "Cond
token count"). There's **no padding knob** — the reference runs at whatever token
count its latent actually has.

This is exactly why §2a says the reference must be the **same size** as the input
and §2c says to encode at **native size**: do both and the reference latent lands
in the same bucket as its target automatically, and everything just works. If you
want a smaller reference (totally fine — smaller = less memory, faster), shrink it
at the **image** level, before encoding, so the latent still lands on a real
bucket. Don't try to cap the token count inside the network.

---

## 4. Thing 2 — the dataset (`configs/datasets/<task>.toml`)

This file is what makes the reference different from the target. It's an ordinary
dataset (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`) with **one extra
knob**: `cond_cache_dir` (and optionally `text_cache_dir`).

colorize's (`configs/easycontrol/colorize.toml`), with notes:

```toml
[general]
caption_extension = '.txt'
keep_tokens = 3

[[datasets]]
batch_size = 1
validation_split = 0.005
validation_seed = 42

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'        # the COLOR targets
  cache_dir = 'post_image_dataset/lora'           # target latents + text — REUSED, not rebuilt
  cond_cache_dir = 'post_image_dataset/easycontrol/colorize/cond'   # ← the reference latents (from prep.py)
  text_cache_dir = 'post_image_dataset/easycontrol/colorize/text'   # ← color-only text cache (from prep.py)
  recursive = true
  flip_aug = false        # latents can't be flipped after the fact, and there's no flipped reference
  num_repeats = 1
```

What the redirects do:

- **`cond_cache_dir`** — the one knob that makes this a control task. The loader
  matches each target to a reference latent in here by stem. This is the
  reference EasyControl feeds into its two-stream forward.
- **`text_cache_dir`** — redirects **only** the text cache (latents still come
  from `cache_dir`). Leave it out entirely if you keep full captions — then the
  loader uses the shared text cache and you can skip `prep.py`'s text stage.
- **`flip_aug = false`** — required. A flipped target would need a flipped
  reference latent, which you never cached. Keep flipping off.

Note that colorize **reuses** the shared `post_image_dataset/lora` cache for the
target latents and text — `make preprocess` already built those, so nothing is
re-encoded. Your adapter only adds the *reference* cache (and maybe a shorter
text cache).

---

## 5. Thing 3 — the method config (`configs/methods/<task>.toml`)

This is almost a copy of `configs/easycontrol/easycontrol.toml`. The only structural
change is `dataset_config`, pointing at your dataset; the rest is just
hyperparameters.

colorize's (`configs/easycontrol/colorize.toml`), the lines that matter:

```toml
dataset_config = "configs/easycontrol/colorize.toml"   # ← your dataset from §4

network_module = "networks.methods.easycontrol"     # SHARED — same network as plain EasyControl

network_dim = 32
network_alpha = 32
network_args = [
    "b_cond_init=-6.0",     # how strongly the reference starts out (see below)
    "cond_scale=1.0",
    "apply_ffn_lora=1",     # 0 → drop the FFN LoRA, about half the trainable params
]

output_name = "anima_colorize_full"   # ← checkpoint name; the inference selector looks for this

use_easycontrol = true
easycontrol_drop_p = 0.0              # reference dropout for image-CFG; default 0.1, colorize wants 0
masked_loss = false

caption_dropout_rate = 0.05           # auto-color floor (§2b)
use_shuffled_caption_variants = true  # full-vs-partial balance (§2b)
easycontrol_cond_noise_max = 0.02     # small — too much noise erases the line art
learning_rate = 2e-5
max_train_epochs = 3
blocks_to_swap = 0                    # recommended for EasyControl
gradient_checkpointing = true
unsloth_offload_checkpointing = true
```

Knobs to think about for your own task:

- **`b_cond_init`** — how much the reference matters at the very start of
  training. `-10` means the reference barely contributes at step 0 (so the model
  starts out like the plain DiT, then learns to lean on the reference); the bench
  in `docs/experimental/easycontrol.md` ("Step-0 baseline equivalence") explains
  why. colorize loosens it to `-6` so the reference kicks in sooner — fine for a
  task with a strong reference. It's learnable either way.
- **`easycontrol_cond_noise_max`** — how much noise is added to the reference
  during training (σ drawn from `U(0, max)`, applied as `cond + σ·ε`). `0` means
  the reference is treated as a perfect blueprint; a higher value degrades it into
  a rough "hint," forcing text to carry the missing detail. colorize uses `0.02`
  (tiny — the line art *is* the signal). The default easycontrol.toml uses `0.3`.
- **`easycontrol_drop_p`** — how often the whole reference is dropped during
  training, for image-CFG. colorize uses `0` (it always wants the reference);
  default is `0.1`.
- **`output_name`** — must be unique; the inference step finds your latest
  checkpoint by this name (§6).

You can also add `configs/gui-methods/<task>.toml` — a standalone version (no
toggle blocks) with a `[variant]` block (`family = "easycontrol"`, `label`,
`description`, `order`) so it shows up in the GUI's EasyControl dropdown. See
`configs/gui-methods/easycontrol.toml`. Skip this if you only run from the command
line.

---

## 6. Thing 4 — make `EASYADAPTER=<task>` work

The `make easycontrol*` commands switch on the `EASYADAPTER` environment
variable. Three small edits make `EASYADAPTER=<task>` use your config, prep, and
checkpoint.

**In `scripts/tasks/training.py`:**

1. Add the name to the allowlist:
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   (`_easyadapter()` checks against this set and errors on a typo.)

2. Route preprocessing to your `prep.py` in `cmd_easycontrol_preprocess`:
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. Training itself needs **no** edit — `cmd_easycontrol` already calls
   `train(_easyadapter() or "easycontrol", extra)`, so once your name is in the
   allowlist, `EASYADAPTER=<task>` runs `configs/methods/<task>.toml` on its own.

4. (Only if your builder needs downloads) add a branch in
   `cmd_easycontrol_download` for your weight-fetch task.

**In `scripts/tasks/inference.py`** (`cmd_test_easycontrol`): the
selector currently hard-codes colorize. Generalize the few colorize-specific
values for your task — the checkpoint name, the output folder, the fallback
reference folder, and the empty-prompt default:

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

Add your own `adapter == "<task>"` cases next to these (the weight name must match
your config's `output_name`). If your task wants an empty-prompt default and a
reference pulled from a specific folder — like colorize — copy the `is_colorize`
branches further down too.

---

## 7. Run it

```bash
# 1. Build the reference cache (build + VAE-encode). Idempotent.
make easycontrol-preprocess EASYADAPTER=<task>
#    Check a few first:
python easycontrol_adapters/<task>/prep.py --limit 8
#    Eyeball the staged reference PNGs under post_image_dataset/<task>_staging/

# 2. Train (DiT frozen, adapter only).
make easycontrol EASYADAPTER=<task>

# 3. Inference — give it a real, in-distribution reference image.
REF_IMAGE=path/to/condition.png make test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

First-time setup: run `make preprocess` once so the shared target latents and
text cache exist in `post_image_dataset/lora` (your adapter reuses them).

### Inference tips (learned from colorize)

- **Give it a real, in-distribution reference.** At inference the reference is
  VAE-encoded as-is — there's no building step. colorize feeds a real screentoned
  B&W page; a plain grayscale photo is out of distribution and looks worse.
  Whatever your builder was *imitating* is what inference expects to receive.
- **`--easycontrol_image_match_size`** — picks the token bucket that matches the
  reference's aspect ratio, so tall pages don't get squashed. colorize forces
  this on.
- **`--easycontrol_scale`** (`EC_SCALE=`, how closely to follow the reference) —
  `1.0` is the trained default; raise it (1.1–1.2) if the reference bleeds
  through too much, lower it (0.7–0.8) for looser output.
- **`--guidance_scale`** — works together with your text setup. colorize: empty
  prompt → low CFG (1.0–1.5, nothing to push toward); text prompt → higher
  (3.0–4.5, which is what makes the prompt actually take effect).

---

## 8. Checklist

- [ ] `easycontrol_adapters/<task>/` with a deterministic, atomic-writing builder
      (`(img, seed) → reference`, same size) and an idempotent `prep.py`
      (build → encode → optional text).
- [ ] Reference encoded at **native size** so its token count matches the
      target's bucket (§3).
- [ ] `configs/datasets/<task>.toml` with `cond_cache_dir` (+ optional
      `text_cache_dir`), `flip_aug = false`, reusing the shared target cache.
- [ ] `configs/methods/<task>.toml` → points at the dataset, unique
      `output_name`, `network_module = "networks.methods.easycontrol"`,
      `use_easycontrol = true`. (Optional GUI variant.)
- [ ] `EASYADAPTER=<task>` added to `_EASYADAPTERS` + a preprocess branch
      (training.py) + generalized checkpoint/output/fallback (inference.py).
- [ ] Eyeballed a `--limit 8` staging batch before the full run.

---

## 9. Where to read more

- **`easycontrol_adapters/colorization/README.md`** — the full colorize design
  notes (caption policy, screentone bands, Phase B roadmap). The reference
  implementation of everything above.
- **`docs/experimental/easycontrol.md`** — the network itself: the two-stream
  forward, the `b_cond` step-0 bench, the inference cache, memory use, limits.
  Read this before touching `network_args`.
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` and the patched
  `Block.forward`. You should **not** need to edit this for a new adapter; if you
  think you do, double-check whether your task's real difference is actually in
  the reference image.
- **`networks/CLAUDE.md`** — the per-module map and dispatch rules.
