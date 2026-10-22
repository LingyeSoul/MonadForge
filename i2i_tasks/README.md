# i2i tasks

One folder per image-to-image task: how its condition (or pair) tree is built from the
dataset. Staging is model-agnostic; each model line encodes and trains from the staged
tree on its own.

```
i2i_tasks/<task>/  ──make i2i-stage TASK=<task>──▶  post_image_dataset/i2i/<name>/staging/  (+ pairs.jsonl)
                                                        │
             ┌──────────────────────────────────────────┴───────────────────┐
             ▼                                                              ▼
  EasyControl (Anima)                                          MOSS i2i (VAE reference on RoPE r)
  make i2i-preprocess TASK=<task>   (= easycontrol-preprocess)  make i2i-preprocess TASK=<task> MODEL=moss
  make easycontrol EASYADAPTER=<task>                           → configs/moss/i2i/<task>/cache.toml
                                                                → target latent + TE + {stem}.moss_ref
                                                                scripts/moss/train.py on that cache
```

`pairs.jsonl` is the model-agnostic pair list (`{stem, target, cond, caption}` per
line, paths relative to it); near_twins / sanitize write it, and
`library.moss.cache.manifest_pairs` reads it. Colorize's MOSS cache reads the staging
tree directly (`scripts/moss/colorize_cache.py`, `library/moss/colorize.py`: vision TE +
OCR captions). The MOSS task list is `library.moss.requests.I2I_CACHE`; the GUI's i2i tab
(`gui/moss/i2i.py`) runs stage → cache → train → test for each.

| Task (`TASK=`) | Folder | Staging builds |
|---|---|---|
| `colorize` | `colorization/` | mangafied B&W page per colour image (`mangafy`, `wb`); colour-only captions (`color_caption.py`) |
| `inpaint` | `inpainting/` | masked condition per image |
| `region` | `region/` | SAM3 masks → paint-to-character condition + positioned captions |
| `near_twins`, `sanitize` | `tools/near_twins/` | in-artist near-twin pair tree |
| `subject`, `subject_edit` | `tools/subject_pairs.py`, `tools/subject_edit_pairs.py` | subject pairs |
| `phash_edit` | `tools/phash_edit_pairs.py` | perceptual-hash aligned edit pairs |

The task registry is `_EASY_ADAPTERS` in `scripts/tasks/training.py` (a `stage` and a
`preprocess` body per task); knobs are the `[staging]` table of
`configs/easycontrol/<task>.toml`, whose `name` picks `post_image_dataset/i2i/<name>/`.
`make easycontrol-staging EASYADAPTER=<task>` is an alias of `i2i-stage`.
`post_image_dataset/easycontrol` is a symlink to `post_image_dataset/i2i` so pre-rename
paths in configs and caches still resolve.

Adding a task: `ADAPTER_GUIDE.md`.
