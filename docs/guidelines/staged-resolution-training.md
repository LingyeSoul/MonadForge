# Staged-resolution training

Staged-resolution training keeps one model, optimizer, LR scheduler, and global
step while switching the active dataset row at configured progress boundaries.
It does not start independent training processes and it does not preprocess data
during training.

## Data contract

Create one fully preprocessed dataset row per resolution tier. Rows may originate
from the same source images, but each row needs its own resized image directory
and latent cache directory. For the common three-stage curriculum:

```toml
staged_resolution = true
staged_resolution_ratios = "20,30,50"
staged_resolution_base_sides = "512,768,1024"
max_train_steps = 6000

[[datasets]]
batch_size = 4
[[datasets.subsets]]
image_dir = "post_image_dataset/resized-512"
cache_dir = "post_image_dataset/lora-512"
num_repeats = 1
recursive = true

[[datasets]]
batch_size = 2
[[datasets.subsets]]
image_dir = "post_image_dataset/resized-768"
cache_dir = "post_image_dataset/lora-768"
num_repeats = 1
recursive = true

[[datasets]]
batch_size = 1
[[datasets.subsets]]
image_dir = "post_image_dataset/resized-1024"
cache_dir = "post_image_dataset/lora-1024"
num_repeats = 1
recursive = true
```

`staged_resolution` is retained as a deprecated shorthand for this exact
three-row layout. New configurations should use the generic schedule below.
The former subprocess behavior, automatic batch-size scaling, and LCM-aligned
save/sample intervals are no longer provided.

This produces these global-step intervals:

```text
0..1199    dataset row 0 (512 tier)
1200..2999 dataset row 1 (768 tier)
3000..5999 dataset row 2 (1024 tier)
```

Use explicit `max_train_steps` for stable percentage boundaries. When
`max_train_epochs` is set, the initial step budget is derived from the first
stage's dataset length and is therefore less intuitive.

Preprocess each row before starting training. For example, run
`scripts/preprocess/resize_images.py` separately with `--target_res 512`,
`--target_res 768`, and `--target_res 1024`, using distinct `--dst` paths, then
run the latent/text cache commands against the matching row paths. Startup fails
if any selected row is empty, has incomplete caches, or does not match the tier
declared by `staged_resolution_base_sides`.

## Arbitrary schedules

The generic TOML surface supports any number of stages and can reuse rows:

```toml
stage_schedule_enabled = true
stage_schedule = [
  { name = "low",  subset_index = 0, start_pct = 0.0, end_pct = 0.25 },
  { name = "mid",  subset_index = 1, start_pct = 0.25, end_pct = 0.60 },
  { name = "high", subset_index = 2, start_pct = 0.60, end_pct = 1.0 },
]
```

Stages must cover `0.0..1.0` without gaps or overlaps. In a multi-row
`DatasetGroup`, `subset_index` selects a dataset row. With one dataset row that
contains multiple local subsets, it selects the local subset instead.

At a boundary, the trainer filters from an immutable full-dataset snapshot,
rebuilds bucket indices, and prepares a fresh DataLoader through Accelerate.
Model and optimizer state remain untouched.

Malformed numeric values, negative row indices, non-finite boundaries, gaps,
overlaps, and rows with no complete batch fail during startup. Values in
`0.0..1.0` are fractions; values greater than `1` through `100` are interpreted
as percentages.

Resumable checkpoints persist the active stage, outer epoch, and stage-local
batch cursor in `train_state.json`. On resume the trainer selects the stage from
the restored global step, verifies it against the saved stage, rebuilds that
row's DataLoader, and skips the batches already consumed in the current epoch.
Checkpoint metadata records the normalized schedule and the unfiltered image
counts for every row.
