# memorization — does a LoRA reproduce its training images?

Phase-0 diagnostic motivated by Bonnaire/Urfin/Biroli/Mézard, *"Why Diffusion
Models Don't Memorize"* (NeurIPS 2025, arXiv:2505.17638). Their picture: score
training has two timescales — `tau_gen` (novel high-quality samples appear,
~capacity-bound, n-independent) and `tau_mem` (training images get reproduced,
grows ~linearly with n). Early-stop in `[tau_gen, tau_mem]` and you generalize;
past `tau_mem` you memorize. Artist LoRAs train on *tiny* n → structurally the
memorization-prone regime, and CMMD (our live val signal) measures distributional
quality, **not** "is this checkpoint xeroxing training frames." This probe adds
that axis.

## What it does

For each chosen training caption `i`: condition on its cached TE, draw a
**held-out** noise seed, render `gen_i`. Compare against the training set in
pooled **PE-Spatial** cosine space (`{stem}_anima_pe_spatial.safetensors` — what
`make preprocess` already writes for REPA; no `preprocess-pe` needed):

```
s_self[i]  = cos(gen_i, train_i)              # closeness to its own source
s_other[i] = max_{j != i} cos(gen_i, train_j) # nearest OTHER training image
```

A *generalizing* LoRA re-renders the caption as a plausible new image in the
style: `s_self ≈ s_other`, so `contrast = mean(s_self - s_other) ≈ 0`. A
*memorizing* LoRA snaps back to the exact source: `contrast >> 0` and the global
nearest neighbour of `gen_i` is its own source (`self_lock`). **The floor is ~0
by construction** — no baseline run required (`--with_base` adds the no-adapter
contrast for an absolute anchor).

Why PE-Spatial and not pooled PE-Core: its 1024 patch tokens, mean-pooled, encode
*composition* ("what's where"), so a verbatim copy snaps to a near-identical
vector while same-style/different-layout images separate. PE-Core's global vector
over-weights style/identity — wrong axis for instance copies. It's also cached on
every artist set already.

## Reading the output

- **`contrast_mean`** and **`self_lock_frac`** are the headline signals.
  `verdict`: GENERALIZING / PARTIAL / MEMORIZING from heuristic thresholds
  (contrast ≥ 0.07 / 0.15; self_lock ≥ 0.2 / 0.5).
- Pooled PE-Spatial cosines are **compressed** (~0.95 floor — all anime images
  cluster), so absolute `s_self`/`s_other` look high; only the *contrast* is
  informative. For the same reason the paper's raw NN-distance-ratio
  (`mem_frac_paper`) rarely fires in feature space — keep it as a reference
  number, not the decision metric.
- `real_real_nn_mean` is the same-style floor: per-train nearest *other* real
  image. Tells you how close "not a copy" already is in this dataset.
- `memorized.png`: most-locked items, **gen | source | nearest-other** per row —
  the human "your LoRA copied this frame" check.

## Usage

```bash
python bench/memorization/probe.py \
    --method lora --preset default \
    --adapter output/ckpt/<your_lora>.safetensors \
    --num_items 64 --label <name>
```

Point `--adapter` at a checkpoint trained on the **current blueprint** — the
probe builds the reference pool from whatever `configs/base.toml` yields, so a
single-artist LoRA scored against the full multi-artist pool dilutes the signal
(its captions are mostly images it never trained on). NN search is always over
the full training set; `--num_items` only caps how many captions are generated.

Run it across several step-count checkpoints of one recipe to watch `contrast`
climb as the run crosses `tau_mem` — that curve is the early-stopping signal the
paper argues for, and the practical takeaway for users: **stop before it rises.**

Caches required: `{stem}_anima_te.safetensors` (conditioning) +
`{stem}_anima_pe_spatial.safetensors` (reference features). Both are written by a
normal `make preprocess`.
