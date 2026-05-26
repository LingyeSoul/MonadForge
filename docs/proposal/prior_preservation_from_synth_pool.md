# Prior Preservation from the distill-mod Synth Pool

**Status:** proposal
**Author:** (drafted with Claude Code)
**Date:** 2026-05-24

## Motivation

`make distill-prep` already spent ~8 GPU-hours producing
`post_image_dataset/distill_mod_synth/` — **1823 teacher-generated clean `x0`
latents** (CFG=2.5, 24 steps, frozen base Anima with `pooled_text_proj`
disabled), each paired *by stem* with a real caption's TE sidecar
(`crossattn_emb` + `pooled`). Today this pool has exactly one consumer:
`make distill-mod` via `--synth_data_dir` (mod-guidance distillation).

This pool is, definitionally, a **prior-preservation set** in the DreamBooth
sense: on-manifold outputs of the *frozen base model*, each with the caption
that produced it. We can reuse it to regularize identity/style LoRA training
against language drift / prior collapse, at zero additional generation cost.

## Background: what prior preservation buys us

Identity/style LoRA on a small dataset overfits — the adapter narrows the
model's output distribution toward the training set, and generic prompts
degrade ("language drift"). DreamBooth's remedy is a prior-preservation loss:
sample the *class* from the frozen base model and train the adapter to keep
reproducing those base outputs alongside the subject images. The base-generated
samples pin the adapter to the model's prior on everything outside the subject.

Mechanically, applying the standard flow-matching loss on a base-generated
latent (with the adapter *active*) pulls the adapter toward "reproduce base
behavior on this prompt" — i.e. toward zero drift on generic content — while
the real subject images carry the identity signal. The mixing ratio trades off
identity strength vs. prior retention.

## What we have vs. what prior preservation needs

| Need | Have? |
|---|---|
| Base-model outputs for generic prompts | ✅ 1823 `x0` latents, CFG=2.5 |
| Caption / conditioning for each | ✅ paired TE sidecar by stem |
| Aspect/resolution coverage | ✅ native buckets (filename-encoded) |
| Clean `x0` target for FM loss | ✅ endpoint saved |
| Source noise / trajectory | ❌ not saved — **but FM loss only needs `x0`** |

The missing trajectory is irrelevant here: flow-matching loss samples its own
noise + timestep per step and regresses velocity toward the clean `x0`. The
endpoint is all the prior-preservation loss consumes. This is the same reason
the pool works for `distill-mod`.

## Design

The blocker: the main `train.py` pipeline reads data through
`BlueprintGenerator` + subsets (`library/datasets/`), whereas the existing
`synth_data_dir` remap lives in `CachedDataset` (`library/datasets/cache.py`),
which is the *distill* reader. So the synth pool can't be dropped in as a
normal subset — it has no source images, only pre-cached latents.

Three integration options, cheapest-correct first:

### Option A — prior-preservation subset reader (recommended)
Add a latent-only subset type that reads `distill_mod_synth/*.npz` directly
(reusing the stem→TE-sidecar pairing logic already in `cache.py:103-128`) and
concatenate it with the main blueprint dataset. The subset carries a
`prior_preservation_ratio` (fraction of each batch, or sampling weight) and a
`prior_loss_weight` scalar applied to its loss contribution.

- **Pro:** no VAE round-trip, no extra disk, reuses existing pairing code.
- **Con:** needs a new subset path in the blueprint/dataset assembly; must
  respect native-shape bucketing so synth and real samples never share a batch
  across token-count families (4032 vs 4200).

### Option B — decode to an image subset
Decode the 1823 latents → pixels, write to an `image_dataset/` subdir with the
captions as `.txt` sidecars, run normal `make preprocess`. Becomes an ordinary
weighted subset with zero pipeline changes.

- **Pro:** trivial integration; uses the existing subset weighting.
- **Con:** VAE encode→decode→re-encode round-trip (lossy), ~extra disk, and a
  preprocess pass. Defeats some of the "free" appeal.

### Option C — config-only, reuse distill reader
Expose the `CachedDataset` synth path to `train.py` for LoRA runs and run a
two-loader training loop (subject loader + prior loader).

- **Con:** `CachedDataset` lacks the blueprint's masking/aug/subset machinery;
  duplicates the training loop. Not recommended.

**Recommendation: Option A.** It's the only one that keeps the pool's
zero-cost, no-round-trip advantage while plugging into the real training loop.

## Open questions

1. **Caption alignment.** The synth captions are the *real dataset's* captions.
   For prior preservation against generic drift that's fine (they span the
   model's content distribution), but if a synth caption overlaps the identity
   trigger, it could fight the subject signal. May want to filter synth stems
   whose captions contain the trigger token.
2. **Mixing ratio + loss weight.** DreamBooth defaults to 1:1 prior:subject
   with `prior_loss_weight=1.0`. With 1823 prior vs. (typically) far fewer
   subject images, we'd cap the prior subset's sampling weight rather than
   exhaust it. Sweep needed.
3. **CFG=2.5 mode bias.** These are CFG=2.5 outputs, so they encode the base
   model's *guided* mode, not the unconditional prior. Mild concern — they're
   still base outputs, but slightly sharpened. Probably fine; flag for the
   ablation.

## Validation plan

- Train one identity LoRA **with** and **without** the prior-preservation
  subset, fixed seed/steps otherwise.
- Primary signal: **CMMD** (`validation_split_num`, paired PE-Core MMD² vs real
  val images) — expect the prior-preserved run to hold CMMD better on a held-out
  *generic* prompt set as training progresses.
- Secondary: qualitative generic-prompt grid (drift check) + subject-fidelity
  grid (confirm identity isn't washed out by the prior weight).
- Guard against the known trap: FM-MSE val loss does **not** track quality on
  Anima (see memory `project_fm_val_loss_uninformative`) — judge on CMMD +
  perceptual, not val MSE.

## Cost

Generation: **already paid** (the 8h is sunk). Integration is Option A's subset
reader + a loss-weight knob. The win is per-LoRA-run drift protection with no
new sampling.
