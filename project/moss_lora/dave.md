# DAVE on MOSS-Image

Port of Anima's DAVE (`docs/inference/dave.md`) to `library/moss/dave.py`. Diversity
tool only: same-prompt seed variety, at the cost of text / hand coherence when the
window is too wide.

## What changed from Anima

- **Image tokens only.** A MOSS block outputs `[image (H·W) | text]`; `μ` is the
  per-channel mean over the image tokens and only those are edited.
- **Seam = `MOSSImageTransformer.layer_hook`**, called after each block in `run_layers`
  on the rows that ran it (under skip-22 the uncond row skips block 22, so the hook at
  22 sees the cond row alone). `None` is the reference path.
- **τ is a step count**: steps `0 .. round(τ·steps)−1`, at least one.
- **Spectrum**: forecast steps run no blocks, so DAVE only lands on full steps. At 50
  steps the default τ 0.10 (5 steps) sits inside the 6-step warmup; `generate.py`
  prints a note when τ reaches past it.

- **Massive channels** are left out of the subtracted mean (see the log): one 0/1
  channel list, `library/moss/calibration/massive_channels.npz`, used with every pool spec.

Surface: `GenerateRequest.dave` (`'' | auto | lo-hi | npz`), `dave_strength` (0.3),
`dave_tau` (0.10, advanced); part of the latent fingerprint. `auto` reads
`dave_alpha.npz` in `library/moss/calibration/`, which does not exist until the probe
below has been run and a mask derived.

## Phases

1. **Premise probe** — `bench/moss_dave/probe_dc_convergence.py` (2 prompts × 8 seeds ×
   30 steps, cond forward). Gate: early-step DC sim ≫ AC sim on the DC-heavy blocks
   (`premise_holds`). If it fails, stop.
2. **Mask** — `bench/dave/derive_alpha_mask.py --probe <run>/per_block.npz --block_cap N
   --out …`, writing `dave_alpha.npz` into `library/moss/calibration/`. Open: the cap
   on 42 blocks (Anima's ≤ 18 was the patch-grid dot artifact; re-eyeball here), and
   whether block 22 belongs in the pool.
3. **Dose / window sweep** — τ {0.06, 0.10, 0.15} × strength {0.3, 0.5, 0.8}, Base and SFT,
   prompts with quoted CJK text and hands. Read per grid.
4. **Ship** — mask committed, `docs/experimental/moss_lora.md` section, bench
   `result.json` cited here.

## Log

**2026-10-18 — premise probe, Base, vanilla** (`bench/moss_dave/results/20261018-1048-vanilla`,
anime_sunset + ja_manga × 8 seeds × 30 steps, CFG 3, skip 22, 1024²). The gate passes
(`premise_holds`: all-block early DC 0.998, AC 0.474, gap +0.52) but the metric is
saturated, so it does not pick a pool:

- DC sim ≥ 0.99 on every block 0–40 in the early window, and ≈ 1.000 on blocks 0–20 at
  **every** step, step 0 included. On MOSS the DC does not converge early; it is the same
  across seeds from the first step.
- The DC holds 0.98–0.99 of the image-token energy on blocks 0–18 (0.6–0.9 on 31–41), at
  every step. Every block counts as DC-heavy. On Anima the power ratio was bimodal.
- The AC sim falls over the run (block 20: 0.96 at step 0 → 0.27 at step 29): the seeds
  separate over the run rather than settling into one layout early.
- Per prompt: anime_sunset AC 0.20–0.70, ja_manga 0.08–0.49 (early window).

Reading: a DC carrying 99 % of the energy and shared from step 0 looks like a large
per-channel offset (massive-activation channels), not the conditioning lock the paper
measures. `derive_alpha_mask.py` on this probe would select almost every block below
the cap, and strength 0.3 there would remove ~30 % of the block output's energy. Not yet
checked: how many channels hold the DC energy, and whether the DC is also shared across
*prompts*.

**2026-10-18 — channel probe** (`bench/moss_dave/results/20261018-1106-channels`, anime_sunset
+ ja_manga + danbooru_tags × 8 seeds × 30 steps). The 99 % DC is massive activations:

- Channel **1356** holds 0.87–0.93 of the early DC energy on blocks 0–21; its |μ| is
  ~1900× the median channel at block 0, ~1260× at 10, ~530× at 20, ~210× at 30. Channels
  1356 and **42** are in the top 4 of 41–42 of the 42 blocks; 2807 / 1833 / 426 follow.
- Cross-prompt cosine of the early DC is 1.00 on every block with all channels in: the
  massive channels are prompt-independent, so they carry no conditioning.
- Top 4 out: power ratio falls to 0.18 (block 0) → 0.29–0.37 (8–30) → 0.47–0.80 (36–41),
  the Anima-like regime. DC sim stays 0.978–0.994 and the DC−AC gap +0.33–0.81, so the
  premise holds on the remaining channels. k = 16 / 64 change almost nothing past k = 4.
- With the top 4 out, the paper's `DC_sim ≥ 0.99` cut no longer separates blocks
  (0.978–0.994 everywhere). Prompt specificity (within-prompt cross-seed DC sim minus
  cross-prompt DC sim) does: 0.01–0.03 on blocks 0–3 (the DC there is prompt-generic,
  like Anima's block 0), 0.08–0.09 on 8–12, 0.05–0.07 on 13–36, 0.09–0.11 on 38–40.

Consequence: `MossDAVE` as written subtracts the mean on every channel, so at strength
0.3 it would move channel 1356 by hundreds of times a normal channel's mean. It must
skip the massive channels (a 0/1 channel mask, not a per-channel scale) before any
eyeball arm.

Done the same day: `MossDAVE` now leaves a channel list out of the subtracted mean
(`library/moss/calibration/massive_channels.npz`, written by
`bench/moss_dave/derive_massive_channels.py --probe <run>/channels.npz --k 4`: the union of
each block's top 4 = 18 channels, 1356 and 42 among them). Derived on Base; whether SFT
has the same channels is unchecked.
