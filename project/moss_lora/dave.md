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
`library/moss/calibration/dave_alpha.npz`: blocks 19–30, picked from the DAVE-on pool
sweep in the log (`derive_alpha_mask.py`'s `DC_sim` cut does not separate MOSS blocks).
ComfyUI: `~/ComfyUI-Anima-DAVE` ≥ 0.2.0 detects the `moss-image` transformer and, with
`mask=auto`, bundles the same pool + the massive-channel list (`moss_dave_alpha.npz`).

## Phases

1. **Premise probe** — done (log): passes once the massive channels are left out.
2. **Mask** — done: blocks 19–30 from a DAVE-on pool sweep, not `derive_alpha_mask.py`
   (log). Block 22 is in the pool; under skip-22 only the cond pass is edited there.
3. **Dose / window sweep** — one arm read (19-30 · s0.3 · τ0.10, Base). Open: τ 0.06 for
   hands, higher dose, SFT (and whether SFT has the same massive channels).
4. **Ship** — done for Base at s0.3 · τ0.10: `dave=auto` + ComfyUI node 0.2.0.

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

**2026-10-18 — DAVE-on pool sweep, Base** (`bench/moss_dave/results/20261018-{1915-dave-8-18,
1958-dave-19-30,1928-dave-8-30,1942-dave-8-40}`, same 3 prompts × 8 seeds × 30 steps as the
channel probe, s 0.3, τ 0.10 = 3 steps, 18 channels left out; vanilla = the channel
probe's `per_block_k4.npz`). Cross-seed AC sim (k4, all-block mean; lower = more diverse):

| | vanilla | 8-18 | 19-30 | 8-30 | 8-40 |
|---|---|---|---|---|---|
| AC sim step 3 | 0.531 | 0.520 | 0.442 | 0.432 | 0.448 |
| AC sim step 29 | 0.406 | 0.379 | 0.209 | 0.205 | 0.192 |
| DC sim steps 20–29 | 0.980 | 0.981 | 0.965 | 0.966 | 0.953 |

- Anima's pool (8–18) barely moves MOSS: the early drop is pulled back as soon as DAVE
  stops. Anima's 8–18 halved its late AC sim (0.538 → 0.265).
- 19–30 alone does what 8–30 does, from step 3 on, and the unlock compounds after the
  hook goes quiet. 8–18 adds only a transient drop in steps 0–2.
- 31–40 add 0.013 of AC drop and 0.013 of DC-sim drop, most on blocks 31–41: seeds
  diverge more in global attributes without more layout unlock.
- The DC-sim drop (0.980 → 0.965) comes with 19–30 and is not avoidable by shrinking
  the pool.

**2026-10-18 — eyeball, 19-30 · s0.3 · τ0.10, Base** (`output/moss_dave/eyeball_1018/`,
ER-SDE 50 steps so τ 0.10 = 5 steps, CFG 3, skip 22; ja_manga / anime_sunset / a two-hand
mug close-up × 4 seeds, paired with vanilla). Text stays legible on all four; bubble
placement, panel framing and drawing style vary where vanilla pins the bubble to the right.
Silver hair / red eyes / black uniform hold; one seed of four turns the sunset sky blue
(the DC-sim drop above). Hand close-ups recompose (off-centre, side, wider) where vanilla
repeats one framing, at a visible cost: a hand half through the mug, a mug opening on the
wrong side, a softer render. Shipped as is (mask committed, ComfyUI node 0.2.0); τ 0.06
(3 steps at 50, the probe's window) is the open follow-up for hands.
