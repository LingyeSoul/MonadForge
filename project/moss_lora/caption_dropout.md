# Caption dropout on MOSS-Image

**Verdict: keep `caption_dropout 0`.** Dropout shifts the LoRA's uncond prediction
and CFG cancels part of the style with it; the cond prediction barely moves.
(2026-10-19)

## Setup

caststation (`channel_(caststation)_layout`, 30 images), the 2026-10-18 recipe: Base,
rank 32 / alpha 128, `weight_svd`, lr 5e-5, channel scaling 0.5, `dc_isolate`, 20 epochs
= 600 steps, seed 0, `--plain_caption_rate 0`.

| Arm | `caption_dropout` | Checkpoint (`output/ckpt_moss/`) |
|---|---|---|
| A | 0 | `caststation_layout_at_r32a128_lr5e5_svd_cs05_dc` (2026-10-18) |
| B | 0.1 (58 empty steps) | `…_dc_cd01` |
| C | 0.05 (29 empty steps) | `…_dc_cd005` |

The empty prompt is `library/moss/asset/empty_prompt.safetensors` (10 chat-template
tokens). Dropout draws from its own RNG (seed + 2), so σ and the shuffle match A step
for step, and C's empty steps are a subset of B's.

A is not a perfectly clean control. The TE cache was re-encoded after A trained
(2026-10-18 15:45, when the plain captions were added), and epoch 1's length-sorted
order differs from B's and from the `_tm` run's (which trained after the re-encode),
so some caption token lengths changed. The CFG 1 read below suggests the gap is small.

Renders: Space prompts 004 (cfg 5, negative `""`) and 017 (cfg 3, negative
`worst quality, bad anatomy`), 4 seeds each, ER-SDE + Spectrum, 50 steps, uncond skip 22.
`output/tests/ab_caption_dropout/` (`pairs*/`: A on top, the other arm below).

## Reads

- **Loss**: the 517 steps that train on the caption in both arms differ by −0.0005 on
  average (epochs 2–20); the 53 empty-prompt steps in that range average +0.016. The
  last epoch's mean loss is 0.2207 (A) vs 0.2199 (B). Loss does not separate the arms.
- **CFG 3 / 5 (default)**: B is softer and more rendered (gradient shading, texture,
  muted colour). A keeps caststation's flat pale fills and line weight. Seed for seed the
  compositions diverge. C falls between the two.
- **CFG 1** (no uncond pass; MOSS breaks up at CFG 1): A and B are almost the same
  image, down to pose and how it breaks up. Dropout leaves the cond prediction alone.
- **Plain CFG** (no block skip, same cfg): the same A/B gap as with skip 22, so the
  skip-22 mismatch (dropout trains the full-depth `""` path, inference uncond skips 22)
  is not the cause.
- 017 is affected too, although its negative is not `""`: what the LoRA learned on the
  empty prompt carries over to a short negative.

## Reading

Training on `""` puts some of the style into the LoRA's uncond prediction. CFG scales
cond − uncond, so the style share inside that difference shrinks. The style is no
longer amplified, and B reads closer to the base model. The cond prediction is unchanged
(CFG 1). That leaves the uncond branch as the only place the arms differ.

Untested: why dropout 0.05–0.1 suited Anima. One guess is its habit of long negative
prompts, which keep the uncond away from `""`.
