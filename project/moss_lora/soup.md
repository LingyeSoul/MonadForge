# Seed soup on MOSS-Image

**Verdict: a 3-seed soup at fixed lr keeps the style.** The ingredients' ΔW cos is
low (~0.17), but the soup does not wash out the way the LR-diverse sincos soup did.
(2026-10-20)

## Setup

caststation (`channel_(caststation)_layout`, 30 images), the caption-dropout A recipe
(`_archive/moss_lora/caption_dropout.md`): Base, rank 32 / alpha 128, `weight_svd`, lr 5e-5, channel
scaling 0.5, `dc_isolate`, 20 epochs = 600 steps, `--plain_caption_rate 0`,
`--caption_dropout 0`. Seeds 0 / 1001 / 1002. Seed 0 was retrained on the current TE
cache rather than reusing A.

| Ingredient | Checkpoint (`output/ckpt_moss/`) |
|---|---|
| s0 | `caststation_soup_lr5e5_s0` |
| s1001 | `caststation_soup_lr5e5_s1001` |
| s1002 | `caststation_soup_lr5e5_s1002` |
| soup | `caststation_soup_lr5e5_3way_r32` (`scripts/soup/build.py --rank 32`, uniform) |

Renders: the caption-dropout prompt set (Space 004 cfg 5 and 017 cfg 3, 4 seeds each),
ER-SDE + Spectrum, 50 steps. `output/tests/ab_caststation_soup/`: one dir per arm,
`sheet*.jpg` with one row per arm, plus `dw_cos.txt` and `soup_build.txt`.

## Reads

- **ΔW cos** (`python -m scripts.soup.dw_cos`): every pair has a per-module mean of 0.14–0.15, a
  median of ≈0.035 and a full-sum of 0.17–0.18. The per-module max is ≈0.75. Energy is
  even (32 / 34 / 34 %).
- **Rank-32 truncation** keeps 98.2 % of the averaged ΔW's energy on average (min 94.7 %,
  the early blocks' `ffn_down`).
- **Images**: the soup keeps caststation's flat pale fills and line weight on both
  prompts, the same as each single seed. It does not drift toward the softer, rendered
  look that caption dropout B produced. The single seeds differ in costume and prop
  detail, and the soup sits between them.
- **Loss**: per-epoch means jump with each seed's σ draw (0.22–0.34), so they do not
  compare across runs. Loss divided by its σ-bin mean (8 quantile bins, pooled over the
  three runs) falls 1–10 % over the 600 steps (s0 −0.10, s1001 −0.01, s1002 −0.07). It
  is still drifting down slowly, with no plateau and no sharp drop.

## Reading

The style lives in the shared component of ΔW. Averaging cancels each seed's own
component and leaves the shared part, so a low full-sum cos does not mean dilution. The
sincos soup more likely failed because its ingredients' energy was unbalanced (the 1e-4
run carried ~82 %), not because their cos was low. The earlier cos gate is replaced by
an energy-balance check (`scripts/soup/dw_cos.py`, before building).

Not shown: that the soup beats a single seed (8 renders by eye only), and how strong the
style is against a no-LoRA render (this set has none).
