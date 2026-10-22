# Seed soup on MOSS-Image

**Verdict (2026-10-20): a 3-seed soup at fixed lr keeps the style.** The ingredients' ΔW
cos is low (~0.17), but the soup does not wash out the way the LR-diverse sincos soup did.
The style lives in the shared component of ΔW; averaging cancels each seed's own part, so
a low cos does not mean dilution. The sincos soup more likely failed on unbalanced energy
(its 1e-4 run carried ~82 %), so the cos gate is replaced by an energy-balance check
(`scripts/soup/dw_cos.py`, before building).

## Setup

caststation (`channel_(caststation)_layout`, 30 images): Base, rank 32 / alpha 128,
`weight_svd`, lr 5e-5, channel scaling 0.5, `dc_isolate`, 20 epochs = 600 steps,
`--plain_caption_rate 0`, `--caption_dropout 0`. Ingredients (`output/ckpt_moss/`)
`caststation_soup_lr5e5_s{0,1001,1002}`; soup `caststation_soup_lr5e5_3way_r32`
(`scripts/soup/build.py --rank 32`, uniform). Renders: Space 004 cfg 5 and 017 cfg 3, 4
seeds each, ER-SDE + Spectrum, 50 steps, in `output/tests/ab_caststation_soup/` (with
`dw_cos.txt`, `soup_build.txt`).

## Reads

- **ΔW cos** (`python -m scripts.soup.dw_cos`): per pair, per-module mean 0.14–0.15,
  median ≈0.035, full-sum 0.17–0.18, per-module max ≈0.75. Energy even (32 / 34 / 34 %).
- **Rank-32 truncation** keeps 98.2 % of the averaged ΔW's energy (min 94.7 %, early
  blocks' `ffn_down`).
- **Images**: the soup keeps caststation's flat pale fills and line weight on both
  prompts, like each single seed, without caption dropout B's softer rendered drift; it
  sits between the seeds on costume and prop detail.
- **Loss**: loss over its σ-bin mean (8 quantile bins, pooled) falls 1–10 % over 600
  steps (s0 −0.10, s1001 −0.01, s1002 −0.07), still drifting down, no plateau.

## Open

- Whether the soup beats a single seed (8 renders by eye only), and style strength
  against a no-LoRA render (this set has none).
