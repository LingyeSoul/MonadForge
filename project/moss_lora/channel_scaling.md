# Channel scaling on MOSS-Image

Port of Anima's channel scaling to the MOSS LoRA. Usage and mechanism:
`docs/moss/channel_scaling.md`. This file is the log and the open items.

Question that started it: the DAVE probe found massive channels holding ~99 % of the
residual's DC (`dave.md`). Is that the same thing Anima's channel scaling rebalances, and
does it reach the LoRA inputs?

## Log

**2026-10-18 — weights only (CPU).** `norm1.weight[1356]` ≈ 0.000–0.03 on blocks 5–20
(median 0.9): the attention inputs mostly drop it. `norm2.weight[1356]` ≈ 0.6. The shared
adaLN (`shared_block_modulation`, one for all 42 blocks) gives the FFN input `1+scale` 0.22
and shift −6.8 on 1356, `1+scale` 0.24 and shift −4.3 on 42 (median |shift| 0.03), flat
over σ 0.1–0.95. Whether the shift cancels the normalized offset or adds to it is not
readable from weights.

**2026-10-18 — LoRA-input probe** (`bench/moss_channel_stats/results/20261018-1120-base`,
Base, `_calib_mixed` 16 × σ {0.1 … 0.9}, 1.6 min). The shift does not cancel:

- `ffn_up_gate`: ch 42 is the top channel on blocks 11–41, 11→26× median; 1356 reaches
  9–13× on blocks 27–28 and 35–41. Top-channel peak-to-mean 2.7 (DC bias).
- `q`: 1356 sits below median on blocks 1–17, rises to 6.7× late; 42 is 4× on blocks 0–5.
- Leverage α=0 → 0.5: q 4.0–18.7 → 2.0–4.4, o 2.6–6.2 → 1.6–2.7, ffn_up_gate 9.3–22.8 →
  3.1–4.9, ffn_down 4.5–34.8 → 2.2–6.2 (block 41 worst, peak-to-mean 9.5). Anima's
  matching groups run about 2× higher (mlp.layer1 11–80, qkv 7.6–37.9).

Reading: the massive channels arrive ~1900× → ≤ 26× after the norm; the one that matters
for a LoRA is 42 into the FFN. The part DAVE leaves out is channel scaling's territory; the
prompt-specific DC DAVE uses is not.

**2026-10-18 — implementation.** `library/lines/adapters.py` (shared with qwen21; inert at
α = 0), scale on the down weight rather than the activation. Calibration shipped as
`library/moss/calibration/channel_stats.safetensors` from the probe above. Default α = 0.

**2026-10-18 — A/B at lr 2e-5.** `caststation_layout_at_r32a128_lr2e5_svd` vs `…_cs05`
(same cache `channel_(caststation)_layout`, 20 epochs, r32 / α128, `weight_svd`, seed 0,
α 0 vs 0.5). Grids `output/tests/ab_caststation_layout_at_r32a128_lr2e5_svd{,_cs05}_50st`
(Base + SFT × 3 prompts × seeds 1, 2, ER-SDE + Spectrum, layout_json).

- Training: loss per epoch equal to the third decimal through epoch 20; peak 12.04 GB,
  1.93 s/step both.
- Weights, `ffn_up_gate` blocks 15 / 25 / 35: `down` drift from the SVD init 6–9 % in both;
  ch 42's column drift / mean column 0.99, 0.61, 0.92 (α 0) vs 0.25, 0.15, 0.17 (α 0.5).
  `q` block 30, ch 922: 1.88 vs 0.58. ΔW norm ratio 0.98–1.01; output leverage on the loudest
  channel 45.8 vs 44.0 (ffn_up_gate median). Per-block ΔW cosine between the runs 0.07–0.92.
- Images: style strength the same by eye. Distance to the no-LoRA image (same prompt and
  seed): cs05 closer in 9 / 12 cells, mean pixel distance 49.4 → 43.7, 32-px 39.2 → 33.6.

Reading: since `up` carries most of the learning, ΔW's channel structure is set by
`up @ down_init` in both runs; what channel scaling removes is the global drift that an
update on the constant ch-42 input causes (a bias on the FFN for every prompt). The
user's read of the grids: it touches less of what the style does not need.

**2026-10-18 — A/B at lr 1e-4.** `caststation_layout_at_r32a128_lr1e4_svd` vs `…_cs05`,
same setup otherwise, on the current cache (the older `caststation_layout_r32a128_svd`
predates the `@artists` captions, so it is not a twin). Grids
`output/tests/ab_caststation_layout_at_r32a128_lr1e4_svd{,_cs05}_50st`, LoRA arm only; the
no-LoRA reference is the 2e-5 grid's ×0.0 (same prompts, seeds, sampler).

- Training: losses within ±0.005 per epoch; 1.87 vs 1.84 s/step.
- Weights: `down` drift 0.27 / 0.28 / 0.23 (α 0) vs 0.30 / 0.29 / 0.25 (α 0.5) on
  ffn_up_gate blocks 15 / 25 / 35; ch-42 column / mean 0.45, 0.71, 0.80 vs 0.19, 0.13,
  0.15. `q` block 30: 1.08 vs 0.42. ΔW norm ratio 0.98–1.03, ΔW cosine 0.02 / 0.17 / 0.48
  / 0.30.
- Distance to no-LoRA: cs05 closer in 5 / 12, mean 50.7 vs 51.2, 32-px 41.4 vs 42.3 —
  the same amount of change.
- Eyeball (user verdict: cs05 much better): both strongly stylised, but α 0 adds
  off-prompt content — glasses + garbled sign (Base p2 s1), garbled sign (SFT p2 s1),
  street and car instead of the park (SFT p2 s2), broken hand pose (Base p0 s1). cs05
  stays on-prompt at the same style strength.

Reading: the mechanism holds (column suppressed, `down` now moves 23–30 %), and at the
lr where plain LoRA loses prompt details, channel scaling keeps them. The distance metric
cannot see this: it measures how much changed, not whether it is right. A first draft of
this entry read the flat distance as "no benefit"; that was the metric, not the images.
The 2e-5 "touches less" and the 1e-4 "stays on-prompt" may be one effect.

**2026-10-18 — defaults flipped** on the lr 1e-4 eyeball: `channel_scaling_alpha` 0 → 0.5,
`lr` 2e-5 → 1e-4 (rank 32 / alpha 128 unchanged). The trainer passes α only for
`algo = lora`.

## Open

1. **Second seed** — train seed 1 for both arms at lr 1e-4 and check the eyeball verdict
   holds; a metric for it would need prompt adherence (tagger / VLM on the off-prompt
   items above), not distance to no-LoRA.
2. **lr 1e-4 + cs05 vs lr 2e-5 plain** on the two complex gallery prompts (ten-character
   ensemble, 4:1 banner) that set the earlier 2e-5 default — the new default has not been
   read on them.
3. **α sweep** — 0.25 / 0.5 / 1.0.
4. **Calibration** — sample saturation (re-probe on `sincos_layout`, compare `s`) and SFT
   (`--model sft`); both 2-min jobs, low priority.
5. If (1) or (2) goes against cs05, revert the defaults to lr 2e-5 + α 0.
