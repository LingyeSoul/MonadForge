# Channel scaling on MOSS-Image

Anima's channel scaling (`docs/optimizations/channel_scaling.md`) ported to the MOSS LoRA:
each target's input channel `c` is rebalanced by `s[c] = mean|x[c]|^α` inside the down
projection, so AdamW's per-step change to `down[:, c]` is weighted by `1/s[c]`. The forward
output is unchanged and the saved file is a plain LoRA. **On by default** (`α = 0.5`,
since 2026-10-18, together with lr 1e-4 at rank 32 / alpha 128); `--channel_scaling_alpha 0`
turns it off.

GUI: Train step → `channel_scaling_alpha`. `lora` only; the trainer skips it for `lokr` /
`loha` (`LoRANetwork` itself refuses the combination).

## Why MOSS has something to rebalance

MOSS's residual stream carries massive-activation channels (1356, 42, …; ~99 % of the
image tokens' DC, identical across seeds and prompts — `project/moss_lora/dave.md`). The
LoRA targets read the residual through RMSNorm + adaLN, which damp them: `norm1.weight[1356]`
≈ 0 on blocks 5–20, and the shared FFN modulation has `1+scale` ≈ 0.22 with a −6.8 / −4.3
shift on 1356 / 42. Measured at the Linear inputs (Base, 16 samples × 5 σ):

| input | what arrives |
|---|---|
| `q/k/v` | 1356 below median on blocks 1–17 (0.3–0.9×), up to 6.7× from block 24 on |
| `ffn_up_gate` | **ch 42 is the top channel on blocks 11–41**, 11→26× median; top-channel peak-to-mean 2.7, so it is on every token (DC bias, not a sink) |
| `o` | mild (leverage 2.6–6.2×) |
| `ffn_down` | 4.5–35× leverage, worst at block 41; peak-to-mean 9.5 (partly sink-like) |

Leverage dominance (max/mean of `mean|x|^(1−α)`), MOSS against Anima's shipped calibration:

| | α = 0 | α = 0.5 |
|---|---|---|
| MOSS `ffn_up_gate` | 9.3–22.8 (med 14.6) | 3.1–4.9 |
| Anima `mlp.layer1` | 11–80 (med 30) | 3.5–9.3 |
| MOSS `q` | 4.0–18.7 (med 5.8) | 2.0–4.4 |
| Anima `self_attn.qkv` | 7.6–37.9 (med 13.6) | 2.8–7.5 |

About half of Anima's imbalance, concentrated in one place: ch 42 into the FFN.

## What it does to a LoRA (caststation, r32 / α128, 20 epochs, seed 0)

Twin runs, identical except `channel_scaling_alpha`. **lr 2e-5:**

- The mechanism lands as designed: the `down` update on ch 42 of `ffn_up_gate` falls from
  0.6–1.0× to 0.15–0.25× of an average column (≈ 1/s, s ≈ 3.7–4.8). Loss curves match to
  three decimals.
- `down` moves only 6–9 % from its `weight_svd` init and `up` (from zero, untouched by the
  rescale) carries most of the learning, so ΔW's size and channel structure barely change.
- Images: the same style strength by eye, but the channel-scaled LoRA stays closer to the
  no-LoRA image in **9 / 12** cells (mean pixel distance 49.4 → 43.7; 32-px composition
  39.2 → 33.6). Reading: an update on a constant input channel acts as a global FFN bias,
  shifting composition and palette on every prompt; rebalancing it leaves more of what the
  style does not need. One seed pair per cell, no repeat-run noise floor yet.

**lr 1e-4:**

- `down` drifts 23–30 % from init (vs 6–9 % at 2e-5) and the ch-42 column is still held to
  0.13–0.19× of an average column (vs 0.45–0.80× without), so the mechanism is intact.
- Distance to the no-LoRA image is even (cs05 closer in 5 / 12, mean 50.7 vs 51.2): both
  move the image away from the base model by the same amount.
- **By eye the channel-scaled LoRA is clearly better.** The α 0 run adds things the prompt
  does not ask for — glasses and a garbled-text sign (Base p2 s1), garbled signage (SFT p2
  s1), a street with a car in place of the park (SFT p2 s2), a broken hand pose (Base p0
  s1) — the detail loss that already ruled lr 1e-4 out (`docs/experimental/moss_lora.md`).
  The cs05 cells stay on-prompt at the same style strength.

![lr 1e-4 twins on the four cells named above: no LoRA | α 0 | α 0.5](assets/channel_scaling_lr1e4_ab.webp)

The cells cited above, ER-SDE + Spectrum 50 steps, CFG 3, layout_json. The no-LoRA column
is the lr 2e-5 grid's ×0.0 (same prompts, seeds and sampler); full grids in
`output/tests/ab_caststation_layout_at_r32a128_lr1e4_svd{,_cs05}_50st`.

Distance to the no-LoRA image measures how much a LoRA changes, not whether the change is
right, so it misses this. One seed pair per cell. On this result the defaults moved to
lr 1e-4 + α 0.5 (from lr 2e-5 + α 0). Open items: `project/moss_lora/channel_scaling.md`.

## Mechanism

`library/lines/adapters.py`: `LoRAAdapter.absorb_channel_scale(s)` stores `down · s` and the
forward uses `down_weight() = down · s⁻¹`. The scale sits on the `(rank, in)` weight, not
the `(T, in)` activation — the same product and the same gradient (`∂L/∂down[:, c] · s⁻¹[c]`),
with no activation-sized op (peak and s/step unchanged). `s = clamp(mean|x|, 1e-6)^α`,
normalized to mean 1 (`channel_scale`). `save` writes `down · s⁻¹` and stamps
`channel_scaling_alpha` in the metadata as provenance; loading never needs the calibration.
`inv_scale` is a non-persistent buffer.

## Calibration

`library/moss/calibration/channel_stats.safetensors` (3.9 MB): `mean|x|` per target over all
tokens (image + text, what the LoRA gradient sees), keyed `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`
(k / v share q's input). Derived on **Base** from `_calib_mixed`, 16 samples × σ
{0.1, 0.3, 0.5, 0.7, 0.9}, the training forward. Regenerate:

```bash
make daemon-run ARGS="bench/moss_channel_stats/probe_lora_inputs.py --label <name>"
cp bench/moss_channel_stats/results/<run>/channel_stats.safetensors library/moss/calibration/
```

The probe also writes `stats.npz` (per-block dominance, leverage, peak-to-mean), `dominance.png`
and the massive channels' rank at each input. `--model sft` profiles SFT; `--channel_stats`
on the trainer points at another file.

Precision matters little: `s` is a square root, so a 10 % error in `mean|x|` is 5 % in `s`,
and the adaLN modulation is near-constant over σ 0.1–0.95, so the calibration's σ grid
need not match the training σ distribution.

## Code

- `library/lines/adapters.py` — `absorb_channel_scale`, `down_weight`, `channel_scale`,
  `load_channel_stats`; `LoRANetwork(channel_stats=, channel_scaling_alpha=)`.
- `library/moss/requests.py` — `TrainRequest.channel_scaling_alpha` / `channel_stats`.
- `bench/moss_channel_stats/probe_lora_inputs.py` — the calibration probe.
- `tests/test_moss_lora.py::test_channel_scaling_is_output_neutral_and_saves_plain` — output
  equality, gradient `= g · s⁻¹`, plain-LoRA save round trip.
