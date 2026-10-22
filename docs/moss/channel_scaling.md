# Channel scaling on MOSS-Image

A training-time fix for one quirk of the MOSS LoRA: a single input channel is 10–25× louder
than every other channel at the FFN input, and AdamW turns it into a bias knob the adapter
reaches for first. Channel scaling rebalances the optimizer's step per input channel. The
forward output is unchanged and the saved file is a plain LoRA.

**On by default** (`channel_scaling_alpha = 0.5`, together with `dc_isolate` and lr 5e-5 at
rank 32 / alpha 128). `--channel_scaling_alpha 0` turns it off. GUI: Train step →
`channel_scaling_alpha`. `lora` only; the trainer skips it for `lokr` / `loha`.

| α | effect on the loudest channel's step |
|---|---|
| `0` | off; a plain LoRA |
| `0.5` | shrunk by √(its loudness): 25× → 5× (the default) |
| `1.0` | fully flattened: every channel moves the output the same amount |

## The problem: one channel, 25× louder, on every token

MOSS's residual stream carries a handful of massive-activation channels (1356, 42, …) that
hold almost all of the image tokens' DC offset, the same on every prompt and seed
(`docs/moss/dave.md`). RMSNorm and the adaLN modulation damp them but do not remove them: at
the FFN input, channel 42 is the loudest of 3072 channels on blocks 11–41, 11 to 26× the
median, with a peak-to-mean of only 2.7, so it is loud on every token. It behaves like a
constant input: a bias line.

![ch 42 and ch 1356 at the FFN input per block; block 35's loudest channels](assets/channel_scaling_loud_channel.png)

Adam moves every weight by roughly the same amount per step, whatever its gradient's size.
Moving `down[:, c]` by one step changes the adapter output by about `|x[c]|` × step, so
through channel 42 one step moves the output 25× further than through a typical channel, on
every token at once. The cheapest way to lower the loss is a global shift of the FFN input
on every prompt, before the adapter has learned anything specific to the style.

![per-channel output leverage of one Adam step, α 0 vs α 0.5](assets/channel_scaling_adam_step.png)

## What channel scaling does

Each input channel gets a scale `s[c] = mean|x[c]|^α` (normalised to mean 1, from a
calibration file), folded into the down projection:

- stored: `down · s`, so the trainable weight on channel `c` is `s[c]` times larger;
- forward: `down · s⁻¹`, so the product is exactly `down · x` as before.

The output and the gradient on `up` are unchanged. A uniform Adam step on the stored weight
is a step of `1 / s[c]` on the effective one, so one step moves the output by
`|x[c]|^(1−α)` instead of `|x[c]|`. At α 0.5 the loudest channel's lead over the average
channel drops from 9–23× to 3–5× on every block. Peak memory, s/step and the loss curve (to
three decimals) are unchanged. The saved file holds `down · s⁻¹`, a plain LoRA; the metadata
stamps `channel_scaling_alpha`.

The scale sits on the `(rank, in)` weight, not the `(tokens, in)` activation, so there is
no activation-sized op (`library/lines/adapters.py`: `absorb_channel_scale`,
`down_weight`, `channel_scale`).

## Measured effect

Twin runs on caststation (r32 / α128, 20 epochs, `weight_svd`, seed 0), identical except
`channel_scaling_alpha`. The `down` update on channel 42 falls from 0.5–1.0× an average
column to 0.13–0.25× (≈ 1/s, s ≈ 3.7–4.8), and the loss curves match.

- **lr 2e-5**: same style strength by eye; the channel-scaled LoRA stays closer to the
  no-LoRA image in 9 / 12 cells (mean pixel distance 49.4 → 43.7).
- **lr 1e-4**: clearly better by eye. Plain LoRA adds things the prompt never asked for
  (glasses and a garbled-text sign, a street with a car in place of the park, a broken hand
  pose); the channel-scaled twin stays on-prompt at the same style strength. Pixel distance
  to the no-LoRA image does not see this difference.

![lr 1e-4 twins on the four cells named above: no LoRA | α 0 | α 0.5](assets/channel_scaling_lr1e4_ab.webp)

One seed per cell. Log and open items: `project/moss_lora/done/channel_scaling.md`.

## The `up` gradient and `--dc_isolate`

Channel scaling only reaches the `down` side. The `up` gradient is `Σ_t g_t · h_tᵀ`, built
from `h = down · x`, which the reparameterisation leaves as it was. Because every token
carries the same DC offset, `h` has a constant component on every token, so much of each
step's `up` gradient is "shift every token's output the same way", spread over all 32 rank
directions. Measured with the trainer's loss (`bench/moss_channel_stats/probe_up_gradient.py`):
that bias share is 80–90 % of the `up` gradient at the FFN and attention inputs, against
40–50 % in the full-fine-tune gradient. Per step the `up` gradient is ~0.8 rank-1, and
channel 42's share of it is 3–6× its share of the full gradient (the entanglement GraLoRA,
arXiv:2505.20355, describes). α does nothing to it.

![bias share of the up gradient: plain LoRA, full fine-tune, dc_isolate](assets/channel_scaling_up_gradient.png)

`--dc_isolate` (**on by default**, `--no-dc_isolate` to turn off) makes the calibrated DC
direction of the input one rank row of `down` and keeps the other 31 rows orthogonal to it
(re-projected after every optimizer step), so one `up` column learns the bias and the rest
get a centered gradient. The bias share on the other 31 columns drops to 5–12 %. It wants a
lower lr: at lr 1e-4 it over-fit (the artist tag rendered as text in 3 / 12 cells); at lr
5e-5 all 12 cells are clean and on-prompt with style strength close to lr 1e-4. That pair is
the default. The α 0.5-only control at lr 5e-5 has not been run, so the lr and the flag are
not separated.

## Default history

All on caststation (30 images, Base, `weight_svd`, 20 epochs), compared by eye. Under AdamW,
raising the scale (alpha / rank) raises the effective lr of `up` by the same factor.

1. **Rank 16, alpha 16, lr 1e-4.** Alpha 90 at lr 1e-4 matched its style but started
   copying the set's signature (watermark and handle); alpha 90 at lr 5e-5 transferred
   partially.
2. **Rank 32, alpha 128, lr 2e-5, no channel scaling.** lr 1e-4 pushed the style hard but
   lost prompt details on the complex gallery prompts; lr 2e-5 kept them.
3. **Channel scaling α 0.5, lr 1e-4**: strong style, on-prompt.
4. **Plus `dc_isolate`, lr 5e-5**: the current default.

## Calibration

Two files under `library/moss/calibration/`, both from
`bench/moss_channel_stats/probe_lora_inputs.py` on Base (`_calib_mixed`, 16 samples ×
σ {0.1, 0.3, 0.5, 0.7, 0.9}, the training forward), keyed
`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}` (k / v share q's input):

- `channel_stats.safetensors` (3.9 MB): `mean|x|` per channel, for `channel_scaling_alpha`.
- `channel_mean.safetensors`: the signed token mean `x̄`, for `dc_isolate`.

```bash
make daemon-run ARGS="bench/moss_channel_stats/probe_lora_inputs.py --label <name>"
cp bench/moss_channel_stats/results/<run>/channel_{stats,mean}.safetensors library/moss/calibration/
```

The probe also writes `stats.npz` (per-block dominance, leverage, peak-to-mean),
`dominance.png` and the massive channels' rank at each input; `--model sft` profiles SFT,
and `--channel_stats` / `--channel_mean` on the trainer point at other files. A 10 % error
in `mean|x|` is 5 % in `s`, and the adaLN modulation is near-constant over σ 0.1–0.95, so
the σ grid need not match training. Regenerate only if the base DiT changes.

## Code

- `library/lines/adapters.py`: `absorb_channel_scale`, `down_weight`, `channel_scale`,
  `isolate_dc`, `reproject_dc`, `load_channel_stats`;
  `LoRANetwork(channel_stats=, channel_scaling_alpha=, channel_mean=, dc_isolate=)`.
- `library/moss/requests.py`: `TrainRequest.channel_scaling_alpha` / `channel_stats` /
  `dc_isolate` / `channel_mean`.
- `bench/moss_channel_stats/probe_lora_inputs.py`: the calibration probe (both files).
- `bench/moss_channel_stats/probe_up_gradient.py`: the `up`-gradient decomposition
  (`--dc_isolate` to measure the isolated init).
- `tests/test_moss_lora.py`: output equality, gradient `= g · s⁻¹`, plain-LoRA round
  trip; the `dc_isolate` orthogonality / re-projection / round-trip test.
