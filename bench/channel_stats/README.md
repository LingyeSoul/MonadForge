# channel_stats

Channel-dominance bench + SmoothQuant-style calibration generator for the
`per_channel_scaling` LoRA path (`configs/methods/lora.toml`).

## What it does

`analyze_lora_input_channels.py` registers `forward_pre_hook` on every
`nn.Linear` in the DiT, collects per-input-channel `mean|x|` over a small
batch of cached samples at 5 flow-matching sigmas, and reports per-module
dominance (`max(mean_abs) / median(mean_abs)`). Optionally dumps the full
per-channel `mean_abs` vectors to a safetensors file that the LoRA factory
absorbs into `lora_down` columns at construction time.

See `channel_dominance_analysis.md` for the original investigation, the
DC-bias-vs-attention-sink decomposition, and the remediation tradeoffs
against GraLoRA (arXiv:2505.20355).

## Usage

```bash
# Regenerate the vendored calibration consumed by channel_scaling_alpha > 0
# (--per_artist: one sample per artist subdir under post_image_dataset/lora/):
python bench/channel_stats/analyze_lora_input_channels.py --per_artist \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --dump_channel_stats networks/calibration/channel_stats.safetensors \
    --out_json bench/channel_stats/results/$(date -u +%Y%m%d-%H%M)-base.json
```

The calibration ships in-tree at `networks/calibration/channel_stats.safetensors`
(~3.5 MB, negative-ignored past the global `*.safetensors` rule) so deploys —
including `custom_nodes/*/_vendor/` trees — work without a separate download.
16 samples × 5 sigmas saturates the calibration in practice; per-artist (71)
broadens coverage without changing per-group dominance numbers meaningfully.

## Then flip it on

```toml
# configs/methods/lora.toml
channel_scaling_alpha = 0.5
```

Sole knob. α=0.0 disables (default); α=0.5 = sqrt balance (SmoothQuant default);
α=1.0 fully flattens per-channel input magnitude. The vendored path is
hardcoded in `networks/lora_anima/factory.py::_CHANNEL_STATS_PATH`.

## When this helps more

See the `per_channel_scaling` audit (memory `project-per-channel-scaling-audit`)
for the regime analysis. Short version: higher rank, plain LoRA (not OrthoLoRA),
shared-A across experts (HydraLoRA/Chimera content pool), and long-to-convergence
single-domain runs see the largest delta. Anima's default 12-epoch OrthoLoRA stack
on diverse data sees the smallest.

## Results directory

`results/<YYYYMMDD-HHMM>[-<label>].json` — bench/_common envelope is not
used here (the bench predates `bench/_common.py`); the JSON layout is
defined inline in the script. Future re-runs may want to migrate.
