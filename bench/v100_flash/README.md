# bench/v100_flash — V100 FlashAttention stability probe

`flash-attention-v100` is a third-party/drop-in FA2-style backend for Volta
(`sm_70`). It can be useful on V100 fp16 training, but it is experimental for
Anima's DiT training path. This probe checks whether a tiny Anima forward,
loss, backward, and gradients stay finite under the selected attention backend.

## Quick checks on a V100

```bash
# Baseline: should be finite.
python -m bench.v100_flash.run_probe --attn_mode torch --device cuda

# Recommended V100-FA2 experiment: self-attn flash, cross-attn torch SDPA.
python -m bench.v100_flash.run_probe --attn_mode flash --stability hybrid --device cuda

# Full flash with extra finite checks around q/k/v and attention outputs.
python -m bench.v100_flash.run_probe --attn_mode flash --stability safe --debug_finite --device cuda
```

Results are written under `bench/v100_flash/results/<timestamp>/result.json`.

## Stability modes

| Mode | Meaning |
|---|---|
| `off` | Normal `attn_mode=flash` behavior. |
| `hybrid` | Keep self-attention on FlashAttention, route cross-attention through torch SDPA. This keeps the largest attention speedup while avoiding a common numerically sensitive path. |
| `safe` | Keep flash but enable finite checks around q/k/v, attention output, projection output, block residuals, loss, and gradients when paired with `--debug_finite` / `ANIMA_DEBUG_FINITE=1`. |

Training accepts the same mode through config/CLI:

```toml
attn_mode = "flash"
v100_flash_stability = "hybrid"  # off | hybrid | safe
```

or temporarily via environment variable:

```bash
ANIMA_V100_FLASH_STABILITY=hybrid ANIMA_DEBUG_FINITE=1 python tasks.py lora-gui tlora
```

Do not use `nan_to_num` to hide NaNs in training. If the probe or training fails,
keep the first `FloatingPointError` location; it tells whether the first non-finite
value appeared before attention, after attention, after a residual add, in the
loss, or in gradients.
