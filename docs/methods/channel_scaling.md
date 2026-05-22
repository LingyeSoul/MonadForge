# Channel Scaling — SmoothQuant-style LoRA pre-scaling

Per-channel input magnitude rebalancing for LoRA-family adapters. Absorbs a calibrated per-channel scale `s[c] = (mean|x[c]|)^α` into `lora_down` columns and applies `x / s` at forward, so the adapter output is unchanged at init but the down-projection's per-column gradient no longer scales with `|x[c]|^2`.

> **For the motivation** (DC-bias outlier channels in the frozen Anima DiT, decomposition into "register-token sinks" vs "stable outlier features", and the GraLoRA alternative weighed against), see **`bench/channel_stats/channel_dominance_analysis.md`**. This doc is the usage reference.

## Quick start

```toml
# configs/methods/lora.toml
channel_scaling_alpha = 0.5
```

Sole user knob:

- `0.0` — disabled (default). No `inv_scale` buffer; identical to a stock LoRA module.
- `0.5` — sqrt balance (SmoothQuant default). Most often what you want.
- `1.0` — fully flatten per-channel input magnitude.

The vendored calibration ships in-tree at `networks/calibration/channel_stats.safetensors` (~3.5 MB), so deploys — including `custom_nodes/*/_vendor/` trees — work without a separate download. No regeneration is needed unless the base DiT changes.

## Mechanism

At network build (`networks/lora_anima/factory.py::_load_channel_scales`):

1. Load the per-module `mean|x|` vector for every adapted Linear that has a calibration entry.
2. Compute `s = (mean|x|.clamp_min(1e-6))^α`, mean-normalize so `s.mean() == 1`.
3. Pass `s` into each `LoRAModule` / `OrthoLoRAModule` / `HydraLoRAModule` / `ChimeraHydraModule` / `StackedExpertsLoRAModule` constructor.
4. `BaseLoRAModule._register_channel_scale` mutates `lora_down[:, c] *= s[c]` in place and registers `inv_scale = 1 / s` as a persistent fp32 buffer.

At forward (`networks/lora_modules/{lora,ortho,hydra,chimera,stacked_experts}.py`):

```python
lx = lora_down_project(x, self.lora_down.weight, self.inv_scale)
# equivalent to F.linear(x * inv_scale, weight), in fp32
```

Output is bit-equivalent at init (because `(x · s^-1) · (down · s) == x · down`), but the per-column gradient becomes uniform across channels:

```
∂L/∂down[:, c]  ∝  |x[c] · s[c]^-1|^2 · |∂L/∂lx|^2
             ≈  uniform across c when α=1, sqrt-balanced when α=0.5
```

`inv_scale` is calibration data — it carries no gradient and never updates during training. It rides through `state_dict` as a persistent buffer.

## Save / load

`inv_scale` is included in `state_dict` next to `lora_down.weight` / `lora_up.weight` / `alpha`. The fused-attn split/refuse round-trip (`networks/lora_anima/loading.py`) treats it as a shared tensor — q/k/v of the same Linear see the same input so they share `inv_scale`. Hydra / Chimera / StackedExperts handle it via their per-pool stacked layouts.

`merge_to` and `LoRAModule.get_weight` undo absorption before computing `up @ down`, so a baked LoRA checkpoint (`make merge`) is correct without needing the calibration file at inference.

Round-trip is covered by `tests/test_per_channel_scaling_roundtrip.py`.

## Regenerating the calibration

```bash
python bench/channel_stats/analyze_lora_input_channels.py --per_artist \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --dump_channel_stats networks/calibration/channel_stats.safetensors \
    --out_json bench/channel_stats/results/$(date -u +%Y%m%d-%H%M)-base.json
```

The script registers `forward_pre_hook` on every `nn.Linear` in the DiT, accumulates per-input-channel `sum|x|` and token count over a small batch of cached samples at 5 flow-matching sigmas, then writes one `mean|x|` vector per LoRA-target Linear. 16 samples × 5 sigmas saturates the calibration in practice; `--per_artist` (71 samples on the current dataset) broadens coverage without changing per-group dominance numbers meaningfully.

Only regenerate when:

- The base DiT weights change (different normalization → different DC-bias channels).
- The set of adapted Linears widens (new attention/MLP layers exposed by the trainer).

See `bench/channel_stats/README.md` for the script flags and the output JSON layout.

## When this helps more

The bench analysis confirms the precondition (20–100× per-channel dominance, DC-bias not attention sinks) holds on the current Anima DiT. The magnitude of the *quality* delta has not been measured; the regime that should see the largest payoff is:

- **Higher rank** (≥64, where GraLoRA's argument bites hardest).
- **Plain LoRA, not OrthoLoRA.** OrthoLoRA's Cayley-rotated SVD parameterization already diffuses dominant-channel direction lock-in; the per-channel rebalance is partly redundant.
- **Shared-A across experts** (HydraLoRA, ChimeraHydra content pool) — the shared `lora_down` amplifies the bias K-fold.
- **Long single-domain runs** — the column-imbalanced gradient compounds over more steps.

Anima's default 12-epoch OrthoLoRA + T-LoRA stack on diverse multi-artist data sees the smallest expected delta, which is why the feature ships opt-in.

## Compatibility

| Adapter | Status |
|---|---|
| LoRA | ✓ |
| OrthoLoRA | ✓ |
| HydraLoRA (shared-A) | ✓ |
| StackedExperts / FeRA (independent-A) | ✓ |
| ChimeraHydra (dual-pool) | ✓ |
| ReFT | n/a — block-level intervention, not a Linear wrap |
| T-LoRA mask | ✓ — orthogonal; mask applies after the down-projection |

## References

- Xiao et al., **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**, ICML 2023 — the original `s = mean|x|^α` absorption trick (this implementation borrows the parameterization, not the quantization goal).
- Dettmers et al., **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**, NeurIPS 2022 — the "outlier features" phenomenon the bench observes in the Anima DiT.
- Jung et al., **GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning**, NeurIPS 2025, arXiv:2505.20355 — the more invasive `k×k`-block alternative; weighed against and rejected for now (same mechanism targeted, far higher integration cost).

## Code

- `networks/lora_anima/factory.py::_load_channel_scales` — calibration load + α-exponentiation.
- `networks/lora_modules/base.py::_absorb_channel_scale` — in-place column scaling + `inv_scale` registration.
- `networks/lora_modules/custom_autograd.py::ScaledLoRADownProjectFn` — fp32 down-projection that folds `inv_scale` into the weight at the matmul (avoids a bf16 `x * inv_scale` activation).
- `networks/lora_anima/loading.py` — q/k/v split/refuse handling for `inv_scale`.
- `tests/test_per_channel_scaling_roundtrip.py` — save → load → rebuild forward-equality check.
- `bench/channel_stats/` — calibration script, README, dominance analysis, historical results.
