# HydraLoRA: Multi-Style Routing via Layer-Local Experts

MoE-style multi-head LoRA with per-module routing. Targets multi-artist training in a single LoRA without style bleed — a standard LoRA trained on multiple artists blends all styles into one shared low-rank subspace, so distinct fingerprints are lost. HydraLoRA attaches several `lora_up` heads per adapted `Linear` and lets a learned router pick a per-sample mixture.

Paper: [arXiv:2605.03252](https://arxiv.org/abs/2605.03252).

> For the structural walkthrough (architecture, forward pass, why RMS-over-rank-R, load-balancing formula, orthogonalized experts and the cold-start deadlock, composition matrix), see `docs/structure/hydralora.md`. This doc is the usage / ops reference.

## File format

Training state dict (runtime form, used inside the trainer):

```
<prefix>.lora_down.weight        # (rank, in_dim)    shared
<prefix>.lora_up_weight          # (E, out, rank)    stacked per-expert
<prefix>.router.weight           # (E, rank)
<prefix>.router.bias             # (E,)
<prefix>.alpha                   # scalar
<prefix>.inv_scale               # (in_dim,)  optional; only when channel_scale is set
```

`save_weights` produces two files side by side:

1. `anima_hydra.safetensors` — standard LoRA (baked-down): expert ups are averaged to a single `lora_up.weight`, routers stripped. ComfyUI drop-in, but routing is lost so it's effectively a uniform-prior approximation.
2. `anima_hydra_moe.safetensors` — full multi-head format: per-expert `lora_ups.N.weight`, routers preserved, attention modules split into separate `q_proj`/`k_proj`/`v_proj` so the ComfyUI custom node can map them to the ComfyUI model's attention key names. Shared tensors (`lora_down`, `alpha`, `router.*`, `inv_scale`) are cloned into each split component.

## Inference

### CLI (router-live)

`inference.py` auto-detects moe files by safetensors-header sniff (`library/inference/models.py:_is_hydra_moe`). When detected, static merge is skipped and the network is attached as dynamic forward hooks — the training-time `HydraLoRAModule.forward` runs for every adapted layer on every denoising step, reproducing the trained router's per-sample, per-layer behavior.

Static merge and router-live are mutually exclusive: mixing hydra moe files with regular LoRA files in one `--lora_weight` list is refused. P-GRAFT composes cleanly — the cutoff step toggles `network.enabled` for both, and `HydraLoRAModule` honors the flag.

Use `make test-hydra` (or `python tasks.py test-hydra`) to run inference against the latest moe output.

### ComfyUI (live routing)

Use the Anima Adapter Loader node (`https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter`), which installs per-Linear forward hooks that reproduce `HydraLoRAModule.forward` exactly — including σ-conditional routing when the checkpoint's router input is wider than `rank`. See `https://github.com/sorryhyun/ComfyUI-Anima_lora-Adapter` for installation, hook mechanics, and changelog.

## Composition with other variants

- T-LoRA — timestep rank masking applies to `lora_down` (shared across experts), so it composes directly; `configs/gui-methods/hydralora.toml` enables both.
- Spectrum — composes cleanly. Cached steps skip all transformer blocks entirely (router included), so hydra just runs fewer times.
- Modulation guidance — orthogonal. Touches AdaLN only, outside the hydra-adapted Linears.

## Router gate

`_compute_gate` takes the rank-R `lx` (after `lora_down`), RMS-pools it over the
sequence, and feeds it to the router (`router.weight` is `(E, rank)`, plus any
appended σ features). The gate is computed before the T-LoRA mask and dropout, so
it is identical at train and inference time. RMS rather than mean pooling: the
layer input is near zero-mean, so a mean over the ~4096-token sequence cancels by
√N and leaves the router a constant input (uniform gates, no router gradient).
RMS is safe in rank-R space because `lora_down` strips the DiT's DC-bias outlier
channels that would dominate it in bf16.
The ComfyUI live-routing hook mirrors this forward.

Refused at load, with a retrain message:

- `_hydra_router.*` keys — the earlier single global router on max-pooled
  `crossattn_emb`.
- `router.weight` wider than `rank` + the σ-feature cap — a router trained on the
  raw `in_dim`-wide layer input.

## Configuration

`configs/gui-methods/hydralora.toml` is the shipped variant. HydraLoRA is one cell of the three-axis routing surface ([`../guidelines/training.md`](../guidelines/training.md#lora-family--the-three-axis-surface)):

- `use_moe_style = "shared_A"` + `route_per_layer = true` + `router_source` (`"sigma"` in the shipped variant) — selects `HydraLoRAModule`. The retired `use_hydra` key raises.
- `num_experts` — code default 4, shipped variant 6. More experts add `(out_dim * rank)` parameters per module.
- `balance_loss_weight` — Switch Transformer load-balancing coefficient (shipped variant 3e-7; the ceiling on Anima is ~5e-5). Raise it if experts collapse; lower it if specialization stays weak.

HydraLoRA requires `cache_llm_adapter_outputs = true` (same as standard LoRA in this repo).

### Hard σ-band partition

`specialize_experts_by_sigma_buckets = true` (with `num_sigma_buckets > 1` and `num_experts % num_sigma_buckets == 0`) partitions the E experts into B σ-bands. For a sample at σ in band b, only the in-band experts can win the gate (out-of-band logits masked to `-inf` before softmax). Soft routing still operates within a band.

- Layout: interleaved. Expert e belongs to band `e mod B` — see `networks/lora_modules/hydra.py::_register_sigma_band_partition`.
- Edges: optional. `sigma_bucket_boundaries = [0.0, 0.5, 0.8, 1.0]` (length B+1, strictly increasing, 0.0 → 1.0) overrides the default uniform `linspace(0, 1, B+1)`. Lets you concentrate capacity in a chosen σ regime — e.g. wide low-σ band, narrow high-σ band — while keeping equal experts per band. With variable bucket widths under uniform σ sampling, narrow buckets see fewer training samples per band; consider oversampling those σ ranges if you want their experts to converge as fast.

Both fields are stamped into safetensors metadata (`ss_specialize_experts_by_sigma_buckets`, `ss_num_sigma_buckets`, `ss_sigma_bucket_boundaries`) so inference (CLI + ComfyUI) reconstructs the partition exactly.
