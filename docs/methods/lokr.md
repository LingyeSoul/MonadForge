# LoKr

LoKr parameterizes an adapter delta as a Kronecker product:

```text
delta_W = kron(w1, w2) * scale
```

MonadForge uses `lycoris-lora==3.4.0` as the LoKr backend. Each factor may be
stored directly or decomposed again into a low-rank pair. `lokr_factor`
controls the Kronecker dimension split and `network_dim` controls the
secondary factor rank when full-matrix mode is off. The Anima integration only
adds module targeting, fused-attention splitting, lifecycle, and checkpoint
metadata around the official implementation.

The source baseline is LyCORIS commit
`5ec93d24fcb8f27d6b16d3d706e69c60404d4b39`. The wrapper uses official
factorization, initialization, canonical state keys, and bypass operations.
LyCORIS 3.4.0's module-level bypass omits `self.scale`; MonadForge supplies
that missing `alpha / rank` factor so the efficient path remains equivalent to
the official regular forward without materializing a full DiT-sized delta.

## AnimaLoraToolkit comparison

AnimaLoraToolkit commit `c6bd6b644e4cd31fe0f98ba563a5856c88413e4e`
does not use LyCORIS in its main `anima_train.py` path. It injects a local
`LoKrLayer` into q/k/v/output and MLP projections, evaluates
`F.linear(dropout(x), kron(w1, w2)) * (alpha / rank)`, and optimizes all adapter
parameters with AdamW through the normal flow-matching MSE loss. Its checkpoint
metadata names `lycoris.kohya`, but the layer itself is custom. The separate
`utils/model_utils.py` LyCORIS branch is an unused example/fallback and is not
called by the main trainer. MonadForge therefore follows the official library
rather than copying that simplified full-factor layer.

## Full-factor training

Use the dedicated switch when both Kronecker factors should remain complete:

```toml
network_dim = 32
network_alpha = 32
use_lokr = true
lokr_factor = 8
decompose_both = false
lokr_full_factor = true
```

`lokr_full_factor` is the compatibility name for LyCORIS `full_matrix`.
LyCORIS produces full `lokr_w1` and `lokr_w2` parameters and forces
`alpha=network_dim`, so the effective training scale is always 1. The
`decompose_both` flag is ignored while full-matrix mode is active.

`network_dim = 114514` is accepted for compatibility with LyCORIS-era configs.
Once both factors become full, official LyCORIS also forces unit scale; it does
not retain `network_alpha / 114514`. Prefer the explicit flag because it states
the intended layout without an opaque sentinel.

Full-factor LoKr still has Kronecker structure. It can reach the maximum rank
allowed by the two factors, but it is not equivalent to unrestricted full
fine-tuning of the base weight.

## V100 eager memory

On V100/fp16, MonadForge keeps the LoKr bypass arithmetic in FP32 for numerical
stability. Two eager-autograd effects otherwise exhaust a 16 GiB card:

1. Official LoKr bypass autograd retains the converted full FP32 input and a
   grouped projection that is close to the base layer's output width.
2. Anima's bounded eager GELU MLP originally recognized plain LoRA modules
   only. Switching the same configuration to LoKr silently restored the
   ordinary `layer1 -> GELU -> layer2` graph and retained full `d_ff`
   activations for every block.

LyCORIS does not materialize `kron(w1, w2)` in this path. The OOM is caused by
retained eager activations, not construction of a full Kronecker delta.

When `torch_compile=false`, the V100 compatibility resolver automatically
enables `use_custom_down_autograd` together with `lora_fp32_compute`. For LoKr,
that switch now uses a chunked bypass that saves the original FP16 input and
factor storage, then rematerializes each FP32 chunk in backward. Full-factor
`w1`/`w2`, decomposed `w2`, and `decompose_both` layouts are supported. The
LoKr-specific default is 1024 rows because its intermediate is much wider than
a normal LoRA rank projection. The same switch also enables a LoKr-aware,
two-linear GELU MLP Function that saves only the original `d_model` input and
rematerializes the base and LoKr work per row chunk.

On July 25, 2026, the unchanged 1024px V100 16 GiB configuration OOMed on its
first forward before the LoKr-aware MLP path was added: PyTorch had 15.30 GiB
allocated, 31.44 MiB free, and failed a 64 MiB allocation in the frozen MLP
`layer1` linear. With both rematerialization paths enabled, the same
configuration completed 6/6 steps twice, saved all 280 native LoKr modules,
and measured a PyTorch peak of 14.2324 GiB allocated and 14.3164 GiB reserved.
The instrumented run averaged 10.93 seconds per optimizer step. Its
27,543,320-parameter BF16 checkpoint is 55,186,240 bytes (52.63 MiB), so
checkpoint parameter volume was not the dominant source of the original OOM.

The corresponding wide-FFN microbenchmark (`4200 x 3072 -> 9216`, full-factor
LoKr, V100) reduced saved tensors from 198.6 MiB to 25.5 MiB. At the default
1024-row chunk, peak reserved workspace fell from 814 MiB to 382 MiB. These
numbers isolate one adapter projection; the full-training peak above includes
the model, optimizer, all activations, and CUDA allocator state.

For the reported 1024px eager configuration, use:

```toml
use_lokr = true
lokr_factor = 8
network_dim = 32
network_alpha = 32
lokr_full_factor = true

torch_compile = false
mixed_precision = "fp16"
use_custom_down_autograd = true
use_timestep_mask = false
```

`use_timestep_mask` must be false for every LoKr layout. T-LoRA masks a shared
`network_dim` bottleneck, while LoKr has separate Kronecker factors and
full-matrix mode has no rank axis at all. MonadForge rejects the combination
instead of silently training unmasked LoKr.

Full-factor parameters and optimizer state still consume more memory than some
decomposed LoKr layouts. If a particular dataset bucket remains too large after
the eager rematerialization path is active, the next memory controls are
`torch_compile=true` with its activation-memory budget, gradient checkpointing,
or a lower resolution.

## Legacy states

Historical snapshots containing `lokr_allow_legacy_dim=true` still parse. The
backend no longer requires it; the WebUI retains it only as an escape hatch for
its explicit-configuration migration prompt. Canonical decomposed keys are
`lokr_w1_a` / `lokr_w1_b` and `lokr_w2_a` / `lokr_w2_b`; the loader accepts
older MonadForge `w1a` / `w1b` / `w2a` / `w2b` keys at the boundary.
