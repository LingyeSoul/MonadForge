# ConvRot int8 base training

ConvRot is an experimental frozen-base compute path for Anima adapter training,
ported from
[`scvxzf1/anima_lora_webui`](https://github.com/scvxzf1/anima_lora_webui).
It rotates selected DiT `Linear` inputs and weights with a grouped Hadamard
transform, stores the frozen base weights as int8, and leaves the adapter
parameters trainable at their configured precision.

The default remains `base_compute = "bf16"`. Enabling ConvRot changes the base
forward used during training; it does not create an int8 adapter checkpoint.

## Modes

| `base_compute` | Frozen base weights | Activations | Intended use |
|---|---:|---:|---|
| `bf16` | bf16/fp16 training dtype | training dtype | Default and quality baseline |
| `w8a16_convrot` | int8 + per-output scale | training dtype | First mode to evaluate for VRAM savings |
| `w8a8_convrot` | int8 + per-output scale | int8 forward with STE backward | More aggressive experiment; validate quality carefully |

Only the captured base `Linear` forward is replaced. LoRA down/up parameters,
optimizer state, gradients, and saved adapter tensors keep their normal path.
After quantization, successfully patched dense base weights are replaced by
`meta` tensors so bf16 and int8 copies are not resident together.

## Enabling

The WebUI exposes the fields in the Performance section. Selecting `bf16`
hides the inactive `convrot_*` controls. A ready-made experimental method is
available as `LoRA + ConvRot W8A16 (VRAM)` from
`configs/gui-methods/lora-convrot-vram.toml`.

Minimal TOML override:

```toml
base_compute = "w8a16_convrot"
convrot_scope = "mlp"
convrot_group_size = 256
convrot_hadamard = "sylvester"
convrot_weight_source = "online_from_bf16"
```

Quality-oriented first comparison:

```toml
base_compute = "w8a16_convrot"
convrot_scope = "mlp"
convrot_group_size = 64
convrot_hadamard = "regular"
```

`convrot_scope` accepts `mlp`, `all`, `attention`, `self_attn_qkv`,
`attention_out`, and comma-separated combinations of the named scopes. The
group size must divide every selected layer's `in_features`. Supported CLI and
WebUI group sizes are 64, 256, and 1024.

The optional size filters are:

```toml
convrot_min_in_features = 4096
convrot_largest_in_features_only = false

# Optional hybrid mode for layers at or above the threshold.
convrot_large_layer_mode = "w8a8"
convrot_large_min_in_features = 8192
```

Use `convrot_large_layer_mode = "none"` and
`convrot_large_min_in_features = 0` to disable the hybrid override.

## Prequantized weights

Online conversion is the default and quantizes the loaded bf16 base once at
startup. To reuse a prequantized payload:

```toml
convrot_weight_source = "prequant_checkpoint"
convrot_prequant_path = "path/to/anima-convrot-prequant.safetensors"
```

The native format is `anima_lora_convrot_prequant_v1`. Each selected layer has
`{layer}.weight` int8 data and a `{layer}.scale` vector. Checkpoint group-size
metadata is checked against `convrot_group_size`. A prequant path is rejected
when the source remains `online_from_bf16`.

## Compatibility

| Feature | Status | Notes |
|---|---|---|
| `torch_compile` | Supported | ConvRot is applied after adapter loading and before block compilation |
| Gradient checkpointing | Supported | Adapter gradients remain in the original activation space |
| `blocks_to_swap` | Supported | Int8 weights and scales follow their owning DiT block; heavier swapping reduces residency but adds transfer latency |
| DoRA | Rejected | DoRA assumes a high-precision writable base-weight path |
| Plain bake/merge | Rejected by default | ConvRot runtime behavior cannot be reproduced by folding only the adapter delta into the bf16 base |
| Distributed training | Experimental | The payload is non-persistent module state; validate on the target topology before a long run |

ConvRot checkpoint metadata includes `ss_base_compute` and `ss_convrot_*` keys.
The standalone DiT merge path reads those keys and refuses a normal bake.
`allow_partial` can deliberately fold only the adapter delta into the bf16
base, but the result does not reproduce the ConvRot training-time base forward.

## Verification

Run the CPU-friendly output/loss/adapter-gradient probe before a real run:

```powershell
.venv\Scripts\python.exe scripts\experiments\convrot_equivalence_probe.py --mode w8a16
.venv\Scripts\python.exe scripts\experiments\convrot_equivalence_probe.py --mode w8a8
```

```bash
.venv/bin/python scripts/experiments/convrot_equivalence_probe.py --mode w8a16
.venv/bin/python scripts/experiments/convrot_equivalence_probe.py --mode w8a8
```

The W8A16 probe hard-fails when output relative L2 exceeds 3% or adapter
gradient relative error exceeds 5%. W8A8 prints the same gates but is not a
hard failure because its quality tradeoff is intentionally more aggressive.
These toy gates do not replace a short training A/B on the target dataset.

For a production decision, compare against the same-seed bf16 run and inspect:

- peak allocated/reserved VRAM after warmup;
- steady-state step time, excluding compile warmup;
- validation loss and fixed-seed samples;
- adapter gradient finiteness and any skipped ConvRot layers in startup logs.

The imported upstream RTX 3080 profile reports a VRAM reduction but no speed
win for W8A16. Treat ConvRot as a capacity option for a larger rank, batch, or
resolution, not as a throughput optimization.

## Runtime tuning

Defaults are selected for correctness. Change these only while measuring:

| Environment variable | Default | Effect |
|---|---|---|
| `ANIMA_CONVROT_RHT` | `dense` | `fwht` is available only with `sylvester` |
| `ANIMA_CONVROT_FUSED` | `1` | Set `0` to use the unfused reference path |
| `ANIMA_CONVROT_INT8_GEMM` | `auto` | W8A8 backend: `auto`, `int_mm`, or `float` |
| `ANIMA_CONVROT_STE_TF32` | `0` | Faster W8A8 backward experiment; may violate the gradient gate |

Keep `ANIMA_CONVROT_STE_TF32=0` for quality comparisons. The selected
`convrot_hadamard` value is written to `ANIMA_CONVROT_HADAMARD` by the training
bootstrap so the CLI/TOML setting wins over an inherited environment value.

## Implementation map

- Training hook: `library/training/convrot.py`
- Runtime patch and free-base handling: `library/runtime/convrot/`
- Block-swap payload carrier: `library/runtime/block_swap_payload.py`
- CLI fields: `library/config/cli_args.py`
- Metadata and merge policy: `library/training/metadata.py`,
  `library/anima/merge.py`
- Numerical probe: `scripts/experiments/convrot_equivalence_probe.py`

