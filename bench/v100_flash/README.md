# bench/v100_flash — V100 FlashAttention stability probe

`flash-attention-v100` is a third-party/drop-in FA2-style backend for Volta
(`sm_70`). MonadForge can import and exercise it, but real V100 testing on Anima
DiT fp16 training showed that it is **not production-stable** for this model:

- `attn_mode="torch"` is stable on Tesla V100-SXM2-16GB.
- `attn_mode="flash"` + `v100_flash_stability="hybrid"` still produced a
  first-step NaN in **self-attention**:
  `self_attn.attention_output backend=flash dtype=torch.float16 shape=(1, 2925, 2048)`.
- `torch_compile=true` is fine on V100 with the stable `attn_mode="torch"` path.
  The Dynamo shape-guard error was specific to the V100 flash kernel path; treat
  `flash-attention-v100` as non-traceable here.

**Recommendation:** use `attn_mode="torch"` for V100 production training. Keep this
probe and the stability modes for diagnostics only.

## Production V100 recipe

Use a V100-specific preset/variant like:

```toml
mixed_precision = "fp16"
attn_mode = "torch"
torch_compile = true
gradient_checkpointing = true
```

Operational notes from the verified V100 environment:

- Use a V100-compatible PyTorch build/venv (for example `torch==2.10.0+cu129` in
  `.venv-v100`). Newer CUDA/PyTorch wheels may omit SM 7.0 kernels and fail with
  `no kernel image is available for execution on the device`.
- `torch_compile=true` can remain enabled with `attn_mode="torch"`; disable it
  only when debugging the unsupported `flash-attention-v100` path.
- Keep the V100 preset hardware-focused. Avoid method-level conflicts such as an
  empty `network_weights = ""` path or init-strategy overrides that fight the
  selected method.
- If a method TOML fully replaces `[[datasets]]`, include `image_dir` and
  `cache_dir`; otherwise the base dataset blueprint paths are dropped.

## Diagnostic quick checks on a V100

These commands are for investigation, not production recommendation:

```bash
# Baseline: should be finite.
python -m bench.v100_flash.run_probe --attn_mode torch --device cuda

# Diagnostic: self-attn flash, cross-attn torch SDPA.
python -m bench.v100_flash.run_probe --attn_mode flash --stability hybrid --debug_finite --device cuda

# Diagnostic: full flash with finite checks around q/k/v and attention outputs.
python -m bench.v100_flash.run_probe --attn_mode flash --stability safe --debug_finite --device cuda
```

Results are written under `bench/v100_flash/results/<timestamp>/result.json`.

If the V100 venv is minimal, install full project deps first (the tiny Anima
fixture still needs normal runtime deps such as `einops`):

```bash
# Example only — use the same venv that contains the V100-compatible torch build.
pip install -r requirements.txt
# or install the project with its normal dependency workflow, without replacing torch.
```

## Stability modes

| Mode | Meaning | V100 Anima fp16 status |
|---|---|---|
| `off` | Normal `attn_mode=flash` behavior. | Not recommended; full flash was not pursued after hybrid failed in self-attn. |
| `hybrid` | Keep self-attention on FlashAttention, route cross-attention through torch SDPA. | Still failed: first non-finite tensor came from self-attn flash output. |
| `safe` | Keep flash but enable finite checks around q/k/v, attention output, projection output, block residuals, loss, and gradients. | Diagnostic only; it cannot make an unstable kernel numerically safe. |

Training accepts the same mode through config/CLI:

```toml
attn_mode = "flash"
v100_flash_stability = "hybrid"  # off | hybrid | safe, diagnostics only on V100
```

or temporarily via environment variable:

```bash
ANIMA_V100_FLASH_STABILITY=hybrid ANIMA_DEBUG_FINITE=1 python tasks.py lora-gui tlora
```

Do not use `nan_to_num` to hide NaNs in training. If the probe or training fails,
keep the first `FloatingPointError` location; it tells whether the first non-finite
value appeared before attention, after attention, after a residual add, in the
loss, or in gradients.

## Import compatibility note

`flash_attn_v100` exposes the public functions used by the main backend:

- `flash_attn_func`
- `flash_attn_varlen_func`

Some releases do **not** expose official FlashAttention internal helpers such as
`_flash_attn_forward`, `_wrapped_flash_attn_forward`, or
`_wrapped_flash_attn_backward`. MonadForge should treat those internal wrappers as
optional: their absence should not block public `flash_attn_func` import, but
features that need the wrapped internals must fall back or report that the V100
fork lacks them.
