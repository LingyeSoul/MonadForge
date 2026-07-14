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

## Legacy states

Historical snapshots containing `lokr_allow_legacy_dim=true` still parse. The
backend no longer requires it; the WebUI retains it only as an escape hatch for
its explicit-configuration migration prompt. Canonical decomposed keys are
`lokr_w1_a` / `lokr_w1_b` and `lokr_w2_a` / `lokr_w2_b`; the loader accepts
older MonadForge `w1a` / `w1b` / `w2a` / `w2b` keys at the boundary.
