# LoKr

LoKr parameterizes an adapter delta as a Kronecker product:

```text
delta_W = kron(w1, w2) * (network_alpha / network_dim)
```

Each factor may be stored directly or decomposed again into a low-rank pair.
`lokr_factor` controls the Kronecker dimension split. `network_dim` controls
the secondary factor rank when full-factor mode is off.

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

This produces full `lokr_w1` and `lokr_w2` parameters while keeping the
training scale at `32 / 32 = 1`.

Do not use `network_dim = 114514` as a full-factor sentinel. Although that
value also prevents factor decomposition, it changes the training scale to
`network_alpha / 114514`. With `network_alpha = 32`, output and initial
gradients are suppressed by approximately 3578.6x.

Full-factor LoKr still has Kronecker structure. It can reach the maximum rank
allowed by the two factors, but it is not equivalent to unrestricted full
fine-tuning of the base weight.

## Legacy states

Historical training states that used the sentinel must not be migrated while
resuming: changing the scale would create a discontinuous jump. Resume them
only with an explicit compatibility override:

```toml
lokr_allow_legacy_dim = true
```

The trainer then preserves the historical scale and emits a warning. New
training rejects the sentinel and directs the user to `lokr_full_factor`.
