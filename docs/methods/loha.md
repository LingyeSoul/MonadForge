# LoHa

LoHa (Low-rank Hadamard product adaptation, FedPara / arXiv:2108.06098)
parameterizes an adapter delta as the elementwise product of two low-rank
reconstructions:

```text
delta_W = (hada_w1_a @ hada_w1_b) ⊙ (hada_w2_a @ hada_w2_b) * (alpha / rank)
```

Two rank-`r` factor pairs give an effective rank up to `r²` at 2× the
parameter count of a rank-`r` LoRA. MonadForge uses `lycoris-lora==4.0.0` as
the backend (same pin as LoKr). The Anima integration only adds
module targeting, fused-attention splitting, lifecycle, and checkpoint
metadata around the official implementation — factorization, initialization
(`hada_w2_a` zero-init ⇒ ΔW=0 at step 0), dropout semantics, and the
canonical `hada_*` state keys are official LyCORIS behavior.

## Selection & config

```toml
# configs/methods/loha.toml — or set in any lora-method config / network_args
network_dim = 32
network_alpha = 32
use_loha = true
```

`make lora ARGS="--method loha"` / `python train.py --method loha --preset
<p>`, or the WebUI `LoHa` variant (`configs/gui-methods/loha.toml`,
`python tasks.py lora-gui loha`). Non-MoE only — `use_loha` composes with
none of the three-axis routing knobs (resolver precedence routes MoE first).

## LyCORIS quirks the wrapper routes around (states at the 4.0.0 pin)

The wrapper (`networks/lora_modules/loha.py`) subclasses the official
`LohaModule` and pins `bypass_mode=True`. Three version-specific behaviors are
handled — they differ from the LoKr wrapper, so don't copy fixes between the
two blindly:

1. **No missing-scale bug in LoHa's bypass** (unlike LoKr 3.4.0). LoHa's
   bypass applies `alpha/rank` — via `get_weight` in 3.4.0, folded into the
   dispatched gamma in 4.0 — so the forward passes `scale=multiplier` only.
   Re-adding `self.scale` (the call form the LoKr wrapper needed before 4.0)
   would double-scale.
2. **`get_diff_weight`/`get_merged_weight` double-apply `self.scale`** —
   same defect as LoKr, **still present in 4.0.0** — both are overridden to
   scale once.
3. **`lycoris.functional.loha.bypass_forward_diff` was broken as shipped in
   3.4.0** (passed `gamma` positionally into a 6-target unpack); 4.0 fixed it
   and it now dispatches to the fused kernels. The fp32 path
   (`lora_fp32_compute`, V100/fp16 only) still uses
   `functional.loha.diff_weight` + `F.linear`, pinned to `backend="torch"` so
   the experimental kernel tiers stay out of that lane. Note the gamma
   convention also differs from LoKr's functional: LoHa's `gamma` is the
   **full** `alpha/rank` scale (no internal rank division).

4.0 also fixes the tucker-core LoHa backward (the w1u/w2u gradients previously
contracted the wrong side's chain) — irrelevant here because Tucker cores are
conv-only and MonadForge's LoHa is Linear-only, but don't re-port the old
backward math from 3.4.0 sources.

The Hadamard product cannot factor through the input, so LoHa's eager
fallback tiers still materialize the full ΔW every forward; the 4.0 fused
LoHa bypass kernel chains the factor products tile-wise instead and never
builds it. The 4.0 fused-kernel story and `LYCORIS_KERNEL_BACKEND` selection
are described in `docs/methods/lokr.md`; LoHa's bypass and `diff_weight`
dispatch ride the same selector.

## Save format & inference

Checkpoints carry canonical LyCORIS keys (`hada_w1_a` / `hada_w1_b` /
`hada_w2_a` / `hada_w2_b` + `alpha` per module) with `ss_network_spec =
"loha"` stamped. Fused `qkv_proj`/`kv_proj` are split into per-component
`q_proj`/`k_proj`/`v_proj` at train time (`split_fused_projections`, shared
with LoKr), so saved keys are ComfyUI-compatible without any defuse surgery
and ComfyUI core's loha weight adapter loads the file natively.

The in-tree inference story matches LoKr: `make merge
ADAPTER_DIR=output/ckpt` bakes the delta with official Hadamard math
(`create_network_from_weights` key-sniffs `hada_*` and selects the loha spec
before the `for_inference` plain-LoRA fallback; `LoHaModule.merge_to` writes
through the split-projection views), then `make test-merge`. Passing a
native LoHa file directly as `--lora_weight` does **not** work — the static
merge hook (`networks/lora_utils.py`) only folds `lora_down`/`lora_up`, so
the adapter would be skipped; `load_dit_model` logs an explicit warning when
it sees native `lokr_*`/`hada_*` keys on that path. Tucker-core checkpoints
(`hada_t1`/`hada_t2`, conv-only upstream) and DoRA/`weight_decompose`
checkpoints (`dora_scale`) are rejected at load — the wrappers cannot
reproduce their semantics, and dropping the keys silently would be worse.

## Interactions

- **Channel scaling** (`channel_scaling_alpha`): auto-disabled — the native
  `hada_*` format has no `inv_scale` slot. The module constructor rejects
  `channel_scale`, `create_modules` skips the injection, and the factory
  warns (same triple defense as LoKr).
- **T-LoRA** (`use_timestep_mask`): silently inert on LoHa modules (the mask
  buffer lives on `BaseLoRAModule`, which LyCORIS wrappers don't inherit) —
  same behavior as LoKr.
- **LoRA+** (`loraplus_lr_ratio`): does not match `hada_*` parameter names;
  all LoHa params train at the base LR.
