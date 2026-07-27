# LoHa

LoHa (Low-rank Hadamard product adaptation, FedPara / arXiv:2108.06098)
parameterizes an adapter delta as the elementwise product of two low-rank
reconstructions:

```text
delta_W = (hada_w1_a @ hada_w1_b) ⊙ (hada_w2_a @ hada_w2_b) * (alpha / rank)
```

Two rank-`r` factor pairs give an effective rank up to `r²` at 2× the
parameter count of a rank-`r` LoRA. MonadForge uses `lycoris-lora==3.4.0` as
the backend (same pin and source baseline as LoKr: LyCORIS commit
`5ec93d24fcb8f27d6b16d3d706e69c60404d4b39`). The Anima integration only adds
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

## LyCORIS 3.4.0 quirks the wrapper routes around

The wrapper (`networks/lora_modules/loha.py`) subclasses the official
`LohaModule` and pins `bypass_mode=True`. Three version-specific behaviors are
handled — they differ from the LoKr wrapper, so don't copy fixes between the
two blindly:

1. **No missing-scale bug in LoHa's bypass** (unlike LoKr). LoHa's
   `bypass_forward_diff` routes through `get_weight`, which already applies
   `alpha/rank` — the forward passes `scale=multiplier` only. Re-adding
   `self.scale` (the LoKr fix) would double-scale.
2. **`get_diff_weight`/`get_merged_weight` double-apply `self.scale`** in
   3.4.0 (same defect as LoKr) — both are overridden to scale once.
3. **`lycoris.functional.loha.bypass_forward_diff` is broken as shipped**
   (passes `gamma` positionally into a 6-target unpack) — never call it. The
   fp32 path (`lora_fp32_compute`, V100/fp16 only) uses
   `functional.loha.diff_weight` + `F.linear` instead. Note the gamma
   convention also differs from LoKr's functional: LoHa's `gamma` is the
   **full** `alpha/rank` scale (no internal rank division).

Unlike LoKr, LoHa's bypass still materializes the full ΔW every forward — a
Hadamard product cannot factor through the input — so bypass only saves the
base-weight read/subtract of the rebuild path, not the materialization.

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
