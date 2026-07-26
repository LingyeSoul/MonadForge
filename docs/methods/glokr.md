# GLoKr: Kronecker delta + BoRA bi-dimensional weight decomposition

Parameter-efficient adapter that combines the LoKr Kronecker factorization
(compact ΔW = kron(w1, w2)·scale) with **BoRA** — bi-dimensional
weight-decomposed adaptation: the merged weight is re-normalized row-wise and
column-wise with two independent trainable magnitude vectors, so magnitude
training is symmetric across both weight-matrix dimensions (the asymmetry
BoRA identifies in DoRA, which trains column magnitudes only).

> Papers: [BoRA: Bi-dimensional Weight-Decomposed Low-Rank Adaptation
> (arXiv 2412.06441)](https://arxiv.org/abs/2412.06441);
> Kronecker adapter lineage: [KronA (arXiv 2212.10650)](https://arxiv.org/abs/2212.10650),
> LyCORIS LoKr. The Kronecker split (factorization, init, alpha/scale
> conventions) is LyCORIS-exact; the module is native (no LyCORIS wrapper)
> because BoRA turns the forward into a weight *replacement* that the additive
> bypass path cannot express.

## How it works

Per adapted Linear (W0 ∈ R^{out×in}, frozen):

```
ΔW  = kron(w1, w2) · scale                 # LoKr delta (zero at init)
V^r = (W0 + ΔW) / ‖W0 + ΔW‖_row            # row-normalize
H   = m_row ⊙ V^r                          # trainable row magnitudes (out, 1)
W′  = m_col ⊙ H / ‖H‖_col                  # col-normalize + col magnitudes (1, in)
y   = x @ (W0 + multiplier·(W′ − W0))^T + b
```

* `m_row` / `m_col` initialize to W0's row/col norms and `w2`'s zero-init leg
  makes ΔW = 0, so **W′ == W0 exactly at step 0**.
* Norms are **detached** from the autograd graph (official DoRA memory
  convention); magnitudes and Kronecker factors still receive gradients
  through the numerators.
* The multiplier **lerps toward W0** (LyCORIS weight-decompose convention) —
  BoRA is non-linear in ΔW, so scaling the raw delta would be wrong.
* Attention projections: fused `qkv_proj`/`kv_proj` are pre-split into
  per-component Linears at network build (`split_attn.py`, the LoKR path) —
  a Kronecker product cannot be sliced at q/k/v output boundaries.

## Quick start

```bash
python train.py --method glokr --preset default
```

`make print-config METHOD=glokr PRESET=default` dumps the merged chain. The
WebUI exposes it as the `GLoKr (BoRA)` variant (`configs/gui-methods/glokr.toml`).

## Config surface (`configs/methods/glokr.toml`)

| Knob | Default | Meaning |
|---|---|---|
| `use_glokr` | `true` | selects the variant (`resolve_network_spec` → `glokr`) |
| `glokr_factor` | `8` | Kronecker split factor (`-1` = LyCORIS auto, closest-to-square) |
| `decompose_both` | `false` | low-rank-factor the small w1 leg too (shared knob with LoKR) |
| `glokr_full_factor` | `false` | full-matrix both legs (scale forced to 1, alpha := dim) |
| `glokr_rs_lora` | `false` | rank-stabilized scale `alpha/sqrt(r)` instead of `alpha/r` |
| `glokr_bora` | `true` | the BoRA decomposition; `false` degrades to a plain additive Kronecker delta (prefer `use_lokr` for that — it has the memory-cheap bypass) |

Mutually exclusive with `use_lokr` (validated). `network_dim`/`network_alpha`
set the Kronecker rank/alpha as usual.

## Save format

Native keys, one module per (split) Linear — no distill-to-LoRA is possible
(`W′ − W0` is full-rank in general):

```
lora_unet_blocks_0_self_attn_q_proj.glokr_w1        # or glokr_w1_a/_b
lora_unet_blocks_0_self_attn_q_proj.glokr_w2_a      # or glokr_w2
lora_unet_blocks_0_self_attn_q_proj.glokr_w2_b
lora_unet_blocks_0_self_attn_q_proj.bora_m_row
lora_unet_blocks_0_self_attn_q_proj.bora_m_col
lora_unet_blocks_0_self_attn_q_proj.alpha
```

Metadata stamps: `ss_network_spec="glokr"` plus `ss_glokr_factor` /
`ss_glokr_rs_lora` / `ss_glokr_bora` / `ss_glokr_full_factor`. The factor is
not shape-recoverable (a network-wide inference fallback exists for stripped
files); `rs_lora` needs no recovery at all — the persisted `alpha` buffer
pre-folds `sqrt(r)` (LyCORIS rs convention), so `alpha/rank` consumers are
exact even without metadata and the stamp is provenance only. The
`ss_network_dim`/`ss_network_alpha` setdefaults mirror LoKR.

## Inference

`inference.py --lora_weight <glokr.safetensors>` detects the checkpoint
(metadata stamp, header key-sniff fallback) and loads it as **kept-live
weight-replacement hooks** — the static merge path only understands
`lora_down/up` keys and would silently drop every GLoKr key (the LoKr
inert-adapter failure mode). `--lora_multiplier` and `--lora_cutoff_step`
work (the network rides the P-GRAFT cutoff slot). Mixing a GLoKr file with
other LoRA files in one `--lora_weight` list is refused — bake first instead.

## Merge / bake

`make merge ADAPTER_DIR=...` works: `GLoKRModule.merge_to` rebuilds W′ from
the checkpoint slice and **replaces** the DiT weight (`W0 + m·(W′−W0)`), so
the baked DiT is exact. Fuse/unfuse is exact too (the delta is stashed —
the BoRA normalization is not invertible from the fused weight alone).

## Compatibility

* **T-LoRA** (`use_timestep_mask`): supported — the shared mask gates w2's
  rank axis in training (no rank axis exists on full factors; w1's rank is
  deliberately not double-gated). Kronecker rank has no importance ordering,
  so treat the combination as experimental.
* **Channel scaling**: structurally refused (Kronecker input axis is a
  *factor* of `in_features`) — module ctor raises, factory warns, same as LoKR.
* **Not supported**: `dropout` / `rank_dropout` (module ctor raises;
  `module_dropout` works), Conv2d, MoE/ortho stacking (resolver precedence:
  ortho/hydra selectors win over `use_glokr`).
* **ComfyUI**: no loader understands the BoRA keys — bake with `make merge`
  for ComfyUI use.

## Cost

Trainable params per Linear: kron factors + `out + in` magnitude scalars
(~2.5–8× fewer than plain LoRA at DiT shapes; see `bench/glokr`). Each
forward materializes W′ (one `out×in` fp32 temp + elementwise chain) — small
next to the token GEMM at DiT sequence lengths, but heavier than plain LoRA's
rank-GEMM bypass; expect a modest step-time increase with gradient
checkpointing (W′ rebuilt on recompute).

## Files

* `networks/lora_modules/glokr.py` — module (math, merge, fuse)
* `networks/__init__.py` — `glokr` spec + kwargs allowlist
* `networks/lora_anima/factory.py` — key-sniff + metadata recovery
* `networks/lora_save.py` — native passthrough save variant
* `library/inference/models.py` — `_is_glokr` peek + kept-live attach
* `bench/glokr/` — Tier-2 bench (invariants + BoRA symmetry falsifier)
* `tests/test_glokr_module.py` — regression pins
