# FeRA — frequency-energy constrained routing

Author-faithful port of Yin et al., *FeRA: Frequency-Energy Constrained Routing for Effective Diffusion Adaptation Fine-Tuning* (arXiv:2511.17979). DiT runs frozen; every adapted Linear is replaced by a small MoE of **independent rank-r experts**, and a **single global router** consumes the spectral state of the current latent `z_t` and emits one `(B, num_experts)` gate that every adapted layer reuses for that step.

Reference: `FeRA/` in the repo root (paper authors' code: `FeRA/fera/{layer,model,utils,config}.py`).

This page is about the author-faithful path (`network_module = networks.methods.fera`). The earlier FEI-on-Hydra variant (the FEI router added to HydraLoRA's σ-router) still ships as a commented-out toggle block inside the same `fera.toml`; see "Two variants in one TOML" below.

## Quick start

```bash
make fera                         # author-faithful FeRA (default fera.toml)
python tasks.py fera              # cross-platform
make fera PRESET=low_vram         # hardware preset override as usual

make test-fera                    # inference smoke against latest output/anima_fera*.safetensors
```

(If `make fera` / `make test-fera` aren't wired yet in your tree, run `python tasks.py fera` — `tasks.py` is the source of truth and the Makefile is a dispatcher.)

## What it actually does

```
                                 z_t  (B, C, H_lat, W_lat)
                                  │
                  FrequencyEnergyIndicator (DoG pyramid → simplex)
                                  │
                          e_t (B, num_bands)
                                  │
                   SoftFrequencyRouter (Linear→ReLU→Linear→softmax/τ)
                                  │
                       w  (B, num_experts)            ◄── one global router
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       ▼                          ▼                          ▼
┌─ FeRALinear ─┐         ┌─ FeRALinear ─┐         ┌─ FeRALinear ─┐
│  attn.qkv    │   …     │  attn.out    │   …     │  mlp.layer1  │   …
│  base(x)+    │         │  base(x)+    │         │  base(x)+    │
│  Σ_k w_k·    │         │  Σ_k w_k·    │         │  Σ_k w_k·    │
│   E_k(x)     │         │   E_k(x)     │         │   E_k(x)     │
└──────────────┘         └──────────────┘         └──────────────┘
  ^ independent             ^ independent             ^ independent
   (down,up)k                (down,up)k                (down,up)k
```

Per adapted Linear:

```
FeRALinear(x) = base(x) + multiplier · Σ_{k=0..E-1} w_k · LoRAExpert_k(x)
LoRAExpert_k(x) = lora_up_k( lora_down_k(x) ) · (alpha / rank)
```

The base Linear is frozen. Each expert has its own independent `(lora_down, lora_up)` pair — **no shared-A pooling** like HydraLoRA. At init `lora_up_k = 0` so the residual contribution is exactly zero at step 0 regardless of routing weights.

## How this differs from the FEI-on-Hydra variant

The repo already shipped FEI as a routing key on top of HydraLoRA (plan.md Phase 1 — `use_fei_router=true` inside `lora.toml`). That variant is a *router-key swap* on the existing Hydra architecture; the author-faithful path is a different network family entirely.

|                        | author-faithful FeRA (this doc)            | FEI-router-on-Hydra                                  |
|------------------------|--------------------------------------------|------------------------------------------------------|
| Network module         | `networks.methods.fera`                    | `networks.lora_anima` (Hydra + FEI router)           |
| Experts                | independent `(down_k, up_k)`               | shared `down`, per-expert `up_k`                     |
| Router                 | one global router on `z_t`                 | per-Linear router on pooled rank-R + FEI features    |
| Default targets        | attn `qkv/q/kv/output_proj` + `mlp.layer{1,2}` | MLP-only (`mlp.layer[12]`)                       |
| Stacks with T-LoRA / OrthoLoRA / ReFT | no (clean port)                 | yes (LoRA-family stack)                              |
| Default `num_bands`    | 3 (paper)                                  | 2 (Anima-validated, see [[project_fera_probe_2band_decision]]) |
| FECL aux loss          | exposed but off by default                 | n/a (collapses to scalar at 2 bands)                 |

Both share the bench-validated bucket-adaptive σ_low rule (`min(H_lat, W_lat) / fei_sigma_low_div`).

## Two variants in one TOML

`configs/methods/fera.toml` ships the author-faithful path uncommented as the default. The FEI-on-Hydra variant lives as a commented toggle block at the bottom of the same file — comment the author block + uncomment the Hydra block to A/B. They use different `network_module` values, so the swap has to be at TOML level rather than at flag level.

The toggle exists so the same config name (`fera`) covers both points in the design space; nothing forces you to use either one.

## Implementation map

| File | Role |
|------|------|
| `networks/methods/fera.py` | `FrequencyEnergyIndicator`, `SoftFrequencyRouter`, `LoRAExpert`, `FeRALinear`, `FeRANetwork`. Standard `create_network` / `create_network_from_weights` surface; save/load with `ss_fera_*` metadata. |
| `library/runtime/fei.py` | `gaussian_blur_2d` + kernel cache reused by `FrequencyEnergyIndicator` so we don't reinstantiate the kernel per step. |
| `library/inference/adapters.py` | `iter_fera_networks` + `set_fera_zt` + `clear_fera_zt`. Parallel slot to `iter_hydra_networks` (different attachment so the two methods don't fight). |
| `library/inference/generation.py` | `set_fera_zt(anima, latents)` at the same hook site as `set_hydra_sigma` / `compute_and_set_hydra_fei`; both denoising loops + `finally:` clear paths. |
| `library/inference/models.py` | `_is_fera(path)` header check + load-time `model._fera_network = network` attach. |
| `train.py` | At the per-step σ/FEI hook block: `if hasattr(network, "prepare_forward") and hasattr(network, "fera_layers"): network.prepare_forward(noisy_model_input)`. |
| `configs/methods/fera.toml` | Default config (author-faithful) + commented FEI-on-Hydra toggle block. |
| `bench/fera/probe_fei.py` | Diagnostic FEI probe — predates the network module; used to settle 2-band vs 3-band on Anima. |

## Parameter count

Per adapted Linear: `E · r · (D_in + D_out)`. Default `E=3, r=4` on Anima's 28 blocks × 5 Linears (qkv_proj + output_proj + q_proj + kv_proj + 2 × MLP layers — varies slightly per block) gives roughly:

```
28 blocks × 5 Linears × 3 experts × 4 rank × (D_in + D_out) avg
≈  28 × 5 × 3 × 4 × (2048 + 2048)  ≈  6.9 M    (experts)
+  router: (3 → 64) + (64 → 3)     ≈  450     (negligible)
```

About 6–7 M trainable params at defaults. Roughly 1.5× a `rank=16` plain LoRA, because the per-expert `(down, up)` pair is independent (no shared-A reuse).

## Knobs (`configs/methods/fera.toml`)

| Param | Default | Notes |
|---|---|---|
| `network_module` | `networks.methods.fera` | The author-faithful path. Switch to unset (i.e. lora_anima) for the FEI-on-Hydra toggle block. |
| `network_dim` (rank) | 4 | Author paper rank. Each expert is `(in → r → out)`. |
| `network_alpha` | 4 | Same as rank ⇒ scale `α/r = 1`. |
| `fera_num_experts` | 3 | Paper default. Independent per-expert `(down, up)`. |
| `fera_num_bands` | 3 | Paper default. Drop to 2 for Anima's bench-validated bimodal split (see [[project_fera_probe_2band_decision]]). |
| `fera_router_tau` | 0.7 | Softmax temperature. Lower → sharper expert specialization. |
| `fera_router_hidden` | 64 | Router MLP hidden width. Author uses 64. |
| `fei_sigma_low_div` | 4.0 | `σ_low = min(H_lat, W_lat) / fei_sigma_low_div`. Default picked from the 2026-05-13 dataset sweep — `div=4` yields the highest router std(e_low) at low/mid t on real training latents. NOT the paper's pixel-domain `min(H, W) / 128` (that's SD2-512-specific). Previous default `8.0` remains a Pareto-defensible alternative; see [[project_fera_probe_2band_decision]]. |
| `fera_fecl_weight` | 0.1 | FECL aux loss weight (paper used 0.1–0.2). Activates the base-pass forward + FECL term inside the loss composer — 2× per-step forward cost. Set to 0 to disable. |
| `fera_target_modules` | `.*\.(qkv_proj\|q_proj\|kv_proj\|output_proj\|layer[12])$` | Anima-naming regex covering self-attn fused QKV + cross-attn q/kv + attn output + MLP. Restrict to a subset to ablate. |
| `fera_ortho` | `false` | Swap free per-expert `(down_k, up_k)` for PSOFT-style weight ortho on each expert: shared frozen top-r SVD bases `Q_basis / P_basis` + per-expert Cayley-rotated `S_q, S_p (E, r, r)` + per-expert diagonal `λ (E, r)`. See [Weight-ortho variant (`fera_ortho`)](#weight-ortho-variant-fera_ortho) below. |
| `fera_ortho_init_std` | `0.02` | Stdev for random init on `S_p, S_q` when `fera_ortho=true`. Larger → experts start at larger rotations from identity; smaller → closer to vanilla deterministic SVD init (zero symmetry breaking). |

Training defaults: `learning_rate = 1e-4`, `max_train_epochs = 4`, `cache_llm_adapter_outputs = true`, `caption_dropout_rate = 0.1`, `compile_mode = "full"`.

## σ_low rule (why not the paper's κ)

The paper picks DoG kernel scale `κ = min(H, W) / 128` — a pixel-domain constant tuned for SD2 at 512×512. Anima trains under constant-token bucketing (`H_lat · W_lat ≈ 4096`) at varied aspect ratios, so a fixed pixel σ would land on different fractions of the latent grid per bucket. Bench probes (`bench/fera/results/20260512-1814-fera-pilot/`, `…20260512-1827-fera-midwide/`) validated the latent-domain rule on inference trajectories; the 2026-05-13 dataset sweep (`bench/fera/probe_fei_dataset.py`, results under `…20260513-1649-dataset-sweep/`) then picked `div=4` over `div=8` for real training-distribution router signal:

```
σ_low = min(H_lat, W_lat) / fei_sigma_low_div     (default fei_sigma_low_div = 4.0)
```

aspect invariance held across 1024², 832×1248, 1248×832 at the inference probe (mean `|Δ FEI|` < 0.02 between mirror buckets). The dataset sweep ranked divisors by population std(e_low) at flow-matching training-input t: **div=4 highest** (0.131 at t=0.05), div=8 second (0.112), paper-style div=128 worst (0.020). This is the same rule the FEI-on-Hydra variant uses.

## 3 bands vs 2 bands

Author paper picks `num_bands = 3` (low / mid / high). On Anima flow-matching latents the mid band is structurally near-empty (`e_mid ≤ 8%` at `σ_mid = 4`, `≤ 1.5%` at `σ_mid = 8`) — see [[project_fera_probe_2band_decision]]. Anima's velocity target `(image − noise)` is bimodal by construction (concentrated at very-low image structure + very-high noise), so the third band carries no routing-useful signal.

We default to 3 here for paper fidelity. If the held-out gate-entropy + per-expert utilization log shows expert utilization collapsing to two experts at convergence (one band always near-zero), drop `fera_num_bands` to 2. The FEI-on-Hydra toggle block ships 2 bands directly.

## FECL

The paper's frequency-energy consistency loss (eq. 10) is wired into the loss composer. When `fera_fecl_weight > 0` and the active network is a FeRANetwork, every training step does:

1. **Main forward** with FeRA routing active → `z_fera` (with grad).
2. **`network.clear_routing()`** to fall every `FeRALinear` through to its frozen base.
3. **Second no-grad forward** under the same autocast → `z_base`.
4. **`network.prepare_forward(z_t)`** to restore the gates — this is load-bearing under gradient checkpointing, which replays the main forward during backward and needs to see the same routing weights it saw originally.
5. **`network.compute_fecl_loss(z_base, z_fera, target)`** stashed in `loss_aux["fecl_loss"]`.
6. Loss composer's `fera_fecl` handler (`library/training/losses.py::_fera_fecl_loss`) multiplies by `network.fecl_weight` and adds it to the total — same `_STAGE_SCALAR_BROADCAST` stage as ortho / hydra-balance / soft-tokens-contrastive.

FECL bandwise distribution: pushes the adapter correction `δ = z_fera − z_base` to concentrate its energy in the same bands the residual `r = z_fera − target` has energy in. Encourages experts to spend their capacity where the FM loss is currently failing rather than perturbing bands that are already correct.

Two caveats:

- The 2× forward cost is real — at default 4 epochs that's an extra full training run worth of compute. The base pass is `no_grad` but still has to do the full DiT forward.
- At `num_bands = 2` the loss degenerates to a single scalar ratio (the two band shares sum to 1, so weighted (Δshare)² is content-free across bands). Setting `fera_fecl_weight > 0` while `fera_num_bands = 2` will train but the FECL contribution can't differentiate experts spectrally — the term becomes purely a magnitude regularizer on the correction. If you want 2-band FeRA, set `fera_fecl_weight = 0`.

Setting `fera_fecl_weight = 0` skips the base-pass entirely (the gate is checked before the second forward in `get_noise_pred_and_target`), so the inactive case has zero overhead.

## Weight-ortho variant (`fera_ortho`)

Replaces each expert's free `(down_k, up_k)` Parameters with a PSOFT-style ([[psoft-integrated-ortholora]]) parameterization:

```
Q_basis : (r, in)   frozen — top-r right singular vectors of base Linear W   (shared across experts)
P_basis : (out, r)  frozen — top-r left singular vectors of base Linear W    (shared across experts)
S_q     : (E, r, r) trainable, random skew seed init                          (per-expert)
S_p     : (E, r, r) trainable, random skew seed init                          (per-expert)
λ       : (E, r)    trainable, zero init                                      (per-expert)

Q_eff_k = cayley(S_q_k) @ Q_basis                # (r, in), orthonormal rows
P_eff_k = P_basis @ cayley(S_p_k)                # (out, r), orthonormal columns
ΔW_k    = P_eff_k @ diag(λ_k) @ Q_eff_k          # rank-r contribution from expert k
```

The Cayley transform `R = (I + A)^{-1}(I - A)` with `A = S - S^T` guarantees `R^T R = I` at every gradient step — no orthogonality regularizer. All `(E+E) × r × r` solves are batched into one `linalg.solve` (`_cayley_effective`), so the per-step overhead is one LU + TRSM launch per FeRALinear regardless of `E`.

**Why shared bases, not disjoint slices.** The OrthoHydra path slices the top-`E·r` singular vectors into `E` disjoint bundles (`docs/structure/ortholora.md`) so experts are structurally orthogonal in output space. On Anima's DiT the singular spectrum decays fast enough that expert `E−1` would inherit a low-σ bundle and have nothing useful to contribute. Shared bases avoid this trap — every expert lives in the same high-energy subspace and only differs by rotation + diagonal scale.

**Symmetry breaking via random skew init.** With identical SVD bases and zero-init `S, λ` every expert is bit-identical and the global router has no per-expert gradient signal — a Kaiming-on-A analog of FeRA's vanilla init. We instead seed `S_p, S_q ∼ N(0, fera_ortho_init_std²)` per expert. λ stays at zero so `ΔW = 0` at init, but each expert lives at a slightly different rotation within the shared basis and the router can pull them apart through λ gradients. Compared to vanilla FeRA's Kaiming on `down_k`: same role (stochastic differentiation), tighter guarantee (Cayley structural orthogonality vs approximate orthogonality from concentration of measure).

**Parameter count.** Per adapted Linear:

```
vanilla FeRA:  E · r · (in + out)                            ≈ E · r · 4096   (≈ 16k at E=4, r=8)
ortho:         E · (2r² + r) + (in + out) · r                ≈ E · 136 + 4k   (≈ 4.5k at E=4, r=8)
                ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
                trainable      frozen buffers (P_basis, Q_basis)
```

Trainable parameters drop ~3 orders of magnitude. Optimizer state shrinks proportionally. Frozen buffer size is similar to vanilla FeRA's lora_down + lora_up, so peak storage doesn't change much — but the trainable surface is tiny.

**What stacks with what.**

| Component | Compat under `fera_ortho=true` |
|---|---|
| Global FEI router | ✅ — the router consumes `z_t` spectral state and doesn't care how the experts are parameterized. Same gating, same shape. |
| FECL aux loss | ✅ — operates on adapter output deltas, independent of expert internals. |
| Spectrum inference | ✅ — saved file ships distilled `(lora_down, lora_up)` keys (see below) so the inference path is identical to vanilla FeRA. |
| ComfyUI custom node | ✅ — same distilled-keys story; the node loads `lora_down / lora_up` and never touches `S_p / S_q / λ`. |
| `torch.compile` | ⚠ — the `linalg.solve` inside `_cayley_effective` may force one graph break per FeRALinear under `compile_mode="full"`. Bench compile-on vs compile-off before relying on it; if it's a regression, the falsafe is to drop to `compile_mode = "default"` and let the rest of the DiT compile while the FeRALinear forward runs eagerly. |

**Save format.** Both layouts are written to the same `.safetensors`:

```
# Distilled — inference, ComfyUI, Spectrum
{lora_name}.lora_down                  (E, r, in)    bf16   ← Q_eff per expert
{lora_name}.lora_up                    (E, out, r)   bf16   ← P_eff_k · λ_k folded in

# Native — training resume only
{lora_name}.S_p                        (E, r, r)     fp32
{lora_name}.S_q                        (E, r, r)     fp32
{lora_name}.lambda_layer               (E, r)        fp32
{lora_name}.P_basis                    (out, r)      fp32   ← redundant with on-disk SVD but
{lora_name}.Q_basis                    (r, in)       fp32     stored to avoid svd_lowrank non-determinism
```

Metadata stamps `ss_fera_ortho` + `ss_fera_ortho_init_std`. `create_network_from_weights` defaults to **non-ortho** (loads the distilled keys, runs the cheap vanilla forward) — opt back into ortho-mode rehydration by passing `fera_ortho=true` in the call kwargs (the trainer's resume path does this automatically via `network_args`).

**What to bench against vanilla FeRA.**

1. **Gate entropy + per-expert utilization** — ortho mode's symmetry breaking comes from random `S_p, S_q` init, not Kaiming on a high-dim `down_k`. Verify experts actually differentiate during training; collapse to a single expert is the failure mode to watch for.
2. **Sample quality** at matched epochs — the question is whether constraining ΔW to live in W's top-r SVD subspace (rotated, scaled) costs expressiveness vs free A/B. Likely a wash on highly-redundant attention/MLP weights; could matter on early conv-like layers.
3. **Optimizer memory + step time** — trainable params drop 3 OOM, so optimizer state shrinks. Per-step Cayley solve adds a tiny FLOP cost. Net should be a small win for memory, a small regression for step time at `compile_mode = "default"` and possibly larger at `compile_mode = "full"` if `linalg.solve` breaks the graph.

## Global-router invariant

The single global router is the architectural commitment: **every adapted Linear sees the same `(B, num_experts)` gate this step**. `prepare_forward(z_t)` runs the router once and writes the same tensor reference into each `FeRALinear._routing_weights`; one Python-level write propagates to all sites.

That has two consequences:

- A single point of failure for routing — if the router collapses (e.g. one expert always wins, gate entropy → 0), every layer collapses together. Watch this in training logs.
- No per-Linear specialization signal beyond what the global gate provides. Compared to Hydra (per-Linear router from its own input), FeRA has less *layerwise* diversity but much more *latent-state* diversity.

The hook site is symmetric to `set_hydra_sigma` / `compute_and_set_hydra_fei` so cudagraph capture sees a stable per-step order.

## Save format

`output/ckpt/<output_name>.safetensors` keys:

```
router.net.0.weight                                        (router MLP, fp32)
router.net.0.bias
router.net.2.weight
router.net.2.bias
lora_unet_<dotted_path>.lora_down                          (stacked Parameter, bf16)
lora_unet_<dotted_path>.lora_up                            shape (E, r, in) / (E, out, r)
…
```

Each adapted Linear gets two flat stacked Parameters (`lora_down`, `lora_up`) — *not* an inner `experts.k.lora_{down,up}.weight` nesting like a ModuleList would produce. The training-side `FeRALinear` exposes them as `nn.Parameter` directly so a single `einsum` consumes all E experts at once (Hydra-style activation-memory layout).

**ComfyUI compatibility — split q/k/v on disk.** ComfyUI's cosmos backbone (`comfy/comfy/ldm/cosmos/predict2.py`) uses split `q_proj` / `k_proj` / `v_proj` Linears while Anima's training-side DiT uses fused `qkv_proj` (self-attn) and `kv_proj` (cross-attn). `save_weights` always writes the **split** layout — the fused `lora_up` is sliced `[Q | K | V]` along its output axis (matching `Attention.compute_qkv`'s `unflatten(..., (3, n_heads, head_dim)).unbind(dim=-3)` order) and the shared `lora_down` is duplicated to each split prefix. `load_state_dict` recognizes split files and re-fuses on the fly so the training-side DiT (which adapts the fused `qkv_proj` / `kv_proj` Linears) receives a single stacked Parameter.

Net effect: one checkpoint loads in both `python inference.py` (re-fused) and ComfyUI's `AnimaFeraLoader` (already split — zero conversion). Disk overhead vs the pre-split layout is ~10% on Anima at default rank (duplicated `lora_down` on the qkv/kv slots; `lora_up` is unchanged in aggregate).

Metadata stamps:

```
ss_network_module       = "networks.methods.fera"
ss_network_spec         = "fera"
ss_fera_rank            = "4"
ss_fera_alpha           = "4.0"
ss_fera_num_experts     = "3"
ss_fera_num_bands       = "3"
ss_fera_router_tau      = "0.7"
ss_fera_router_hidden   = "64"
ss_fei_sigma_low_div    = "4.0"
ss_fera_fecl_weight     = "0.1"
ss_fera_target_modules  = "..." (the regex used)
```

`create_network_from_weights` reads these stamps so the loader doesn't need the original TOML to rebuild the network. The frozen base Linear weights are *not* saved (they belong to the DiT, not the adapter — see "Frozen base ownership" below).

## Frozen base ownership

`FeRALinear` keeps a reference to the original `nn.Linear` (`self.base_layer`) but assigns it via `object.__setattr__` so it bypasses `nn.Module`'s child-tracking. The base layer's weights stay where they always were — owned by the DiT — and don't leak into `FeRANetwork.state_dict()`. This means:

- The trained FeRA file only carries router + expert deltas (~6–7 MB on Anima at default config).
- Loading FeRA against a DiT that doesn't match the architecture the FeRA was trained against will silently produce broken outputs — no shape check on the base, since FeRA has no record of what the base was.
- Merging FeRA into the DiT is not supported (`is_mergeable() == False`). A router-mixed contribution isn't a single ΔW unless you freeze the gate at inference time — TBD if a static-gate merge mode is ever needed.

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `train.py` checks `hasattr(network, "prepare_forward") and hasattr(network, "fera_layers")` and fires `prepare_forward(noisy_model_input)` at the σ/FEI hook site. |
| Standard inference | ✅ | `library/inference/models.py::_is_fera` detects FeRA checkpoints by header; `load_dit_model` rehydrates the network and attaches it as `model._fera_network`. `set_fera_zt` fires per step in `generation.py`. |
| Spectrum inference | ⚠ | Per-step `set_fera_zt` is wired, but cached-step skip semantics need a closer look — on a Spectrum cached step the FEI/gate is updated but the cached features may have been forecast from a different gate distribution. Bench against `--spectrum` before relying on it. |
| `torch.compile` | ✅ | `FeRALinear.forward` is a simple base + accumulator loop over experts; shape-static once bucket is fixed. Compile gets the same treatment as plain LoRA. |
| `blocks_to_swap` | ✅ | FeRALinear replaces the original Linear in-place, so block swap moves the FeRALinear and its experts together. |
| `gradient_checkpointing` | ✅ | The adapter is a thin Linear-replacement; checkpointing at block granularity wraps it correctly. |
| Modulation guidance | ✅ orthogonal | AdaLN path is untouched. |
| T-LoRA / OrthoLoRA / ReFT | ❌ not stacked | This is the author-faithful path — no stacking. Use the FEI-on-Hydra toggle block (`lora.toml`-style) if you want stacking. |
| DCW (scalar / v4) | ✅ orthogonal | Sampler-level correction; composes with anything upstream of the Euler step. |
| ComfyUI | ⚠ custom node | Stock weight-patcher can't load FeRA keys (router + experts are not in LoRA convention). Would need an `AnimaFeraLoader` node analogous to the existing `AnimaAdapterLoader`. Not shipped yet. |
| HydraLoRA-moe loaded simultaneously | ❌ | `library/inference/models.py` refuses to load both — they're alternative router schemes, pick one. |

## What to measure

The only reason author-faithful FeRA is worth running on Anima is that the global router on latent spectral state captures *per-prompt* routing variance the σ-router can't — populations at the same σ get the same Hydra gate but different FEI gates. The whole bet hinges on whether this content-aware variance translates to a quality lift.

1. **Router gate entropy across training.** Should stabilize above zero, with consistent per-prompt-type variation (scenery vs portrait vs flat-style routes to different gate distributions). Collapse → one expert always wins → FeRA reduces to a plain LoRA with extra unused params.
2. **Per-expert utilization on a held-out prompt set.** Histogram of `argmax_k w_k` (or weighted utilization) across prompts. Useful answer: experts specialize by *content type*, not by σ-stage (that's what Hydra does).
3. **Per-prompt routing stability across seeds.** Two seeds of the same prompt should produce similar gate distributions (gate is a function of `z_t`, which differs by seed but converges to similar spectral shape). If gates drift wildly seed-to-seed, the router is noise-sensitive — tighten `fera_router_tau` or grow `fera_router_hidden`.
4. **A/B vs the FEI-on-Hydra toggle.** Same dataset, matched epochs/lr. Compare CLIP-similarity to held-out reference + subjective quality. The author-faithful path is heavier (independent A) and globally-routed; FEI-on-Hydra is lighter and locally-routed. Whichever wins tells us which axis matters more on Anima.
5. **Sample quality vs `make lora`.** The hard test — is FeRA better than the plain LoRA-family default? Use the same prompt set used for `make test` and look at structural quality + prompt following + style coherence. FM val-MSE is uninformative on Anima (see [[project_fm_val_loss_uninformative]]).

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `fera_num_experts` | 3 | 2, 3, 4, 6 | Paper: 3. Watch for expert utilization saturating ⇒ too many. |
| `fera_num_bands` | 3 | 2, 3 | 2 = Anima-validated. 3 only useful if router actually splits along mid-band. |
| `network_dim` (rank) | 4 | 2, 4, 8, 16 | Independent per-expert → rank multiplied by `E`. Going to 16 at `E=3` is ~Hydra-default territory. |
| `fera_router_tau` | 0.7 | 0.3, 0.7, 1.0, 2.0 | Lower τ → sharper specialization but more sensitive to FEI noise. |
| `fera_router_hidden` | 64 | 32, 64, 128 | Router input is only `num_bands` floats; the bottleneck is usually expressive enough. |
| `fei_sigma_low_div` | 4.0 | 2, 4, 8, 16 | Higher → tighter low band (more high-freq picked up there). 2026-05-13 dataset sweep picked 4 as the discriminative-signal winner over 8 on training latents; both 4 and 8 are in the Pareto region. |
| `fera_target_modules` | all attn + MLP | MLP-only, attn-only | Ablate which sites benefit from FeRA gating. MLP-only mirrors the FEI-on-Hydra default. |
| `fera_fecl_weight` | 0.1 | 0, 0.1, 0.2 | Activates the FECL base-pass + composer term (2× per-step forward cost). Only meaningful at `num_bands ≥ 3` (at 2 bands the term collapses to a magnitude regularizer). |
| `multiplier` (inference) | 1.0 | 0.0, 0.5, 1.0, 1.5 | `0.0` short-circuits to frozen base for clean ablation. Per-layer multiplier control isn't exposed. |

## Files

- `networks/methods/fera.py` — network module.
- `configs/methods/fera.toml` — TOML with default author block + commented FEI-on-Hydra toggle block.
- `library/runtime/fei.py` — DoG kernels + 2-band helper, shared with FEI-on-Hydra.
- `bench/fera/probe_fei.py` — pre-network-module bench that settled 2-band on Anima.
- `bench/fera/results/20260512-1814-fera-pilot/` — 3-bucket probe.
- `bench/fera/results/20260512-1827-fera-midwide/` — wider-σ_mid probe.
- `FeRA/` — paper authors' reference implementation (read-only, for diffing against our port).
- `plan.md` — earlier 2-phase plan; Phase 1 (FEI-on-Hydra) is what currently lives behind the toggle block.

## Citation

```
@article{yin2025fera,
  title={FeRA: Frequency-Energy Constrained Routing for Effective Diffusion Adaptation Fine-Tuning},
  author={Yin, Bo and Hu, Xiaobin and Zhou, Xingyu and Jiang, Peng-Tao and Liao, Yue
          and Zhu, Junwei and Zhang, Jiangning and Tai, Ying and Wang, Chengjie
          and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2511.17979},
  year={2025}
}
```
