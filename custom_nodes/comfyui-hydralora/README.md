# Anima Adapter / Postfix Loaders (ComfyUI)

Two ComfyUI custom nodes that load Anima-trained interventions and dispatch them through ComfyUI's patching system. Each node does one thing; chain them with the MODEL socket when a workflow needs both.

Algorithm-level notes live in the main docs tree (`docs/methods/hydra-lora.md`, `docs/methods/reft.md`, `docs/experimental/postfix.md`). This README covers only what's ComfyUI-specific: detection, installation paths, and the node's changelog.

## Install

Drop `custom_nodes/comfyui-hydralora/` (this directory) into your ComfyUI `custom_nodes/`, restart ComfyUI. The two nodes appear as **Anima Adapter Loader** and **Anima Postfix Loader** in the loaders menu.

## The loaders

### Anima Adapter Loader

| Input | Purpose |
|-------|---------|
| `adapter` | safetensors file holding any mix of LoRA / HydraLoRA / ReFT keys |
| `strength_lora` | scales LoRA + HydraLoRA delta (set 0 to disable both while keeping ReFT) |
| `strength_reft` | scales ReFT residual edit (set 0 to disable ReFT while keeping LoRA) |

Sniffs the safetensors header and routes each component independently — you get correct behavior whether the file contains plain LoRA, a `*_moe.safetensors` hydra checkpoint (σ-conditional or FeRA-style FEI-conditional), a ReFT-only file, or any combination. The two strength sliders are useful for ablation ("is it the LoRA or the ReFT doing the anatomy fix?") and for dialing back either branch when one overshoots.

### Anima Postfix Loader

| Input | Purpose |
|-------|---------|
| `postfix` | safetensors file with prefix / postfix / cond keys |
| `strength_postfix` | scales the postfix / prefix delta |

Mode (prefix / postfix / cond) is auto-detected from the file's keys. When chaining with the adapter loader, put the postfix loader *after* the adapter loader so the postfix wrapper sees the model with adapter modifications already in place.

## How each component applies

**Plain LoRA** → `ModelPatcher.add_patches`, the standard ComfyUI weight-patch path.

**HydraLoRA** (live routing) → per-Linear `forward_hook` installed via `ModelPatcher.add_object_patch` on each adapted Linear's `_forward_hooks`. The hook replays `HydraLoRAModule.forward` exactly: rank-R `lora_down` projection, RMS pool over the sequence dim, optional sinusoidal(σ) concatenated onto the pooled vector, `Linear(rank + sigma_feature_dim, E)` router, softmax, gate-weighted expert `lora_up` blend. Routing is data-driven, so `strength_lora` is a single slider — per-expert controls would not be meaningful under live routing.

σ-conditional routing: a forward pre-hook on `diffusion_model` records the current `timesteps` into shared state on each denoising call; every hydra hook reads it to build the sinusoidal σ features. Detected automatically from `router.weight.shape[1] > rank` (minus any FEI dim, when applicable — see below).

FeRA-style FEI routing (`make exp-fera` checkpoints): when the checkpoint's safetensors metadata declares `ss_use_fei_router=true`, the same pre-hook also computes the per-step 2-band Laplacian energy (`e_low, e_high`) of the current latent and stashes it as `(B, 2)` simplex features. The hook concatenates them onto the pooled router input *after* any σ features, matching the training-time `_compute_gate` order `[pooled, sinusoidal(σ), FEI]`. The σ-band partition path and FEI router compose freely — they touch different parts of the router-input layout — though shipped FeRA configs leave σ-band off. FEI compute is one separable Gaussian per denoising step on the (B, C, H, W) latent, negligible vs the DiT forward.

**ReFT** → per-block `forward_hook` installed via `ModelPatcher.add_object_patch` on `diffusion_model.blocks.<idx>._forward_hooks`. The hook adds `R^T · (ΔW · h + b) · scale · strength` to the block output.

**Prefix / postfix / cond** → `ModelPatcher.add_object_patch` on `diffusion_model.forward`, splicing learned vectors into the T5-compatible crossattn embedding *after* the LLM adapter + pad-to-512 step. Positive-batch rows only via `cond_or_uncond` from `transformer_options` (CFG-safe).

## Why forward hooks, not `forward` override

For both HydraLoRA and ReFT we install a `forward_hook` rather than overriding `block.forward` / `linear.forward`. Overriding `forward` strands weights on CPU under ComfyUI's cast-weights path: ComfyUI relies on walking the real `forward` to drive its `comfy_cast_weights` machinery, and replacing the method confused it — blocks ended up with `comfy_cast_weights=False` and their Linears stayed on CPU, producing a device mismatch at runtime. A hook leaves `forward` untouched, traces cleanly through `torch.compile`, and is properly reverted on `unpatch_model`.

## Code layout

| File | Role |
|------|------|
| `adapter.py` | LoRA / Hydra / ReFT loading, parsing, hook install |
| `postfix.py` | Prefix / postfix / cond context splicing |
| `nodes.py` | The `AnimaAdapterLoader` node |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` |

## Changelog

### 3.2.0 — 2026-05-14 — AnimaFeraLoader handles plan2 `stacked_experts_global_fei`

Plan2 reshaped the LoRA-family routing surface into three axes (`use_moe_style` / `route_per_layer` / `router_source`); the FeRA cell of that matrix (`independent_A` / `route_per_layer=False` / `router_source="fei"`) saves as `*_moe.safetensors` with `ss_network_spec=stacked_experts_global_fei`. Different on-disk shape from the older `networks.methods.fera` format:

- Router under `global_router.net.*` (not `router.net.*`).
- Per-Linear experts as **split** `lora_unet_*.lora_downs.{i}.weight` / `.lora_ups.{i}.weight` (not stacked flat `lora_down` / `lora_up` Parameters).
- FEI is fixed 2-band, `[e_low, e_high]` ordering (matches `library/runtime/fei.py::compute_fei_2band`) rather than the author-faithful N-band `[high, ..., low]`.

`AnimaFeraLoader` now auto-routes to the right parser based on metadata (`ss_network_spec` / `ss_network_module`) or a key sniff (`global_router.net.*` + `.lora_downs.{i}.weight`). Inference semantics are identical between the two formats — global router on the latent's FEI emits one `(B, num_experts)` gate per step, every adapted Linear adds `Σ_k w_k · U_k @ D_k @ x`. The pre-hook now dispatches the FEI compute by `cfg["fei_kind"]` so both orderings stay bit-correct.

`AnimaAdapterLoader` also got an early-exit guard: feeding it a `stacked_experts_global_fei` file now raises with a clear "use AnimaFeraLoader" message instead of producing the previous "Hydra live-routing skipped 280 prefix(es): missing lora_down/lora_ups" + "no recognizable keys" pair, which gave no hint about the right node.

### 3.0.0 — 2026-05-12 — Split adapter/postfix into two nodes + FeRA FEI router

**Breaking — workflow update required.** The single `AnimaAdapterLoader` with `use_adapter` / `use_postfix` toggle booleans is gone. In its place:

- `AnimaAdapterLoader` now applies LoRA / HydraLoRA / ReFT only (inputs: `model`, `adapter`, `strength_lora`, `strength_reft`).
- `AnimaPostfixLoader` (new node) applies prefix / postfix / cond context splicing (inputs: `model`, `postfix`, `strength_postfix`).

Chain them when a workflow needs both — `MODEL → AnimaAdapterLoader → AnimaPostfixLoader → MODEL` (or only the one you need). Each node now does one thing; bypass them via ComfyUI's standard "bypass node" feature when you want to A/B with adapter-only vs postfix-only. Existing workflows that referenced `AnimaAdapterLoader` with eight inputs will need to be rewired: re-pick the adapter node (it now has four inputs) and add a fresh `AnimaPostfixLoader` if the workflow was using a postfix.

Also lands FeRA-style FEI routing support (the second half of this release, below).

#### FeRA-style FEI router support

Catches the node up to the training-side FeRA-on-Hydra path (`make exp-fera`, `configs/methods/fera.toml`). Before this, loading an `anima_hydra_fei*_moe.safetensors` succeeded structurally but produced wrong gates: the node inferred `sigma_feature_dim = router_in - rank` and happily fed sinusoidal(σ) into router columns the trainer had reserved for FEI features, so the router routed on a completely different signal than it was trained on. There's no way to distinguish the two cases from `router.weight.shape` alone — needed safetensors metadata.

Applied in `adapter.py`:

1. `load_adapter` now reads `ss_use_fei_router`, `ss_fei_feature_dim`, and `ss_fei_sigma_low_div` from the safetensors metadata and stashes them on the parsed hydra bundle. Malformed values fall back to a clean default with a warning rather than crashing.
2. `_make_router_pre_hook` (renamed from `_make_sigma_pre_hook`) extends the existing diffusion-model pre-hook: when FEI is enabled, in addition to recording `timesteps` from `args[1]`, it also computes the 2-band Laplacian energy of `args[0]` (the latent, squeezed of any T=1 dim) using `σ_low = min(H_lat, W_lat) / fei_sigma_low_div` — bucket-invariant by construction. Stashed as `(B, 2)` simplex into the same shared state read by every per-Linear hook.
3. `_make_hydra_hook` extends `_compute_gate` to concat FEI features onto the pooled router input *after* the existing sinusoidal(σ) slice, matching training's `[pooled, σ, FEI]` order. Defensive zero-pad path keeps router shape valid if the pre-hook hasn't fired.
4. `_apply_hydra_live_to_model` derives `sigma_feature_dim = router_in - rank - fei_feature_dim` so old σ-only checkpoints collapse to the original split (no behavior change), FEI-only checkpoints get `σ_dim=0, fei_dim=2`, and a future σ+FEI sweep cell gets both correctly. The "σ-conditional yes/no" log line also accounts for FEI now.

Compute helpers (`_compute_fei_2band`, `_gaussian_blur_2d`, `_gaussian_kernel_1d`, `_fei_sigma_low`) mirror `library/runtime/fei.py` and are inlined in `adapter.py` to keep the node standalone.

### 2.2.0 — 2026-05-02 — σ-band partition reconstruction + perf cleanup

Catches the node up to the training-side σ-band specialization (commit `bf37e3e`). When `specialize_experts_by_sigma_buckets = true` is on at training, the expert→band lookup buffer (`_expert_band`) is registered non-persistent, so it doesn't ride along in the safetensors and the hook had no way to re-derive it. Inference therefore ran soft routing across all `E` experts, silently ignoring the partition baked into the router weights.

Applied in `adapter.py`:

1. `load_adapter` now opens the safetensors metadata alongside the weights and propagates `ss_specialize_experts_by_sigma_buckets`, `ss_num_sigma_buckets`, and (optional) `ss_sigma_bucket_boundaries` into the parsed hydra bundle. Divisibility (`num_experts % num_buckets == 0`) is validated; mismatches log a warning and disable the partition rather than crashing.
2. `_make_hydra_hook` rebuilds `expert_band` from `num_sigma_buckets` using the **interleaved** `e mod B` rule, matching the training-side switch in `_register_sigma_band_partition`. Out-of-band expert logits are masked to `-inf` before softmax.
3. Custom σ-bucket edges (`ss_sigma_bucket_boundaries`, length `B+1`, monotone `0.0 → 1.0`) override the default uniform `linspace`, so checkpoints with capacity concentrated on a chosen σ regime — e.g. `[0.0, 0.5, 0.8, 1.0]` for late-step refinement — bucket samples the same way training did.
4. Hot-path fp32 casts (`.float()` on `lora_down`, `lora_ups`, `router_w`, `router_b`, `inv_scale`, and on `sigma`) are hoisted out of the per-call hydra hook into device-migration (one-shot) and a normalized `sigma_pre_hook` (once per denoising step). Eliminates the per-Linear-per-compile `DeviceCopy` warning torch.compile was emitting; behavior is unchanged.

### 2.1.1 — 2026-04-29 — CPU-stranding fix on lowvram path

Capturing σ via `add_object_patch("diffusion_model.forward", …)` stranded sub-Linears (e.g. cosmos `x_embedder.proj`) on CPU under ComfyUI's lowvram-aware load path — the same failure mode that retired the old `block.forward` override in favor of `_forward_hooks`. Replaced the wrapper with a forward pre-hook on `diffusion_model._forward_pre_hooks`; the hook records `args[1]` (timesteps) into the shared σ state read by each hydra hook, leaving `forward` untouched.

### 2.1.0 — 2026-04-21 — σ-input catch-up + plain-LoRA fall-through

Training had moved σ from an additive `sigma_mlp` bias on router logits to a direct router-input feature: `router = Linear(rank + sigma_feature_dim, E)` with sinusoidal(σ) concatenated onto the pooled rank-R vector (see `docs/methods/hydra-lora.md` §Fixes, 2026-04-20). The node hadn't been updated — it still looked for `sigma_mlp.*` keys and refused routers whose second dim wasn't exactly `rank`, so every σ-conditional hydra checkpoint skipped all hydra modules. In mixed checkpoints (`hydra_router_layers` = mlp only), the `elif` fall-through to plain LoRA also didn't fire, so cross_attn / self_attn adapters went unapplied too.

Applied in `adapter.py`:

1. `_parse_hydra` drops `sigma_mlp.*` parsing and filters to modules with `lora_ups` so plain-LoRA prefixes stop surfacing as `missing lora_down/lora_ups` skip warnings.
2. `_apply_hydra_live_to_model` derives `sigma_feature_dim = router_w.shape[1] - rank` (≥ 0) instead of refusing non-rank router inputs.
3. `_make_hydra_hook` concatenates sinusoidal(σ) onto the pooled rank-R router input (broadcast when σ is shape `(1,)` vs CFG-doubled batch); additive bias path removed.
4. `apply_adapter` runs the plain-LoRA path whenever `bundle["lora"]` is present, not only when hydra is absent. The two paths target disjoint prefixes (`_extract_lora_sd` skips `lora_ups.*`, `_parse_hydra` requires `lora_ups`), so coexistence is safe.

### 2.0.0 — 2026-04-20 — rank-R router rewiring

Live-routing hook updated to mirror the training-time forward exactly: RMS pool over the sequence dim of the post-`lora_down` rank-R signal, not mean-pool over the raw layer input. Corresponding training fix is in `docs/methods/hydra-lora.md` §Fixes (2026-04-20 entry) — pre-fix routers never learned, so old checkpoints are refused at load.
