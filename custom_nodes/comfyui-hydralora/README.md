# Anima Adapter / FeRA / Soft-Token Loaders (ComfyUI)

Three ComfyUI custom nodes that load Anima-trained interventions and dispatch them through ComfyUI's patching system. Each node does one thing; chain them with the MODEL socket when a workflow needs more than one.

Algorithm-level notes live in the main docs tree (`docs/methods/hydra-lora.md`, `docs/methods/reft.md`, `docs/experimental/soft_tokens.md`). This README covers only what's ComfyUI-specific: detection, installation paths, and the node's changelog.

> **Retired:** the **Anima Postfix Loader** was removed when the postfix training method was archived (soft tokens superseded it — see the repo's `_archive/postfix/`). Older changelog entries below still reference it as history.

## Install

Drop `custom_nodes/comfyui-hydralora/` (this directory) into your ComfyUI `custom_nodes/`, restart ComfyUI. The nodes appear as **Anima Adapter Loader**, **Anima FeRA Loader**, and **Anima Soft Tokens Loader** in the loaders menu.

## The loaders

### Anima Adapter Loader

| Input | Purpose |
|-------|---------|
| `adapter` | safetensors file holding any mix of LoRA / HydraLoRA / ReFT keys |
| `strength_lora` | scales LoRA + HydraLoRA delta (set 0 to disable both while keeping ReFT) |
| `strength_reft` | scales ReFT residual edit (set 0 to disable ReFT while keeping LoRA) |

Sniffs the safetensors header and routes each component independently — you get correct behavior whether the file contains plain LoRA, a `*_moe.safetensors` hydra checkpoint (σ-conditional or FeRA-style FEI-conditional), a ReFT-only file, or any combination. The two strength sliders are useful for ablation ("is it the LoRA or the ReFT doing the anatomy fix?") and for dialing back either branch when one overshoots.

### Anima Soft Tokens Loader

| Input | Purpose |
|-------|---------|
| `soft_tokens` | safetensors file with `tokens` + `t_offsets.weight` keys (`make exp-soft-tokens`) |
| `strength` | scales the spliced soft tokens (0 = no-op) |

SoftREPA-parameterization soft tokens (Lee et al., arXiv:2503.08250): a bank of per-layer, per-timestep-bucket learned vectors is spliced into the crossattn embedding *inside* the first `n_layers` DiT blocks. Each block gets its own splice via a `forward_pre_hook` that rewrites the block's `crossattn_emb` argument — soft tokens use a *different* per-layer vector at each block; a `diffusion_model` pre-hook records the per-step sigma and precomputes the bank. Applies to the whole batch (both CFG branches) — soft tokens are part of the conditioning the trainer always saw. `n_layers` / `K` / `n_t_buckets` / splice position are read from the checkpoint (tensor shapes + `ss_splice_position`). Chain after the adapter loader when a workflow needs more than one.

## How each component applies

**Plain LoRA** → `ModelPatcher.add_patches`, the standard ComfyUI weight-patch path.

**HydraLoRA** (live routing) → per-Linear `forward_hook` installed via `ModelPatcher.add_object_patch` on each adapted Linear's `_forward_hooks`. The hook replays `HydraLoRAModule.forward` exactly: rank-R `lora_down` projection, RMS pool over the sequence dim, optional sinusoidal(σ) concatenated onto the pooled vector, `Linear(rank + sigma_feature_dim, E)` router, softmax, gate-weighted expert `lora_up` blend. Routing is data-driven, so `strength_lora` is a single slider — per-expert controls would not be meaningful under live routing.

σ-conditional routing: a forward pre-hook on `diffusion_model` records the current `timesteps` into shared state on each denoising call; every hydra hook reads it to build the sinusoidal σ features. Detected automatically from `router.weight.shape[1] > rank` (minus any FEI dim, when applicable — see below).

FeRA-style FEI routing (`make exp-fera` checkpoints): when the checkpoint's safetensors metadata declares `ss_use_fei_router=true`, the same pre-hook also computes the per-step 2-band Laplacian energy (`e_low, e_high`) of the current latent and stashes it as `(B, 2)` simplex features. The hook concatenates them onto the pooled router input *after* any σ features, matching the training-time `_compute_gate` order `[pooled, sinusoidal(σ), FEI]`. The σ-band partition path and FEI router compose freely — they touch different parts of the router-input layout — though shipped FeRA configs leave σ-band off. FEI compute is one separable Gaussian per denoising step on the (B, C, H, W) latent, negligible vs the DiT forward.

**ChimeraHydra dual-pool routing** (`make chimera` checkpoints, files named `*_chimera.safetensors`): the per-Linear router is narrowed to `K_c` outputs and reads pooled rank-R `lx` only (no σ/FEI columns). A network-level FreqRouter MLP (`Linear → SiLU → Linear → softmax/τ`, weights under `freq_router.net.*`) runs once per denoising step on `concat(FEI(z_t), sinusoidal(σ))` and broadcasts `π_f ∈ (B, K_f)` to every chimera Linear via shared state. Each per-Linear hook concatenates `[π_c, π_f]` over the full `E = K_c + K_f` experts and dispatches the standard Hydra einsum/bmm. Detected from `ss_use_chimera_hydra=true` plus the chimera-specific `ss_num_experts_content` / `ss_num_experts_freq` / `ss_chimera_*` metadata.

When chimera was trained with `content_router_source = "crossattn"` (`ss_chimera_content_router_source="crossattn"` in metadata), the per-Linear content router is replaced by a single network-level `ContentRouter` MLP fed pooled post-LLM-adapter `crossattn_emb`. A second `forward_hook` is installed on `diffusion_model.llm_adapter._forward_hooks` that pools its output to `(B, D)`, runs the MLP, and writes `π_c` into the same shared state as `π_f`. Per-Linear chimera hooks then broadcast that `π_c` instead of running their own pooled-`lx` softmax. The per-Linear `router.weight`/`router.bias` keys are absent from the file in this mode. See changelog 3.5.0.

**ReFT** → per-block `forward_hook` installed via `ModelPatcher.add_object_patch` on `diffusion_model.blocks.<idx>._forward_hooks`. The hook adds `R^T · (ΔW · h + b) · scale · strength` to the block output.

**Soft tokens** → per-block `forward_pre_hook` installed via `ModelPatcher.add_object_patch` on each of the first `n_layers` `diffusion_model.blocks.<idx>._forward_pre_hooks`, plus one `diffusion_model._forward_pre_hooks` pre-hook. The block pre-hook rewrites the block's `crossattn_emb` positional arg (overwriting the K padding-tail slots for `end_of_sequence`, or scattering after the real text tokens for `front_of_padding`); `forward` itself is untouched, same invariant as Hydra/ReFT. The model-level pre-hook recovers the `[0, 1]` sigma from comfy's `sigma × 1000` FLOW timesteps (`ModelSamplingDiscreteFlow` multiplier), bucketizes it, and precomputes the `(n_layers, B, K, D)` token bank the block hooks index. All hook installs go through `get_model_object`, so soft tokens compose with a prior adapter pre-hook on the same `_forward_pre_hooks` dict rather than clobbering it.

## Why forward hooks, not `forward` override

For both HydraLoRA and ReFT we install a `forward_hook` rather than overriding `block.forward` / `linear.forward`. Overriding `forward` strands weights on CPU under ComfyUI's cast-weights path: ComfyUI relies on walking the real `forward` to drive its `comfy_cast_weights` machinery, and replacing the method confused it — blocks ended up with `comfy_cast_weights=False` and their Linears stayed on CPU, producing a device mismatch at runtime. A hook leaves `forward` untouched, traces cleanly through `torch.compile`, and is properly reverted on `unpatch_model`.

## Code layout

| File | Role |
|------|------|
| `adapter.py` | LoRA / Hydra / ReFT loading, parsing, hook install |
| `fera.py` | Author-faithful + plan2 stacked-experts FeRA loading |
| `soft_tokens.py` | SoftREPA soft-token bank loading + per-block splice pre-hooks |
| `nodes.py` | `AnimaAdapterLoader` / `AnimaFeraLoader` / `AnimaSoftTokensLoader` |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` |
| `_vendor/` | Generated by `scripts/sync_vendor.py` — bundled copy of the router-compute kernels so the node works when not sitting inside the anima_lora repo |

The pure-compute router math (FEI 2-band / FEI n-band high-to-low, σ sinusoidal features, σ-band partition mask) lives in `library/inference/router_compute.py` in the main repo. `adapter.py` resolves it live when the node is inside anima_lora, falls back to `_vendor/library/inference/router_compute.py` when standalone. Trained router weights are bit-sensitive to these kernels, so the vendored copy must stay in lockstep with the live tree — re-run `make vendor-sync` (or `python scripts/sync_vendor.py`) before publishing a new node version.

## Changelog

### 3.7.0 — 2026-05-20 — Retire the Anima Postfix Loader

The postfix training method was archived (soft tokens superseded it — see the repo's `_archive/postfix/`), so `AnimaPostfixLoader` and `postfix.py` were removed. The node package now ships three loaders: `AnimaAdapterLoader`, `AnimaFeraLoader`, `AnimaSoftTokensLoader`. Existing workflows that referenced the postfix loader will need to drop that node. Soft tokens (`AnimaSoftTokensLoader`) cover the per-block crossattn-splice use case going forward. No change to the other loaders.

### 3.6.1 — 2026-05-20 — Postfix `cond+ortho` (v4) support + drop the `forward` override

`AnimaPostfixLoader` now loads the current `mode=cond` checkpoints (`make exp-postfix`, output `anima_postfix_ortho_v4`). Two parts:

- **cond+ortho format.** The trainer's cond head was rewritten (commit `e989d64`) to `LayerNorm → Linear → GELU → Linear` emitting `K(K-1)/2 + 1` scalars — a Cayley rotation seed `S(c)` + magnitude `λ(c)` — over a frozen `ortho_basis`, with maxabs-pooling of the content tokens. The node reconstructed the *old* 2-layer `K×D` format and crashed on `cond_mlp.2.weight` (now a GELU). It now mirrors `networks/methods/postfix.py::append_postfix` exactly: maxabs-pool → `postfix(c) = Cayley(S(c) − S(c)ᵀ) @ ortho_basis · λ(c)` (verified bit-for-bit against the trainer). Legacy 2-layer non-ortho cond checkpoints are no longer loadable (they're already unloadable on the trainer side). `postfix` (free-param) and `prefix` paths are unchanged.
- **No more `diffusion_model.forward` override.** Postfix previously replaced the model forward to run `preprocess_text_embeds` itself — which stranded the DiT's own `x_embedder.proj` on CPU under ComfyUI's dynamic-VRAM / cast-weights staging walk (`mat2 is on cpu`), the same failure mode that retired the hydra σ-capture `forward` wrapper in 2.1.1. The model already runs the LLM adapter inside `forward` and hands the same post-adapter `crossattn_emb` to every block, so the splice moved to a per-block `with_kwargs` `forward_pre_hook` on every block (reading `cond_or_uncond` from `transformer_options` to keep positive-only routing). `forward` is left intact — same hook-not-override invariant as Hydra/ReFT/soft-tokens. Outputs are unchanged on setups where the node already worked.

### 3.6.0 — 2026-05-20 — Soft-token inference (`AnimaSoftTokensLoader`)

New node `AnimaSoftTokensLoader` runs SoftREPA-parameterization soft tokens (Lee et al., arXiv:2503.08250) at inference — the `networks/methods/soft_tokens.py` checkpoints from `make exp-soft-tokens` were previously training-only on the ComfyUI side.

- Detection: the file's `tokens` `(n_layers, K, D)` + `t_offsets.weight` `(n_t_buckets, n_layers·D)` tensors. `n_layers` / `K` / `n_t_buckets` are inferred from the shapes; splice position from `ss_splice_position` (`end_of_sequence` default, or `front_of_padding`).
- Application: a per-block `forward_pre_hook` on the first `n_layers` `diffusion_model.blocks.<idx>._forward_pre_hooks` rewrites each block's `crossattn_emb` arg with that block's spliced bank; a `diffusion_model._forward_pre_hooks` pre-hook records the per-step sigma and precomputes the `(n_layers, B, K, D)` bank. `forward` is never overridden (Hydra/ReFT invariant).
- Sigma convention: comfy hands the FLOW model `timesteps = sigma × 1000` (`ModelSamplingDiscreteFlow`, multiplier 1000), so the pre-hook divides by 1000 to recover the `[0, 1]` sigma the trainer's t-bucket index uses (`train.py` draws `[0,1]`-scaled timesteps).
- Applies to the whole batch (both CFG branches), matching training — not positive-only like postfix.
- Composes with the adapter / postfix loaders: hook installs go through `get_model_object`, so a prior adapter pre-hook on `diffusion_model._forward_pre_hooks` is preserved rather than clobbered.

### 3.5.0 — 2026-05-19 — ChimeraHydra global ContentRouter (`content_router_source="crossattn"`)

`AnimaAdapterLoader` now supports chimera checkpoints trained with the network-level ContentRouter — one MLP per network, fed pooled post-LLM-adapter `crossattn_emb`, broadcasting `π_c` to every chimera Linear. The per-Linear pooled-`lx_c` softmax is replaced by a global "caption regime" axis (analogous to the freq pool's FreqRouter). Mutually exclusive with the default per-Linear path; selected at training time via `content_router_source = "crossattn"` in `configs/methods/chimera.toml`.

- Detection key: `ss_chimera_content_router_source == "crossattn"` in safetensors metadata. The loader parses top-level `content_router.net.{0,2}.weight/bias` into `chimera_data["content_router"]` and honors `ss_chimera_content_router_layer_norm` for the parameterless LN flag.
- The per-Linear `router.weight`/`router.bias` keys (shape `(K_c, rank)`) are **absent** under this mode — the loader no longer requires them on chimera prefixes when `content_router_source == "crossattn"`. Other modes (per-Linear router, default) are unchanged.
- New application hook: `_make_content_router_llm_adapter_hook` is installed as a `forward_hook` on `diffusion_model.llm_adapter._forward_hooks`. It captures the post-T5 features `(B, L_text, D)`, zero-pads to 512 (matches `Anima.preprocess_text_embeds`), RMS-pools over the sequence dim, optionally LayerNorms over D, runs `Linear → SiLU → Linear → softmax/τ`, and writes `π_c` into the same shared state the FreqRouter already uses. Per-Linear chimera hooks broadcast `π_c` from that state (uniform `1/K_c` fallback on the very first compile-cache miss).
- CFG batching composes naturally — cond and uncond rows go through one `diffusion_model.forward` and the hook produces per-row gates because their text differs.
- Composes with `AnimaPostfixLoader` (postfix splices `crossattn_emb` at the block level, which fires after the llm_adapter hook, so the content router always sees the unmodified post-T5 features).
- Hard error if the file claims crossattn but is missing `content_router.net.*`, or if the loaded DiT has no `llm_adapter` (non-Anima base).
- Single-A (3.3.0) and dual-A (3.4.0) chimera formats both pick this up; the parser is one helper (`_parse_chimera_content_router`) shared across both branches.

### 3.4.0 — 2026-05-15 — ChimeraHydra dual-A on-disk format

ChimeraHydra was revised on the training side (`networks/lora_modules/chimera.py`) to carry **two independent A's per Linear** — one per pool — instead of sharing a single rank-R basis across the content + freq pools. Each pool now also has its own per-pool B stack on disk. Free orthogonality on both sides of `ΔW` at init (top `(K_c+K_f)·r` left singular vectors split into per-pool sub-stacks; top `2r` right singular vectors split between the two A's), with Cayley rotations diverging the pools during training.

`AnimaAdapterLoader` now detects and loads the dual-A on-disk shape:

- New keys per Linear: `lora_down_c.weight` + `lora_down_f.weight` (each `(r, in)`), `lora_ups_c.{i}.weight` (i in 0..K_c-1) + `lora_ups_f.{j}.weight` (j in 0..K_f-1), shared `router.weight`/`router.bias` (`(K_c, r)`), `alpha`, optional `inv_scale`. Top-level `freq_router.net.*` unchanged.
- New per-Linear hook (`_make_chimera_dual_a_hook`): independent `lx_c = down_c · x` and `lx_f = down_f · x` projections; content router pools `lx_c` only (proposal-faithful — pooling `lx_f` would cross-couple the pools); gate-weighted per-pool ups `out_c = bmm(lx_c, comb_c.T) + bmm(lx_f, comb_f.T)`. FreqRouter pre-hook is unchanged (same input shape, same `[FEI, sinusoidal(σ)]` concat order).
- Detection key: `ss_use_chimera_hydra=true` **plus** any `.lora_down_c.weight` on disk. The legacy single-A chimera format (3.3.0) continues to load through the existing `_make_chimera_hook` path — the two paths are mutually exclusive by key shape.
- No `alpha/rank` scaling at inference (mirrors training, where chimera bakes `lambda_{c,f}` into the saved per-pool weights via the sqrt-split in `_convert_chimera_dual_a_to_hydra`). Apply via the `strength` slider only.

T-LoRA's content-branch rank mask remains training-only — inference runs full rank on both pools at every t.

### 3.3.1 — 2026-05-15 — router-compute kernels share live source-of-truth with anima_lora

`adapter.py` and `fera.py` no longer carry parallel reimplementations of the FEI / σ / σ-band kernels. They now import from `library/inference/router_compute.py` in the parent repo (live), falling back to `_vendor/library/inference/router_compute.py` when the node is installed standalone. The vendored copy is regenerated by `scripts/sync_vendor.py` and ships with each release. No checkpoint or workflow change — the kernels are bit-identical to the previous in-node copies, pinned by `tests/test_router_compute.py`.

Why: the trained router weights are bit-sensitive to band ordering (high→low for author-faithful FeRA, low→high for plan2 stacked-experts) and the σ frequency schedule. Two copies meant two places for a silent drift to enter. Now there's exactly one impl on disk, with the node consuming it through a vendor handshake that mirrors the existing `comfyui-anima-tagger` / `comfyui-anima-directedit` pattern.

### 3.3.0 — 2026-05-15 — AnimaAdapterLoader handles ChimeraHydra dual-pool routing

`ChimeraHydra` (see `networks/lora_modules/chimera.py` + `docs/proposal/chimera_hydra.md`) splits HydraLoRA's expert pool into a **content pool** (`K_c`, routed per-Linear by the content router on pooled rank-R `lx`) and a **frequency pool** (`K_f`, routed once per step by a network-level `FreqRouter` MLP on `concat(FEI(z_t), sinusoidal(σ))`). The combined gate `[π_c | π_f]` flows into the standard Hydra einsum, so the additive composition `Σ π_c · B_c(Ax) + Σ π_f · B_f(Ax)` reduces to one batched matmul.

Save format mirrors HydraLoRA-MoE (shared `lora_down` + per-expert `lora_ups.{i}`, q/k/v defused) **plus** top-level `freq_router.net.*` keys for the network-level freq router. The per-Linear content router shrinks to `(K_c, rank)` — no σ/FEI columns. Files are written next to the base adapter as `*_chimera.safetensors`.

Detection: `ss_use_chimera_hydra=true` in safetensors metadata. The loader reads `ss_num_experts_content` / `ss_num_experts_freq` / `ss_chimera_fei_feature_dim` / `ss_chimera_sigma_feature_dim` / `ss_chimera_fei_sigma_low_div`, captures `freq_router.net.{0,2}.weight/bias`, and installs a chimera-flavored pre-hook + per-Linear hook. The pre-hook runs FreqRouter on the current latent + timestep once per step and stashes `π_f` in shared state; the per-Linear hook concatenates `[π_c, π_f]` and dispatches the standard Hydra einsum/bmm. T-LoRA's content-branch mask is training-only — chimera at inference runs full rank at every t.

σ-band partition is unsupported for chimera (the FreqRouter owns the σ axis by construction) and skipped even if metadata claims it.

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
