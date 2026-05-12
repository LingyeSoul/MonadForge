# custom_nodes/comfyui-hydralora/

ComfyUI custom nodes that dispatch Anima-trained interventions (LoRA / HydraLoRA / ReFT / prefix / postfix / cond) through ComfyUI's patching system. Exists because vanilla ComfyUI's weight-patcher silently drops non-LoRA keys (`reft_*`, `lora_ups`, postfix vectors), so a Hydra/ReFT/postfix checkpoint loaded with a stock LoRA loader produces wrong output with no warning.

Three single-purpose nodes (adapter + postfix split in v3.0.0, FeRA added in v3.1.0):

  - `AnimaAdapterLoader` — LoRA / HydraLoRA / ReFT (`adapter.py`).
  - `AnimaFeraLoader` — author-faithful FeRA (`fera.py`).
  - `AnimaPostfixLoader` — prefix / postfix / cond context splicing (`postfix.py`).

Chain them `MODEL → <adapter or fera> → AnimaPostfixLoader → MODEL` when a workflow needs both; the postfix wrapper sees the model with adapter modifications already in place. `AnimaAdapterLoader` and `AnimaFeraLoader` are mutually exclusive — author-faithful FeRA and HydraLoRA-moe are alternative router schemes (see `library/inference/models.py`). Pre-v3.0.0 the adapter + postfix were one toggle-bool node — see README §3.0.0 for the rationale.

Full user-facing docs and changelog live in `README.md`. This file is for code-level edits to the node.

## Files

| File | Role |
|------|------|
| `adapter.py` | LoRA / Hydra / ReFT key parsing, classification, hook install (incl. FEI compute helpers for the FEI-on-Hydra path). |
| `fera.py` | Author-faithful FeRA (`networks.methods.fera`) parsing + apply. Reuses `_gaussian_blur_2d` from `adapter.py` for the FEI compute; bands ordering / router shape are FeRA-specific. |
| `postfix.py` | Prefix / postfix / cond context splicing on `diffusion_model.forward`. |
| `nodes.py` | `AnimaAdapterLoader` + `AnimaFeraLoader` + `AnimaPostfixLoader` ComfyUI node definitions. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |

## Application paths (which key goes where)

Each node sniffs its safetensors header and routes each component independently — paths are disjoint:

| Component | Application path |
|-----------|-----------------|
| Plain LoRA | `ModelPatcher.add_patches` (standard ComfyUI weight patch). |
| HydraLoRA | Per-Linear `forward_hook` installed via `ModelPatcher.add_object_patch` on each adapted Linear's `_forward_hooks`. |
| FeRA (author-faithful) | One global `forward_pre_hook` on `diffusion_model._forward_pre_hooks` computes per-step FEI + router gates; per-Linear `forward_hook`s on each adapted Linear's `_forward_hooks` add the gated stacked-expert correction. Same hook-not-override invariant as Hydra. |
| ReFT | Per-block `forward_hook` installed via `ModelPatcher.add_object_patch` on `diffusion_model.blocks.<idx>._forward_hooks`. |
| Prefix / postfix / cond | `ModelPatcher.add_object_patch` on `diffusion_model.forward`, splicing learned vectors into the T5-compatible crossattn embedding **after** the LLM adapter + pad-to-512 step. CFG-safe via `cond_or_uncond` from `transformer_options`. |

## Critical invariant: forward_hook, never override `forward`

For Hydra and ReFT, install a `forward_hook` — do **not** replace `block.forward` / `linear.forward`. Overriding `forward` strands weights on CPU under ComfyUI's cast-weights path: ComfyUI walks the real `forward` to drive its `comfy_cast_weights` machinery, and replacing the method confuses it — blocks end up with `comfy_cast_weights=False` and their Linears stay on CPU, producing a device mismatch at runtime. A hook leaves `forward` untouched, traces cleanly through `torch.compile`, and is properly reverted on `unpatch_model`.

Prefix/postfix is the exception (it patches `diffusion_model.forward` itself), but that's the model-level forward, not a per-Linear / per-block one — same rule, different scope.

## Router-input layout (σ + FEI)

Routing is data-driven: `router = Linear(rank + sigma_dim + fei_dim, E)` with the input built as `[pooled, sinusoidal(σ), FEI]` — concat order matters and must match `networks/lora_modules/hydra.py::_compute_gate`. A forward pre-hook on `diffusion_model._forward_pre_hooks` records the current `timesteps` (`args[1]`) and — for FeRA-style FEI routing — the per-step 2-band Laplacian energy of `args[0]` (the latent, squeezed of any T=1 dim) into shared state on each denoising call. Every Hydra hook reads from that state to build its router input.

**The shape alone is ambiguous.** `router.weight.shape[1] - rank` does not tell you whether the extra columns are σ or FEI — only the safetensors metadata can. `load_adapter` reads `ss_use_fei_router` / `ss_fei_feature_dim` / `ss_fei_sigma_low_div`, and `_apply_hydra_live_to_model` derives `sigma_feature_dim = router_in - rank - fei_feature_dim`. Without that metadata flag the node assumes FEI is off and treats all extra columns as σ-feature columns (the historical behavior for σ-only checkpoints).

Old `sigma_mlp.*` checkpoints are not supported (see README §2.1.0).

## Author-faithful FeRA (`fera.py`)

`AnimaFeraLoader` is for `networks.methods.fera` checkpoints — a different network family from the FEI-on-Hydra path above. Three architectural differences that drive the split:

1. **Single global router** owned by the network, not per-Linear. `_make_fera_pre_hook` computes FEI on the latent + runs the 2-layer `Linear → ReLU → Linear → softmax/τ` router once per `diffusion_model` forward and writes `fera_state["gates"]` of shape `(B, num_experts)`. Every per-Linear hook reads that same gate.
2. **Independent stacked experts** — `lora_down: (E, r, in)` and `lora_up: (E, out, r)` are flat Parameters, not Linear submodules. Saved keys end in `.lora_down` / `.lora_up` (no `.weight` suffix). `_make_fera_hook` does `einsum("...i,eri->...er")` for the down projection, multiplies by the broadcast gates, then `einsum("...er,eor->...o")` for the up — bit-identical to `FeRALinear.forward`.
3. **Multi-band FEI ordering.** The author network's `FrequencyEnergyIndicator` returns `[high, ..., low]` (high freq first), which differs from `adapter.py::_compute_fei_2band`'s `[e_low, e_high]`. `fera.py::_compute_fei_nband` matches the author-faithful ordering exactly — router weights are sensitive to band order. Don't share band-compute code with `adapter.py` even though both use the same Gaussian blur.

Detection prefers `ss_network_module == "networks.methods.fera"` metadata; falls back to a key sniff for `router.net.*` + stacked `lora_unet_*.lora_down`/`.lora_up`. `AnimaFeraLoader` and `AnimaAdapterLoader` should not both target the same checkpoint — author-faithful FeRA and HydraLoRA-moe are alternative router schemes (mirrors the inference loader's `fera_mode`/`hydra_mode` check).

**Key resolution — direct walk, not `model_lora_keys_unet`.** ComfyUI's `comfy.lora.model_lora_keys_unet` was designed around the older UNet convention where Q/K/V live in separate Linears; it doesn't enumerate fused projections by default and (more importantly for us) doesn't always reach modules inside the LLM adapter sub-tree. `fera.py::_build_fera_key_map` walks `diffusion_model.named_modules()` directly and emits one `lora_unet_<dotted>` entry per `nn.Linear`, mirroring the training-side `FeRANetwork._scan_targets` convention exactly. The save format also writes split q/k/v on disk (see `networks/methods/fera.py::_split_fused_state_dict`), so checkpoint prefixes match ComfyUI's split modules without any conversion at load time.

Caveat: `is_mergeable() == False` on the training side because a router-mixed output isn't a single ΔW — there's no merge-into-DiT path. Stay on the live-routing node.

## Coexistence

Plain-LoRA and Hydra paths target disjoint key prefixes (`_extract_lora_sd` skips `lora_ups.*`, `_parse_hydra` requires `lora_ups`), so a mixed checkpoint where only some Linears are Hydra-routed runs both paths in the same load without conflict. Don't reintroduce mutual-exclusion checks.

Author-faithful FeRA is a different story: its prefixes overlap with Hydra's at the `lora_unet_*` level but the suffixes differ (`.lora_down` vs `.lora_down.weight`). The two paths still won't collide on parsing (each parser keys off its own suffix), but installing both on the same Linear would chain two hooks — additive composition with no semantic basis. Pick one loader per workflow.

## Publishing

This node ships as a ComfyUI Registry package — bump version in `pyproject.toml`, push to GitHub, then `comfy node publish --token $COMFY_REG`. The token is in `anima_lora/.env`.
