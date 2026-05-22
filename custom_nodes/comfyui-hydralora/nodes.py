"""Anima ComfyUI custom nodes.

Three single-purpose loader nodes that each take a MODEL, apply one kind
of Anima-trained intervention, and return a MODEL. Chain them when a
workflow needs more than one.

  - ``AnimaAdapterLoader``: LoRA / HydraLoRA / ReFT (auto-detected from
    the safetensors keys + metadata). Installs ComfyUI weight patches for
    plain LoRA, per-Linear forward hooks for HydraLoRA live routing
    (σ-conditional and/or FeRA-style FEI-conditional on the Hydra stack),
    and per-block forward hooks for ReFT.
  - ``AnimaFeraLoader``: author-faithful FeRA (Yin et al., arXiv:2511.17979)
    — global router on the latent's spectral energy + per-Linear stacked
    independent experts. Different network family from
    ``AnimaAdapterLoader``'s Hydra/FEI variant: incompatible save format,
    mutually exclusive with HydraLoRA-moe (load one, not both).
  - ``AnimaSoftTokensLoader``: SoftREPA-parameterization soft tokens.
    Per-block forward pre-hooks splice per-layer x per-timestep-bucket
    learned tokens into the crossattn embedding inside the first n_layers
    DiT blocks; a diffusion_model pre-hook records the per-step sigma.

``AnimaFeraLoader`` was added in v3.1.0; ``AnimaSoftTokensLoader`` in
v3.6.0. The ``AnimaPostfixLoader`` node was retired when the postfix
training method was archived (the prefix / postfix / cond splice has no
live trainer; see the repo's _archive/postfix/).
"""

import folder_paths

from .adapter import apply_adapter
from .fera import apply_fera
from .soft_tokens import apply_soft_tokens


class AnimaAdapterLoader:
    """Apply an Anima adapter (LoRA / HydraLoRA / ReFT) to a MODEL.

    Auto-detects which components the safetensors file contains and
    routes each to its correct application path:

      - Plain LoRA → ``ModelPatcher.add_patches``
      - HydraLoRA → per-Linear ``forward_hook`` (live router replay,
        including σ-conditional bias and FeRA-style FEI routing when
        the checkpoint's metadata declares ``ss_use_fei_router=true``)
      - ReFT → per-block ``forward_hook`` on the DiT's blocks

    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": (
                    loras,
                    {
                        "tooltip": (
                            "Anima adapter file. May contain any combination "
                            "of LoRA, HydraLoRA (*_moe.safetensors), "
                            "ChimeraHydra (*_chimera.safetensors — dual-pool "
                            "content + frequency routing), and ReFT "
                            "(residual-stream) weights."
                        )
                    },
                ),
                "strength_lora": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength for LoRA / Hydra weight patches.",
                    },
                ),
                "strength_reft": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength for ReFT residual-stream edits.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima adapter loader. Auto-detects LoRA / HydraLoRA / ChimeraHydra "
        "/ ReFT in the safetensors file. HydraLoRA installs per-Linear "
        "forward hooks that compute the trained per-sample router gate "
        "from each Linear's input and blend per-expert lora_up heads — "
        "full live routing including σ-conditional bias and FeRA-style "
        "FEI-conditional content routing when the checkpoint declares "
        "it. ChimeraHydra (*_chimera.safetensors, ss_use_chimera_hydra=true) "
        "additionally runs a network-level FreqRouter on FEI+σ each step, "
        "splits experts into content (K_c, per-Linear) + frequency (K_f, "
        "global) pools, and dispatches the concat gate through the same "
        "Hydra einsum. ReFT installs per-block forward hooks."
    )

    def apply(self, model, adapter, strength_lora, strength_reft):
        new_model = model.clone()
        file_path = folder_paths.get_full_path("loras", adapter)
        apply_adapter(new_model, file_path, strength_lora, strength_reft)
        return (new_model,)


class AnimaFeraLoader:
    """Apply a FeRA adapter to a MODEL — either author-faithful or plan2.

    FeRA (Yin et al., arXiv:2511.17979): one **global router** consumes
    the latent's Frequency-Energy Indicator each denoising step and emits
    a single ``(B, num_experts)`` gate that every adapted Linear reuses
    for that step. Each adapted Linear carries **independent** stacked
    low-rank experts (``lora_down: (E, r, in)``, ``lora_up: (E, out, r)``)
    and adds ``Σ_k w_k · U_k @ D_k @ x`` to the frozen base.

    Loads two save formats with identical inference semantics:

      * Author-faithful (``networks.methods.fera``) — N-band FEI,
        ``router.net.*``, stacked-Parameter ``lora_down``/``lora_up``.
      * Plan2 stacked-experts (``networks.lora_anima`` with
        ``ss_network_spec=stacked_experts_global_fei``) — 2-band FEI,
        ``global_router.net.*``, per-expert split
        ``lora_downs.{i}.weight`` / ``lora_ups.{i}.weight``.

    Distinct from ``AnimaAdapterLoader``'s FEI-on-Hydra variant: that
    one routes per-Linear on Hydra's shared-A stack, this one routes
    globally on independent experts. Mutually exclusive with HydraLoRA-
    moe at the inference layer — load one, not both.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": (
                    loras,
                    {
                        "tooltip": (
                            "FeRA checkpoint — either author-faithful "
                            "(networks.methods.fera; router.net.* + "
                            "lora_unet_*.lora_down/lora_up) or plan2 "
                            "stacked_experts_global_fei (global_router.net.* + "
                            "lora_unet_*.lora_downs.{i}.weight / .lora_ups.{i}.weight, "
                            "typically named *_moe.safetensors). Both use "
                            "an independent-A stacked-expert layout with a "
                            "single network-level FEI router."
                        )
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Scales the gated expert correction added to "
                            "each adapted Linear (mirrors the training-side "
                            "multiplier; 0 short-circuits to the frozen base)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima FeRA loader (author-faithful — Yin et al., arXiv:2511.17979). "
        "Installs a single model-level forward_pre_hook that computes the "
        "per-step Frequency-Energy Indicator and global router gates, plus "
        "per-Linear forward_hooks that add the gated stacked-expert "
        "correction. Mutually exclusive with HydraLoRA — for FEI-on-Hydra "
        "checkpoints, use AnimaAdapterLoader."
    )

    def apply(self, model, adapter, strength):
        new_model = model.clone()
        file_path = folder_paths.get_full_path("loras", adapter)
        apply_fera(new_model, file_path, strength)
        return (new_model,)


class AnimaSoftTokensLoader:
    """Apply Anima soft tokens (SoftREPA parameterization) to a MODEL.

    Splices per-layer, per-timestep-bucket learned soft tokens into the
    T5-compatible crossattn embedding **inside** the first ``n_layers`` DiT
    blocks (the same surface anima_lora's trainer monkey-patches). A
    ``forward_pre_hook`` on each block rewrites its ``crossattn_emb`` argument;
    a ``diffusion_model`` pre-hook records the per-step sigma and precomputes
    the bank, so ``forward`` is never overridden (same invariant as Hydra/ReFT).

    Applies to the whole batch (both CFG branches) — soft tokens are part of
    the conditioning the trainer always saw. Chain after ``AnimaAdapterLoader``
    when a workflow needs more than one intervention.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "soft_tokens": (
                    loras,
                    {
                        "tooltip": (
                            "Soft-token file (tokens + t_offsets.weight keys, "
                            "from `make exp-soft-tokens`)."
                        )
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength multiplier for the spliced soft tokens.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima soft-token loader. Splices per-layer x per-t learned soft "
        "tokens into the crossattn embedding inside the first n_layers DiT "
        "blocks via forward pre-hooks. Applies to both CFG branches. Chain "
        "after an adapter loader when a workflow needs both."
    )

    def apply(self, model, soft_tokens, strength):
        new_model = model.clone()
        file_path = folder_paths.get_full_path("loras", soft_tokens)
        apply_soft_tokens(new_model, file_path, strength)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaAdapterLoader": AnimaAdapterLoader,
    "AnimaFeraLoader": AnimaFeraLoader,
    "AnimaSoftTokensLoader": AnimaSoftTokensLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaAdapterLoader": "Anima Adapter Loader",
    "AnimaFeraLoader": "Anima FeRA Loader",
    "AnimaSoftTokensLoader": "Anima Soft Tokens Loader",
}
