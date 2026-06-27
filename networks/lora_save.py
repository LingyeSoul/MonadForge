"""Save-pipeline orchestrator for the LoRA / Ortho / Hydra family.

The per-variant save logic — Cayley distillation, MoE write layout,
qkv defuse — lives on the variant's module class in
``networks/lora_modules/`` (``OrthoLoRAModule.distill_save_state_dict``,
``HydraLoRAModule.build_moe_state_dict``, etc). This file is the thin
ordering layer that calls them and writes the resulting file(s).

Ordering of the conversion pipeline is load-bearing:

  1. ``ChimeraHydraLoRAModule.distill_save_state_dict``
     (gated on co-located ``.Q_basis_c`` + ``.Q_basis_f`` — covers both the
     Cayley and OrthoInit chimera parameterizations)
  2. ``StackedExpertsLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 3-D ``.S_q``)
  3. ``OrthoHydraLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 2-D ``.S_q``)
  4. ``OrthoLoRAModule.distill_save_state_dict``
     (gated on 2-D ``.S_p``)
  4b. ``OrthoInitLoRAModule.distill_save_state_dict``
     (gated on ``.P_init`` — a name no other variant uses, so order vs the
     ``.S_p``-keyed steps above is independent; placed here for readability)
  5. legacy sig-type OrthoLoRA → standard LoRA
     (gated on ``.base_lambda``; kept here because it touches the
     deprecated ``lora_deprecated.OrthoLoRAModule`` save layout that no
     live module class owns)

The ``.S_p`` / ``.S_q`` dimensionality is the discriminator — every step
checks both dims explicitly so the matchers never overlap on the same
prefix. Step 5 handles legacy checkpoints from
``lora_deprecated.OrthoLoRAModule``; current training never emits those
keys, but the converter is kept so old artifacts remain re-bakeable.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from library.log import setup_logging
from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    OrthoHydraLoRAModule,
    OrthoInitLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
)
from networks.lora_modules.lora import defuse_and_bake_standard

setup_logging()
logger = logging.getLogger(__name__)


# Legacy sig-type OrthoLoRA → standard LoRA via 2r-dim SVD. Kept here (not on a
# module class) because the live ``OrthoLoRAModule`` never emits these keys —
# they belong to the deprecated ``lora_deprecated.OrthoLoRAModule``.


def _convert_legacy_ortho_to_lora(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".base_lambda"):
            prefixes.add(key[: -len(".base_lambda")])

    for prefix in prefixes:
        P = state_dict[f"{prefix}.p_layer.weight"]  # (out, r)
        Q = state_dict[f"{prefix}.q_layer.weight"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]
        P_base = state_dict[f"{prefix}.base_p_weight"]
        Q_base = state_dict[f"{prefix}.base_q_weight"]
        lam_base = state_dict[f"{prefix}.base_lambda"]
        alpha = state_dict.get(f"{prefix}.alpha")
        rank = Q.shape[0]

        # ΔW = P·diag(λ)·Q − P_base·diag(λ_base)·Q_base is rank ≤ 2r. SVD
        # works in the small 2r-dim column/row space instead of on the full
        # (out × in) matrix: ΔW = [P|P_base] @ M @ [Q; Q_base], then SVD of M.
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dtype = dtype if dtype is not None else P.dtype

        P_cat = torch.cat([P, P_base], dim=1).float().to(svd_device)  # (out, 2r)
        Q_cat = torch.cat([Q, Q_base], dim=0).float().to(svd_device)  # (2r, in)
        lam_diag = torch.diag(lam.squeeze(0).float().to(svd_device))
        lam_base_diag = torch.diag(lam_base.squeeze(0).float().to(svd_device))

        M = torch.zeros(2 * rank, 2 * rank, device=svd_device)
        M[:rank, :rank] = lam_diag
        M[rank:, rank:] = -lam_base_diag

        Qp, Rp = torch.linalg.qr(P_cat)
        Qq, Rq = torch.linalg.qr(Q_cat.T)

        core = Rp @ M @ Rq.T
        Uc, Sc, Vhc = torch.linalg.svd(core)

        lora_up = (
            (Qp @ Uc[:, :rank] * Sc[:rank].sqrt().unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Sc[:rank].sqrt().unsqueeze(1) * Vhc[:rank, :] @ Qq.T)
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        for suffix in (
            "p_layer.weight",
            "q_layer.weight",
            "lambda_layer",
            "base_p_weight",
            "base_q_weight",
            "base_lambda",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


def _convert_lokr_to_standard_lora(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
    lora_rank: int = 128,
) -> None:
    """Convert LoKR factor keys to standard ``lora_down``/``lora_up`` format.

    Materialises the Kronecker-product delta from each prefix's ``lokr_w1`` /
    ``lokr_w2`` (or ``w1a``/``w1b``/``w2a``/``w2b`` decomposed pairs), applies
    ``inv_scale`` when present, SVD-splits the result, and replaces the
    original factor keys in *state_dict* so the file is ComfyUI-compatible.
    Uses the per-module ``alpha`` (set to ``lora_dim`` in shipped LoKR presets,
    but user-overridable via ``network_alpha``) as the SVD rank cap when
    present; falls back to *lora_rank* only when no ``.alpha`` key exists.
    """
    prefixes: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in list(state_dict.keys()):
        if key.endswith(".lokr_w1"):
            p = key[: -len(".lokr_w1")]
            prefixes.setdefault(p, {})["lokr_w1"] = state_dict[key]
        elif key.endswith(".lokr_w2"):
            p = key[: -len(".lokr_w2")]
            prefixes.setdefault(p, {})["lokr_w2"] = state_dict[key]
        elif key.endswith(".w1a"):
            p = key[: -len(".w1a")]
            prefixes.setdefault(p, {})["w1a"] = state_dict[key]
        elif key.endswith(".w2a"):
            p = key[: -len(".w2a")]
            prefixes.setdefault(p, {})["w2a"] = state_dict[key]
        elif key.endswith(".w1b"):
            p = key[: -len(".w1b")]
            prefixes.setdefault(p, {})["w1b"] = state_dict[key]
        elif key.endswith(".w2b"):
            p = key[: -len(".w2b")]
            prefixes.setdefault(p, {})["w2b"] = state_dict[key]

    if not prefixes:
        return

    svd_device = "cuda" if torch.cuda.is_available() else "cpu"
    converted = 0

    for prefix, factors in prefixes.items():
        # Preserve source dtype when no explicit dtype was requested — matches
        # ``_convert_legacy_ortho_to_lora`` and the standard write path. Read
        # before the ``.float()`` SVD casts below.
        save_dtype = dtype if dtype is not None else next(iter(factors.values())).dtype

        w1 = factors.get("lokr_w1")
        if w1 is None:
            w1a = factors.get("w1a")
            w1b = factors.get("w1b")
            if w1a is None or w1b is None:
                continue
            w1 = w1a.float().to(svd_device) @ w1b.float().to(svd_device)
        else:
            w1 = w1.float().to(svd_device)

        w2 = factors.get("lokr_w2")
        if w2 is None:
            w2a = factors.get("w2a")
            w2b = factors.get("w2b")
            if w2a is None or w2b is None:
                continue
            w2 = w2a.float().to(svd_device) @ w2b.float().to(svd_device)
        else:
            w2 = w2.float().to(svd_device)

        delta = torch.kron(w1, w2)
        inv_scale_key = f"{prefix}.inv_scale"
        if inv_scale_key in state_dict:
            inv_s = state_dict[inv_scale_key].float().to(svd_device)
            delta = delta * inv_s.unsqueeze(0)

        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        alpha_key = f"{prefix}.alpha"
        if alpha_key in state_dict:
            target_rank = int(state_dict[alpha_key].item())
        else:
            target_rank = lora_rank
        max_rank = min(target_rank, S.shape[0])
        S_sqrt = S[:max_rank].sqrt()
        lora_up = (
            (U[:, :max_rank] * S_sqrt.unsqueeze(0)).to(save_dtype).cpu().contiguous()
        )
        lora_down = (
            (S_sqrt.unsqueeze(1) * Vh[:max_rank, :]).to(save_dtype).cpu().contiguous()
        )

        for suffix in ("lokr_w1", "lokr_w2", "w1a", "w1b", "w2a", "w2b", "inv_scale"):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        state_dict[f"{prefix}.alpha"] = torch.tensor(max_rank, dtype=save_dtype)
        converted += 1

    if converted:
        logger.info(f"LoKR → standard LoRA: converted {converted} module(s)")


# Back-compat shim: tests/test_global_router.py imports this name directly
# to exercise the StackedExperts MoE writer in isolation.


def _build_stacked_experts_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Thin shim → :meth:`StackedExpertsLoRAModule.build_moe_state_dict`."""
    return StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
) -> None:
    """Run the full save pipeline: distill chain + variant write.

    Mutates ``state_dict`` in place.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Distill chain. Order is load-bearing — see module docstring.
    ChimeraHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    StackedExpertsLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoInitLoRAModule.distill_save_state_dict(state_dict, dtype)
    _convert_legacy_ortho_to_lora(state_dict, dtype)

    # Variant dispatch:
    #   * stacked_experts_global_fei: independent-A → *_moe.safetensors
    #   * chimera_hydra_moe: dual-A per-pool + freq_router.* → *_chimera.safetensors
    #   * hydra_moe / ortho_hydra_to_hydra: shared-A Hydra → *_moe.safetensors
    #   * standard: defuse qkv → *.safetensors
    # Auto-fallback: any surviving ``.lora_up_weight`` key implies a Hydra
    # payload — kept for callers that don't plumb ``save_variant`` through.
    is_lokr_variant = save_variant == "lokr"
    is_stacked_experts_variant = save_variant == "stacked_experts_global_fei"
    is_chimera_variant = save_variant == "chimera_hydra_moe"
    is_hydra_variant = (
        save_variant in ("hydra_moe", "ortho_hydra_to_hydra")
        or (
            not is_chimera_variant
            and any(k.endswith(".lora_up_weight") for k in state_dict.keys())
        )
    ) and not is_stacked_experts_variant

    if is_stacked_experts_variant:
        se_file = os.path.splitext(file)[0] + "_moe.safetensors"
        se_sd = StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(se_sd, se_file, metadata or {})
        logger.info(f"StackedExperts full format saved to {se_file}")
        return

    if is_chimera_variant:
        chimera_file = os.path.splitext(file)[0] + "_chimera.safetensors"
        chimera_sd = ChimeraHydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(chimera_sd, chimera_file, metadata or {})
        logger.info(f"ChimeraHydra full format saved to {chimera_file}")
        return

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = HydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    if is_lokr_variant:
        # LoKR → standard LoRA: materialise kron delta and SVD-split so the
        # file is ComfyUI-compatible (lora_down/lora_up format).
        _convert_lokr_to_standard_lora(state_dict, dtype)
        defuse_and_bake_standard(state_dict)

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key].detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library.training.hashing import precalculate_safetensors_hashes

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
        return

    # Standard (lora / ortho) write path.
    defuse_and_bake_standard(state_dict)

    if dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key].detach().clone().to("cpu").to(dtype)
            state_dict[key] = v

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)
