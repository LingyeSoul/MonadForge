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


def _collect_lokr_prefixes(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Group LoKR factor keys by prefix.

    Handles both the MonadForge-internal naming (``lokr_w1`` / ``lokr_w2`` full,
    ``w1a`` / ``w1b`` / ``w2a`` / ``w2b`` decomposed) and the ComfyUI/LyCORIS
    native naming (``lokr_w1_a`` / ``lokr_w1_b`` / ``lokr_w2_a`` / ``lokr_w2_b``).
    """
    prefixes: Dict[str, Dict[str, torch.Tensor]] = {}
    suffix_map = (
        "lokr_w1",
        "lokr_w2",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2_a",
        "lokr_w2_b",
        "w1a",
        "w1b",
        "w2a",
        "w2b",
    )
    for key in list(state_dict.keys()):
        for suf in suffix_map:
            if key.endswith(f".{suf}"):
                p = key[: -len(suf) - 1]
                # Normalize internal names (w1a/w1b/w2a/w2b) to a canonical key
                # so the reconstruction helpers below can treat both layouts
                # uniformly. Native names are kept as-is.
                canon = suf
                if suf == "w1a":
                    canon = "lokr_w1_a"
                elif suf == "w1b":
                    canon = "lokr_w1_b"
                elif suf == "w2a":
                    canon = "lokr_w2_a"
                elif suf == "w2b":
                    canon = "lokr_w2_b"
                prefixes.setdefault(p, {})[canon] = state_dict[key]
                break
    return prefixes


def _reconstruct_lokr_factor(
    factors: Dict[str, torch.Tensor], which: str
) -> torch.Tensor:
    """Reconstruct a full ``w1`` / ``w2`` matrix from full or decomposed keys.

    ``which`` is ``"w1"`` or ``"w2"``. Decomposed pairs are ``lokr_w1_a @
    lokr_w1_b`` (and likewise for w2); the full key is ``lokr_w1`` / ``lokr_w2``.
    """
    full_key = f"lokr_{which}"
    a_key = f"lokr_{which}_a"
    b_key = f"lokr_{which}_b"
    if full_key in factors:
        return factors[full_key]
    return factors[a_key] @ factors[b_key]


def _materialize_lokr_delta(
    factors: Dict[str, torch.Tensor],
    inv_scale: Optional[torch.Tensor],
    scale: float,
    device: str,
) -> torch.Tensor:
    """Materialize the full LoKR delta including scale and inv_scale.

    ``delta = kron(w1, w2) * scale``, with input-column scaling ``*
    inv_scale[None, :]`` applied when *inv_scale* is present (the forward-time
    ``x * inv_scale`` expressed as delta-column scaling — see
    ``LoKRModule._apply_inv_scale_to_full_delta``).
    """
    w1 = _reconstruct_lokr_factor(factors, "w1").float().to(device)
    w2 = _reconstruct_lokr_factor(factors, "w2").float().to(device)
    delta = torch.kron(w1, w2) * scale
    if inv_scale is not None:
        delta = delta * inv_scale.float().to(device).unsqueeze(0)
    return delta


def _convert_lokr_to_standard_lora(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
    lora_rank: int = 0,
    network_dim: Optional[int] = None,
) -> None:
    """Convert LoKR factor keys to standard ``lora_down``/``lora_up`` format.

    Materialises the full Kronecker delta (including the training ``scale =
    alpha / network_dim`` and per-channel ``inv_scale`` when present), SVD-splits
    it, and writes ComfyUI-compatible ``lora_down`` / ``lora_up`` keys whose
    ``(alpha/rank) * (up @ down)`` reproduces the trained delta exactly at full
    rank (``alpha`` is written as ``rank`` so ComfyUI's scale term is 1.0 — the
    factors carry the full scaling).

    ``lora_rank`` caps the SVD truncation rank per module (0 = full rank,
    lossless; values like 128/256 trade accuracy for file size — see
    ``docs/compose/plans/2026-06-27-lokr-full-rank-fix.md`` for the energy
    retention table). ``network_dim`` is the training ``lora_dim`` used to
    derive ``scale``; when None it is read from the per-module ``.alpha`` key
    (assuming ``alpha == lora_dim``, the common LyCORIS-style default). When
    neither is available, scale defaults to 1.0.
    """
    prefixes = _collect_lokr_prefixes(state_dict)
    if not prefixes:
        return

    svd_device = "cuda" if torch.cuda.is_available() else "cpu"
    converted = 0

    for prefix, factors in prefixes.items():
        save_dtype = dtype if dtype is not None else next(iter(factors.values())).dtype

        # Derive the training scale = alpha / lora_dim. alpha is per-module
        # (network_alpha); lora_dim is network-level and not in the per-module
        # state_dict — callers pass it via network_dim, else we assume the
        # LyCORIS convention alpha == lora_dim (scale = 1).
        alpha_key = f"{prefix}.alpha"
        alpha_val = (
            float(state_dict[alpha_key].item()) if alpha_key in state_dict else 1.0
        )
        if network_dim is not None and network_dim > 0:
            scale = alpha_val / network_dim
        else:
            scale = 1.0

        inv_scale_key = f"{prefix}.inv_scale"
        inv_scale = state_dict.get(inv_scale_key)

        delta = _materialize_lokr_delta(factors, inv_scale, scale, svd_device)

        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        # rank cap: 0 = full rank (lossless); explicit value truncates.
        max_avail = S.shape[0]
        if lora_rank and lora_rank > 0:
            max_rank = min(lora_rank, max_avail)
        else:
            max_rank = max_avail
        S_sqrt = S[:max_rank].sqrt()
        lora_up = (
            (U[:, :max_rank] * S_sqrt.unsqueeze(0)).to(save_dtype).cpu().contiguous()
        )
        lora_down = (
            (S_sqrt.unsqueeze(1) * Vh[:max_rank, :]).to(save_dtype).cpu().contiguous()
        )

        for suffix in (
            "lokr_w1",
            "lokr_w2",
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2_a",
            "lokr_w2_b",
            "w1a",
            "w1b",
            "w2a",
            "w2b",
            "inv_scale",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        # alpha == rank ⇒ ComfyUI scale = alpha/rank = 1.0, so the up/down
        # product (which already carries the trained scale) is applied as-is.
        state_dict[f"{prefix}.alpha"] = torch.tensor(max_rank, dtype=save_dtype)
        converted += 1

    if converted:
        logger.info(f"LoKR → standard LoRA: converted {converted} module(s)")


def _convert_lokr_to_native_lokr(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
    network_dim: Optional[int] = None,
) -> bool:
    """Rewrite LoKR factor keys to the ComfyUI/LyCORIS native lokr layout.

    Follows the LyCORIS convention (matches reference checkpoints such as
    ``chen_bin_anima_epoch72.safetensors``): **factors are saved RAW** — the
    unmodified training parameters — and the training ``scale = alpha /
    network_dim`` is recovered at load time by ComfyUI's ``LokrAdapter``, which
    applies ``alpha / rank`` whenever any factor is decomposed (see
    ``comfy/weight_adapter/lokr.py`` ``calculate_weight``).

    Per factor:
      * Decomposed (``w1a``/``w1b``/``w2a``/``w2b``) → renamed to ComfyUI's
        ``lokr_w1_a``/``lokr_w1_b``/``lokr_w2_a``/``lokr_w2_b``, contents raw.
      * Full (``lokr_w1``/``lokr_w2``) → kept as-is, contents raw.
      * ``alpha`` → the training alpha (``int64`` when integral, else ``float32``),
        so ComfyUI's ``alpha / rank`` restores the trained scale exactly.

    One edge case breaks the raw-factor convention: when **both** factors are
    full (``lokr_w1`` + ``lokr_w2``, no decomposition), ComfyUI's loader sets
    ``dim = None`` and forces the load scale to ``1.0`` — it cannot recover the
    training scale. LyCORIS sidesteps this by forcing ``alpha = lora_dim``
    (scale 1.0) on the full-full path; MonadForge does not, so when the training
    scale differs from 1.0 we fold it into ``lokr_w2`` (the only place the
    loader will look). ``alpha`` is still written as the training value for
    provenance (the loader ignores it on this path).

    Returns True if any module was converted. Raises if any module carries an
    ``inv_scale`` key — native lokr format cannot represent per-channel input
    scaling (the Kronecker structure breaks it; see ``LoKRModule`` docstring at
    ``lokr.py:126``). Callers with ``inv_scale`` must use
    ``_convert_lokr_to_standard_lora`` instead.
    """
    prefixes = _collect_lokr_prefixes(state_dict)
    if not prefixes:
        return False

    # Refuse inv_scale — native lokr cannot represent it.
    inv_prefixes = [p for p in prefixes if f"{p}.inv_scale" in state_dict]
    if inv_prefixes:
        raise ValueError(
            f"Cannot emit native lokr format: {len(inv_prefixes)} module(s) "
            f"carry inv_scale (first: {inv_prefixes[0]}). Native lokr cannot "
            "represent per-channel input scaling — use the SVD-to-standard-lora "
            "path (save_variant='lokr' with inv_scale, or extract script "
            "--format lora). Re-train without channel_scaling_alpha to get a "
            "native-lokr-compatible checkpoint."
        )

    converted = 0
    for prefix, factors in prefixes.items():
        save_dtype = dtype if dtype is not None else next(iter(factors.values())).dtype

        alpha_key = f"{prefix}.alpha"
        alpha_val = (
            float(state_dict[alpha_key].item()) if alpha_key in state_dict else 1.0
        )
        if network_dim is not None and network_dim > 0:
            scale = alpha_val / network_dim
        else:
            scale = 1.0

        use_w1 = "lokr_w1" in factors
        use_w2 = "lokr_w2" in factors
        # ComfyUI forces load scale = 1.0 when both factors are full (dim=None),
        # so a non-unit training scale can only be carried by the factors.
        full_full_needs_fold = use_w1 and use_w2 and abs(scale - 1.0) > 1e-12

        # Drop any stale keys (handles mixed internal/naming layouts).
        for suffix in (
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w2_a",
            "lokr_w2_b",
            "w1a",
            "w1b",
            "w2a",
            "w2b",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        # w1: raw passthrough (full key already correct; decomposed → rename).
        if use_w1:
            state_dict[f"{prefix}.lokr_w1"] = (
                factors["lokr_w1"].to(save_dtype).cpu().contiguous()
            )
        else:
            state_dict[f"{prefix}.lokr_w1_a"] = (
                factors["lokr_w1_a"].to(save_dtype).cpu().contiguous()
            )
            state_dict[f"{prefix}.lokr_w1_b"] = (
                factors["lokr_w1_b"].to(save_dtype).cpu().contiguous()
            )

        # w2: raw passthrough, except the full-full + scale≠1 edge case above,
        # where the loader's forced scale=1.0 means the factor must carry it.
        if use_w2:
            w2 = factors["lokr_w2"]
            if full_full_needs_fold:
                w2 = (w2.float() * scale).to(save_dtype)
            state_dict[f"{prefix}.lokr_w2"] = w2.to(save_dtype).cpu().contiguous()
        else:
            state_dict[f"{prefix}.lokr_w2_a"] = (
                factors["lokr_w2_a"].to(save_dtype).cpu().contiguous()
            )
            state_dict[f"{prefix}.lokr_w2_b"] = (
                factors["lokr_w2_b"].to(save_dtype).cpu().contiguous()
            )

        # alpha = training alpha (LyCORIS convention): ComfyUI restores scale via
        # alpha/rank on the decomposed path. int64 matches reference checkpoints;
        # fall back to float32 for non-integral alphas to avoid truncation.
        if alpha_val == int(alpha_val):
            state_dict[alpha_key] = torch.tensor(int(alpha_val), dtype=torch.int64)
        else:
            state_dict[alpha_key] = torch.tensor(alpha_val, dtype=torch.float32)
        converted += 1

    if converted:
        logger.info(f"LoKR → native lokr: converted {converted} module(s)")
    return True


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
        # LoKR save path. Two sub-paths:
        #   * No inv_scale keys → native lokr format (lokr_w1 + lokr_w2_a/b),
        #     ComfyUI's LokrAdapter loads it directly. Scale folded into factors.
        #   * inv_scale present (channel_scaling was on, or pre-fix training
        #     state) → native lokr can't represent per-channel scaling, so SVD
        #     to standard lora (lora_down/up) which can bake inv_scale in.
        # ``ss_network_dim`` carries the training lora_dim needed to recover
        # scale = alpha/network_dim. It's written by the metadata builder.
        meta_dim = metadata.get("ss_network_dim") if metadata else None
        network_dim = int(float(meta_dim)) if meta_dim else None

        has_inv_scale = any(k.endswith(".inv_scale") for k in state_dict.keys())
        if has_inv_scale:
            logger.warning(
                "LoKR checkpoint has inv_scale (channel_scaling_alpha was on "
                "during training). Native lokr format cannot represent it — "
                "falling back to SVD-to-standard-lora. Re-train with "
                "channel_scaling_alpha=0 (lokr auto-disables it) to emit a "
                "native lokr file."
            )
            _convert_lokr_to_standard_lora(
                state_dict, dtype, lora_rank=0, network_dim=network_dim
            )
            defuse_and_bake_standard(state_dict)
        else:
            _convert_lokr_to_native_lokr(state_dict, dtype, network_dim=network_dim)

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
