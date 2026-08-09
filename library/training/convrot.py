"""Training bootstrap for the optional ConvRot frozen-base compute path."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


def _is_enabled(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _reject_dora(args: Any, network_kwargs: Mapping[str, Any] | None) -> None:
    net_args = getattr(args, "network_args", None) or []
    joined = " ".join(str(item) for item in net_args).lower()
    if "dora_wd" in joined or "use_dora=true" in joined or "dora=true" in joined:
        raise ValueError(
            "base_compute ConvRot is not supported with DoRA. "
            "Use plain LoRA or disable ConvRot."
        )
    kwargs = network_kwargs or {}
    if _is_enabled(kwargs.get("dora_wd")) or _is_enabled(kwargs.get("use_dora")):
        raise ValueError(
            "base_compute ConvRot is not supported with DoRA. "
            "Use plain LoRA or disable ConvRot."
        )


def maybe_apply_convrot_base(
    args: Any,
    network: Any,
    *,
    unet: Any = None,
    network_kwargs: Mapping[str, Any] | None = None,
) -> bool:
    """Patch frozen base Linear forwards when ``base_compute`` selects ConvRot.

    This must run after ``network.apply_to`` captures each original Linear and
    before ``compile_blocks`` traces the adapter-patched forward.
    """
    from library.runtime.convrot.apply import apply_convrot_to_lora_network
    from library.runtime.convrot.checks import (
        assert_convrot_block_swap_mutex,
        convrot_mode_from_base_compute,
        normalize_base_compute,
        warn_convrot_blocks_to_swap,
    )

    base_compute = normalize_base_compute(getattr(args, "base_compute", "bf16"))
    args.base_compute = base_compute
    mode = convrot_mode_from_base_compute(base_compute)
    if mode is None:
        return False

    assert_convrot_block_swap_mutex(
        base_compute=base_compute,
        block_swap_transfer_dtype=getattr(args, "block_swap_transfer_dtype", "bf16"),
    )
    swap_warning = warn_convrot_blocks_to_swap(
        base_compute=base_compute,
        blocks_to_swap=getattr(args, "blocks_to_swap", 0),
    )
    if swap_warning:
        logger.warning("[convrot] %s", swap_warning)

    _reject_dora(args, network_kwargs)

    group_size = int(getattr(args, "convrot_group_size", 256) or 256)
    scope = str(getattr(args, "convrot_scope", "mlp") or "mlp")
    hadamard = str(getattr(args, "convrot_hadamard", "sylvester") or "sylvester")
    weight_source = str(
        getattr(args, "convrot_weight_source", "online_from_bf16") or "online_from_bf16"
    )
    prequant_path = getattr(args, "convrot_prequant_path", None) or None
    min_in_features = int(getattr(args, "convrot_min_in_features", 0) or 0)
    largest_only = bool(getattr(args, "convrot_largest_in_features_only", False))
    large_mode_raw = getattr(args, "convrot_large_layer_mode", None)
    large_layer_mode = (
        None
        if str(large_mode_raw or "").strip().lower() in {"", "none", "off"}
        else str(large_mode_raw).strip()
    )
    large_min_raw = getattr(args, "convrot_large_min_in_features", None)
    large_min_in_features = (
        int(large_min_raw) if large_min_raw not in (None, "", 0, "0") else None
    )

    kind = hadamard.strip().lower()
    if kind in {"regular", "reg", "paper", "convrot"}:
        os.environ["ANIMA_CONVROT_HADAMARD"] = "regular"
        kind_resolved = "regular"
    else:
        os.environ["ANIMA_CONVROT_HADAMARD"] = "sylvester"
        kind_resolved = "sylvester"

    # Adapter parameters live on ``network``. Only the referenced DiT base
    # weights are frozen and replaced by runtime-only quantized payloads.
    if unet is not None:
        unet.requires_grad_(False)
    else:
        for lora in getattr(network, "unet_loras", None) or []:
            for base in getattr(lora, "org_module_ref", None) or []:
                weight = getattr(base, "weight", None)
                if weight is not None and bool(getattr(weight, "requires_grad", False)):
                    weight.requires_grad_(False)

    result = apply_convrot_to_lora_network(
        network,
        mode=mode,
        scope=scope,
        group_size=group_size,
        weight_source=weight_source,
        prequant_path=prequant_path,
        unet=unet,
        min_in_features=min_in_features,
        largest_in_features_only=largest_only,
        large_layer_mode=large_layer_mode,
        large_min_in_features=large_min_in_features,
    )
    args._convrot_apply_result = result
    logger.info(
        "[convrot] applied mode=%s scope=%s group=%s hadamard=%s source=%s "
        "patched=%d skipped=%d min_in=%d largest_only=%s large_mode=%s prequant=%s",
        mode,
        scope,
        result.group_size,
        kind_resolved,
        weight_source,
        result.patched_count,
        result.skipped_count,
        result.min_in_features,
        result.largest_in_features_only,
        result.large_layer_mode,
        prequant_path,
    )
    return True
