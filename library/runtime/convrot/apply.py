"""Apply ConvRot W8A* base forwards onto LoRA-wrapped frozen Linears."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from library.runtime.convrot.checks import (
    is_dora_module,
    raise_if_compiled,
    validate_base_linear_for_convrot,
)
from library.runtime.convrot.free_base import (
    free_base_weights_for_patches,
    free_linear_weight_storage,
    is_base_weight_freed,
)
from library.runtime.convrot.linear_w8a16 import w8a16_forward_from_buffers
from library.runtime.convrot.linear_w8a8 import w8a8_forward_from_buffers
from library.runtime.convrot.prequant import (
    PrequantCheckpoint,
    load_prequant_checkpoint,
    resolve_effective_group_size,
)
from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.rht import (
    assert_group_divides,
    hadamard_kind,
    normalized_hadamard,
    rht_backend,
)
from library.runtime.convrot.scope import classify_convrot_linear_module
from library.runtime.block_swap_payload import BlockSwapManagedTensor

logger = logging.getLogger(__name__)

ConvRotMode = Literal["w8a16", "w8a8"]

_BUFFER_Q = "_convrot_quantized_weight"
_BUFFER_SCALE = "_convrot_scale"
_BUFFER_HADAMARD = "_convrot_hadamard"
_ATTR_GROUP = "_convrot_group_size"
_ATTR_MODE = "_convrot_mode"
_ATTR_WEIGHT_SOURCE = "_convrot_weight_source"
# W8A8 stores int8 as contiguous [K,N] (= weight.T) so torch._int_mm skips a
# per-step transpose+copy. W8A16 keeps classic [N,K] for dequant F.linear.
_ATTR_WEIGHT_LAYOUT = "_convrot_weight_layout"


@dataclass(frozen=True)
class ConvRotLoRABaseForwardPatch:
    lora_name: str
    name: str
    family: str
    block_idx: int
    shape: tuple[int, ...]
    group_size: int
    mode: str
    bf16_bytes: int
    payload_bytes: int
    weight_source: str = "online_from_bf16"
    skipped_reason: str | None = None


@dataclass(frozen=True)
class ConvRotApplyResult:
    patches: list[ConvRotLoRABaseForwardPatch]
    skipped: list[ConvRotLoRABaseForwardPatch]
    mode: str
    scope: str
    group_size: int
    weight_source: str = "online_from_bf16"
    prequant_path: str | None = None
    freed_modules: int = 0
    freed_bytes: int = 0
    min_in_features: int = 0
    largest_in_features_only: bool = False
    large_layer_mode: str | None = None
    large_min_in_features: int | None = None

    @property
    def patched_count(self) -> int:
        return len(self.patches)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def _set_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor, persistent=False)


def _set_swap_payload(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    current = module._modules.get(name)
    if isinstance(current, BlockSwapManagedTensor):
        current.weight = tensor.detach().contiguous()
        return
    module.add_module(name, BlockSwapManagedTensor(tensor))


def _swap_payload(module: nn.Module, name: str) -> torch.Tensor:
    payload = module._modules.get(name)
    if not isinstance(payload, BlockSwapManagedTensor):
        raise RuntimeError(f"missing ConvRot block payload {name!r}")
    return payload.weight


def _resolve_base_linear(lora: nn.Module) -> nn.Linear | None:
    refs = getattr(lora, "org_module_ref", None)
    if refs:
        base = refs[0]
        if isinstance(base, nn.Linear):
            return base
        return None
    org_forward = getattr(lora, "org_forward", None)
    owner = getattr(org_forward, "__self__", None)
    if isinstance(owner, nn.Linear):
        return owner
    return None


def _maybe_attach_hadamard(
    lora: nn.Module,
    *,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    shared_cache: dict[tuple[int, str, str, str], torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Precompute dense RHT matrix once (compile-friendly; skip env in hot path).

    FWHT backend stays dynamic (no fixed matmul). Dense is the 3080 default.
    Prefer bf16 on CUDA so group_rht can skip a per-call dtype cast when acts
    are bf16 (training default).

    When ``shared_cache`` is provided, all LoRAs with the same
    ``(group_size, device, dtype)`` share one Hadamard buffer (P1.7) instead of
    duplicating ``group×group`` per module.
    """
    if rht_backend() != "dense":
        return None
    store_dtype = dtype if device.type == "cuda" else torch.float32
    cache_key = (int(group_size), str(device), str(store_dtype), hadamard_kind())
    if shared_cache is not None and cache_key in shared_cache:
        h = shared_cache[cache_key]
        _set_buffer(lora, _BUFFER_HADAMARD, h)
        return h
    try:
        h = normalized_hadamard(
            int(group_size),
            device=device if device.type != "meta" else "cpu",
            dtype=store_dtype,
            kind=hadamard_kind(),
        )
    except ValueError:
        return None
    h = h.contiguous()
    if shared_cache is not None:
        shared_cache[cache_key] = h
    _set_buffer(lora, _BUFFER_HADAMARD, h)
    return h


def _install_org_forward(
    lora: nn.Module,
    *,
    payload_owner: nn.Module,
    mode: ConvRotMode,
    group_size: int,
) -> None:
    # Close over precomputed Hadamard when present so forward avoids cache/env.
    hadamard = getattr(lora, _BUFFER_HADAMARD, None)
    weight_layout = getattr(lora, _ATTR_WEIGHT_LAYOUT, "nk")

    if mode == "w8a16":

        def _base_forward(
            x: torch.Tensor,
            *,
            _lora=lora,
            _owner=payload_owner,
            _gs=group_size,
            _h=hadamard,
        ) -> torch.Tensor:
            h = getattr(_lora, _BUFFER_HADAMARD, None)
            if h is None:
                h = _h
            return w8a16_forward_from_buffers(
                x,
                _swap_payload(_owner, _BUFFER_Q),
                _swap_payload(_owner, _BUFFER_SCALE),
                group_size=_gs,
                hadamard=h,
            )

    else:

        def _base_forward(
            x: torch.Tensor,
            *,
            _lora=lora,
            _owner=payload_owner,
            _gs=group_size,
            _h=hadamard,
            _layout=weight_layout,
        ) -> torch.Tensor:
            h = getattr(_lora, _BUFFER_HADAMARD, None)
            if h is None:
                h = _h
            layout = getattr(_lora, _ATTR_WEIGHT_LAYOUT, _layout)
            return w8a8_forward_from_buffers(
                x,
                _swap_payload(_owner, _BUFFER_Q),
                _swap_payload(_owner, _BUFFER_SCALE),
                group_size=_gs,
                hadamard=h,
                weight_layout=str(layout),
            )

    lora.org_forward = _base_forward


def _resolve_layer_mode(
    *,
    default_mode: ConvRotMode,
    in_features: int,
    large_layer_mode: str | None,
    large_min_in_features: int | None,
) -> ConvRotMode:
    if not large_layer_mode or large_min_in_features is None:
        return default_mode
    if in_features < int(large_min_in_features):
        return default_mode
    text = str(large_layer_mode).strip().lower().removesuffix("_convrot")
    if text not in {"w8a16", "w8a8"}:
        raise ValueError(
            f"unsupported convrot_large_layer_mode={large_layer_mode!r}; "
            "expected w8a16 | w8a8"
        )
    return text  # type: ignore[return-value]


def _filter_candidates_by_size(
    candidates: list[tuple[nn.Module, nn.Linear, int, str, str]],
    *,
    min_in_features: int,
    largest_in_features_only: bool,
) -> tuple[
    list[tuple[nn.Module, nn.Linear, int, str, str]],
    list[ConvRotLoRABaseForwardPatch],
]:
    """Apply P1-G size filters; return (kept, size-skipped stubs)."""
    skipped: list[ConvRotLoRABaseForwardPatch] = []
    kept: list[tuple[nn.Module, nn.Linear, int, str, str]] = []
    min_in = max(0, int(min_in_features))

    for lora, base, block_idx, family, original_name in candidates:
        in_f = int(base.in_features)
        lora_name = str(getattr(lora, "lora_name", original_name))
        if min_in > 0 and in_f < min_in:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=tuple(int(d) for d in base.weight.shape),
                    group_size=0,
                    mode="",
                    bf16_bytes=int(base.weight.numel()) * 2,
                    payload_bytes=0,
                    skipped_reason=f"min_in_features:{in_f}<{min_in}",
                )
            )
            continue
        kept.append((lora, base, block_idx, family, original_name))

    if largest_in_features_only and kept:
        max_in = max(int(base.in_features) for _, base, *_ in kept)
        trimmed: list[tuple[nn.Module, nn.Linear, int, str, str]] = []
        for lora, base, block_idx, family, original_name in kept:
            in_f = int(base.in_features)
            if in_f < max_in:
                lora_name = str(getattr(lora, "lora_name", original_name))
                skipped.append(
                    ConvRotLoRABaseForwardPatch(
                        lora_name=lora_name,
                        name=str(original_name),
                        family=family,
                        block_idx=block_idx,
                        shape=tuple(int(d) for d in base.weight.shape),
                        group_size=0,
                        mode="",
                        bf16_bytes=int(base.weight.numel()) * 2,
                        payload_bytes=0,
                        skipped_reason=f"largest_in_features_only:{in_f}<{max_in}",
                    )
                )
                continue
            trimmed.append((lora, base, block_idx, family, original_name))
        kept = trimmed

    return kept, skipped


def _resolve_payload(
    *,
    weight_source: str,
    base_module: nn.Linear,
    original_name: str,
    group_size: int,
    prequant: PrequantCheckpoint | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_source == "online_from_bf16":
        return rotate_and_quantize_weight(base_module.weight.detach(), group_size)

    assert prequant is not None
    layer = prequant.get(str(original_name))
    if layer is None:
        raise KeyError(
            f"prequant checkpoint missing layer {original_name!r} "
            f"(file has {prequant.layer_count} layers; path={prequant.path})"
        )
    q = layer.quantized_weight
    scale = layer.scale
    if tuple(q.shape) != tuple(base_module.weight.shape):
        raise ValueError(
            f"prequant shape mismatch for {original_name!r}: "
            f"file {tuple(q.shape)} vs live Linear {tuple(base_module.weight.shape)}"
        )
    assert_group_divides(int(q.shape[1]), group_size)
    # Keep buffers on the same device as the live base (usually CUDA).
    device = base_module.weight.device
    if device.type == "meta":
        # Already freed somehow; stay on CPU and let forward move as needed.
        return q.contiguous(), scale.to(torch.float32).contiguous()
    return (
        q.to(device=device).contiguous(),
        scale.to(device=device, dtype=torch.float32).contiguous(),
    )


def apply_convrot_to_lora_network(
    network: nn.Module,
    *,
    mode: ConvRotMode = "w8a16",
    scope: str = "mlp",
    group_size: int = 256,
    weight_source: str = "online_from_bf16",
    prequant_path: str | None = None,
    dry_run: bool = False,
    allow_zero_patches: bool = False,
    free_base_weights: bool = True,
    unet: nn.Module | None = None,
    prequant_group_size_strict: bool = True,
    min_in_features: int = 0,
    largest_in_features_only: bool = False,
    large_layer_mode: str | None = None,
    large_min_in_features: int | None = None,
) -> ConvRotApplyResult:
    """Patch LoRA ``org_forward`` with ConvRot W8A* base path.

    Only replaces ``lora.org_forward``; never swaps child ``nn.Linear`` modules.
    When ``free_base_weights=True`` (default), dense bf16 ``Linear.weight``
    storage for patched modules is released after quant so GPU no longer holds
    dual-resident bf16+int8 payloads.

    ``weight_source``:

    * ``online_from_bf16`` — RHT+quant live frozen weights (default).
    * ``prequant_checkpoint`` — load rotated int8 + scale from ``prequant_path``
      (native ``anima_lora_convrot_prequant_v1`` or weight/weight_scale pairs).

    P1 size / mixed-mode knobs:

    * ``min_in_features`` — skip Linears with ``in_features`` below this (0=off).
    * ``largest_in_features_only`` — among scope hits, only patch max ``in_features``.
    * ``large_layer_mode`` + ``large_min_in_features`` — override mode for big
      layers (e.g. default w8a16, large mlp.layer1 → w8a8).
    """
    if mode not in {"w8a16", "w8a8"}:
        raise ValueError(f"unsupported convrot mode={mode!r}")
    if weight_source not in {"online_from_bf16", "prequant_checkpoint"}:
        raise ValueError(f"unsupported convrot_weight_source={weight_source!r}")

    large_mode_norm: str | None = None
    large_min: int | None = None
    if large_layer_mode is not None and str(large_layer_mode).strip():
        large_mode_norm = (
            str(large_layer_mode).strip().lower().removesuffix("_convrot")
        )
        if large_mode_norm not in {"w8a16", "w8a8"}:
            raise ValueError(
                f"unsupported convrot_large_layer_mode={large_layer_mode!r}"
            )
        if large_min_in_features is None:
            raise ValueError(
                "convrot_large_layer_mode requires convrot_large_min_in_features"
            )
        large_min = int(large_min_in_features)
        if large_min <= 0:
            raise ValueError("convrot_large_min_in_features must be > 0")
    elif large_min_in_features is not None and int(large_min_in_features) > 0:
        raise ValueError(
            "convrot_large_min_in_features requires convrot_large_layer_mode"
        )

    prequant: PrequantCheckpoint | None = None
    effective_group = int(group_size)
    if weight_source == "prequant_checkpoint":
        if not prequant_path:
            raise ValueError(
                "convrot_weight_source=prequant_checkpoint requires convrot_prequant_path"
            )
        prequant = load_prequant_checkpoint(prequant_path)
        effective_group = resolve_effective_group_size(
            prequant,
            requested_group_size=int(group_size),
            strict=prequant_group_size_strict,
        )
    elif prequant_path:
        raise ValueError(
            "convrot_prequant_path is only valid with "
            "convrot_weight_source=prequant_checkpoint"
        )

    raise_if_compiled(network, context="apply_convrot_to_lora_network")
    if unet is not None:
        raise_if_compiled(unet, context="apply_convrot_to_lora_network(unet)")

    patches: list[ConvRotLoRABaseForwardPatch] = []
    skipped: list[ConvRotLoRABaseForwardPatch] = []
    group_size = int(effective_group)
    min_in = max(0, int(min_in_features))

    candidates: list[tuple[nn.Module, nn.Linear, int, str, str]] = []
    for lora in getattr(network, "unet_loras", []) or []:
        original_name = getattr(lora, "original_name", None)
        if not original_name:
            continue
        classified = classify_convrot_linear_module(str(original_name), scope=scope)
        if classified is None:
            continue
        block_idx, family = classified
        lora_name = str(getattr(lora, "lora_name", original_name))

        if is_dora_module(lora):
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=(),
                    group_size=group_size,
                    mode=mode,
                    bf16_bytes=0,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason="dora_unsupported",
                )
            )
            continue

        base_module = _resolve_base_linear(lora)
        if base_module is None:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=(),
                    group_size=group_size,
                    mode=mode,
                    bf16_bytes=0,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason="no_org_module_ref",
                )
            )
            continue

        candidates.append(
            (lora, base_module, block_idx, family, str(original_name))
        )

    kept, size_skipped = _filter_candidates_by_size(
        candidates,
        min_in_features=min_in,
        largest_in_features_only=bool(largest_in_features_only),
    )
    for item in size_skipped:
        skipped.append(
            ConvRotLoRABaseForwardPatch(
                lora_name=item.lora_name,
                name=item.name,
                family=item.family,
                block_idx=item.block_idx,
                shape=item.shape,
                group_size=group_size,
                mode=mode,
                bf16_bytes=item.bf16_bytes,
                payload_bytes=0,
                weight_source=weight_source,
                skipped_reason=item.skipped_reason,
            )
        )

    # Share one Hadamard buffer across all modules with the same
    # (group_size, device, dtype, kind) — group=256 → 256×256 bf16 ≈ 128 KiB total.
    hadamard_share: dict[tuple[int, str, str, str], torch.Tensor] = {}
    # Incremental free-base accounting (per-module free inside the loop below).
    incremental_freed_bytes = 0
    incremental_freed_module_ids: set[int] = set()

    for lora, base_module, block_idx, family, original_name in kept:
        lora_name = str(getattr(lora, "lora_name", original_name))
        layer_mode = _resolve_layer_mode(
            default_mode=mode,
            in_features=int(base_module.in_features),
            large_layer_mode=large_mode_norm,
            large_min_in_features=large_min,
        )

        try:
            validate_base_linear_for_convrot(
                base_module, group_size=group_size, name=str(original_name)
            )
        except (TypeError, ValueError) as exc:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=tuple(int(d) for d in base_module.weight.shape),
                    group_size=group_size,
                    mode=layer_mode,
                    bf16_bytes=int(base_module.weight.numel()) * 2,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason=str(exc),
                )
            )
            continue

        weight = base_module.weight.detach()
        bf16_bytes = int(weight.numel()) * 2
        payload_bytes = int(weight.numel()) + int(weight.shape[0]) * 4
        patch = ConvRotLoRABaseForwardPatch(
            lora_name=lora_name,
            name=str(original_name),
            family=family,
            block_idx=block_idx,
            shape=tuple(int(d) for d in weight.shape),
            group_size=group_size,
            mode=layer_mode,
            bf16_bytes=bf16_bytes,
            payload_bytes=payload_bytes,
            weight_source=weight_source,
        )

        if dry_run:
            if weight_source == "prequant_checkpoint":
                try:
                    _resolve_payload(
                        weight_source=weight_source,
                        base_module=base_module,
                        original_name=str(original_name),
                        group_size=group_size,
                        prequant=prequant,
                    )
                except (KeyError, ValueError) as exc:
                    skipped.append(
                        ConvRotLoRABaseForwardPatch(
                            lora_name=lora_name,
                            name=str(original_name),
                            family=family,
                            block_idx=block_idx,
                            shape=patch.shape,
                            group_size=group_size,
                            mode=layer_mode,
                            bf16_bytes=bf16_bytes,
                            payload_bytes=0,
                            weight_source=weight_source,
                            skipped_reason=f"prequant:{exc}",
                        )
                    )
                    continue
            patches.append(patch)
            continue

        try:
            q, scale = _resolve_payload(
                weight_source=weight_source,
                base_module=base_module,
                original_name=str(original_name),
                group_size=group_size,
                prequant=prequant,
            )
        except (KeyError, ValueError) as exc:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=patch.shape,
                    group_size=group_size,
                    mode=layer_mode,
                    bf16_bytes=bf16_bytes,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason=f"prequant:{exc}",
                )
            )
            continue

        patches.append(patch)
        # W8A8: store [K,N] so torch._int_mm avoids per-step t().contiguous()
        # (same nbytes as [N,K]; W8A16 keeps [N,K] for dequant F.linear).
        if layer_mode == "w8a8":
            q_store = q.t().contiguous()
            setattr(lora, _ATTR_WEIGHT_LAYOUT, "kn")
        else:
            q_store = q.contiguous()
            setattr(lora, _ATTR_WEIGHT_LAYOUT, "nk")
        _set_swap_payload(base_module, _BUFFER_Q, q_store)
        # P1.10: W8A16 can store scale as bf16 on CUDA (dequant/(gy*scale) skip
        # a cast). W8A8 keeps float32 — bf16 scale alone pushed seed0 grad_rel
        # over the 5% full-ckpt gate (P1.11d diagnosis vs p18 3/3).
        if (
            layer_mode == "w8a16"
            and base_module.weight.device.type == "cuda"
        ):
            scale_dtype = torch.bfloat16
        else:
            scale_dtype = torch.float32
        _set_swap_payload(
            base_module,
            _BUFFER_SCALE,
            scale.to(dtype=scale_dtype).contiguous(),
        )
        setattr(lora, _ATTR_GROUP, group_size)
        setattr(lora, _ATTR_MODE, layer_mode)
        setattr(lora, _ATTR_WEIGHT_SOURCE, weight_source)
        _maybe_attach_hadamard(
            lora,
            group_size=group_size,
            device=base_module.weight.device,
            shared_cache=hadamard_share,
        )
        _install_org_forward(
            lora,
            payload_owner=base_module,
            mode=layer_mode,
            group_size=group_size,
        )
        # Free this base weight's bf16 storage as soon as its int8 payload is
        # installed. Batching the free until after the whole loop keeps the full
        # bf16 base AND every accumulated payload resident at once (a transient
        # apply peak of ~2.2GB on top of steady state). Freeing incrementally
        # bounds the overlap to one module. ``free_linear_weight_storage`` is
        # idempotent (0 for already-freed / meta), so shared Linears and the
        # trailing safety-net call below stay correct.
        if not dry_run and free_base_weights:
            freed_now = free_linear_weight_storage(base_module)
            if freed_now:
                incremental_freed_bytes += freed_now
                incremental_freed_module_ids.add(id(base_module))

    # Safety net: free any patched base weight missed above (e.g. a shared
    # Linear deduped out of the incremental path). ``free_linear_weight_storage``
    # is idempotent (0 bytes for already-freed / meta), so bytes never double
    # count; module count is deduped by base-module id across both paths.
    if not dry_run and free_base_weights and patches:
        free_stats = free_base_weights_for_patches(network, patches)
        incremental_freed_bytes += int(free_stats.get("freed_bytes", 0))
        incremental_freed_module_ids.update(
            id(lora.org_module_ref[0])
            for lora in getattr(network, "unet_loras", []) or []
            if getattr(lora, "org_module_ref", None)
            and isinstance(lora.org_module_ref[0], torch.nn.Linear)
            and is_base_weight_freed(lora.org_module_ref[0])
        )

    freed_modules = len(incremental_freed_module_ids)
    freed_bytes = incremental_freed_bytes

    result = ConvRotApplyResult(
        patches=patches,
        skipped=skipped,
        mode=mode,
        scope=scope,
        group_size=group_size,
        weight_source=weight_source,
        prequant_path=str(prequant_path) if prequant_path else None,
        freed_modules=freed_modules,
        freed_bytes=freed_bytes,
        min_in_features=min_in,
        largest_in_features_only=bool(largest_in_features_only),
        large_layer_mode=large_mode_norm,
        large_min_in_features=large_min,
    )

    if not dry_run and result.patched_count == 0 and not allow_zero_patches:
        skip_reasons = sorted(
            {item.skipped_reason or "unknown" for item in skipped}
        ) or ["no_matching_scope"]
        raise RuntimeError(
            "apply_convrot_to_lora_network: patched=0 "
            f"(mode={mode} scope={scope} group={group_size} "
            f"weight_source={weight_source} min_in={min_in} "
            f"largest_only={bool(largest_in_features_only)}); "
            f"skipped={result.skipped_count} reasons={skip_reasons}. "
            "Phase 1 requires plain LoRA modules with org_module_ref on scope targets."
        )

    mode_mix = ""
    if large_mode_norm and large_min is not None:
        n_large = sum(1 for p in patches if p.mode == large_mode_norm)
        mode_mix = f" large_mode={large_mode_norm}@{large_min}({n_large})"

    logger.info(
        "[convrot] mode=%s scope=%s group=%s source=%s patched=%d skipped=%d "
        "freed_modules=%d freed_mb=%.1f min_in=%d largest_only=%s%s dry_run=%s "
        "prequant=%s",
        mode,
        scope,
        group_size,
        weight_source,
        result.patched_count,
        result.skipped_count,
        freed_modules,
        freed_bytes / (1024 * 1024),
        min_in,
        bool(largest_in_features_only),
        mode_mix,
        dry_run,
        prequant_path,
    )
    return result
