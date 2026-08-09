"""Metadata stamps and merge-policy helpers for ConvRot training."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

SS_BASE_COMPUTE = "ss_base_compute"
SS_CONVROT_GROUP_SIZE = "ss_convrot_group_size"
SS_CONVROT_SCOPE = "ss_convrot_scope"
SS_CONVROT_HADAMARD = "ss_convrot_hadamard"
SS_CONVROT_WEIGHT_SOURCE = "ss_convrot_weight_source"
SS_CONVROT_MODE = "ss_convrot_mode"

CONVROT_METADATA_KEYS = (
    SS_BASE_COMPUTE,
    SS_CONVROT_GROUP_SIZE,
    SS_CONVROT_SCOPE,
    SS_CONVROT_HADAMARD,
    SS_CONVROT_WEIGHT_SOURCE,
    SS_CONVROT_MODE,
)


def stamp_convrot_metadata(
    metadata: MutableMapping[str, Any],
    *,
    base_compute: str,
    group_size: int,
    scope: str,
    weight_source: str,
    mode: str | None = None,
    hadamard: str | None = None,
) -> None:
    """Write ConvRot ss_* keys into adapter metadata dict (in-place)."""
    metadata[SS_BASE_COMPUTE] = str(base_compute)
    metadata[SS_CONVROT_GROUP_SIZE] = str(int(group_size))
    metadata[SS_CONVROT_SCOPE] = str(scope)
    if hadamard is not None:
        kind = str(hadamard).strip().lower()
        if kind in {"regular", "reg", "paper", "convrot"}:
            kind = "regular"
        else:
            kind = "sylvester"
        metadata[SS_CONVROT_HADAMARD] = kind
    metadata[SS_CONVROT_WEIGHT_SOURCE] = str(weight_source)
    if mode is not None:
        metadata[SS_CONVROT_MODE] = str(mode)
    elif base_compute.endswith("_convrot"):
        # w8a16_convrot -> w8a16
        metadata[SS_CONVROT_MODE] = str(base_compute).removesuffix("_convrot")


def metadata_indicates_convrot(metadata: Mapping[str, Any] | None) -> bool:
    if not metadata:
        return False
    base = str(metadata.get(SS_BASE_COMPUTE, "bf16") or "bf16").strip().lower()
    if base in {"w8a16_convrot", "w8a8_convrot", "w8a16", "w8a8"}:
        return True
    mode = str(metadata.get(SS_CONVROT_MODE, "") or "").strip().lower()
    return mode in {"w8a16", "w8a8"}


def raise_if_merge_with_convrot(
    metadata: Mapping[str, Any] | None = None,
    *,
    base_compute: str | None = None,
    context: str = "merge",
) -> None:
    """Refuse merge/fuse when ConvRot base was used (default policy)."""
    active = False
    if base_compute is not None:
        text = str(base_compute).strip().lower()
        active = text not in {"", "bf16", "none", "off", "fp16"}
    if not active and metadata_indicates_convrot(metadata):
        active = True
    if active:
        raise RuntimeError(
            f"{context}: refused for ConvRot base_compute. "
            "Merge/fuse assumes high-precision writable Linear.weight. "
            "Dequantize base to bf16 / train with base_compute=bf16 before merge, "
            "or use a dedicated dequant+fold tool."
        )
