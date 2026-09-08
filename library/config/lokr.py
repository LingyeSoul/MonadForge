"""Explicit LoKr kernel selection and LyCORIS 4.0 Triton shape contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

LoKrBackend = Literal["torch", "triton"]
LOKR_BACKENDS: tuple[LoKrBackend, ...] = ("torch", "triton")
LOKR_OBSOLETE_KEYS = frozenset({"bypass", "ypass", "use_triton", "apply_bypass"})


def parse_lokr_backend(value: object) -> LoKrBackend:
    if value == "torch":
        return "torch"
    if value == "triton":
        return "triton"
    raise ValueError(
        f"lokr_backend={value!r}: expected 'torch' or 'triton'; "
        "LoKr uses an explicit per-network backend, not LYCORIS_KERNEL_BACKEND."
    )


def validate_lokr_legacy_options(options: Mapping[str, object]) -> None:
    obsolete = sorted(LOKR_OBSOLETE_KEYS.intersection(options))
    if obsolete:
        raise ValueError(
            f"Unsupported LoKr options: {', '.join(obsolete)}. Remove these keys "
            "and select lokr_backend='torch' or 'triton'; linear LoKr always "
            "uses bypass mode."
        )


def validate_lokr_triton_shape(
    name: str, in_features: int, out_features: int, factor: int
) -> None:
    """Check reconstructed factors, independent of rank/full-matrix storage."""
    from lycoris.functional.general import factorization

    a, c = factorization(out_features, factor)
    b, d = factorization(in_features, factor)
    pads = tuple(max(16, 1 << (v - 1).bit_length()) for v in (a, b, c, d))
    if max(pads) > 128:
        raise ValueError(
            f"lokr_backend='triton', lokr_factor={factor}: {name} "
            f"(in_features={in_features}, out_features={out_features}) produces "
            f"padded Kronecker factors {pads}, exceeding the LyCORIS 4.0 "
            "Triton limit of 128 per axis. Select lokr_backend='torch' or "
            "change lokr_factor so every targeted layer fits (the standard "
            "Anima attention and MLP layers fit with lokr_factor=64). "
            "Changing network_dim or image resolution does not fix this limit."
        )
