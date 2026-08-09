"""Group-wise Regular Hadamard Transform (RHT) for ConvRot.

Supports two Hadamard constructions (normalized so ``R^{-1} = R^{T}``):

* **sylvester** (default, power-of-two orders): recursive
  ``H_{2n} = [[H_n, H_n], [H_n, -H_n]]``. Equivalent to FWHT. First column is
  all-ones and can amplify DiT row-wise outliers (ConvRot paper §3.2).
* **regular** (paper-aligned, orders ``4^k``): Kronecker
  ``H_{4^{k+1}} = H_{4^k} ⊗ H_4`` from a base regular ``H_4`` with minimal
  column discrepancy ``√n``. Select with ``ANIMA_CONVROT_HADAMARD=regular``.

Dense matmul backend is default (``ANIMA_CONVROT_RHT=dense``); FWHT only
matches the **sylvester** construction.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

_HADAMARD_CACHE: dict[tuple[str, int, str, torch.dtype], torch.Tensor] = {}
_RHT_ENV = "ANIMA_CONVROT_RHT"
_HADAMARD_KIND_ENV = "ANIMA_CONVROT_HADAMARD"

HadamardKind = Literal["sylvester", "regular"]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def is_power_of_four(n: int) -> bool:
    if not is_power_of_two(n):
        return False
    # n = 2^k with k even  ⇔  n = 4^{k/2}
    k = n.bit_length() - 1
    return k % 2 == 0


def assert_group_divides(
    in_features: int,
    group_size: int,
    *,
    kind: HadamardKind | None = None,
) -> None:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    kind = kind or hadamard_kind()
    if kind == "regular":
        if not is_power_of_four(group_size):
            raise ValueError(
                f"regular Hadamard group_size must be a power of four "
                f"(4^k, e.g. 4/16/64/256/1024), got {group_size}"
            )
    elif not is_power_of_two(group_size):
        raise ValueError(
            f"convrot group_size must be a power of two, got {group_size}"
        )
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )


def hadamard_kind() -> HadamardKind:
    raw = str(os.environ.get(_HADAMARD_KIND_ENV, "sylvester") or "sylvester").strip().lower()
    if raw in {"regular", "reg", "convrot", "paper"}:
        return "regular"
    return "sylvester"


def sylvester_hadamard(n: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard matrix of order ``n`` (entries ±1)."""
    if not is_power_of_two(n):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = torch.ones((1, 1), dtype=dtype)
    size = 1
    while size < n:
        top = torch.cat([h, h], dim=1)
        bottom = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bottom], dim=0)
        size *= 2
    return h


def regular_hadamard_base4(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unnormalized regular Hadamard of order 4 (row/col sums = ±2 = ±√4).

    One standard regular H_4 (three +1 and one −1 per row/column)::

        [[+1, +1, +1, -1],
         [+1, +1, -1, +1],
         [+1, -1, +1, +1],
         [-1, +1, +1, +1]]
    """
    return torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
    )


def regular_hadamard(n: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unnormalized regular Hadamard of order ``n = 4^k`` via Kronecker of H_4.

    Matches ConvRot Theorem 3.3: ``H_{4^{k+1}} = H_{4^k} ⊗ H_4``.
    Each row/column sum has absolute value ``√n`` (minimal column discrepancy).
    """
    if not is_power_of_four(n):
        raise ValueError(
            f"regular Hadamard order must be 4^k, got {n}"
        )
    h = regular_hadamard_base4(dtype=dtype)
    size = 4
    while size < n:
        h = torch.kron(h, regular_hadamard_base4(dtype=dtype))
        size *= 4
    return h


def unnormalized_hadamard(
    n: int,
    *,
    kind: HadamardKind | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    kind = kind or hadamard_kind()
    if kind == "regular":
        return regular_hadamard(n, dtype=dtype)
    return sylvester_hadamard(n, dtype=dtype)


def column_discrepancy(h: torch.Tensor) -> float:
    """``||H^T 1||_∞`` for unnormalized ±1 Hadamard."""
    return float(h.to(torch.float64).sum(dim=0).abs().max().item())


def normalized_hadamard(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    kind: HadamardKind | None = None,
) -> torch.Tensor:
    """Normalized Hadamard: ``H / sqrt(n)``, orthogonal (and symmetric for these constructions)."""
    kind = kind or hadamard_kind()
    assert_group_divides(n, n, kind=kind)
    key = (kind, n, str(device) if device is not None else "cpu", dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h = unnormalized_hadamard(n, kind=kind, dtype=torch.float32)
    h = h.mul_(1.0 / (n**0.5)).to(dtype=dtype)
    if device is not None:
        h = h.to(device=device)
    _HADAMARD_CACHE[key] = h
    return h


def clear_hadamard_cache() -> None:
    _HADAMARD_CACHE.clear()


def rht_backend() -> str:
    """Return active RHT backend: ``dense`` (default) or ``fwht`` (sylvester only)."""
    raw = str(os.environ.get(_RHT_ENV, "dense") or "dense").strip().lower()
    if raw in {"fwht", "fast", "fht"}:
        return "fwht"
    return "dense"


def group_rht(
    x: torch.Tensor,
    group_size: int,
    *,
    hadamard: torch.Tensor | None = None,
    kind: HadamardKind | None = None,
) -> torch.Tensor:
    """Apply group-wise normalized RHT on the last dimension of ``x``."""
    if x.shape[-1] == 0:
        return x
    kind = kind or hadamard_kind()
    assert_group_divides(int(x.shape[-1]), group_size, kind=kind)
    # FWHT matches Sylvester only.
    if hadamard is None and rht_backend() == "fwht":
        if kind != "sylvester":
            raise ValueError(
                "ANIMA_CONVROT_RHT=fwht only supports sylvester Hadamard; "
                "use dense backend with regular, or set ANIMA_CONVROT_HADAMARD=sylvester"
            )
        return group_fwht(x, group_size)

    *leading, dim = x.shape
    n_groups = dim // group_size
    work_dtype = x.dtype if x.is_floating_point() else torch.float32
    if hadamard is None:
        h = normalized_hadamard(
            group_size, device=x.device, dtype=work_dtype, kind=kind
        )
    else:
        if tuple(hadamard.shape) != (group_size, group_size):
            raise ValueError(
                f"hadamard must be [{group_size}, {group_size}], got {tuple(hadamard.shape)}"
            )
        # Avoid redundant host/device + dtype copies on the hot path.
        if hadamard.device == x.device and hadamard.dtype == work_dtype:
            h = hadamard
        else:
            h = hadamard.to(device=x.device, dtype=work_dtype)
    grouped = x.reshape(*leading, n_groups, group_size)
    if not grouped.is_floating_point():
        grouped = grouped.to(work_dtype)
    rotated = torch.matmul(grouped, h)
    return (
        rotated.reshape(*leading, dim).to(dtype=x.dtype)
        if x.is_floating_point()
        else rotated.reshape(*leading, dim)
    )


def group_fwht(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Group-wise Fast Walsh–Hadamard Transform (normalized, **Sylvester** order)."""
    if x.shape[-1] == 0:
        return x
    assert_group_divides(int(x.shape[-1]), group_size, kind="sylvester")
    *leading, dim = x.shape
    n_groups = dim // group_size
    work_dtype = torch.float32
    work = x.to(work_dtype).reshape(-1, n_groups, group_size)
    out = work.clone()
    h = 1
    while h < group_size:
        paired = out.view(-1, n_groups, group_size // (2 * h), 2, h)
        a = paired[..., 0, :]
        b = paired[..., 1, :]
        s = a + b
        d = a - b
        paired = torch.stack((s, d), dim=-2)
        out = paired.reshape(-1, n_groups, group_size)
        h *= 2
    scale = group_size**-0.5
    out = out.mul_(scale)
    out = out.reshape(*leading, dim)
    if x.is_floating_point() and x.dtype != work_dtype:
        return out.to(dtype=x.dtype)
    return out


def group_rht_weight(
    weight: torch.Tensor,
    group_size: int,
    *,
    hadamard: torch.Tensor | None = None,
    kind: HadamardKind | None = None,
) -> torch.Tensor:
    """Rotate a Linear weight ``[out, in]`` on the in-feature axis."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D weight, got shape={tuple(weight.shape)}")
    return group_rht(weight, group_size, hadamard=hadamard, kind=kind)
