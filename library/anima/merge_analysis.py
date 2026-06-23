"""Weight-space interference analysis for multi-LoRA merges.

When ``scripts/merge_loras.py`` fuses N adapters it sums their weight-deltas
``ΔW_i = (α_i/r_i)·up_i @ down_i``.  The energy of that sum is **not** just the
sum of energies — it carries pairwise cross-terms::

    ‖Σ_i ΔW_i‖²_F = Σ_i ‖ΔW_i‖²_F  +  2·Σ_{i<j} ⟨ΔW_i, ΔW_j⟩_F
                                       └──────── interference ────────┘

- ``⟨ΔW_i, ΔW_j⟩ > 0`` → **constructive**: the two LoRAs push the same
  direction in weight space (reinforcing — risk of overdrive into high-freq
  noise once summed).
- ``< 0`` → **destructive**: they partially cancel — one LoRA erodes the
  other's learned effect.
- ``≈ 0`` → orthogonal/independent — the ``--normalize global`` √N-quadrature
  assumption holds.

The merge script already derives a normalization scale *assuming* orthogonality
but never reports the actual interference, so two stylistically-opposed LoRAs
get silently cancelled with no diagnostic.  This module measures it exactly and
cheaply via the Gram-trace identity — no ``out×in`` ΔW is ever materialized
(mirrors ``merge_loras.fro2``)::

    ⟨ΔW_i, ΔW_j⟩_F = Σ( (up_iᵀ up_j) ⊙ (down_i down_jᵀ) )

The per-LoRA scale ``(α/r)·w`` is folded into the up factor before measuring, so
the energy-weighted aggregate reflects what the merge actually writes (pairwise
*cosine* is scale-invariant, but the per-module weighting of the global cosine
is not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import torch

# Loaded-LoRA tuple as returned by ``scripts/merge_loras.load_lora``:
# (downs, ups, alphas) keyed by module stem.
LoadedLoRA = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]

_EPS = 1e-12


def _scaled_up(up: torch.Tensor, down: torch.Tensor, alpha: float | None, w: float):
    """``up`` with ``(α/r)·w`` folded in — matches ``merge_loras`` line ``ups_cat``."""
    r = down.shape[0]
    s = (alpha if alpha is not None else float(r)) / r
    return up * (s * w)


def _gram_inner(up_i, down_i, up_j, down_j) -> float:
    """⟨up_i@down_i, up_j@down_j⟩_F via the Gram-trace identity (no out×in)."""
    return ((up_i.T @ up_j) * (down_i @ down_j.T)).sum().item()


@dataclass
class InterferenceReport:
    """Pairwise + aggregate weight-space interference for a multi-LoRA merge."""

    names: list[str]
    weights: list[float]
    n_inputs: int
    n_modules: int  # union of module stems
    n_shared_modules: int  # stems present in ≥2 inputs (interference only defined here)
    # Global energy-weighted cosine per input pair, keyed by (i, j) with i < j.
    pair_cosine: dict[tuple[int, int], float]
    # ‖Σ ΔW‖² / Σ‖ΔW_i‖² over the whole weight space. 1 = orthogonal,
    # >1 = net constructive, <1 = net destructive.
    overall_energy_ratio: float
    # Per-module energy-excess index (2·Σ_{i<j}⟨⟩ / Σ‖·‖²): <0 destructive layer.
    module_index: dict[str, float] = field(default_factory=dict)

    @property
    def worst_pair(self) -> tuple[tuple[int, int], float] | None:
        """The input pair with the most negative (destructive) cosine, if any."""
        if not self.pair_cosine:
            return None
        return min(self.pair_cosine.items(), key=lambda kv: kv[1])

    def _verdict(self, ratio: float) -> str:
        if ratio > 1.05:
            return "constructive"
        if ratio < 0.95:
            return "destructive"
        return "orthogonal"

    def summary_line(self) -> str:
        v = self._verdict(self.overall_energy_ratio)
        worst = self.worst_pair
        tail = ""
        if worst is not None:
            (i, j), c = worst
            tail = f" — most opposed: {self.names[i]}↔{self.names[j]} cos={c:+.3f}"
        return (
            f"interference: energy ratio {self.overall_energy_ratio:.3f} ({v}), "
            f"{self.n_shared_modules}/{self.n_modules} shared modules{tail}"
        )


def analyze(
    loaded: list[LoadedLoRA],
    names: list[str],
    weights: list[float] | None = None,
) -> InterferenceReport:
    """Compute weight-space interference across ``loaded`` LoRAs.

    ``loaded`` / ``names`` / ``weights`` are parallel lists (one entry per input
    adapter); ``loaded[k]`` is the ``(downs, ups, alphas)`` tuple from
    ``merge_loras.load_lora``. ``weights`` defaults to all-1.0.
    """
    n = len(loaded)
    if n < 2:
        raise ValueError("interference analysis needs at least 2 LoRAs")
    if weights is None:
        weights = [1.0] * n

    # Pre-scale every module's up factor by (α/r)·w once.
    scaled: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
    for (downs, ups, alphas), w in zip(loaded, weights):
        per: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for stem, down in downs.items():
            if stem not in ups:
                continue
            per[stem] = (_scaled_up(ups[stem], down, alphas.get(stem), w), down)
        scaled.append(per)

    stems = sorted({s for per in scaled for s in per})

    # Global accumulators: per-input energy N_i, per-pair inner product S_ij.
    norm_sq = [0.0] * n
    inner = {pair: 0.0 for pair in combinations(range(n), 2)}
    module_index: dict[str, float] = {}
    n_shared = 0

    for stem in stems:
        present = [k for k in range(n) if stem in scaled[k]]
        # Per-module energy bookkeeping for the local interference index.
        m_norm = {}
        for k in present:
            up_k, down_k = scaled[k][stem]
            nrm = _gram_inner(up_k, down_k, up_k, down_k)
            m_norm[k] = nrm
            norm_sq[k] += nrm
        m_cross = 0.0
        for i, j in combinations(present, 2):
            up_i, down_i = scaled[i][stem]
            up_j, down_j = scaled[j][stem]
            s = _gram_inner(up_i, down_i, up_j, down_j)
            inner[(i, j)] += s
            m_cross += s
        if len(present) >= 2:
            n_shared += 1
            denom = sum(m_norm.values()) + _EPS
            module_index[stem] = 2.0 * m_cross / denom

    pair_cosine = {}
    for (i, j), s in inner.items():
        denom = (norm_sq[i] * norm_sq[j]) ** 0.5 + _EPS
        pair_cosine[(i, j)] = s / denom

    total_norm = sum(norm_sq)
    total_cross = sum(inner.values())
    overall_energy_ratio = (total_norm + 2.0 * total_cross) / (total_norm + _EPS)

    return InterferenceReport(
        names=list(names),
        weights=list(weights),
        n_inputs=n,
        n_modules=len(stems),
        n_shared_modules=n_shared,
        pair_cosine=pair_cosine,
        overall_energy_ratio=overall_energy_ratio,
        module_index=module_index,
    )


def format_report(report: InterferenceReport, *, top_modules: int = 8) -> str:
    """Human-readable multi-line interference report for CLI / GUI log output."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("LoRA merge — weight-space interference analysis")
    lines.append("=" * 64)
    for k, (nm, w) in enumerate(zip(report.names, report.weights)):
        lines.append(f"  [{k}] {nm}  (weight {w:g})")
    lines.append("")
    lines.append(report.summary_line())
    lines.append(
        "  energy ratio = ‖Σ ΔW‖² / Σ‖ΔW_i‖²  "
        "(1.0 = orthogonal, >1 reinforce, <1 cancel)"
    )
    lines.append("")

    lines.append("Pairwise interference (global energy-weighted cosine):")
    for (i, j), c in sorted(report.pair_cosine.items()):
        if c > 0.05:
            tag = "constructive"
        elif c < -0.05:
            tag = "destructive"
        else:
            tag = "orthogonal"
        lines.append(f"  {report.names[i]} ↔ {report.names[j]}: {c:+.3f}  ({tag})")
    lines.append("")

    if report.module_index:
        worst = sorted(report.module_index.items(), key=lambda kv: kv[1])[:top_modules]
        lines.append(f"Most destructive modules (of {report.n_shared_modules} shared):")
        any_neg = False
        for stem, idx in worst:
            if idx < 0:
                any_neg = True
                lines.append(f"  {idx:+.3f}  {stem}")
        if not any_neg:
            lines.append("  (none — no module has net cancellation)")
    lines.append("=" * 64)
    return "\n".join(lines)
