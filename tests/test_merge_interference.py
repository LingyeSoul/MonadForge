"""Invariant tests for LoRA merge weight-space interference analysis.

Covers the three reference regimes of ``library.anima.merge_analysis.analyze``:
identical deltas (cos +1, ratio →N²/N), negated deltas (cos -1, cancellation),
and orthogonal deltas (cos 0, ratio 1). Also checks the Gram-trace inner product
matches a dense ``⟨ΔW_i, ΔW_j⟩_F`` and that ``--weights`` scaling is respected.
"""

from __future__ import annotations

import torch

from library.anima import merge_analysis as ma


def _lora(down: torch.Tensor, up: torch.Tensor, stem: str = "blk.0.attn"):
    """Build a one-module loaded-LoRA tuple with alpha == rank (scale 1)."""
    r = down.shape[0]
    return ({stem: down}, {stem: up}, {stem: float(r)})


def _dense_inner(up_i, down_i, up_j, down_j) -> float:
    return torch.tensordot(up_i @ down_i, up_j @ down_j, dims=2).item()


def test_gram_inner_matches_dense():
    torch.manual_seed(0)
    up_i, down_i = torch.randn(8, 4), torch.randn(4, 6)
    up_j, down_j = torch.randn(8, 4), torch.randn(4, 6)
    got = ma._gram_inner(up_i, down_i, up_j, down_j)
    assert abs(got - _dense_inner(up_i, down_i, up_j, down_j)) < 1e-4


def test_identical_loras_constructive():
    torch.manual_seed(1)
    down, up = torch.randn(4, 6), torch.randn(8, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, up)], ["a", "b"])
    assert rep.pair_cosine[(0, 1)] == 1.0 or abs(rep.pair_cosine[(0, 1)] - 1.0) < 1e-5
    # Two identical deltas: ‖2ΔW‖² / 2‖ΔW‖² = 4/2 = 2.
    assert abs(rep.overall_energy_ratio - 2.0) < 1e-4
    assert rep.n_shared_modules == 1


def test_negated_loras_destructive():
    torch.manual_seed(2)
    down, up = torch.randn(4, 6), torch.randn(8, 4)
    rep = ma.analyze([_lora(down, up), _lora(down, -up)], ["a", "b"])
    assert abs(rep.pair_cosine[(0, 1)] - (-1.0)) < 1e-5
    # Perfect cancellation: ‖ΔW - ΔW‖² = 0.
    assert abs(rep.overall_energy_ratio) < 1e-4
    assert rep.worst_pair[0] == (0, 1)


def test_orthogonal_loras_independent():
    # Disjoint output rows → ⟨ΔW_i, ΔW_j⟩ = 0 regardless of down.
    down = torch.randn(4, 6)
    up_a = torch.zeros(8, 4)
    up_a[:4] = torch.randn(4, 4)
    up_b = torch.zeros(8, 4)
    up_b[4:] = torch.randn(4, 4)
    rep = ma.analyze([_lora(down, up_a), _lora(down, up_b)], ["a", "b"])
    assert abs(rep.pair_cosine[(0, 1)]) < 1e-5
    assert abs(rep.overall_energy_ratio - 1.0) < 1e-4


def test_weight_scaling_changes_energy_not_cosine():
    torch.manual_seed(3)
    down, up = torch.randn(4, 6), torch.randn(8, 4)
    loaded = [_lora(down, up), _lora(down, -up)]
    base = ma.analyze(loaded, ["a", "b"])
    scaled = ma.analyze(loaded, ["a", "b"], weights=[1.0, 0.5])
    # Cosine is scale-invariant; still anti-aligned.
    assert abs(scaled.pair_cosine[(0, 1)] - base.pair_cosine[(0, 1)]) < 1e-5
    # But partial cancellation now leaves residual energy (ratio > 0).
    assert scaled.overall_energy_ratio > 0.1
