"""Unit tests for ConvRot group Regular / Sylvester Hadamard Transform."""

from __future__ import annotations

import pytest
import torch

from library.runtime.convrot.rht import (
    assert_group_divides,
    column_discrepancy,
    group_fwht,
    group_rht,
    group_rht_weight,
    hadamard_kind,
    is_power_of_four,
    is_power_of_two,
    normalized_hadamard,
    regular_hadamard,
    regular_hadamard_base4,
    sylvester_hadamard,
)


def test_is_power_of_two() -> None:
    assert is_power_of_two(1)
    assert is_power_of_two(64)
    assert is_power_of_two(256)
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(100)


def test_is_power_of_four() -> None:
    assert is_power_of_four(1)
    assert is_power_of_four(4)
    assert is_power_of_four(16)
    assert is_power_of_four(64)
    assert is_power_of_four(256)
    assert is_power_of_four(1024)
    assert not is_power_of_four(2)
    assert not is_power_of_four(8)
    assert not is_power_of_four(32)
    assert not is_power_of_four(100)


def test_assert_group_divides_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError, match="power of two"):
        assert_group_divides(256, 100, kind="sylvester")
    with pytest.raises(ValueError, match="power of four"):
        assert_group_divides(256, 32, kind="regular")
    with pytest.raises(ValueError, match="not divisible"):
        assert_group_divides(100, 64, kind="sylvester")


def test_normalized_sylvester_is_orthogonal_and_symmetric() -> None:
    for n in (2, 4, 8, 16, 64):
        h = normalized_hadamard(n, dtype=torch.float64, kind="sylvester")
        eye = torch.eye(n, dtype=torch.float64)
        assert torch.allclose(h @ h.T, eye, atol=1e-10)
        assert torch.allclose(h, h.T, atol=1e-12)


def test_normalized_regular_is_orthogonal_and_symmetric() -> None:
    for n in (4, 16, 64, 256):
        h = normalized_hadamard(n, dtype=torch.float64, kind="regular")
        eye = torch.eye(n, dtype=torch.float64)
        assert torch.allclose(h @ h.T, eye, atol=1e-9)
        assert torch.allclose(h, h.T, atol=1e-12)


def test_regular_has_minimal_column_discrepancy() -> None:
    # Paper: regular attains ||H^T 1||_∞ = √n; Sylvester has a column sum n.
    for n in (4, 16, 64, 256):
        reg = regular_hadamard(n, dtype=torch.float64)
        syl = sylvester_hadamard(n, dtype=torch.float64)
        d_reg = column_discrepancy(reg)
        d_syl = column_discrepancy(syl)
        assert abs(d_reg - (n**0.5)) < 1e-6, (n, d_reg)
        assert d_syl >= n - 1e-6  # all-ones column
        assert d_reg < d_syl


def test_regular_base4_entries_pm_one() -> None:
    h = regular_hadamard_base4()
    assert set(h.unique().tolist()) == {-1.0, 1.0}
    # each row/col sum abs == 2
    assert torch.allclose(h.sum(dim=1).abs(), torch.full((4,), 2.0))
    assert torch.allclose(h.sum(dim=0).abs(), torch.full((4,), 2.0))


def test_group_rht_sylvester_is_involutory() -> None:
    torch.manual_seed(0)
    group = 8
    x = torch.randn(3, 5, 32, dtype=torch.float64)
    y = group_rht(x, group, kind="sylvester")
    z = group_rht(y, group, kind="sylvester")
    assert y.shape == x.shape
    assert torch.allclose(z, x, atol=1e-6, rtol=1e-6)


def test_group_rht_regular_is_involutory() -> None:
    torch.manual_seed(1)
    group = 16
    x = torch.randn(2, 4, 64, dtype=torch.float64)
    y = group_rht(x, group, kind="regular")
    z = group_rht(y, group, kind="regular")
    assert torch.allclose(z, x, atol=1e-6, rtol=1e-6)


def test_group_rht_weight_matches_rowwise_feature_rht() -> None:
    torch.manual_seed(1)
    w = torch.randn(7, 16, dtype=torch.float64)
    group = 4
    rotated = group_rht_weight(w, group, kind="regular")
    for i in range(w.shape[0]):
        assert torch.allclose(
            rotated[i], group_rht(w[i], group, kind="regular"), atol=1e-10
        )


def test_group_rht_batch_shapes() -> None:
    torch.manual_seed(2)
    group = 16
    for shape in [(16,), (4, 16), (2, 3, 32), (1, 1, 64)]:
        x = torch.randn(*shape)
        y = group_rht(x, group, kind="sylvester")
        assert y.shape == x.shape


def test_sylvester_entries_are_pm_one() -> None:
    h = sylvester_hadamard(8, dtype=torch.float32)
    assert set(h.unique().tolist()) == {-1.0, 1.0}


def test_group_fwht_matches_dense_sylvester() -> None:
    torch.manual_seed(0)
    for g in (8, 16, 64):
        x = torch.randn(2, 3, g * 4, dtype=torch.float64)
        dense = group_rht(x, g, kind="sylvester")
        fast = group_fwht(x, g)
        rel = (fast - dense).norm() / dense.norm().clamp_min(1e-12)
        assert rel.item() < 1e-6


def test_hadamard_kind_env(monkeypatch) -> None:
    monkeypatch.delenv("ANIMA_CONVROT_HADAMARD", raising=False)
    assert hadamard_kind() == "sylvester"
    monkeypatch.setenv("ANIMA_CONVROT_HADAMARD", "regular")
    assert hadamard_kind() == "regular"


def test_regular_and_sylvester_differ_for_same_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float64)
    ys = group_rht(x, 16, kind="sylvester")
    yr = group_rht(x, 16, kind="regular")
    # Both orthogonal transforms of same x → same energy, different direction.
    assert torch.allclose(ys.norm(), yr.norm(), rtol=1e-6)
    rel = (ys - yr).norm() / ys.norm().clamp_min(1e-12)
    assert rel.item() > 0.1
