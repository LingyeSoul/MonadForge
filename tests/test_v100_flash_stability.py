from __future__ import annotations

import types

import pytest
import torch

from networks.attention_dispatch import AttentionParams


@pytest.fixture(scope="module")
def train_mod():
    import train

    return train


def test_v100_flash_doc_detection(train_mod):
    mod = types.SimpleNamespace(
        __doc__="Flash Attention for Tesla V100 v2.8.3 (backend: v26.06)",
        __version__="2.8.3",
    )

    doc, is_v100 = train_mod._flash_attn_v100_doc(mod)

    assert is_v100 is True
    assert "Tesla V100" in doc


def test_v100_flash_stability_env_resolution(monkeypatch, train_mod):
    args = types.SimpleNamespace(v100_flash_stability=None)
    monkeypatch.setenv("ANIMA_V100_FLASH_STABILITY", "hybrid")

    assert train_mod._resolve_v100_flash_stability(args) == "hybrid"

    args.v100_flash_stability = "safe"
    assert train_mod._resolve_v100_flash_stability(args) == "safe"


def test_v100_flash_stability_invalid_env_falls_back(monkeypatch, train_mod, caplog):
    args = types.SimpleNamespace(v100_flash_stability=None)
    monkeypatch.setenv("ANIMA_V100_FLASH_STABILITY", "turbo")

    with caplog.at_level("WARNING"):
        resolved = train_mod._resolve_v100_flash_stability(args)

    assert resolved == "off"
    assert any("ANIMA_V100_FLASH_STABILITY" in rec.getMessage() for rec in caplog.records)


def test_hybrid_specializes_cross_attention_without_mutating_original():
    params = AttentionParams.create_attention_params(
        "flash",
        v100_flash_stability="hybrid",
        debug_finite_checks=True,
    )

    self_params = params.for_attention_kind(is_selfattn=True)
    cross_params = params.for_attention_kind(is_selfattn=False)

    assert self_params is params
    assert self_params.attn_mode == "flash"
    assert cross_params is not params
    assert cross_params.attn_mode == "torch"
    assert cross_params.v100_flash_stability == "hybrid"
    assert cross_params.debug_finite_checks is True
    assert params.attn_mode == "flash"


def test_safe_mode_keeps_flash_for_both_attention_kinds():
    params = AttentionParams.create_attention_params("flash", v100_flash_stability="safe")

    assert params.for_attention_kind(is_selfattn=True).attn_mode == "flash"
    assert params.for_attention_kind(is_selfattn=False).attn_mode == "flash"


def test_debug_finite_check_raises_on_nonfinite_tensor():
    from library.anima.models import _assert_finite_tensor

    x = torch.tensor([1.0, float("nan")])
    with pytest.raises(FloatingPointError, match="unit-test"):
        _assert_finite_tensor(x, "unit-test")
