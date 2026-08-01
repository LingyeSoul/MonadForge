from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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
    assert any(
        "ANIMA_V100_FLASH_STABILITY" in rec.getMessage() for rec in caplog.records
    )


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
    params = AttentionParams.create_attention_params(
        "flash", v100_flash_stability="safe"
    )

    assert params.for_attention_kind(is_selfattn=True).attn_mode == "flash"
    assert params.for_attention_kind(is_selfattn=False).attn_mode == "flash"


def test_debug_finite_check_raises_on_nonfinite_tensor():
    from library.anima.models import _assert_finite_tensor

    x = torch.tensor([1.0, float("nan")])
    with pytest.raises(FloatingPointError, match="unit-test"):
        _assert_finite_tensor(x, "unit-test")


def test_debug_finite_check_is_fullgraph_dynamic_shape_safe():
    from library.anima.models import _assert_finite_tensor

    graphs = []

    class Checked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._block_index = 1

        def forward(self, tensor):
            _assert_finite_tensor(tensor, "compiled-unit-test", block=self)
            return tensor * 2

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    checked = Checked()
    compiled = torch.compile(checked, backend=backend, fullgraph=True)
    for width in (3, 5):
        tensor = torch.ones(2, width)
        torch._dynamo.mark_dynamic(tensor, 1, min=1, max=8)
        assert torch.equal(compiled(tensor), tensor * 2)
        checked._block_index += 1

    assert len(graphs) == 1


def test_v100_public_api_import_survives_missing_private_wrappers(monkeypatch):
    public_func = object()
    public_varlen_func = object()
    facade = types.ModuleType("flash_attn")
    facade.__path__ = []
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_func = public_func
    interface.flash_attn_varlen_func = public_varlen_func
    monkeypatch.setitem(sys.modules, "flash_attn", facade)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", interface)

    module_name = "_v100_attention_dispatch_import_test"
    path = Path(__file__).parents[1] / "networks" / "attention_dispatch.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    assert module.flash_attn is facade
    assert module.flash_attn_func is public_func
    assert module.flash_attn_varlen_func is public_varlen_func
    assert module._flash_attn_forward is None
    assert module._wrapped_flash_attn_forward is None
    assert module._wrapped_flash_attn_backward is None


def test_v100_flash_provider_rejects_bf16_at_dispatch_boundary(monkeypatch):
    from networks import attention_dispatch

    monkeypatch.setattr(attention_dispatch, "flash_attn_func", object())
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", object())

    monkeypatch.setattr(attention_dispatch, "flash_attn_v100_provider", True)
    assert attention_dispatch.flash_attn_available_for_dtype(torch.float16)
    assert not attention_dispatch.flash_attn_available_for_dtype(torch.bfloat16)

    monkeypatch.setattr(attention_dispatch, "flash_attn_v100_provider", False)
    assert attention_dispatch.flash_attn_available_for_dtype(torch.bfloat16)
    assert not attention_dispatch.flash_attn_available_for_dtype(torch.float32)


def test_llm_adapter_v100_bf16_falls_back_to_sdpa(monkeypatch):
    from library.anima.models import LLMAdapterAttention
    from networks import attention_dispatch

    def forbidden_flash(*_args, **_kwargs):
        raise AssertionError("V100 Flash must not receive BF16 LLM-adapter tensors")

    sdpa_dtypes = []

    def fake_sdpa(q, _k, _v, attn_mask=None):
        assert attn_mask is not None
        sdpa_dtypes.append(q.dtype)
        return q

    monkeypatch.setattr(attention_dispatch, "flash_attn_v100_provider", True)
    monkeypatch.setattr(attention_dispatch, "flash_attn_func", forbidden_flash)
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", forbidden_flash)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    attention = LLMAdapterAttention(8, 8, n_heads=2, head_dim=4).to(
        dtype=torch.bfloat16
    )
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output = attention(x, q_mask=mask, kv_mask=mask)

    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert sdpa_dtypes == [torch.bfloat16]


def test_llm_adapter_v100_fp16_keeps_varlen_flash(monkeypatch):
    from library.anima.models import LLMAdapterAttention
    from networks import attention_dispatch

    flash_dtypes = []

    def fake_varlen(q, _k, _v, *_args, **_kwargs):
        flash_dtypes.append(q.dtype)
        return q

    def forbidden_sdpa(*_args, **_kwargs):
        raise AssertionError("V100 FP16 LLM-adapter tensors should keep using Flash")

    monkeypatch.setattr(attention_dispatch, "flash_attn_v100_provider", True)
    monkeypatch.setattr(attention_dispatch, "flash_attn_func", fake_varlen)
    monkeypatch.setattr(attention_dispatch, "flash_attn_varlen_func", fake_varlen)
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", forbidden_sdpa
    )

    attention = LLMAdapterAttention(8, 8, n_heads=2, head_dim=4).to(dtype=torch.float16)
    x = torch.randn(2, 3, 8, dtype=torch.float16)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output = attention(x, q_mask=mask, kv_mask=mask)

    assert output.shape == x.shape
    assert output.dtype == torch.float16
    assert flash_dtypes == [torch.float16]
