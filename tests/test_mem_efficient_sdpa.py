from __future__ import annotations

from contextlib import contextmanager

import torch

from networks import attention_dispatch
from networks.attention_dispatch import AttentionParams, dispatch_attention
from networks.methods.easycontrol_attention import _extended_target_attention


def test_mem_efficient_mode_forces_efficient_backend(monkeypatch):
    selected = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        selected.append(backend)
        yield

    def fake_sdpa(q, k, v, **kwargs):
        assert q.shape == (1, 2, 4, 8)  # dispatcher converted BLHD -> BHLD
        assert k.shape == q.shape
        assert v.shape == q.shape
        assert kwargs["attn_mask"] is None
        assert kwargs["dropout_p"] == 0.0
        return q

    monkeypatch.setattr(attention_dispatch, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    q = torch.randn(1, 4, 2, 8)
    params = AttentionParams.create_attention_params("mem_efficient")
    assert params.supports_fp32 is False
    out = dispatch_attention([q, q, q], attn_params=params)

    assert selected == [attention_dispatch.SDPBackend.EFFICIENT_ATTENTION]
    assert out.shape == (1, 4, 16)


def test_mem_efficient_mode_normalizes_variable_length_mask(monkeypatch):
    selected = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        selected.append(backend)
        yield

    def fake_sdpa(q, k, v, **kwargs):
        mask = kwargs["attn_mask"]
        assert mask.dtype == torch.bool
        assert mask.shape == (2, 1, 1, 4)
        return q

    monkeypatch.setattr(attention_dispatch, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    params = AttentionParams.create_attention_params_from_mask(
        "mem_efficient", img_len=0, attention_mask=mask
    )
    q = torch.randn(2, 4, 2, 8)
    out = dispatch_attention([q, q, q], attn_params=params)

    assert selected == [attention_dispatch.SDPBackend.EFFICIENT_ATTENTION]
    assert out.shape == (2, 4, 16)


def test_easycontrol_mem_efficient_mode_forces_efficient_backend(monkeypatch):
    selected = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        selected.append(backend)
        yield

    def fake_sdpa(q, k, v, **kwargs):
        assert q.shape == (1, 2, 4, 8)
        assert k.shape == (1, 2, 7, 8)
        assert v.shape == k.shape
        assert kwargs["attn_mask"].shape == (1, 1, 1, 7)
        return q

    monkeypatch.setattr(attention_dispatch, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    target = torch.randn(1, 4, 2, 8)
    cond = torch.randn(1, 3, 2, 8)
    params = AttentionParams.create_attention_params("mem_efficient")
    out = _extended_target_attention(
        target,
        target,
        target,
        cond,
        cond,
        b_param=torch.tensor(0.0),
        scale=None,
        attn_params=params,
    )

    assert selected == [attention_dispatch.SDPBackend.EFFICIENT_ATTENTION]
    assert out.shape == (1, 4, 16)


def test_sdpa_kernel_context_is_compile_traceable():
    def attention(q, k, v):
        # MATH is available on CPU; tracing behavior is identical to selecting
        # EFFICIENT_ATTENTION on CUDA and keeps this invariant test GPU-free.
        with attention_dispatch.sdpa_kernel(attention_dispatch.SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q = torch.randn(1, 2, 4, 8)
    explanation = torch._dynamo.explain(attention)(q, q, q)

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
