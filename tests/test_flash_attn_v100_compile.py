from __future__ import annotations

import os

import pytest
import torch


def _v100_flash_available() -> bool:
    if os.environ.get("ANIMA_TEST_GPU") != "1" or not torch.cuda.is_available():
        return False
    try:
        import flash_attn.flash_attn_interface as interface
    except ImportError:
        return False
    return interface.flash_attn_func.__module__.startswith("flash_attn_v100.")


pytestmark = pytest.mark.skipif(
    not _v100_flash_available(),
    reason="requires ANIMA_TEST_GPU=1, CUDA, and flash-attention-v100",
)


def test_v100_flash_dense_inductor_fullgraph_dynamic_backward():
    from networks.attention_dispatch import (
        flash_attn_func,
        flash_attn_v100_compat_active,
    )

    assert flash_attn_v100_compat_active

    def attention_block(x):
        batch, _, seq_len, _, width = x.shape
        qkv = x.reshape(batch, seq_len, 2, width // 2)
        branch = flash_attn_func(qkv, qkv, qkv)
        branch = branch.reshape(batch, 1, seq_len, 1, width)
        return x.float() + branch.float()

    compiled = torch.compile(
        attention_block,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )
    for seq_len in (2048, 2160):
        x = torch.randn(
            1,
            1,
            seq_len,
            1,
            128,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        torch._dynamo.mark_dynamic(x, 2, min=2048, max=2160)
        output = compiled(x)
        output.square().mean().backward()
        assert torch.isfinite(output).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_v100_flash_varlen_inductor_fullgraph_backward():
    from networks.attention_dispatch import flash_attn_varlen_func

    def varlen(q, k, v, cu_seqlens):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            17,
            17,
        )

    compiled = torch.compile(varlen, backend="inductor", fullgraph=True)
    q = torch.randn(33, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = q.detach().clone().requires_grad_(True)
    v = q.detach().clone().requires_grad_(True)
    cu_seqlens = torch.tensor([0, 16, 33], device="cuda", dtype=torch.int32)
    output = compiled(q, k, v, cu_seqlens)
    output.float().square().mean().backward()

    assert output.shape == q.shape
    assert torch.isfinite(output).all()
    assert all(tensor.grad is not None for tensor in (q, k, v))
    assert all(torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


def test_v100_flash_attention_with_lse_zero_dropout_eager_and_fullgraph():
    from networks.attention_dispatch import attention_with_lse

    def with_lse(q):
        return attention_with_lse(q, q, q, attn_mode="flash")

    q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float16)
    compiled = torch.compile(with_lse, backend="inductor", fullgraph=True)
    for implementation in (with_lse, compiled):
        out, lse = implementation(q)
        assert out.shape == q.shape
        assert lse.shape == (q.shape[0], q.shape[2], q.shape[1])
        assert torch.isfinite(out).all()
        assert torch.isfinite(lse).all()
