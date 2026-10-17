"""Invariants for the cross-attn per-key logit bias (``_ctx_k_bias``).

The buffer backs the frontload_text_boost arm-(d) allocation probe
(bench/frontload_text_boost/). Three things must hold:

1. bias = None (the default) is EXACT identity — the buffer is never read
   into the math, so every path rendered before it existed is unchanged.
2. bias = b reproduces softmax(QK^T·scale + b)V exactly (the SDPA additive
   mask lands in the logits, post QK-norm, pre-softmax).
3. self-attention modules don't grow the buffer, and set_xattn_kbias
   reaches every block's cross_attn (set AND clear).
"""

import math
import types

import torch

from library.anima.models import Attention
from networks.attention_dispatch import AttentionParams
from library.inference.adapters import set_xattn_kbias


def _make_attn(seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    return Attention(query_dim=16, context_dim=12, n_heads=2, head_dim=8).eval()


def _params() -> AttentionParams:
    return AttentionParams.create_attention_params("torch")


def _manual(attn: Attention, x, ctx, bias=None) -> torch.Tensor:
    """Reference softmax attention from the module's own q/k/v."""
    q, k, v = attn.compute_qkv(x, ctx)  # [B, L, H, D]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, L, D]
    logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(attn.head_dim)
    if bias is not None:
        logits = logits + bias.float()[None, None, None, :]
    out = torch.softmax(logits, dim=-1) @ v.float()
    out = out.to(v.dtype).transpose(1, 2).flatten(2)
    return attn.output_proj(out)


def test_default_none_is_exact_identity():
    attn = _make_attn()
    assert attn._ctx_k_bias is None
    torch.manual_seed(1)
    x, ctx = torch.randn(2, 5, 16), torch.randn(2, 7, 12)
    with torch.no_grad():
        out = attn(x, _params(), ctx)
        attn._ctx_k_bias = torch.zeros(7)
        out_zero_bias = attn(x, _params(), ctx)
        attn._ctx_k_bias = None
        out_cleared = attn(x, _params(), ctx)
    # zero bias is mathematically identity (kernel choice may differ)…
    assert torch.allclose(out, out_zero_bias, atol=1e-5)
    # …and clearing restores the exact original path.
    assert torch.equal(out, out_cleared)


def test_bias_matches_manual_softmax():
    attn = _make_attn()
    torch.manual_seed(2)
    x, ctx = torch.randn(2, 5, 16), torch.randn(2, 7, 12)
    bias = torch.tensor([0.0, -0.7, -0.7, 0.0, -1.4, 0.0, 0.0])
    with torch.no_grad():
        attn._ctx_k_bias = bias
        out = attn(x, _params(), ctx)
        expected = _manual(attn, x, ctx, bias=bias)
    assert torch.allclose(out, expected, atol=1e-5)


def test_large_negative_bias_excludes_key():
    """−inf-ish bias on one key ≈ that key removed from the context."""
    attn = _make_attn()
    torch.manual_seed(3)
    x, ctx = torch.randn(1, 4, 16), torch.randn(1, 6, 12)
    bias = torch.zeros(6)
    bias[3] = -1e9
    with torch.no_grad():
        attn._ctx_k_bias = bias
        out = attn(x, _params(), ctx)
        expected = _manual(attn, x, ctx, bias=bias)
    assert torch.allclose(out, expected, atol=1e-5)


def test_selfattn_has_no_buffer():
    torch.manual_seed(0)
    attn = Attention(query_dim=16, context_dim=None, n_heads=2, head_dim=8)
    assert not hasattr(attn, "_ctx_k_bias")


def test_setter_reaches_every_block_and_clears():
    blocks = [types.SimpleNamespace(cross_attn=_make_attn(i)) for i in range(3)]
    model = types.SimpleNamespace(blocks=blocks)
    bias = torch.randn(7)
    set_xattn_kbias(model, bias)
    assert all(b.cross_attn._ctx_k_bias is bias for b in blocks)
    set_xattn_kbias(model, None)
    assert all(b.cross_attn._ctx_k_bias is None for b in blocks)
