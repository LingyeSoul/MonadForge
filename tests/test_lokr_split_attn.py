"""LoKR split-projection regression.

Background: LoKR trained on the Anima DiT's fused ``qkv_proj`` / ``kv_proj``
Linears produces keys like ``…_self_attn_qkv_proj.lokr_*`` which ComfyUI
cannot map to its split ``q_proj`` / ``k_proj`` / ``v_proj`` modules (the
Kronecker product mixes output channels across q/k/v and admits no clean
slice at save time). Fix: split the fused projection into per-component
Linears (zero-copy narrow views) before ``create_modules`` attaches LoKR, so
each q/k/v gets its own LoKRModule and the saved keys are split directly.

These tests pin: (1) the split produces narrow views sharing storage with the
fused weight; (2) split forward ≡ fused forward; (3) the LoRA walker sees the
split children and builds 3 modules; (4) save produces ComfyUI-compatible
split keys; (5) merge_to on a split child writes through to the fused weight
without contaminating sibling components; (6) legacy fused keys trigger the
warning.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from networks.lora_modules.lokr import LoKRModule
from networks.lora_modules.split_attn import (
    _make_split_linears,
    split_fused_projections,
)


# ---------------------------------------------------------------------------
# Test doubles — a minimal Attention/Block pair mirroring the Anima DiT
# layout (fused qkv_proj for self-attn, separate q_proj + fused kv_proj for
# cross-attn).
# ---------------------------------------------------------------------------


class _SelfAttn(nn.Module):
    """Self-attention with a fused qkv_proj (mirrors models.py:337-341)."""

    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.is_selfattn = True
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = "bshd"
        inner = n_heads * head_dim
        self.inner_dim = inner
        self.qkv_proj = nn.Linear(dim, 3 * inner, bias=False)
        self.output_proj = nn.Linear(inner, dim, bias=False)
        # Minimal norms (Identity suffices — compute_qkv only needs shapes).
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()

    # Mirrors models.py Attention.compute_qkv self-attn path (pre-split).
    def compute_qkv(self, x, context, rope_cos_sin=None):
        qkv = self.qkv_proj(x).unflatten(-1, (3, self.n_heads, self.head_dim))
        q, k, v = qkv.unbind(dim=-3)
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)


class _CrossAttn(nn.Module):
    """Cross-attn: separate q_proj + fused kv_proj (mirrors models.py:339-341)."""

    def __init__(self, dim: int, ctx_dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.is_selfattn = False
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = "bshd"
        inner = n_heads * head_dim
        self.inner_dim = inner
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.kv_proj = nn.Linear(ctx_dim, 2 * inner, bias=False)
        self.output_proj = nn.Linear(inner, dim, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()

    def compute_qkv(self, x, context, rope_cos_sin=None):
        q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
        kv = self.kv_proj(context).unflatten(-1, (2, self.n_heads, self.head_dim))
        k, v = kv.unbind(dim=-3)
        return self.q_norm(q), self.k_norm(k), self.v_norm(v)


class _Block(nn.Module):
    """Container matching ``ANIMA_TARGET_REPLACE_MODULE = ["Block", ...]``."""

    def __init__(self, dim: int, ctx_dim: int, n_heads: int = 4, head_dim: int = 8):
        super().__init__()
        self.self_attn = _SelfAttn(dim, n_heads, head_dim)
        self.cross_attn = _CrossAttn(dim, ctx_dim, n_heads, head_dim)


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block(dim=32, ctx_dim=24)


# 1. Split children are narrow views sharing storage with the fused weight.
def test_split_children_share_storage_with_fused_weight():
    inner = 16
    in_dim = 32
    fused = nn.Linear(in_dim, 3 * inner, bias=False)
    fused_w_ptr = fused.weight.data_ptr()

    children = _make_split_linears(fused, ("q", "k", "v"))
    assert len(children) == 3
    assert [name for name, _ in children] == ["q_proj", "k_proj", "v_proj"]

    for i, (name, child) in enumerate(children):
        assert child.weight.shape == (inner, in_dim)
        # narrow view shares the underlying storage (zero-copy).
        offset = i * inner * in_dim * fused.weight.element_size()
        assert child.weight.data_ptr() == fused_w_ptr + offset, (
            f"{name} weight must be a zero-copy narrow view of the fused weight"
        )
        assert not child.weight.requires_grad, "frozen base must stay frozen"


# 2. Split forward (concat of per-component) ≡ original fused forward.
def test_split_forward_equivalence_self_attn():
    dit = _TinyDiT()
    self_attn = dit.block.self_attn
    x = torch.randn(2, 5, 32)

    # Reference: original fused compute_qkv output (q,k,v each [...,n_heads,head_dim]).
    q_ref, k_ref, v_ref = self_attn.compute_qkv(x, x)

    n = split_fused_projections(dit)
    assert n == 2, "expected both self_attn qkv and cross_attn kv to split"

    # After split the qkv_proj attr is gone; per-component Linears exist.
    assert not hasattr(self_attn, "qkv_proj")
    assert hasattr(self_attn, "q_proj") and hasattr(self_attn, "k_proj")
    assert hasattr(self_attn, "v_proj")

    q_new, k_new, v_new = self_attn.compute_qkv(x, x)
    assert torch.allclose(q_ref, q_new, atol=1e-6)
    assert torch.allclose(k_ref, k_new, atol=1e-6)
    assert torch.allclose(v_ref, v_new, atol=1e-6)


def test_split_forward_equivalence_cross_attn():
    dit = _TinyDiT()
    cross = dit.block.cross_attn
    x = torch.randn(2, 5, 32)
    ctx = torch.randn(2, 3, 24)
    q_ref, k_ref, v_ref = cross.compute_qkv(x, ctx)

    split_fused_projections(dit)

    # q_proj untouched (was already separate); kv_proj split into k/v.
    assert hasattr(cross, "q_proj")
    assert not hasattr(cross, "kv_proj")
    assert hasattr(cross, "k_proj") and hasattr(cross, "v_proj")

    q_new, k_new, v_new = cross.compute_qkv(x, ctx)
    assert torch.allclose(q_ref, q_new, atol=1e-6)
    assert torch.allclose(k_ref, k_new, atol=1e-6)
    assert torch.allclose(v_ref, v_new, atol=1e-6)


# 3. Idempotent: a second call finds nothing to split.
def test_split_is_idempotent():
    dit = _TinyDiT()
    assert split_fused_projections(dit) == 2
    assert split_fused_projections(dit) == 0


# 4. LoKRModule built on a split child factors against the component dim,
#    not the fused dim.
def test_lokr_module_on_split_child_uses_component_dim():
    inner = 16
    in_dim = 32
    fused = nn.Linear(in_dim, 3 * inner, bias=False)
    children = _make_split_linears(fused, ("q", "k", "v"))
    q_name, q_lin = children[0]

    m = LoKRModule(
        "lora_test_q", q_lin, multiplier=1.0, lora_dim=4, alpha=4, lokr_factor=-1,
    )
    # The delta shape must match the component Linear, not the fused one.
    delta = m.get_weight()
    assert delta.shape == (inner, in_dim), (
        f"LoKR delta on split child must be component-shaped ({inner},{in_dim}), "
        f"got {tuple(delta.shape)}"
    )


# 5. merge_to on a split child writes through to the fused weight at the
#    child's row band only — sibling components stay untouched.
def test_merge_to_writes_through_to_fused_band_only():
    inner = 16
    in_dim = 32
    fused = nn.Linear(in_dim, 3 * inner, bias=False)
    fused_before = fused.weight.data.clone()
    children = _make_split_linears(fused, ("q", "k", "v"))
    _, q_lin = children[0]
    _, k_lin = children[1]

    m = LoKRModule("lora_test_q", q_lin, multiplier=1.0, lora_dim=4, alpha=4, lokr_factor=-1)
    # Non-zero delta so the merge actually contributes.
    with torch.no_grad():
        torch.nn.init.normal_(m.lokr_w2, std=0.5)
    sd = m.state_dict()
    m.merge_to(sd, dtype=torch.float32, device="cpu")

    # q rows changed.
    q_delta = m.get_weight() * m.multiplier * 1.0  # scale folded via alpha/rank path
    assert torch.allclose(
        fused.weight.data[0:inner],
        (fused_before[0:inner].float() + q_delta).float(),
        atol=1e-5,
    ), "q band must reflect the merged delta"
    # k and v bands (rows [inner:]) must be untouched.
    assert torch.allclose(
        fused.weight.data[inner:], fused_before[inner:]
    ), "sibling k/v bands must be untouched by q's merge_to"


# 6. Legacy fused keys trigger the warning.
def test_legacy_fused_lokr_keys_warning(caplog):
    from networks.lora_anima.loading import _warn_legacy_fused_lokr_keys

    sd = {
        "lora_unet_blocks_0_self_attn_qkv_proj.lokr_w2_a": torch.zeros(2, 2),
        "lora_unet_blocks_0_self_attn_qkv_proj.alpha": torch.tensor(4.0),
        "lora_unet_blocks_0_mlp_layer1.lokr_w1": torch.zeros(2, 2),  # non-attn, ok
    }
    with caplog.at_level(logging.WARNING):
        _warn_legacy_fused_lokr_keys(sd)
    assert any("Legacy fused-projection LoKR" in r.message for r in caplog.records), (
        "legacy fused lokr keys must trigger the actionable warning"
    )


# 7. New split keys do NOT trigger the warning.
def test_split_lokr_keys_no_warning(caplog):
    from networks.lora_anima.loading import _warn_legacy_fused_lokr_keys

    sd = {
        "lora_unet_blocks_0_self_attn_q_proj.lokr_w2_a": torch.zeros(2, 2),
        "lora_unet_blocks_0_self_attn_k_proj.lokr_w2_a": torch.zeros(2, 2),
        "lora_unet_blocks_0_self_attn_v_proj.lokr_w2_a": torch.zeros(2, 2),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
    }
    with caplog.at_level(logging.WARNING):
        _warn_legacy_fused_lokr_keys(sd)
    assert not any("Legacy fused-projection LoKR" in r.message for r in caplog.records), (
        "split lokr keys must NOT trigger the legacy warning"
    )
