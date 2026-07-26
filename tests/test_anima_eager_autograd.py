"""Regression tests for eager-only Anima fusion helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from library.anima.eager_autograd import (
    _lokr_owner,
    eager_fused_lora_mlp_tensors,
    eager_fused_lokr_mlp_tensors,
    eager_rotary_qk,
)
from networks.lora_modules.lokr import LoKRModule


def _lora_linear_reference(
    x,
    base_weight,
    down_weight,
    up_weight,
    inv_scale,
    rank_mask,
    residual_scale,
):
    rank_x = x.float()
    if inv_scale.numel() != 0:
        rank_x = rank_x * inv_scale.float()
    rank = F.linear(rank_x, down_weight.float()) * rank_mask.float()
    delta = F.linear(rank, up_weight.float()) * residual_scale
    return F.linear(x.to(base_weight.dtype), base_weight) + delta.to(base_weight.dtype)


def test_fused_lora_mlp_forward_and_grads_match_reference():
    torch.manual_seed(20)
    rows, d_model, d_ff, rank = 11, 7, 13, 3
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    inv1 = torch.rand(d_model) + 0.5
    inv2 = torch.rand(d_ff) + 0.5
    base1 = torch.randn(d_ff, d_model)
    base2 = torch.randn(d_model, d_ff)
    grad_out = torch.randn(rows, d_model)

    initial = (
        torch.randn(rows, d_model),
        torch.randn(rank, d_model),
        torch.randn(d_ff, rank),
        torch.randn(rank, d_ff),
        torch.randn(d_model, rank),
    )

    def run(fused):
        x, down1, up1, down2, up2 = [
            tensor.clone().requires_grad_() for tensor in initial
        ]
        if fused:
            out = eager_fused_lora_mlp_tensors(
                x,
                base1,
                down1,
                up1,
                inv1,
                mask,
                0.75,
                base2,
                down2,
                up2,
                inv2,
                mask,
                1.25,
                chunk_size=4,
            )
        else:
            hidden = F.gelu(
                _lora_linear_reference(
                    x, base1, down1, up1, inv1, mask, 0.75
                )
            )
            out = _lora_linear_reference(
                hidden, base2, down2, up2, inv2, mask, 1.25
            )
        grads = torch.autograd.grad(out, (x, down1, up1, down2, up2), grad_out)
        return out, grads

    expected_out, expected_grads = run(False)
    actual_out, actual_grads = run(True)

    assert torch.allclose(actual_out, expected_out, rtol=1e-5, atol=2e-6)
    for actual, expected in zip(actual_grads, expected_grads):
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_fused_lora_mlp_does_not_save_full_d_ff_activations():
    torch.manual_seed(21)
    rows, d_model, d_ff, rank = 9, 5, 17, 2
    x = torch.randn(rows, d_model, requires_grad=True)
    base1 = torch.randn(d_ff, d_model)
    base2 = torch.randn(d_model, d_ff)
    down1 = torch.randn(rank, d_model, requires_grad=True)
    up1 = torch.randn(d_ff, rank, requires_grad=True)
    down2 = torch.randn(rank, d_ff, requires_grad=True)
    up2 = torch.randn(d_model, rank, requires_grad=True)
    empty_inv = torch.ones(rank)[:0]
    mask = torch.ones(1, rank)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        eager_fused_lora_mlp_tensors(
            x,
            base1,
            down1,
            up1,
            empty_inv,
            mask,
            1.0,
            base2,
            down2,
            up2,
            empty_inv,
            mask,
            1.0,
            chunk_size=3,
        )

    assert (rows, d_model) in saved_shapes
    assert (rows, d_ff) not in saved_shapes


def _lokr_factors(layout, up, uq, vp, vq):
    rank = 2
    if layout == "full":
        return (
            torch.randn(up, uq, requires_grad=True),
            None,
            None,
            torch.randn(vp, vq, requires_grad=True),
            None,
            None,
        )
    if layout == "w2_decomposed":
        return (
            torch.randn(up, uq, requires_grad=True),
            None,
            None,
            None,
            torch.randn(vp, rank, requires_grad=True),
            torch.randn(rank, vq, requires_grad=True),
        )
    if layout == "w1_decomposed":
        return (
            None,
            torch.randn(up, rank, requires_grad=True),
            torch.randn(rank, uq, requires_grad=True),
            torch.randn(vp, vq, requires_grad=True),
            None,
            None,
        )
    return (
        None,
        torch.randn(up, rank, requires_grad=True),
        torch.randn(rank, uq, requires_grad=True),
        None,
        torch.randn(vp, rank, requires_grad=True),
        torch.randn(rank, vq, requires_grad=True),
    )


def _lokr_linear_reference(
    x,
    base_weight,
    factors,
    scalar,
    residual_scale,
):
    w1, w1a, w1b, w2, w2a, w2b = factors
    c = w1.float() if w1 is not None else w1a.float().matmul(w1b.float())
    grouped = x.float().reshape(x.shape[0], c.shape[1], -1)
    if w2 is not None:
        projected = F.linear(grouped, w2.float())
    else:
        projected = F.linear(
            F.linear(grouped, w2b.float()),
            w2a.float(),
        )
    delta = F.linear(projected.transpose(-1, -2), c)
    delta = delta.transpose(-1, -2).reshape(x.shape[0], -1)
    delta = delta * scalar.float() * residual_scale
    return F.linear(x.to(base_weight.dtype), base_weight) + delta.to(
        base_weight.dtype
    )


@pytest.mark.parametrize(
    "layout",
    ["full", "w2_decomposed", "w1_decomposed", "decompose_both"],
)
def test_fused_lokr_mlp_forward_and_grads_match_reference(layout):
    torch.manual_seed(23)
    rows, d_model, d_ff = 11, 12, 18
    base1 = torch.randn(d_ff, d_model)
    base2 = torch.randn(d_model, d_ff)
    grad_out = torch.randn(rows, d_model)

    def run(fused):
        x = torch.randn(rows, d_model, requires_grad=True)
        factors1 = _lokr_factors(layout, 3, 3, 6, 4)
        factors2 = _lokr_factors(layout, 3, 3, 4, 6)
        scalar1 = torch.tensor(0.625, requires_grad=True)
        scalar2 = torch.tensor(0.875, requires_grad=True)
        if fused:
            out = eager_fused_lokr_mlp_tensors(
                x,
                base1,
                *factors1,
                scalar1,
                0.75,
                base2,
                *factors2,
                scalar2,
                1.25,
                chunk_size=4,
            )
        else:
            hidden = F.gelu(
                _lokr_linear_reference(
                    x,
                    base1,
                    factors1,
                    scalar1,
                    0.75,
                )
            )
            out = _lokr_linear_reference(
                hidden,
                base2,
                factors2,
                scalar2,
                1.25,
            )
        requested = (
            x,
            *(factor for factor in factors1 if factor is not None),
            scalar1,
            *(factor for factor in factors2 if factor is not None),
            scalar2,
        )
        return out, torch.autograd.grad(out, requested, grad_out)

    torch.manual_seed(24)
    expected_out, expected_grads = run(False)
    torch.manual_seed(24)
    actual_out, actual_grads = run(True)

    assert torch.allclose(actual_out, expected_out, rtol=1e-5, atol=2e-6)
    for actual, expected in zip(actual_grads, expected_grads):
        assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_fused_lokr_mlp_does_not_save_full_d_ff_activations():
    torch.manual_seed(25)
    rows, d_model, d_ff = 9, 12, 18
    x = torch.randn(rows, d_model, requires_grad=True)
    base1 = torch.randn(d_ff, d_model)
    base2 = torch.randn(d_model, d_ff)
    factors1 = _lokr_factors("full", 3, 3, 6, 4)
    factors2 = _lokr_factors("full", 3, 3, 4, 6)
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        eager_fused_lokr_mlp_tensors(
            x,
            base1,
            *factors1,
            torch.tensor(1.0),
            1.0,
            base2,
            *factors2,
            torch.tensor(1.0),
            1.0,
            chunk_size=3,
        )

    assert (rows, d_model) in saved_shapes
    assert (rows, d_ff) not in saved_shapes


def test_lokr_mlp_owner_accepts_full_factor_without_decomposed_attributes():
    base = torch.nn.Linear(8, 12, bias=False)
    module = LoKRModule(
        "m",
        base,
        lora_dim=2,
        alpha=2,
        lokr_factor=2,
        full_factor=True,
    )
    module.fp32_compute = True
    module.use_custom_down_autograd = True
    module.train()
    module.apply_to()

    assert _lokr_owner(base) is module


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
def test_eager_rotary_qk_matches_formula_and_gradients(layout):
    torch.manual_seed(22)
    seq, batch, heads, dim = 7, 2, 3, 8
    shape = (seq, batch, heads, dim) if layout == "sbhd" else (batch, seq, heads, dim)
    seq_axis = 0 if layout == "sbhd" else 1
    rope_shape = (seq, 1, 1, dim) if layout == "sbhd" else (1, seq, 1, dim)
    angles = torch.randn(rope_shape)
    cos = angles.cos()
    sin = angles.sin()
    q0 = torch.randn(shape)
    k0 = torch.randn(shape)
    grad_q = torch.randn(shape)
    grad_k = torch.randn(shape)

    q_ref = q0.clone().requires_grad_()
    k_ref = k0.clone().requires_grad_()
    out_q_ref = q_ref * cos + _rotate_half(q_ref) * sin
    out_k_ref = k_ref * cos + _rotate_half(k_ref) * sin
    expected_grads = torch.autograd.grad(
        (out_q_ref, out_k_ref), (q_ref, k_ref), (grad_q, grad_k)
    )

    # The Function intentionally mutates fresh non-leaf RMSNorm outputs.
    q_leaf = q0.clone().requires_grad_()
    k_leaf = k0.clone().requires_grad_()
    q = q_leaf + 0.0
    k = k_leaf + 0.0
    out_q, out_k = eager_rotary_qk(
        q,
        k,
        cos,
        sin,
        seq_axis=seq_axis,
        rot_dim=dim,
        chunk_size=3,
    )
    actual_grads = torch.autograd.grad((out_q, out_k), (q_leaf, k_leaf), (grad_q, grad_k))

    assert torch.equal(out_q, out_q_ref)
    assert torch.equal(out_k, out_k_ref)
    assert torch.equal(actual_grads[0], expected_grads[0])
    assert torch.equal(actual_grads[1], expected_grads[1])
