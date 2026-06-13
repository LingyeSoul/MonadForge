"""Tests for the BYG straight-through clean/one-step blend (Eq. 4)."""

from __future__ import annotations

import pytest
import torch

from library.training.forward.ste import ste_clean_blend


def _latent(b=2, c=4, h=3, w=5, **kw):
    return torch.randn(b, c, 1, h, w, **kw)


def test_forward_value_equals_clean():
    y0 = _latent()
    y1 = _latent()
    out = ste_clean_blend(y0, y1)
    assert torch.allclose(out, y0), "forward value must equal the clean estimate"


def test_gradient_routes_only_through_onestep():
    y0 = _latent(requires_grad=True)
    y1 = _latent(requires_grad=True)
    out = ste_clean_blend(y0, y1)
    out.pow(2).sum().backward()
    # Gradient must flow through y_onestep and NOT through y0_clean (stop-grad).
    assert y0.grad is None, "no gradient should reach the clean estimate"
    assert y1.grad is not None and torch.any(y1.grad != 0)
    # d(out)/d(y1) is identity, so grad == d(loss)/d(out) == 2*out == 2*y0.
    assert torch.allclose(y1.grad, 2.0 * y0.detach())


def test_rejects_4d():
    with pytest.raises(ValueError, match="5D"):
        ste_clean_blend(torch.randn(2, 4, 3, 5), torch.randn(2, 4, 3, 5))


def test_rejects_nonsingleton_dim2():
    with pytest.raises(ValueError, match="dim 2"):
        ste_clean_blend(torch.randn(2, 4, 2, 3, 5), torch.randn(2, 4, 2, 3, 5))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="match shape"):
        ste_clean_blend(_latent(h=3, w=5), _latent(h=4, w=5))
