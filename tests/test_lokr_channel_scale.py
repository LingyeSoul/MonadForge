"""Channel-scaling regression for LoKR.

LoKR's Kronecker factorization ``ΔW = kron(w1, w2)`` cannot absorb a
full-length ``channel_scale`` (length ``in_features``) into ``w1`` whose
in-axis is ``in_a`` — a *factor* of ``in_features``. The previous code
called ``_register_channel_scale(lokr_w1, channel_scale)`` which asserted
``channel_scale.shape[0] == weight.shape[1]`` (2048 != 8 in practice) and
crashed at construction; and even where it didn't crash the ``get_weight`` /
``_reconstruct_delta_from_sd`` paths multiplied ``w1`` by a length-``in_features``
``inv_scale``, broadcasting/semantics-broken.

Fix: LoKR does NOT absorb channel_scale into the factors. It registers
``inv_scale`` for forward-time input rebalancing (``_rebalance`` does
``x * inv_scale`` at the full ``in_features`` length), and applies the
mathematically equivalent input-column scaling only after materializing a full
Kron delta for get_weight/merge/fuse. These tests pin all affected paths.
"""

from __future__ import annotations

import torch

from networks.lora_modules.lokr import LoKRModule


def _make_calibration(in_dim: int, seed: int = 0) -> torch.Tensor:
    """Synthetic mean_abs vector with one dominant channel (matches the real
    DC-bias-outlier bench profile)."""
    torch.manual_seed(seed)
    stats = torch.rand(in_dim, dtype=torch.float32) * 0.3 + 0.05
    stats[in_dim // 2] = 50.0  # the dominant channel
    return stats


def _build(
    in_dim=12,
    out_dim=20,
    rank=4,
    calibration=None,
    decompose_both=False,
    lokr_factor=-1,
) -> tuple[torch.nn.Linear, LoKRModule]:
    """Build a LoKR module + its base Linear.

    Dims chosen so ``_factorization`` splits cleanly and both factors stay
    full (``lokr_w1`` / ``lokr_w2``) at ``rank=4``: in_a*in_b=3*4, out_a*out_b=4*5.
    """
    torch.manual_seed(123)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    m = LoKRModule(
        "lora_test_block",
        base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
        decompose_both=decompose_both,
        lokr_factor=lokr_factor,
    )
    return base, m


# 1. Construction does not crash, buffer is correct length.
def test_construction_with_channel_scale_registers_inv_scale():
    in_dim = 12
    calibration = _make_calibration(in_dim)
    _, m = _build(in_dim=in_dim, calibration=calibration)

    assert m._has_channel_scale
    assert "inv_scale" in dict(m.named_buffers())
    assert m.inv_scale.shape[0] == in_dim, (
        "inv_scale must span full in_features for forward _rebalance"
    )


# 2. Forward numerics: _rebalance path is the only place channel_scale acts,
#    and it acts at full in_features length (correct under Kron reshape).
def test_forward_applies_rebalance_correctly():
    in_dim, out_dim = 12, 20
    calibration = _make_calibration(in_dim)
    base, m = _build(in_dim=in_dim, out_dim=out_dim, calibration=calibration)
    # Non-zero w2 so the adapter contributes (init zeros lokr_w2).
    with torch.no_grad():
        torch.nn.init.normal_(m.lokr_w2, std=0.1)
    m.apply_to()
    m.eval()

    x = torch.randn(2, 5, in_dim) * 4.0
    x[..., in_dim // 2] += 30.0  # exercise the dominant channel

    with torch.no_grad():
        out = m(x)
        org = m.org_forward(x)

    # Manually reconstruct the expected adapter output:
    # x_r = x * inv_scale; reshape (..., in_a, in_b); apply the row-major
    # kron identity: (x @ kron(w1, w2).T).reshape(..., out_a, out_b)
    # == w1 @ X @ w2.T.
    inv = m.inv_scale.to(x)
    x_r = x * inv
    w1 = m.lokr_w1.to(x)
    w2 = m.lokr_w2.to(x)
    out_a, in_a = w1.shape
    out_b, in_b = w2.shape
    x_mat = x_r.reshape(*x_r.shape[:-1], in_a, in_b)
    delta = torch.einsum("oi,...ij,bj->...ob", w1, x_mat, w2)
    lx = delta.reshape(*x_r.shape[:-1], out_a * out_b)
    expected = org + lx * m.multiplier * m.scale

    assert torch.allclose(out, expected, atol=1e-5), (
        "LoKR forward diverged from manual kron(_rebalance(x)) reconstruction"
    )

    # The efficient forward path must match the materialized get_weight used by
    # fuse/merge, where inv_scale is expressed as input-column scaling and
    # therefore applies to raw x.
    kron_expected = org + (x @ m.get_weight().to(x).t())
    assert torch.allclose(out, kron_expected, atol=1e-5), (
        "LoKR forward must match get_weight() applied to raw inputs"
    )


# 3. get_weight must express forward-time rebalance as full-delta column scaling.
def test_get_weight_applies_inv_scale_to_materialized_delta_only():
    in_dim, out_dim = 12, 20
    calibration = _make_calibration(in_dim)
    _, m = _build(in_dim=in_dim, out_dim=out_dim, calibration=calibration)

    w1_before = m.lokr_w1.detach().clone()
    w2_before = m.lokr_w2.detach().clone()
    raw_kron = torch.kron(m.lokr_w1.float(), m.lokr_w2.float())
    expected = raw_kron * m.inv_scale.float().unsqueeze(0) * m.scale * m.multiplier

    w = m.get_weight()
    assert w.shape == (out_dim, in_dim)
    assert torch.allclose(w, expected, atol=1e-6), (
        "get_weight must materialize the x*inv_scale effect as input-column scaling"
    )
    assert torch.allclose(m.lokr_w1, w1_before), "w1 factor must not be absorbed"
    assert torch.allclose(m.lokr_w2, w2_before), "w2 factor must not be absorbed"


# 4. Resume round-trip: ones-placeholder rebuild + load_state_dict restores forward.
def test_resume_roundtrip_preserves_forward():
    in_dim, out_dim = 12, 20
    calibration = _make_calibration(in_dim)
    base, m = _build(in_dim=in_dim, out_dim=out_dim, calibration=calibration)
    with torch.no_grad():
        torch.nn.init.normal_(m.lokr_w2, std=0.1)
    m.apply_to()
    m.eval()

    sd = m.state_dict()
    assert "inv_scale" in sd, "inv_scale must persist for resume"

    x = torch.randn(2, 5, in_dim) * 4.0
    with torch.no_grad():
        out_orig = m(x)

    # Simulate factory resume: rebuild with a ones placeholder (length in_dim)
    # then load_state_dict overwrites inv_scale. This used to crash because
    # _register_channel_scale asserted channel_scale.len == w1.in_a.
    placeholder = torch.ones(in_dim)
    _, m2 = _build(in_dim=in_dim, out_dim=out_dim, calibration=placeholder)
    missing, unexpected = m2.load_state_dict(sd, strict=False)
    assert "inv_scale" not in missing, "inv_scale was not restored on resume"
    m2.apply_to()
    m2.eval()
    with torch.no_grad():
        out_new = m2(x)
    assert torch.allclose(out_new, out_orig, atol=1e-6), (
        "resumed LoKR forward diverged from original"
    )


# 5. merge_to injects the same materialized delta as get_weight.
def test_merge_to_matches_get_weight():
    in_dim, out_dim = 12, 20
    calibration = _make_calibration(in_dim)
    base, m = _build(in_dim=in_dim, out_dim=out_dim, calibration=calibration)
    with torch.no_grad():
        torch.nn.init.normal_(m.lokr_w2, std=0.1)

    sd = m.state_dict()
    # Snapshot the pristine base weight, then merge.
    w0 = base.weight.detach().clone()
    m.merge_to(sd, dtype=torch.float32, device="cpu")
    merged = base.weight.detach().clone()

    expected_delta = m.get_weight()
    assert torch.allclose(merged - w0, expected_delta, atol=1e-5), (
        "merge_to delta must equal get_weight, including inv_scale column scaling"
    )


def test_merge_to_preserves_forward_behavior():
    in_dim, out_dim = 12, 20
    calibration = _make_calibration(in_dim)
    base, m = _build(in_dim=in_dim, out_dim=out_dim, calibration=calibration)
    with torch.no_grad():
        torch.nn.init.normal_(m.lokr_w2, std=0.1)
    m.org_forward = base.forward
    m.eval()

    x = torch.randn(2, 5, in_dim) * 4.0
    with torch.no_grad():
        expected = m(x)

    # Keep org_module intact (no apply_to) so merge_to can inject into the raw
    # Linear weight, then compare the raw base output after merge.
    m.merge_to(m.state_dict(), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        actual = base(x)

    assert torch.allclose(actual, expected, atol=1e-5), (
        "merged LoKR weight must preserve the unfused rebalanced forward"
    )


def test_fuse_unfuse_preserves_forward_and_restores_weight():
    in_dim, out_dim = 12, 20
    calibration = _make_calibration(in_dim)
    base, m = _build(in_dim=in_dim, out_dim=out_dim, calibration=calibration)
    with torch.no_grad():
        torch.nn.init.normal_(m.lokr_w2, std=0.1)
    m.apply_to()
    m.eval()

    x = torch.randn(2, 5, in_dim) * 4.0
    w0 = base.weight.detach().clone()
    with torch.no_grad():
        expected = m(x)

    m.fuse_weight()
    with torch.no_grad():
        fused = base(x)
    assert torch.allclose(fused, expected, atol=1e-5), (
        "fused LoKR weight must preserve the unfused rebalanced forward"
    )

    m.unfuse_weight()
    assert torch.allclose(base.weight, w0, atol=1e-6), (
        "unfuse_weight must restore the original base weight"
    )
