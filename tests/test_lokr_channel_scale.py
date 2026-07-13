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

import pytest
import torch
from safetensors import safe_open

from networks.lora_anima.factory import create_network, create_network_from_weights
from networks.lora_modules.lokr import LoKRModule
from networks.lora_save import (
    _convert_lokr_to_native_lokr,
    _convert_lokr_to_standard_lora,
)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(512, 512, bias=False)


class _TinyDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = Block()


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


# --- Effective rank control tests ---


def test_decomposition_respects_lora_dim():
    """lora_dim < min(out_b, in_b) must trigger w2 decomposition."""
    # 1024×1024, lokr_factor=-1 → factorization (32, 32)
    # min(32, 32) = 32, so lora_dim=4 should decompose w2
    base = torch.nn.Linear(1024, 1024, bias=False)
    m = LoKRModule("test", base, lora_dim=4, alpha=4, lokr_factor=-1)
    assert not m._use_w2, "w2 must be decomposed when lora_dim < min(out_b, in_b)"
    assert hasattr(m, "w2a") and hasattr(m, "w2b")
    assert m.w2a.shape == (32, 4), f"w2a shape should be (32, 4), got {m.w2a.shape}"
    assert m.w2b.shape == (4, 32), f"w2b shape should be (4, 32), got {m.w2b.shape}"


def test_decomposition_full_rank_when_lora_dim_large():
    """lora_dim >= min(out_b, in_b) keeps w2 full."""
    base = torch.nn.Linear(1024, 1024, bias=False)
    m = LoKRModule("test", base, lora_dim=32, alpha=32, lokr_factor=-1)
    assert m._use_w2, "w2 must be full when lora_dim >= min(out_b, in_b)"
    assert m.lokr_w2.shape == (32, 32)


def test_full_factor_is_independent_from_lora_dim_and_keeps_unit_scale():
    """Full-factor mode must not need an oversized lora_dim sentinel."""
    base = torch.nn.Linear(2048, 2048, bias=False)
    m = LoKRModule(
        "test",
        base,
        lora_dim=32,
        alpha=32,
        lokr_factor=8,
        full_factor=True,
    )
    assert m._use_w1 and m._use_w2
    assert m.lokr_w1.shape == (8, 8)
    assert m.lokr_w2.shape == (256, 256)
    assert m.scale == 1.0


def test_without_full_factor_same_shape_uses_rank_32_w2():
    base = torch.nn.Linear(2048, 2048, bias=False)
    m = LoKRModule("test", base, lora_dim=32, alpha=32, lokr_factor=8)
    assert not m._use_w2
    assert m.w2a.shape == (256, 32)
    assert m.w2b.shape == (32, 256)


def test_effective_rank_scales_with_lora_dim():
    """Effective rank should scale with lora_dim, not be fixed at full rank.

    LoKR's effective rank = rank(w1) * rank(w2).  When w2 is decomposed as
    w2a(lora_dim) @ w2b(lora_dim), its rank is lora_dim; when full, it's
    min(out_b, in_b).  We check the parameter shapes directly rather than
    calling _get_w2() (which returns the product w2a@w2b at full shape).
    """
    base = torch.nn.Linear(1024, 1024, bias=False)
    # _factorization(1024, -1) → (32, 32), so w1 is always (32, 32) rank 32.
    configs = [
        (4, False, 32 * 4),  # lora_dim=4 → w2 decomposed, eff_rank=128
        (8, False, 32 * 8),  # lora_dim=8 → w2 decomposed, eff_rank=256
        (16, False, 32 * 16),  # lora_dim=16 → w2 decomposed, eff_rank=512
        (32, True, 32 * 32),  # lora_dim=32 → w2 full, eff_rank=1024
    ]
    for lora_dim, expect_full_w2, expected_eff_rank in configs:
        m = LoKRModule("test", base, lora_dim=lora_dim, alpha=lora_dim, lokr_factor=-1)
        assert m._use_w2 is expect_full_w2, (
            f"lora_dim={lora_dim}: expected _use_w2={expect_full_w2}"
        )
        # w1 is always full (32, 32)
        r1 = min(m.lokr_w1.shape[0], m.lokr_w1.shape[1])
        # w2 rank: full → min(shape), decomposed → lora_dim
        if m._use_w2:
            r2 = min(m.lokr_w2.shape[0], m.lokr_w2.shape[1])
        else:
            r2 = lora_dim
        eff_rank = r1 * r2
        assert eff_rank == expected_eff_rank, (
            f"lora_dim={lora_dim}: expected eff_rank={expected_eff_rank}, got {eff_rank}"
        )


# ---------------------------------------------------------------------------
# Save-pipeline regression: scale correctness in the LoKR → standard LoRA
# (SVD) and native-lokr conversion paths. These pin the black-image fix.
# ---------------------------------------------------------------------------


def _trained_lokr_state_dict(
    lora_dim: int, alpha: float, network_dim: int, with_inv_scale: bool, seed: int = 0
):
    """Build a trained-looking LoKR state_dict and its ground-truth delta.

    Returns ``(sd, true_delta)`` where ``sd`` keys are prefixed with ``P.`` and
    ``true_delta`` is the full materialized delta the saved file must reproduce
    under ComfyUI's load formula.
    """
    torch.manual_seed(seed)
    base = torch.nn.Linear(512, 512, bias=False)
    m = LoKRModule("test", base, lora_dim=lora_dim, alpha=alpha, lokr_factor=-1)
    if with_inv_scale:
        cs = torch.rand(512) * 0.5 + 0.5
        m._register_lokr_inv_scale(cs, 512)
    # Simulate training: non-zero w2.
    with torch.no_grad():
        if m._use_w2:
            m.lokr_w2.normal_(0, 0.1)
        else:
            m.w2b.normal_(0, 0.1)
    m.apply_to()
    true_delta = m.get_weight()
    sd = {f"P.{k}": v.clone() for k, v in m.state_dict().items()}
    return sd, true_delta, m


def test_svd_conversion_scale_correctness_with_inv_scale():
    """SVD-to-standard-lora must reproduce the trained delta including scale
    and inv_scale. Verifies the ComfyUI formula (alpha/rank)*(up@down) matches
    get_weight() — the core black-image fix."""
    sd, true_delta, m = _trained_lokr_state_dict(
        lora_dim=32, alpha=32, network_dim=32, with_inv_scale=True
    )
    # network_dim=alpha → scale=1 here; the dim != alpha case is covered below.
    _convert_lokr_to_standard_lora(sd, dtype=torch.float32, network_dim=32)

    up = sd["P.lora_up.weight"]
    down = sd["P.lora_down.weight"]
    alpha = sd["P.alpha"].item()
    rank = up.shape[1]
    # ComfyUI's standard LoRA load formula.
    comfy_delta = (alpha / rank) * (up @ down) * m.multiplier
    rel = (true_delta - comfy_delta).abs().max().item() / max(
        true_delta.abs().max().item(), 1e-12
    )
    # Full-rank SVD in fp32 has ~1e-3 relative error on these magnitudes.
    assert rel < 1e-3, f"SVD delta diverges from truth: rel={rel:.2e}"
    # alpha must equal rank so ComfyUI's scale term is 1.0 (factors carry scale).
    assert alpha == rank, f"alpha={alpha} must equal rank={rank} for scale=1.0"


def test_svd_conversion_large_network_dim_scale():
    """The real black-image case: network_dim=114514 (Prodigy), alpha=32.
    scale = 32/114514 ≈ 2.8e-4. Without the fix, alpha was overwritten to rank
    and ComfyUI computed scale=1.0 → delta blown up ~3579x → black image."""
    sd, true_delta, m = _trained_lokr_state_dict(
        lora_dim=114514, alpha=32, network_dim=114514, with_inv_scale=True
    )
    _convert_lokr_to_standard_lora(
        sd, dtype=torch.float32, network_dim=114514, lora_rank=0
    )
    up = sd["P.lora_up.weight"]
    down = sd["P.lora_down.weight"]
    alpha = sd["P.alpha"].item()
    rank = up.shape[1]
    comfy_delta = (alpha / rank) * (up @ down) * m.multiplier
    rel = (true_delta - comfy_delta).abs().max().item() / max(
        true_delta.abs().max().item(), 1e-12
    )
    assert rel < 1e-3, f"large-dim scale bug: rel={rel:.2e}"


def test_svd_conversion_drops_factor_keys():
    """After SVD conversion, no lokr_* / w*a / inv_scale keys survive."""
    sd, _, _ = _trained_lokr_state_dict(
        lora_dim=32, alpha=32, network_dim=32, with_inv_scale=True
    )
    _convert_lokr_to_standard_lora(sd, dtype=torch.float32, network_dim=32)
    for k in sd:
        assert not any(
            k.endswith(s)
            for s in (
                ".lokr_w1",
                ".lokr_w2",
                ".lokr_w1_a",
                ".lokr_w2_a",
                ".w1a",
                ".w2a",
                ".w1b",
                ".w2b",
                ".inv_scale",
            )
        ), f"leftover factor key after SVD conversion: {k}"
    assert any(k.endswith(".lora_down.weight") for k in sd)
    assert any(k.endswith(".lora_up.weight") for k in sd)


def test_native_lokr_decomposed_raw_factors_restore_scale():
    """Decomposed w2 path (LyCORIS convention): factors saved RAW and alpha kept
    as the training alpha, so ComfyUI's load formula ``kron(w1, w2a@w2b) *
    (alpha/rank) * multiplier`` reproduces ``get_weight()``.

    Uses alpha=16, lora_dim=4 → training scale = 16/4 = 4.0 (a non-trivial
    scale, unlike the old alpha==dim test which silently exercised a 1.0 no-op).
    """
    torch.manual_seed(0)
    base = torch.nn.Linear(512, 512, bias=False)
    # lora_dim=4 < min(out_b,in_b) → w2 decomposed; w1 stays full.
    m = LoKRModule("test", base, lora_dim=4, alpha=16, lokr_factor=-1)
    with torch.no_grad():
        m.w2b.normal_(0, 0.1)
    true_delta = m.get_weight()  # = kron(w1,w2) * scale * multiplier
    sd = {f"P.{k}": v.clone() for k, v in m.state_dict().items()}
    assert not any(k.endswith(".inv_scale") for k in sd), (
        "fixture must have no inv_scale"
    )
    raw_w2a = sd["P.w2a"].clone()
    raw_w2b = sd["P.w2b"].clone()

    _convert_lokr_to_native_lokr(sd, dtype=torch.float32, network_dim=4)

    # ComfyUI-expected key names present; no internal-naming leaks.
    assert "P.lokr_w1" in sd, "lokr_w1 full key must survive"
    assert "P.lokr_w2_a" in sd, "decomposed w2a must be renamed to lokr_w2_a"
    assert "P.lokr_w2_b" in sd, "decomposed w2b must be renamed to lokr_w2_b"
    assert not any(k.endswith(".w2a") or k.endswith(".w2b") for k in sd), (
        "internal w2a/w2b naming leaked"
    )

    # Factors are RAW (LyCORIS convention) — not folded with scale.
    assert torch.equal(sd["P.lokr_w2_a"].float(), raw_w2a.float())
    assert torch.equal(sd["P.lokr_w2_b"].float(), raw_w2b.float())

    # alpha = training alpha (16), int64 to match reference checkpoints.
    assert sd["P.alpha"].dtype == torch.int64, (
        f"alpha dtype {sd['P.alpha'].dtype} != int64 (reference convention)"
    )
    assert sd["P.alpha"].item() == 16, (
        f"alpha must be training value 16, got {sd['P.alpha'].item()}"
    )

    # ComfyUI load formula: kron(w1, w2a@w2b) * (alpha/rank) * multiplier.
    rank = sd["P.lokr_w2_b"].shape[0]
    alpha = sd["P.alpha"].item()
    w1 = sd["P.lokr_w1"].float()
    w2 = sd["P.lokr_w2_a"].float() @ sd["P.lokr_w2_b"].float()
    comfy_delta = torch.kron(w1, w2) * (alpha / rank) * m.multiplier
    rel = (true_delta - comfy_delta).abs().max().item() / max(
        true_delta.abs().max().item(), 1e-12
    )
    assert rel < 1e-5, f"decomposed native lokr scale restore wrong: rel={rel:.2e}"


def test_native_lokr_refuses_inv_scale():
    """Native lokr format cannot represent inv_scale — must raise."""
    sd, _, _ = _trained_lokr_state_dict(
        lora_dim=32, alpha=32, network_dim=32, with_inv_scale=True
    )
    try:
        _convert_lokr_to_native_lokr(sd, dtype=torch.float32, network_dim=32)
    except ValueError:
        return
    raise AssertionError("native lokr conversion must refuse inv_scale")


def test_native_lokr_full_full_folds_scale_into_w2():
    """Full-full path: ComfyUI's loader sets dim=None → forced scale=1.0, so a
    non-unit training scale must be folded into lokr_w2 (the only place the
    loader reads). With scale=1.0 no fold occurs (factors stay raw)."""
    torch.manual_seed(0)
    base = torch.nn.Linear(512, 512, bias=False)
    # lora_dim=32 ≥ min(out_b,in_b) → both factors full.
    m = LoKRModule("test", base, lora_dim=32, alpha=16, lokr_factor=-1)
    with torch.no_grad():
        m.lokr_w2.normal_(0, 0.1)
    true_delta = m.get_weight()  # scale = 16/32 = 0.5
    sd = {f"P.{k}": v.clone() for k, v in m.state_dict().items()}
    raw_w2 = sd["P.lokr_w2"].clone()

    _convert_lokr_to_native_lokr(sd, dtype=torch.float32, network_dim=32)

    assert "P.lokr_w1" in sd and "P.lokr_w2" in sd, "full keys must survive"
    # scale=0.5≠1 → w2 must be folded (raw * 0.5).
    folded_w2 = raw_w2.float() * (16.0 / 32.0)
    assert torch.allclose(sd["P.lokr_w2"].float(), folded_w2, atol=1e-6), (
        "full-full + scale≠1 must fold scale into lokr_w2"
    )
    # ComfyUI applies scale=1.0 on the full path, so kron(w1, w2_saved) * mult.
    comfy_delta = (
        torch.kron(sd["P.lokr_w1"].float(), sd["P.lokr_w2"].float()) * m.multiplier
    )
    rel = (true_delta - comfy_delta).abs().max().item() / max(
        true_delta.abs().max().item(), 1e-12
    )
    assert rel < 1e-5, f"full-full native lokr fold wrong: rel={rel:.2e}"


def test_native_lokr_full_full_scale_one_keeps_factors_raw():
    """Full-full path with scale=1.0 (alpha==network_dim): no fold needed,
    factors stay raw — the LyCORIS/reference default for full-full checkpoints."""
    torch.manual_seed(0)
    base = torch.nn.Linear(512, 512, bias=False)
    m = LoKRModule("test", base, lora_dim=32, alpha=32, lokr_factor=-1)
    with torch.no_grad():
        m.lokr_w2.normal_(0, 0.1)
    sd = {f"P.{k}": v.clone() for k, v in m.state_dict().items()}
    raw_w1 = sd["P.lokr_w1"].clone()
    raw_w2 = sd["P.lokr_w2"].clone()

    _convert_lokr_to_native_lokr(sd, dtype=torch.float32, network_dim=32)

    assert torch.equal(sd["P.lokr_w1"].float(), raw_w1.float()), "w1 must be raw"
    assert torch.equal(sd["P.lokr_w2"].float(), raw_w2.float()), (
        "w2 must be raw when scale=1.0 (no fold)"
    )


def test_native_lokr_keys_load_into_monadforge_runtime_names():
    """LyCORIS/ComfyUI native decomposed keys must load into LoKRModule.

    The runtime module stores decomposed Parameters as w2a/w2b, but native LoKR
    checkpoints store lokr_w2_a/lokr_w2_b. Loading must normalize those names so
    strict=False does not silently leave the learned factors at initialization.
    """
    torch.manual_seed(0)
    lora_name = "lora_unet_block_proj"
    w1 = torch.randn(16, 16)
    w2a = torch.randn(32, 4)
    w2b = torch.randn(4, 32)
    native_sd = {
        f"{lora_name}.lokr_w1": w1,
        f"{lora_name}.lokr_w2_a": w2a,
        f"{lora_name}.lokr_w2_b": w2b,
        f"{lora_name}.alpha": torch.tensor(4),
    }

    network, normalized_sd = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={"ss_network_spec": "lokr"},
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )

    assert network._network_spec.name == "lokr"
    assert f"{lora_name}.w2a" in normalized_sd
    assert f"{lora_name}.w2b" in normalized_sd
    assert f"{lora_name}.lokr_w2_a" not in normalized_sd
    assert f"{lora_name}.lokr_w2_b" not in normalized_sd

    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = network.load_state_dict(normalized_sd, strict=False)
    assert not any(
        key.endswith("w2a") or key.endswith("w2b") for key in info.missing_keys
    )
    assert not any(
        "lokr_w2_a" in key or "lokr_w2_b" in key for key in info.unexpected_keys
    )
    mod = network.unet_loras[0]
    assert torch.equal(mod.w2a.detach(), w2a)
    assert torch.equal(mod.w2b.detach(), w2b)


def test_lokr_save_weights_stamps_full_factor_dim_and_alpha_on_empty_metadata(tmp_path):
    """LoKR save needs ss_network_dim to preserve native-lokr scale."""
    net = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=16,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        use_lokr="true",
        lokr_factor="32",
        lokr_full_factor="true",
    )

    out = tmp_path / "lokr.safetensors"
    net.save_weights(str(out), torch.float32, metadata={})

    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata() or {}
    assert meta["ss_network_spec"] == "lokr"
    assert meta["ss_network_dim"] == "4"
    assert meta["ss_network_alpha"] == "16"
    assert meta["ss_lokr_full_factor"] == "true"


@pytest.mark.parametrize(
    ("network_dim", "network_alpha", "extra_kwargs"),
    [
        pytest.param(32, 16, {}, id="naturally-full"),
        pytest.param(
            114514,
            32,
            {"lokr_allow_legacy_dim": "true"},
            id="legacy-sentinel-resume",
        ),
    ],
)
def test_full_layout_round_trip_stamps_actual_layout(
    tmp_path, network_dim, network_alpha, extra_kwargs
):
    """A full tensor layout must win over the opt-in config flag in metadata."""
    net = create_network(
        multiplier=1.0,
        network_dim=network_dim,
        network_alpha=network_alpha,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        use_lokr="true",
        lokr_factor="32",
        **extra_kwargs,
    )
    source = net.unet_loras[0]
    assert source._use_w1 and source._use_w2
    assert net.cfg.lokr_full_factor is False
    net.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    with torch.no_grad():
        source.lokr_w2.normal_(0, 0.1)
    expected_delta = source.get_weight().clone()

    out = tmp_path / "naturally-full.safetensors"
    net.save_weights(str(out), torch.float32, metadata={})
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata() or {}
    assert meta["ss_lokr_full_factor"] == "true"

    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert not any(key.endswith(("w2a", "w2b")) for key in info.missing_keys)
    assert not any(key.endswith("lokr_w2") for key in info.unexpected_keys)
    actual_delta = restored.unet_loras[0].get_weight()
    assert torch.allclose(actual_delta, expected_delta, atol=1e-6, rtol=1e-5)


def test_full_factor_checkpoint_stamp_restores_runtime_layout():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.lokr_w1": torch.randn(16, 16),
        f"{lora_name}.lokr_w2": torch.randn(32, 32),
        f"{lora_name}.alpha": torch.tensor(4),
    }
    network, _ = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={
            "ss_network_spec": "lokr",
            "ss_lokr_full_factor": "true",
        },
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )
    assert network.cfg.lokr_full_factor is True
    assert network.unet_loras[0]._use_w2 is True


def test_full_factor_checkpoint_stamp_rejects_decomposed_factor_keys():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.lokr_w1_a": torch.randn(16, 4),
        f"{lora_name}.lokr_w1_b": torch.randn(4, 16),
        f"{lora_name}.lokr_w2": torch.randn(32, 32),
        f"{lora_name}.alpha": torch.tensor(4),
    }
    with pytest.raises(RuntimeError, match="contains decomposed factor keys"):
        create_network_from_weights(
            multiplier=1.0,
            file=None,
            weights_sd=native_sd,
            metadata={
                "ss_network_spec": "lokr",
                "ss_lokr_full_factor": "true",
            },
            ae=None,
            text_encoders=[],
            unet=_TinyDiT(),
        )


def test_legacy_unstamped_full_factor_checkpoint_is_inferred():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.lokr_w1": torch.randn(16, 16),
        f"{lora_name}.lokr_w2": torch.randn(32, 32),
        f"{lora_name}.alpha": torch.tensor(4),
    }
    network, _ = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={"ss_network_spec": "lokr"},
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )
    assert network.cfg.lokr_full_factor is True
    assert network.unet_loras[0]._use_w2 is True
