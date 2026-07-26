"""GLoKr (Kronecker delta + BoRA weight decomposition) regressions.

CPU-only, cross-platform-stable (no SVD/LAPACK in the init path — magnitude
init is plain row/col norms). Mirrors ``test_lokr_channel_scale.py``'s
structure: module invariants first, then the factory/save round-trip.

Load-bearing invariants pinned here:
  * W' == W0 exactly at init (zero-init w2 leg + magnitudes = W0 norms).
  * eval forward == org_forward + x @ get_weight().T (delta reconstruction).
  * merge_to == W0 + get_weight() under the multiplier lerp convention
    (multiplier scales W' − W0, NOT the raw Kronecker delta — BoRA is
    non-linear in ΔW).
  * fuse/unfuse is exact (delta stashed — the BoRA normalization is not
    invertible from the fused weight alone).
  * T-LoRA ``_timestep_mask`` gates w2's rank axis in training mode only.
  * Save stamps ``ss_network_spec="glokr"`` + factor/rs_lora/bora scalars,
    and the factory rebuilds bit-identical modules from keys + stamps.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from networks import resolve_network_spec  # noqa: E402
from networks.lora_anima.factory import (  # noqa: E402
    create_network,
    create_network_from_weights,
)
from networks.lora_modules.glokr import GLoKRModule  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)


class _TinyDiT(torch.nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512) -> None:
        super().__init__()
        self.block = Block(in_dim, out_dim)

    def reset_mod_guidance(self) -> None:
        pass


def _make_module(
    *,
    in_dim: int = 512,
    out_dim: int = 512,
    rank: int = 4,
    alpha: float = 16,
    factor: int = -1,
    multiplier: float = 1.0,
    bora: bool = True,
    **kw,
) -> tuple[torch.nn.Linear, GLoKRModule]:
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    base.weight.requires_grad_(False)
    module = GLoKRModule(
        "lora_test",
        base,
        multiplier=multiplier,
        lora_dim=rank,
        alpha=alpha,
        glokr_factor=factor,
        bora=bora,
        **kw,
    )
    return base, module


def _make_delta_nonzero(module: GLoKRModule, std: float = 0.1) -> None:
    with torch.no_grad():
        if module.use_w2:
            module.glokr_w2.normal_(0, std)
        else:
            module.glokr_w2_b.normal_(0, std)
        if module.bora:
            module.bora_m_row.mul_(1.05)
            module.bora_m_col.mul_(0.95)


# ---------------------------------------------------------------------------
# Module invariants
# ---------------------------------------------------------------------------


def test_layout_and_state_keys():
    _, module = _make_module(rank=4, alpha=16, factor=-1)
    assert module.use_w1 is True
    assert module.use_w2 is False
    assert module.scale == 4.0
    assert module.glokr_w1.shape == (16, 16)
    assert module.glokr_w2_a.shape == (32, 4)
    assert module.glokr_w2_b.shape == (4, 32)
    assert module.bora_m_row.shape == (512, 1)
    assert module.bora_m_col.shape == (1, 512)
    assert set(module.state_dict()) == {
        "alpha",
        "glokr_w1",
        "glokr_w2_a",
        "glokr_w2_b",
        "bora_m_row",
        "bora_m_col",
    }


def test_rs_lora_scale_and_self_describing_alpha():
    _, module = _make_module(rank=4, alpha=16, factor=-1, rs_lora=True)
    assert module.scale == 8.0  # 16 / sqrt(4)
    # LyCORIS rs convention: persisted alpha = scale * rank, so any alpha/rank
    # consumer recovers the training scale from tensors alone (no rs stamp).
    assert module.alpha.item() == 32.0


def test_full_factor_forces_unit_scale():
    _, module = _make_module(rank=4, alpha=16, factor=8, full_factor=True)
    assert module.use_w1 and module.use_w2
    assert module.scale == 1.0
    assert module.alpha.item() == 4
    assert set(module.state_dict()) == {
        "alpha",
        "glokr_w1",
        "glokr_w2",
        "bora_m_row",
        "bora_m_col",
    }


def test_init_is_exact_identity():
    base, module = _make_module()
    module.apply_to()
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        assert torch.allclose(base(x), module.org_forward(x), atol=1e-6)


def test_eval_forward_matches_weight_reconstruction():
    base, module = _make_module(multiplier=0.75)
    _make_delta_nonzero(module)
    module.apply_to()
    module.eval()
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        y = base(x)
        ref = module.org_forward(x) + x @ module.get_weight().t()
    # One fused GEMM vs base-GEMM + delta-GEMM: fp32 non-associativity only.
    assert torch.allclose(y, ref, atol=1e-3, rtol=1e-4)


def test_multiplier_zero_is_identity():
    base, module = _make_module(multiplier=0.0)
    _make_delta_nonzero(module)
    module.apply_to()
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        assert torch.allclose(base(x), module.org_forward(x), atol=1e-6)


def test_merge_matches_live_delta():
    base, module = _make_module(multiplier=0.75)
    _make_delta_nonzero(module)
    original = base.weight.detach().clone()
    expected = module.get_weight().clone()

    module.merge_to(module.state_dict(), dtype=torch.float32, device="cpu")

    assert torch.allclose(base.weight, original + expected, atol=1e-6, rtol=1e-5)


def test_fuse_unfuse_preserves_forward_and_restores_weight():
    base, module = _make_module(multiplier=0.75)
    _make_delta_nonzero(module)
    module.apply_to()
    module.eval()
    x = torch.randn(2, 3, 512)
    original = base.weight.detach().clone()
    with torch.no_grad():
        expected = base(x)

    module.fuse_weight()
    with torch.no_grad():
        assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)

    module.unfuse_weight()
    assert torch.allclose(base.weight, original, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)


def test_zero_init_trains_w2_chain_end_and_magnitudes_first():
    base, module = _make_module(rank=4, alpha=4, factor=16)
    module.apply_to()
    base(torch.randn(2, 3, 512)).square().mean().backward()

    # ΔW = 0 at init ⇒ grad reaches w2_b (chain end) and both magnitudes,
    # while w2_a's grad is zero (∂ΔW/∂w2_a ∝ w2_b = 0).
    assert torch.count_nonzero(module.glokr_w2_b.grad) > 0
    assert torch.count_nonzero(module.glokr_w2_a.grad) == 0
    assert torch.count_nonzero(module.bora_m_row.grad) > 0
    assert torch.count_nonzero(module.bora_m_col.grad) > 0


def test_timestep_mask_gates_w2_rank_in_training_only():
    base, module = _make_module()
    _make_delta_nonzero(module)
    with torch.no_grad():
        # Neutralize magnitude perturbation so gating the delta ⇒ identity.
        module.bora_m_row.copy_(base.weight.float().norm(dim=1, keepdim=True))
        module.bora_m_col.copy_(base.weight.float().norm(dim=0, keepdim=True))
    module.apply_to()
    x = torch.randn(2, 3, 512)

    module.train()
    module._timestep_mask = torch.zeros_like(module._timestep_mask)
    with torch.no_grad():
        gated = base(x)
    assert torch.allclose(gated, module.org_forward(x), atol=1e-5)

    module.eval()
    with torch.no_grad():
        ungated = base(x)
    assert not torch.allclose(ungated, module.org_forward(x), atol=1e-5)


def test_bora_off_is_plain_additive_kron_delta():
    base, module = _make_module(bora=False)
    assert not hasattr(module, "bora_m_row")
    _make_delta_nonzero(module)
    module.apply_to()
    module.eval()
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        y = base(x)
        ref = module.org_forward(x) + x @ module.get_weight().t()
    assert torch.allclose(y, ref, atol=1e-3, rtol=1e-4)


def test_channel_scale_rejected():
    base = torch.nn.Linear(12, 20, bias=False)
    with pytest.raises(ValueError, match="channel scaling"):
        GLoKRModule(
            "lora_test", base, lora_dim=4, alpha=4, channel_scale=torch.ones(12)
        )


@pytest.mark.parametrize("kw", [{"dropout": 0.1}, {"rank_dropout": 0.5}])
def test_unsupported_dropouts_rejected(kw):
    base = torch.nn.Linear(12, 20, bias=False)
    with pytest.raises(ValueError):
        GLoKRModule("lora_test", base, lora_dim=4, alpha=4, **kw)


# ---------------------------------------------------------------------------
# Registry / factory / save round-trip
# ---------------------------------------------------------------------------


def test_resolver_selects_glokr_spec():
    assert resolve_network_spec({"use_glokr": "true"}).name == "glokr"


def _build_network(unet, **overrides):
    kwargs = dict(
        use_glokr="true",
        glokr_factor="16",
        channel_scaling_alpha="0",
    )
    kwargs.update(overrides)
    return create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=16,
        vae=None,
        text_encoders=[],
        unet=unet,
        **kwargs,
    )


@pytest.mark.parametrize("rs_lora", [False, True])
def test_checkpoint_save_and_reload_preserves_delta(tmp_path, rs_lora):
    network = _build_network(_TinyDiT(), glokr_rs_lora=str(rs_lora).lower())
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    source = network.unet_loras[0]
    _make_delta_nonzero(source)
    expected = source.get_weight().clone()

    out = tmp_path / f"glokr-{rs_lora}.safetensors"
    network.save_weights(str(out), torch.float32, metadata={})

    with safe_open(str(out), framework="pt") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
    assert metadata["ss_network_spec"] == "glokr"
    assert metadata["ss_glokr_factor"] == "16"
    assert metadata["ss_glokr_rs_lora"] == str(rs_lora).lower()
    assert metadata["ss_glokr_bora"] == "true"
    assert any(key.endswith(".glokr_w2_a") for key in keys)
    assert any(key.endswith(".bora_m_row") for key in keys)
    assert not any(
        key.endswith((".lora_down.weight", ".lora_up.weight")) for key in keys
    )

    # The BoRA delta is W0-dependent (row/col norms of the merged weight), so
    # reload parity must be measured on a target DiT sharing the source's W0.
    target = _TinyDiT()
    with torch.no_grad():
        target.block.proj.weight.copy_(source.org_module_ref[0].weight)
    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=target,
    )
    assert restored._network_spec.name == "glokr"
    # rs_lora is never recovered at load — the persisted alpha pre-folds
    # sqrt(r) (self-describing), so the rebuild always uses alpha/rank.
    assert restored.cfg.glokr_rs_lora is False
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert not info.missing_keys
    assert not info.unexpected_keys
    assert torch.allclose(
        restored.unet_loras[0].get_weight(), expected, atol=1e-5, rtol=1e-5
    )


def test_for_inference_reload_keeps_glokr_spec_and_merges(tmp_path):
    """The for_inference flatten must NOT downgrade GLoKr to plain lora —
    merge/bake needs GLoKRModule.merge_to's replacement semantics."""
    network = _build_network(_TinyDiT())
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    source = network.unet_loras[0]
    _make_delta_nonzero(source)
    expected_delta = source.get_weight().clone()
    out = tmp_path / "glokr.safetensors"
    network.save_weights(str(out), torch.float32, metadata={})

    target = _TinyDiT()
    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=target,
        for_inference=True,
    )
    assert restored._network_spec.name == "glokr"

    # NB: expected_delta was computed against the SOURCE DiT's W0. The BoRA
    # weight is W0-dependent, so merge parity must be measured on a target
    # sharing that W0.
    with torch.no_grad():
        target.block.proj.weight.copy_(network.unet_loras[0].org_module_ref[0].weight)
        w0 = target.block.proj.weight.detach().clone()
    restored.merge_to(None, target, weights_sd, torch.float32, "cpu")
    assert torch.allclose(
        target.block.proj.weight, w0 + expected_delta, atol=1e-5, rtol=1e-5
    )


def test_kron_layout_and_init_match_lycoris_ground_truth():
    """Pin the LyCORIS-exactness claim against the installed lycoris-lora:
    identical factor shapes for the same (dims, rank, factor), kaiming-bounded
    non-zero legs, zero w2 chain end. Guards against kron-order flips,
    transposed factors, and init-convention drift (all of which the
    self-referential forward/merge parity tests cannot see)."""
    from lycoris.modules.lokr import LokrModule as LycorisLokrModule

    base_ours = torch.nn.Linear(512, 384, bias=False)
    base_ref = torch.nn.Linear(512, 384, bias=False)
    ours = GLoKRModule("m", base_ours, lora_dim=4, alpha=16, glokr_factor=8, bora=False)
    ref = LycorisLokrModule(
        "m", base_ref, lora_dim=4, alpha=16, factor=8, bypass_mode=True
    )

    assert ours.use_w1 == ref.use_w1 and ours.use_w2 == ref.use_w2
    assert ours.glokr_w1.shape == ref.lokr_w1.shape
    assert ours.glokr_w2_a.shape == ref.lokr_w2_a.shape
    assert ours.glokr_w2_b.shape == ref.lokr_w2_b.shape
    assert ours.scale == ref.scale

    # Init: kaiming_uniform(a=sqrt(5)) bound is 1/sqrt(fan_in); zero w2_b leg.
    for ours_t, ref_t in (
        (ours.glokr_w1, ref.lokr_w1),
        (ours.glokr_w2_a, ref.lokr_w2_a),
    ):
        bound = 1.0 / (ours_t.shape[1] ** 0.5)
        assert ours_t.abs().max().item() <= bound * 1.0001
        # Same distribution family as the reference (uniform in ±bound).
        assert ref_t.abs().max().item() <= bound * 1.0001
        assert ours_t.std().item() == pytest.approx(ref_t.std().item(), rel=0.35)
    assert torch.count_nonzero(ours.glokr_w2_b) == 0
    assert torch.count_nonzero(ref.lokr_w2_b) == 0


def test_unstamped_rs_lora_checkpoint_reloads_at_training_scale(tmp_path):
    """The regression the self-describing alpha convention exists for: a
    metadata-STRIPPED rs_lora checkpoint must still reproduce the exact
    training-time delta (alpha buffer pre-folds sqrt(r))."""
    network = _build_network(_TinyDiT(), glokr_rs_lora="true")
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    source = network.unet_loras[0]
    _make_delta_nonzero(source)
    expected = source.get_weight().clone()
    native_sd = {k: v.detach().clone() for k, v in network.state_dict().items()}

    target = _TinyDiT()
    with torch.no_grad():
        target.block.proj.weight.copy_(source.org_module_ref[0].weight)
    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,  # no metadata at all
        ae=None,
        text_encoders=[],
        unet=target,
    )
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert not info.missing_keys and not info.unexpected_keys
    assert torch.allclose(
        restored.unet_loras[0].get_weight(), expected, atol=1e-5, rtol=1e-5
    )


def test_unstamped_key_sniff_still_routes_to_glokr(tmp_path):
    """Metadata-stripped checkpoints (load_file drops __metadata__) must still
    rebuild as GLoKr from the key shapes alone (factor inferred)."""
    network = _build_network(_TinyDiT())
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    _make_delta_nonzero(network.unet_loras[0])
    native_sd = {k: v.detach().clone() for k, v in network.state_dict().items()}

    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )
    assert restored._network_spec.name == "glokr"
    # On a single 512×512 shape, factor=-1 and factor=16 produce the SAME
    # split (16, 32), so the network-wide inference legitimately settles on
    # -1 — what matters is that the rebuilt layout loads cleanly below.
    assert restored.cfg.glokr_factor in (-1, 16)
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert not info.missing_keys
    assert not info.unexpected_keys
