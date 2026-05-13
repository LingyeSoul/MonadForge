"""Tests for the in-handler FECL math (plan2 task #5).

The FeRA Frequency-Energy Consistency Loss used to live as
``FeRANetwork.compute_fecl_loss``. Task #5 ported it into the loss
registry handler ``library.training.losses._fera_fecl_loss`` so it
works on the new ``stacked_experts_global_fei`` LoRANetwork spec
without dragging in ``methods/fera.py``. Numerical-parity tests pin
the port to the same scalar the legacy network produces.
"""

from __future__ import annotations

import argparse
import types

import pytest
import torch

from library.training.losses import (
    LossContext,
    _fera_fecl_loss,
    build_loss_composer,
)


@pytest.fixture(scope="module")
def fera_network():
    """A tiny FeRANetwork instance, just so we can call its
    ``compute_fecl_loss`` as the reference oracle.

    Build minimally: bypass ``apply_to`` (no DiT to patch) and only
    populate the bits FECL needs — ``num_bands``, ``fei_sigma_low_div``,
    and a ``fei_indicator`` with ``_band_sigmas``.
    """
    from networks.methods.fera import FeRANetwork

    # FeRANetwork wants a unet for the target scan. Hand it a tiny stub
    # with no Linears to scan — ``apply_to`` is never called, so this is
    # fine. The init still constructs the router + FrequencyEnergyIndicator
    # which is all we need for ``compute_fecl_loss``.
    stub_unet = torch.nn.Module()
    # ``fei_sigma_low_div=16`` keeps the 3-band σ pyramid small enough that
    # the second band's kernel (σ = 2·σ_low) fits inside the 64×64 test
    # latents without exceeding the reflect-pad cap. The numerical-parity
    # check is what matters; the production default 4.0 is exercised via
    # the cfg-only tests above.
    net = FeRANetwork(
        unet=stub_unet,
        rank=4,
        alpha=4.0,
        num_experts=3,
        num_bands=3,
        fei_sigma_low_div=16.0,
    )
    return net


def _make_inputs(B=2, C=4, H=64, W=64, seed=0):
    torch.manual_seed(seed)
    z_base = torch.randn(B, C, H, W)
    z_fera = torch.randn(B, C, H, W)
    z_target = torch.randn(B, C, H, W)
    return z_base, z_fera, z_target


def _make_ctx(model_pred, target, network, aux):
    """LossContext factory with the FECL-irrelevant fields stubbed out."""
    return LossContext(
        args=argparse.Namespace(),
        batch={},
        model_pred=model_pred,
        target=target,
        timesteps=torch.zeros(model_pred.shape[0]),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(model_pred.shape[0]),
        network=network,
        aux=aux,
    )


def test_fera_fecl_handler_matches_legacy(fera_network):
    """Numerical parity: the new in-handler FECL must produce the same
    scalar as ``FeRANetwork.compute_fecl_loss`` on identical inputs.
    """
    z_base, z_fera, z_target = _make_inputs(seed=1)

    # Reference: legacy FeRA path.
    ref = fera_network.compute_fecl_loss(
        z_base=z_base, z_fera=z_fera, z_target=z_target
    )

    # New path: build a LossContext + a fake network carrying the gate.
    # ``fecl_weight=1.0`` so the handler returns the raw FECL scalar.
    cfg = types.SimpleNamespace(
        fera_fecl_weight=1.0, fera_num_bands=3, fei_sigma_low_div=16.0
    )
    network = types.SimpleNamespace(fecl_weight=1.0, cfg=cfg)
    ctx = _make_ctx(z_fera, z_target, network, {"fera": {"z_base": z_base}})
    new = _fera_fecl_loss(ctx)
    assert torch.allclose(new, ref, atol=1e-6), (
        f"FECL parity broken: legacy={ref.item():.6e} vs new={new.item():.6e}"
    )


def test_fera_fecl_handler_returns_zero_when_disabled():
    """Weight=0 short-circuits to a zero scalar without computing anything."""
    z_base, z_fera, z_target = _make_inputs(seed=2)
    network = types.SimpleNamespace(fecl_weight=0.0)
    ctx = _make_ctx(z_fera, z_target, network, {"fera": {"z_base": z_base}})
    out = _fera_fecl_loss(ctx)
    assert out.shape == ()
    assert out.item() == 0.0


def test_fera_fecl_handler_returns_zero_when_aux_missing():
    """Weight>0 but no z_base in aux → 0 scalar (FECL is silently off)."""
    _, z_fera, z_target = _make_inputs(seed=3)
    network = types.SimpleNamespace(fecl_weight=0.5)
    ctx = _make_ctx(z_fera, z_target, network, {})
    out = _fera_fecl_loss(ctx)
    assert out.shape == ()
    assert out.item() == 0.0


def test_fera_fecl_handler_applies_weight():
    """The handler must scale the raw scalar by ``fecl_weight`` — that's
    the single scaling-knob location.
    """
    z_base, z_fera, z_target = _make_inputs(seed=4)

    cfg = types.SimpleNamespace(
        fera_fecl_weight=2.0, fera_num_bands=3, fei_sigma_low_div=16.0
    )
    # Reference at weight=1
    cfg_ref = types.SimpleNamespace(
        fera_fecl_weight=1.0, fera_num_bands=3, fei_sigma_low_div=16.0
    )
    net_ref = types.SimpleNamespace(fecl_weight=1.0, cfg=cfg_ref)
    net_w2 = types.SimpleNamespace(fecl_weight=2.0, cfg=cfg)

    def _ctx(net):
        return _make_ctx(z_fera, z_target, net, {"fera": {"z_base": z_base}})

    ref = _fera_fecl_loss(_ctx(net_ref))
    w2 = _fera_fecl_loss(_ctx(net_w2))
    assert torch.allclose(w2, 2.0 * ref, atol=1e-6)


def test_legacy_pre_computed_path_still_works():
    """Back-compat: when the trainer stashes a pre-computed ``fecl_loss``
    in ``ctx.aux`` (the FeRANetwork legacy path), the handler must use it
    instead of re-running the band decomposition.
    """
    _, z_fera, z_target = _make_inputs(seed=5)
    pre_scalar = torch.tensor(0.7)
    network = types.SimpleNamespace(fecl_weight=0.5)
    ctx = _make_ctx(z_fera, z_target, network, {"fecl_loss": pre_scalar})
    out = _fera_fecl_loss(ctx)
    assert torch.allclose(out, 0.5 * pre_scalar)


def test_build_loss_composer_activates_fera_fecl_on_stacked_experts():
    """The composer must activate ``fera_fecl`` when the network is a
    LoRANetwork with ``cfg.use_moe_style == 'independent_A'`` and
    ``fecl_weight > 0``, even though it isn't a FeRANetwork.
    """
    cfg = types.SimpleNamespace(
        use_moe_style="independent_A", fera_fecl_weight=0.5
    )
    network = types.SimpleNamespace(
        cfg=cfg,
        fecl_weight=0.5,
        _ortho_reg_weight=0.0,
        _balance_loss_weight=0.0,
        contrastive_weight=0.0,
    )
    args = argparse.Namespace(
        method="lora",
        functional_loss_weight=0.0,
        multiscale_loss_weight=0.0,
        repa_weight=0.0,
        use_repa=False,
    )
    composer = build_loss_composer(args, network)
    assert "fera_fecl" in composer.active_losses
