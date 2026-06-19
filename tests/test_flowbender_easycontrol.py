"""FlowBender (EasyControl colorize) invariants — Tier-2 new-method requirement.

Two load-bearing claims:

1. **3-tile extended attention is correct.** The target attends over
   ``[target_k ; cond_k ; feedback_k]`` with two independent logit biases; the
   ``_extended_target_attention`` output must equal a hand-rolled full-softmax
   reference over the concatenated key axis.

2. **Step-0 / gate-off equivalence.** With ``b_feedback`` at a large negative
   logit the feedback rows contribute ≈0 mass → the 3-tile output collapses onto
   the 2-tile (cond-only) output. This is what makes a freshly-initialized
   FlowBender run bit-(≈)equivalent to plain EasyControl, so feedback is a pure
   *addition* the model learns to use rather than a perturbation of the baseline.

Plus the network construction contract (feedback pool built only under
``use_flowbender``; ``set_feedback`` preconditions; metadata round-trip).

The math tests run on CPU in uniform fp32 via the masked-SDPA fallback (no
flash-attn / GPU needed); the LSE autograd Function mirrors the proven 2-tile
``_ExtendedSelfAttnLSEFunc`` and is exercised by training.
"""

from __future__ import annotations

import torch

from networks.attention_dispatch import AttentionParams
from networks.methods.easycontrol import EasyControlNetwork
from networks.methods.easycontrol_attention import _extended_target_attention


def _bshd(B, S, H, D, gen):
    return torch.randn(B, S, H, D, generator=gen, dtype=torch.float32)


def _reference_extended_attention(q, kt, vt, kc, vc, kf, vf, bc, bf, scale):
    """Full-softmax reference over [target; cond; feedback] with key-axis bias."""
    B, St, H, D = q.shape
    Sc, Sf = kc.shape[1], (kf.shape[1] if kf is not None else 0)
    k_tiles = [kt, kc] + ([kf] if kf is not None else [])
    v_tiles = [vt, vc] + ([vf] if vf is not None else [])
    k = torch.cat(k_tiles, dim=1).permute(0, 2, 1, 3)  # B,H,S,D
    v = torch.cat(v_tiles, dim=1).permute(0, 2, 1, 3)
    qh = q.permute(0, 2, 1, 3)  # B,H,St,D
    scores = (qh @ k.transpose(-1, -2)) * scale  # B,H,St,S
    bias_tiles = [torch.zeros(St), torch.full((Sc,), float(bc))]
    if kf is not None:
        bias_tiles.append(torch.full((Sf,), float(bf)))
    scores = scores + torch.cat(bias_tiles).view(1, 1, 1, -1)
    out = torch.softmax(scores, dim=-1) @ v  # B,H,St,D
    return out.permute(0, 2, 1, 3).reshape(B, St, H * D)


def _sdpa_params(scale):
    # attn_mode != "flash" → _extended_target_attention takes the masked-SDPA
    # fallback (no flash-attn needed), exercising the 3-tile concat+bias math.
    return AttentionParams.create_attention_params("torch", softmax_scale=scale)


def test_three_tile_matches_reference():
    gen = torch.Generator().manual_seed(0)
    B, H, D = 2, 4, 8
    St, Sc, Sf = 6, 5, 5
    scale = D**-0.5
    q = _bshd(B, St, H, D, gen)
    kt, vt = _bshd(B, St, H, D, gen), _bshd(B, St, H, D, gen)
    kc, vc = _bshd(B, Sc, H, D, gen), _bshd(B, Sc, H, D, gen)
    kf, vf = _bshd(B, Sf, H, D, gen), _bshd(B, Sf, H, D, gen)
    b_cond = torch.tensor(-1.3)
    b_feedback = torch.tensor(0.7)

    out = _extended_target_attention(
        q,
        kt,
        vt,
        kc,
        vc,
        b_param=b_cond,
        scale=scale,
        attn_params=_sdpa_params(scale),
        feedback_k=kf,
        feedback_v=vf,
        b_feedback_param=b_feedback,
    )
    ref = _reference_extended_attention(q, kt, vt, kc, vc, kf, vf, -1.3, 0.7, scale)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_feedback_gate_off_collapses_to_two_tile():
    """b_feedback → −∞ ⇒ 3-tile output == 2-tile (cond-only) output."""
    gen = torch.Generator().manual_seed(1)
    B, H, D = 1, 4, 8
    St, Sc, Sf = 7, 6, 6
    scale = D**-0.5
    q = _bshd(B, St, H, D, gen)
    kt, vt = _bshd(B, St, H, D, gen), _bshd(B, St, H, D, gen)
    kc, vc = _bshd(B, Sc, H, D, gen), _bshd(B, Sc, H, D, gen)
    kf, vf = _bshd(B, Sf, H, D, gen), _bshd(B, Sf, H, D, gen)
    b_cond = torch.tensor(-2.0)
    params = _sdpa_params(scale)

    two_tile = _extended_target_attention(
        q, kt, vt, kc, vc, b_param=b_cond, scale=scale, attn_params=params
    )
    three_tile_off = _extended_target_attention(
        q,
        kt,
        vt,
        kc,
        vc,
        b_param=b_cond,
        scale=scale,
        attn_params=params,
        feedback_k=kf,
        feedback_v=vf,
        b_feedback_param=torch.tensor(-30.0),
    )
    assert torch.allclose(two_tile, three_tile_off, atol=1e-6)

    # And a default init logit (−10) is already negligibly close.
    three_tile_init = _extended_target_attention(
        q,
        kt,
        vt,
        kc,
        vc,
        b_param=b_cond,
        scale=scale,
        attn_params=params,
        feedback_k=kf,
        feedback_v=vf,
        b_feedback_param=torch.tensor(-10.0),
    )
    assert torch.allclose(two_tile, three_tile_init, atol=1e-3)


def _rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-8)


def test_lse3_forward_and_backward_match_sdpa_reference():
    """The custom 3-tile autograd Function matches a differentiable full-softmax
    reference — forward AND all input grads, including the analytic b_cond /
    b_feedback gradients. Gated on CUDA + flash-attn (the LSE path's requirement).
    """
    import pytest

    from networks import attention_dispatch as ad
    from networks.methods.easycontrol_attention import _ExtendedSelfAttnLSEFunc3

    if not torch.cuda.is_available() or ad._wrapped_flash_attn_forward is None:
        pytest.skip("LSE 3-tile path requires CUDA + flash-attn")

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    B, H, D = 2, 4, 32
    St, Sc, Sf = 16, 12, 12
    scale = D**-0.5

    def mk(S):
        return torch.randn(B, S, H, D, generator=gen, device=dev, dtype=torch.float32)

    q32, kt32, vt32 = mk(St), mk(St), mk(St)
    kc32, vc32 = mk(Sc), mk(Sc)
    kf32, vf32 = mk(Sf), mk(Sf)
    bc32 = torch.tensor(-1.1, device=dev)
    bf32 = torch.tensor(0.6, device=dev)
    dout = torch.randn(B, St, H * D, generator=gen, device=dev, dtype=torch.float32)

    # --- fp32 differentiable reference ---
    leaves32 = [
        t.clone().requires_grad_() for t in (q32, kt32, vt32, kc32, vc32, kf32, vf32)
    ]
    qr, ktr, vtr, kcr, vcr, kfr, vfr = leaves32
    bcr = bc32.clone().requires_grad_()
    bfr = bf32.clone().requires_grad_()
    k = torch.cat([ktr, kcr, kfr], 1).permute(0, 2, 1, 3)
    v = torch.cat([vtr, vcr, vfr], 1).permute(0, 2, 1, 3)
    scores = (qr.permute(0, 2, 1, 3) @ k.transpose(-1, -2)) * scale
    bias = torch.cat(
        [torch.zeros(St, device=dev), bcr.expand(Sc), bfr.expand(Sf)]
    ).view(1, 1, 1, -1)
    out_ref = (
        (torch.softmax(scores + bias, dim=-1) @ v)
        .permute(0, 2, 1, 3)
        .reshape(B, St, H * D)
    )
    (out_ref * dout).sum().backward()

    # --- fp16 LSE Function (flash requires fp16/bf16) ---
    leaves16 = [
        t.clone().half().requires_grad_()
        for t in (q32, kt32, vt32, kc32, vc32, kf32, vf32)
    ]
    q16, kt16, vt16, kc16, vc16, kf16, vf16 = leaves16
    bc16 = bc32.clone().requires_grad_()
    bf16 = bf32.clone().requires_grad_()
    out16 = _ExtendedSelfAttnLSEFunc3.apply(
        q16, kt16, vt16, kc16, vc16, kf16, vf16, bc16, bf16, scale
    ).reshape(B, St, H * D)
    (out16.float() * dout).sum().backward()

    # Forward + per-input grad relative errors (fp16 flash vs fp32 reference).
    assert _rel_err(out16.float(), out_ref) < 2e-2
    for g16, l32, name in zip(leaves16, leaves32, "q kt vt kc vc kf vf".split()):
        assert _rel_err(g16.grad.float(), l32.grad) < 5e-2, name
    # The analytic bias gradients are the FlowBender-specific addition.
    assert (
        abs(bc16.grad.item() - bcr.grad.item()) / (abs(bcr.grad.item()) + 1e-6) < 5e-2
    )
    assert (
        abs(bf16.grad.item() - bfr.grad.item()) / (abs(bfr.grad.item()) + 1e-6) < 5e-2
    )


def _make_net(use_flowbender, **kw):
    return EasyControlNetwork(
        num_blocks=2,
        hidden_size=16,
        num_heads=4,
        mlp_ratio=2.0,
        cond_lora_dim=4,
        cond_lora_alpha=4.0,
        b_cond_init=-10.0,
        cond_scale=1.0,
        apply_ffn_lora=True,
        use_flowbender=use_flowbender,
        **kw,
    )


def test_feedback_pool_only_built_under_flowbender():
    off = _make_net(False)
    assert off.b_feedback is None and off.feedback_lora_qkv is None
    assert off._use_flowbender is False

    on = _make_net(True, b_feedback_init=-7.0, flowbender_p_un=0.2)
    assert on._use_flowbender is True
    assert on._flowbender_p_un == 0.2
    assert len(on.b_feedback) == on.num_blocks
    assert all(float(b.detach()) == -7.0 for b in on.b_feedback)
    assert len(on.feedback_lora_qkv) == on.num_blocks
    # Feedback pool ≈ doubles the cond pool's params.
    assert sum(p.numel() for p in on.parameters()) > 1.8 * sum(
        p.numel() for p in off.parameters()
    )


def test_set_feedback_preconditions():
    import pytest

    off = _make_net(False)
    with pytest.raises(RuntimeError, match="without use_flowbender"):
        off.set_feedback(torch.zeros(1, 16, 4, 4))

    on = _make_net(True)
    # apply_to has not run → no patched blocks yet.
    with pytest.raises(RuntimeError, match="before apply_to"):
        on.set_feedback(torch.zeros(1, 16, 4, 4))


def test_flowbender_metadata_roundtrip():
    on = _make_net(True, b_feedback_init=-6.0, flowbender_p_un=0.15)
    md = on.metadata_fields()
    assert md["ss_use_flowbender"] == "1"
    assert md["ss_b_feedback_init"] == "-6.0"
    assert md["ss_flowbender_p_un"] == "0.15"
    off = _make_net(False)
    assert off.metadata_fields()["ss_use_flowbender"] == "0"


def test_flowbender_save_load_roundtrip(tmp_path):
    """create_network_from_weights rebuilds the feedback pool from metadata and
    round-trips the feedback state_dict keys (b_feedback + feedback_lora_*)."""
    from networks.methods.easycontrol import create_network_from_weights

    net = _make_net(True, b_feedback_init=-6.0, flowbender_p_un=0.2)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    file = tmp_path / "fb.safetensors"
    net.save_weights(str(file), dtype=torch.float32, metadata=None)

    # The metadata stamp must drive feedback-pool reconstruction on load.
    net2, sd = create_network_from_weights(1.0, str(file), None, [], None)
    assert net2._use_flowbender is True
    assert net2._flowbender_p_un == 0.2
    assert net2.b_feedback is not None and len(net2.b_feedback) == net.num_blocks
    net2.load_weights(str(file))

    a, b = net.state_dict(), net2.state_dict()
    fb_keys = [k for k in a if "feedback" in k or "b_feedback" in k]
    assert fb_keys, "no feedback keys saved"
    for k in set(a) & set(b):
        torch.testing.assert_close(
            a[k].float(), b[k].float(), rtol=1e-5, atol=1e-5
        )
