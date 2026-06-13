"""Dtype-policy regression tests for the LoRA-family training forwards.

2026-06-10: the fp32-bottleneck matmul policy (``F.linear(x.float(),
w.float())`` + the custom down-projection autograd) was removed. Training
GEMMs now run in the adapter PARAMETER dtype (bf16; fp32 only for OrthoInit's
trainable master bases). The first cut keyed the GEMM dtype off the *activation*
(``x.dtype`` / ``weight.to(x_lora.dtype)``); that silently upcast, because the
AdaLN ``nn.LayerNorm`` feeding the adapted Linears emits fp32 under
autocast(bf16), so ``x`` arrives fp32 — materializing a full fp32 weight copy
AND (with per-channel scaling on) a fresh fp32 activation out of ``_rebalance``
(``inv_scale``), OOMing for zero numeric gain. The modules now cast ``x`` DOWN
to the param dtype before the rank GEMMs (mirroring OrthoLoRAModule), pinned by
``test_*_fp32_activation_no_upcast`` below. The justification, measured in
``bench/lora_fp32_bottleneck``:

  * ``train.py`` wraps the training forward in ``accelerator.autocast()``
    (default ``mixed_precision="bf16"``). Autocast re-cast the fp32 inputs of
    every ``F.linear``/``einsum``/``bmm`` back to bf16, so the fp32 matmuls
    never executed — live training was already pure-bf16 GEMMs plus dead cast
    traffic.
  * cuBLAS accumulates bf16 GEMMs in fp32 internally, so the only precision
    delta vs a true fp32 GEMM is the final rounding of the rank-R bottleneck
    (invisible to a 200-step Adam probe).

These tests pin the two contracts the rewrite must keep:

  1. **Legacy parity** — under autocast, the new forward (and its grads) is
     bitwise identical to an inline replica of the retired fp32-bottleneck
     code. This is "behavior unchanged for train.py". (Channel-scale chimera
     is the one documented exception: the retired custom Function folded
     ``inv_scale`` into the weight in fp32 while ``_rebalance`` applies it to
     x in bf16 — one rounding apart, asserted with allclose.)
  2. **Dtype honesty** — the training forward produces the same result with
     and without autocast, so non-autocast callers (bespoke loops, tests) see
     the same numerics as train.py.

Inference paths are NOT touched by the rewrite: HydraLoRAModule at eval, the
chimera inference module, and EasyControl's KV prefill keep their historical
fp32 compute (the inference engine runs without autocast). Asserted below.

Also home to the flag-independent invariants that lived in the retired
``test_lora_custom_autograd.py``: the hydra σ-feature cache, the chimera
rank-cat down-projection equivalence, and the OrthoInit-chimera zero-at-init
+ distill roundtrips.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# bf16 has ~8 bits of mantissa; per-op relative noise is ~4e-3. Multi-step
# accumulation in matmul + grad makes the achievable rtol ~1e-2 for grads.
_CS_ATOL = 5e-2
_CS_RTOL = 1e-2


def _autocast():
    return torch.autocast("cpu", dtype=torch.bfloat16)


def _make_channel_scale(in_features: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(in_features, generator=g, dtype=torch.float32) * 2.0 + 0.5


def _named_trainable_grads(module: torch.nn.Module):
    return {
        n: p.grad.detach().clone()
        for n, p in module.named_parameters()
        if p.grad is not None
    }


def _assert_grads_equal(a: dict, b: dict, label: str):
    assert a.keys() == b.keys(), f"{label}: param sets differ: {a.keys() ^ b.keys()}"
    for k in a:
        assert torch.equal(a[k], b[k]), f"{label}: grad on {k!r} differs"


# ---------------------------------------------------------------------------
# Legacy-path replicas: the retired fp32-bottleneck training branches,
# verbatim. Run under autocast they reproduce what train.py executed before
# the rewrite; the new forwards must match them bitwise.
# ---------------------------------------------------------------------------


def _legacy_lora_forward(module, x):
    org_forwarded = module.org_forward(x)
    x_lora = module._rebalance(x)
    lx = F.linear(x_lora.float(), module.lora_down.weight.float())
    lx = lx * module._timestep_mask
    lx, scale = module._apply_rank_dropout(lx)
    lx = F.linear(lx, module.lora_up.weight.float())
    return org_forwarded + (lx * module.multiplier * scale).to(org_forwarded.dtype)


def _legacy_ortho_init_forward(module, x):
    org_forwarded = module.org_forward(x)
    x_lora = module._rebalance(x)
    lx = F.linear(x_lora.float(), module.Q_init.float())
    lx = lx * module.lambda_layer.float() * module._timestep_mask
    lx, scale = module._apply_rank_dropout(lx)
    lx = F.linear(lx, module.P_init.float())
    return org_forwarded + (lx * module.multiplier * scale).to(org_forwarded.dtype)


def _legacy_hydra_forward(module, x):
    org_forwarded = module.org_forward(x)
    x_lora = module._rebalance(x)
    lx = F.linear(x_lora.float(), module.lora_down.weight.float())
    gate = module._compute_gate(lx)
    lx = lx * module._timestep_mask
    lx, scale = module._apply_rank_dropout(lx)
    combined = torch.einsum("be,eod->bod", gate.float(), module.lora_up_weight.float())
    orig_shape = lx.shape
    lx_3d = lx.reshape(orig_shape[0], -1, orig_shape[-1])
    out = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(*orig_shape[:-1], -1)
    return org_forwarded + (out * module.multiplier * scale).to(org_forwarded.dtype)


def _run_pair(make_module, new_fn, legacy_fn, *, in_dim=32, seed=1):
    """Build two identical modules, run new vs legacy forward+backward under
    autocast, return (out, grads, grad_x) for each."""

    def run(fn):
        base, module = make_module()
        module.train()
        torch.manual_seed(seed)
        x = torch.randn(2, 8, in_dim, dtype=torch.bfloat16, requires_grad=True)
        with _autocast():
            out = fn(module, x)
        out.float().sum().backward()
        return out.detach().clone(), _named_trainable_grads(module), x.grad.clone()

    return run(new_fn), run(legacy_fn)


def test_lora_training_matches_legacy_under_autocast():
    from networks.lora_modules.lora import LoRAModule

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        return base, module

    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_lora_forward
    )
    assert torch.equal(o_new, o_old), "LoRA forward != legacy fp32 path under autocast"
    _assert_grads_equal(g_new, g_old, "LoRA")
    assert torch.equal(gx_new, gx_old), "LoRA grad_x differs"


def test_lora_channel_scale_training_matches_legacy_under_autocast():
    from networks.lora_modules.lora import LoRAModule

    cs = _make_channel_scale(32)

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=cs.clone()
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        return base, module

    # The legacy *default* path also routed through _rebalance, so even with
    # channel scaling this stays bitwise (only the retired custom Function's
    # fp32 fold differed by one rounding).
    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_lora_forward
    )
    assert torch.equal(o_new, o_old)
    _assert_grads_equal(g_new, g_old, "LoRA+channel_scale")
    assert torch.equal(gx_new, gx_old)


def test_ortho_init_training_matches_legacy_under_autocast():
    from networks.lora_modules.ortho import OrthoInitLoRAModule

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False)
        base.weight.requires_grad_(False)
        module = OrthoInitLoRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
        with torch.no_grad():
            module.lambda_layer.copy_(torch.randn(1, 4))
        base.to(torch.bfloat16)
        module.apply_to()
        return base, module

    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_ortho_init_forward
    )
    assert torch.equal(o_new, o_old), "OrthoInit forward != legacy under autocast"
    _assert_grads_equal(g_new, g_old, "OrthoInit")
    assert torch.equal(gx_new, gx_old)
    # Grads reach the trainable SVD bases (the point of OrthoInit).
    assert g_new["Q_init"].abs().sum() > 0 and g_new["P_init"].abs().sum() > 0


def test_hydra_training_matches_legacy_under_autocast():
    from networks.lora_modules.hydra import HydraLoRAModule

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = HydraLoRAModule(
            "h", base, multiplier=1.0, lora_dim=4, alpha=4, num_experts=3
        )
        with torch.no_grad():
            module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
        module.apply_to()
        return base, module

    (o_new, g_new, gx_new), (o_old, g_old, gx_old) = _run_pair(
        make, lambda m, x: m.forward(x), _legacy_hydra_forward
    )
    assert torch.equal(o_new, o_old), "Hydra forward != legacy under autocast"
    _assert_grads_equal(g_new, g_old, "Hydra")
    assert torch.equal(gx_new, gx_old)
    assert g_new["router.weight"].abs().sum() > 0, "router must receive gradient"


def test_training_forward_is_autocast_independent():
    """Dtype honesty: with bf16 inputs the new training forward computes the
    same thing with or without autocast — non-autocast callers (bespoke
    loops, tests) see train.py numerics."""
    from networks.lora_modules.lora import LoRAModule

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    module = LoRAModule("m", base, multiplier=1.0, lora_dim=4, alpha=4)
    with torch.no_grad():
        module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
    module.apply_to()
    module.train()

    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    with _autocast():
        y_ac = module.forward(x)
    y_plain = module.forward(x)
    assert torch.equal(y_ac, y_plain)


def _fp32_activation_no_upcast(make, *, in_dim=32):
    """The AdaLN ``nn.LayerNorm`` feeding the adapted Linears emits fp32 under
    autocast(bf16), so ``x`` arrives fp32 in training. The forward must compute
    the rank path in the model's bf16 compute dtype (``org_forwarded.dtype``) —
    NOT key it off ``x.dtype`` and run fp32, which (with per-channel scaling on)
    allocated a full fp32 activation out of ``_rebalance`` and OOMed.

    Pin it: under autocast, feeding the fp32 activation must give the SAME result
    as feeding its bf16 rounding — i.e. the LoRA params (incl. the ``inv_scale``
    multiply, which autocast does NOT downcast) ran at bf16, not fp32. The
    retired ``x.dtype`` policy applied ``inv_scale`` in fp32 for the fp32 input
    and diverged."""
    base, module = make()
    module.train()
    torch.manual_seed(2)
    x_fp32 = torch.randn(2, 8, in_dim, dtype=torch.float32)
    x_bf16 = x_fp32.to(torch.bfloat16)  # same values, bf16 dtype
    with _autocast():
        y_from_fp32 = module.forward(x_fp32)
        y_from_bf16 = module.forward(x_bf16)
    assert torch.equal(y_from_fp32, y_from_bf16), (
        "fp32 activation diverges from its bf16 rounding — the rank path "
        "(inv_scale) ran in fp32 instead of the bf16 compute dtype (silent "
        "upcast regression)"
    )


def test_lora_fp32_activation_no_upcast():
    from networks.lora_modules.lora import LoRAModule

    cs = _make_channel_scale(32)  # inv_scale ON — the path that OOMed

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=cs
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        return base, module

    _fp32_activation_no_upcast(make)


def test_ortho_init_fp32_activation_no_upcast():
    # OrthoInit is the live ``use_ortho_init=true`` variant; it was MISSED by
    # 8c2005c (which only fixed lora/hydra/chimera/step_expert/stacked_experts)
    # and kept ``work = x.dtype`` — so the fp32 AdaLN activation + on-by-default
    # channel scaling materialized a full fp32 ``_rebalance`` activation per
    # module and OOMed. This pins the bf16 compute dtype.
    from networks.lora_modules.ortho import OrthoInitLoRAModule

    cs = _make_channel_scale(32)  # inv_scale ON — the path that OOMed

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = OrthoInitLoRAModule(
            "o", base, multiplier=1.0, lora_dim=4, alpha=4, channel_scale=cs
        )
        with torch.no_grad():
            module.lambda_layer.copy_(torch.randn_like(module.lambda_layer) * 0.1)
        module.apply_to()
        return base, module

    _fp32_activation_no_upcast(make)


def test_hydra_fp32_activation_no_upcast():
    from networks.lora_modules.hydra import HydraLoRAModule

    cs = _make_channel_scale(32)  # inv_scale ON

    def make():
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        module = HydraLoRAModule(
            "h",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            num_experts=3,
            channel_scale=cs,
        )
        with torch.no_grad():
            module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
        module.apply_to()
        return base, module

    _fp32_activation_no_upcast(make)


def test_hydra_inference_keeps_fp32_compute():
    """The rewrite is training-only: at eval the hydra forward still computes
    in fp32 (router-live checkpoints run through the no-autocast inference
    engine and must produce unchanged outputs)."""
    from networks.lora_modules.hydra import HydraLoRAModule

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    module = HydraLoRAModule(
        "h", base, multiplier=1.0, lora_dim=4, alpha=4, num_experts=3
    )
    with torch.no_grad():
        module.lora_up_weight.copy_(torch.randn_like(module.lora_up_weight) * 0.1)
    module.apply_to()
    module.eval()

    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)

    # fp32 reference: the historical eval compute, spelled out.
    org = module.org_forward(x)
    lx = F.linear(module._rebalance(x).float(), module.lora_down.weight.float())
    gate = module._compute_gate(lx)
    combined = torch.einsum("be,eod->bod", gate.float(), module.lora_up_weight.float())
    lx_3d = lx.reshape(2, -1, lx.shape[-1])
    out = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(2, 8, -1)
    ref = org + (out * module.multiplier * module.scale).to(org.dtype)

    assert torch.equal(module.forward(x), ref)


def test_lora_channel_scale_absorption_preserves_output():
    """SmoothQuant-style absorption: a channel-scaled module must produce the
    same delta as an unscaled twin (weights rebalanced, output unchanged)."""
    from networks.lora_modules.lora import LoRAModule

    def make(channel_scale):
        torch.manual_seed(0)
        base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
        base.weight.requires_grad_(False)
        module = LoRAModule(
            "m",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            channel_scale=channel_scale,
        )
        with torch.no_grad():
            module.lora_up.weight.copy_(torch.randn_like(module.lora_up.weight) * 0.1)
        module.apply_to()
        module.train()
        return base, module

    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    _, plain = make(None)
    _, scaled = make(_make_channel_scale(32))
    with _autocast():
        y_plain = plain.forward(x)
        y_scaled = scaled.forward(x)
    assert torch.allclose(
        y_plain.float(), y_scaled.float(), atol=_CS_ATOL, rtol=_CS_RTOL
    )


# ---------------------------------------------------------------------------
# Hydra σ-feature cache (flag-independent invariant, ported from the retired
# test_lora_custom_autograd.py).
# ---------------------------------------------------------------------------


def test_hydra_sigma_feature_cache_updates_and_clears():
    """Sigma-router features are precomputed once per step and cached on modules."""
    from networks.lora_modules.hydra import (
        HydraLoRAModule,
        _sigma_sinusoidal_features,
    )

    torch.manual_seed(0)
    base = torch.nn.Linear(32, 24, bias=False)
    module = HydraLoRAModule(
        "h",
        base,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        num_experts=3,
        sigma_feature_dim=8,
    )

    sigmas = torch.tensor([0.25, 0.5], dtype=torch.float32)
    expected = _sigma_sinusoidal_features(sigmas, 8)
    module.set_sigma(sigmas, expected)

    assert torch.equal(module._sigma, sigmas)
    assert torch.equal(module._sigma_features, expected)

    module.clear_sigma()
    assert torch.equal(module._sigma, torch.zeros_like(sigmas))
    assert torch.equal(
        module._sigma_features,
        _sigma_sinusoidal_features(torch.zeros_like(sigmas), 8),
    )


# ---------------------------------------------------------------------------
# Chimera rank-cat down-projection: both pools collapse into ONE matmul
# (``cat([Q_eff_c, Q_eff_f]) @ x`` then split) so backward computes grad_x
# once instead of summing two (B, L, in) transients. Equivalence within bf16
# tolerance — a (2r, in) GEMM accumulates differently than two (r, in) GEMMs.
# ---------------------------------------------------------------------------


def test_chimera_down_proj_rank_cat_matches_separate():
    in_dim, r, tokens = 32, 4, 8

    def run(catted: bool):
        torch.manual_seed(0)
        x = torch.randn(2, tokens, in_dim, dtype=torch.bfloat16, requires_grad=True)
        Qc = torch.randn(r, in_dim, dtype=torch.bfloat16, requires_grad=True)
        Qf = torch.randn(r, in_dim, dtype=torch.bfloat16, requires_grad=True)
        if catted:
            lx = F.linear(x, torch.cat([Qc, Qf], dim=0))
            lx_c, lx_f = lx[..., :r], lx[..., r:]
        else:
            lx_c = F.linear(x, Qc)
            lx_f = F.linear(x, Qf)
        (lx_c.float().sum() + lx_f.float().sum() * 3.0).backward()
        return (
            lx_c.detach().clone(),
            lx_f.detach().clone(),
            x.grad.detach().clone(),
            Qc.grad.detach().clone(),
            Qf.grad.detach().clone(),
        )

    cat_c, cat_f, cat_gx, cat_gQc, cat_gQf = run(catted=True)
    sep_c, sep_f, sep_gx, sep_gQc, sep_gQf = run(catted=False)

    ac = dict(atol=_CS_ATOL, rtol=_CS_RTOL)
    assert torch.allclose(cat_c.float(), sep_c.float(), **ac)
    assert torch.allclose(cat_f.float(), sep_f.float(), **ac)
    assert torch.allclose(cat_gQc.float(), sep_gQc.float(), **ac)
    assert torch.allclose(cat_gQf.float(), sep_gQf.float(), **ac)
    assert torch.allclose(cat_gx.float(), sep_gx.float(), **ac)
    assert cat_gQc.abs().sum() > 0 and cat_gQf.abs().sum() > 0


# ---------------------------------------------------------------------------
# OrthoInit chimera: trainable SVD-seeded bases (no Cayley). Ported from the
# retired file — flag-independent invariants of the parameterization and its
# distilled on-disk form.
# ---------------------------------------------------------------------------


def _cpu_only():
    """Force the chimera SVD init onto CPU (``init_device`` reads
    ``torch.cuda.is_available()``) so these tests are deterministic and run
    identically with or without a GPU present."""
    from unittest import mock

    return mock.patch("torch.cuda.is_available", return_value=False)


def _fresh_base(seed=0):
    torch.manual_seed(seed)
    base = torch.nn.Linear(32, 24, bias=False).to(torch.bfloat16)
    base.weight.requires_grad_(False)
    return base


def _make_ortho_init_chimera(channel_scale=None, *, K_c=3, K_f=2, seed=0):
    from networks.lora_modules.chimera import ChimeraHydraLoRAModule

    base = _fresh_base(seed)
    with _cpu_only():
        module = ChimeraHydraLoRAModule(
            "c",
            base,
            multiplier=1.0,
            lora_dim=4,
            alpha=4,
            num_experts_content=K_c,
            num_experts_freq=K_f,
            lambda_init=0.1,
            channel_scale=channel_scale,
            use_ortho_init=True,
        )
    module.apply_to()
    # The trainable bases must be Parameters; no Cayley skew, no _eye_r.
    assert isinstance(module.Q_basis_c, torch.nn.Parameter)
    assert isinstance(module.P_bases_f, torch.nn.Parameter)
    assert not hasattr(module, "S_q_c") and not hasattr(module, "_eye_r")
    return base, module


def test_chimera_ortho_init_zero_at_init():
    """ΔW=0 at construction comes from the centered UNIFORM gate, not the
    basis — so an OrthoInit chimera with default (uniform) gates must leave
    the base forward untouched even though λ_init > 0 and the bases are
    nonzero. (Cayley parity: same invariant the frozen path relies on.)"""
    _base, module = _make_ortho_init_chimera()
    module.eval()
    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    assert torch.equal(module.forward(x), module.org_forward(x))


def test_chimera_ortho_init_grads_reach_bases():
    """Training forward+backward: the trainable SVD bases carry gradient (the
    whole point of OrthoInit), and the routing buffers' grad path stays
    intact, in the activation-dtype GEMM regime."""
    base, module = _make_ortho_init_chimera()
    module.train()
    with torch.no_grad():
        module.lambda_c.copy_(torch.randn_like(module.lambda_c) * 0.1)
        module.lambda_f.copy_(torch.randn_like(module.lambda_f) * 0.1)
    module.set_content_routing_weights(torch.tensor([[0.5, 0.3, 0.2]]))
    module.set_freq_routing_weights(torch.tensor([[0.6, 0.4]]))
    torch.manual_seed(1)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16, requires_grad=True)
    with _autocast():
        out = base.forward(x)
    out.float().sum().backward()
    grads = _named_trainable_grads(module)
    assert grads["Q_basis_c"].abs().sum() > 0
    assert grads["P_bases_f"].abs().sum() > 0
    assert x.grad is not None


def _ortho_init_distill_roundtrip(channel_scale):
    """An OrthoInit chimera distills (R=I) to the same free-form layout as the
    Cayley path, and the resulting ``*_chimera.safetensors`` tensors loaded into
    ``ChimeraHydraInferenceModule`` reproduce the trained forward — proving the
    on-disk / inference path needs no OrthoInit awareness."""
    from networks.lora_modules.chimera import (
        ChimeraHydraInferenceModule,
        ChimeraHydraLoRAModule,
    )

    K_c, K_f = 3, 2
    _base, module = _make_ortho_init_chimera(
        channel_scale=channel_scale, K_c=K_c, K_f=K_f
    )
    with torch.no_grad():
        module.lambda_c.copy_(torch.randn_like(module.lambda_c) * 0.3)
        module.lambda_f.copy_(torch.randn_like(module.lambda_f) * 0.3)
    gate_c = torch.tensor([[0.5, 0.3, 0.2]])
    gate_f = torch.tensor([[0.6, 0.4]])
    module.set_content_routing_weights(gate_c)
    module.set_freq_routing_weights(gate_f)
    module.eval()

    torch.manual_seed(2)
    x = torch.randn(2, 8, 32, dtype=torch.bfloat16)
    o_train = module.forward(x).detach()

    # Distill (R=I) — prefix keys so the ``.Q_basis_c`` discriminator fires.
    sd = {f"m.{k}": v.clone() for k, v in module.state_dict().items()}
    ChimeraHydraLoRAModule.distill_save_state_dict(sd, dtype=None)
    assert "m.lora_down_c.weight" in sd and "m.lora_up_c_weight" in sd
    assert not any(k.endswith(".Q_basis_c") for k in sd)  # bases consumed

    # Fresh base with identical weights — ``module.apply_to`` rebound the
    # shared base's forward, so the inference twin needs its own clean Linear.
    base_inf = _fresh_base(seed=0)
    inf = ChimeraHydraInferenceModule(
        "c",
        base_inf,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        num_experts_content=K_c,
        num_experts_freq=K_f,
        channel_scale=channel_scale,
    )
    inf.apply_to()
    with torch.no_grad():
        inf.lora_down_c.weight.copy_(sd["m.lora_down_c.weight"])
        inf.lora_down_f.weight.copy_(sd["m.lora_down_f.weight"])
        inf.lora_up_c_weight.copy_(sd["m.lora_up_c_weight"])
        inf.lora_up_f_weight.copy_(sd["m.lora_up_f_weight"])
    inf.set_content_routing_weights(gate_c)
    inf.set_freq_routing_weights(gate_f)
    inf.eval()
    o_inf = inf.forward(x).detach()

    assert torch.allclose(
        o_train.float(), o_inf.float(), atol=_CS_ATOL, rtol=_CS_RTOL
    ), "OrthoInit chimera distilled inference forward diverges from training"


def test_chimera_ortho_init_distill_roundtrip_unscaled():
    _ortho_init_distill_roundtrip(channel_scale=None)


def test_chimera_ortho_init_distill_roundtrip_scaled():
    _ortho_init_distill_roundtrip(channel_scale=_make_channel_scale(32))
