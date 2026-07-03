"""Tests for the M2 network registry and save pipeline.

Covers:

* ``resolve_network_spec`` precedence and mutual-exclusion rules.
* The ``networks.lora_save`` pipeline round-trips a synthetic state_dict
  for each save_variant, emitting the expected file(s) and preserving
  tensor shapes through the per-variant conversion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from networks import (
    NETWORK_KWARGS,
    NETWORK_REGISTRY,
    NetworkSpec,
    all_network_kwargs,
    resolve_network_spec,
)
from networks import lora_save


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


EXPECTED_VARIANTS = {
    "lora",
    "ortho",
    "ortho_init",
    "hydra",
    "ortho_hydra",
}


def test_registry_has_expected_variants():
    assert EXPECTED_VARIANTS.issubset(NETWORK_REGISTRY.keys())
    for name, spec in NETWORK_REGISTRY.items():
        assert isinstance(spec, NetworkSpec)
        assert spec.name == name


def test_all_network_kwargs_matches_allowlist():
    """`all_network_kwargs()` must be exactly the sorted ``NETWORK_KWARGS`` set.

    The two were unified (the per-variant ``kwarg_flags`` split was collapsed
    into one flat allowlist); this pins them together so the forwarding list
    and the schema-validation set never drift apart.
    """
    assert set(all_network_kwargs()) == set(NETWORK_KWARGS)
    assert list(all_network_kwargs()) == sorted(NETWORK_KWARGS)


def test_hydra_router_kwargs_registered():
    """Regression pin: the bug that motivated the M2 finish.

    `router_targets` + σ-conditional router kwargs must stay in the allowlist
    so they flow through the argparse schema and into `create_network`. If any
    drops off, the router silently defaults to uniform MoE over every target
    module.
    """
    must_have = {
        "router_targets",
        "sigma_feature_dim",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
    }
    missing = must_have - set(NETWORK_KWARGS)
    assert not missing, f"allowlist missing hydra router kwargs: {missing}"


def test_repa_kwargs_registered():
    """REPA v2 kwargs must stay in the allowlist (else use_repa is silently
    inert + the config test rejects the key). See library/training/repa.py."""
    must_have = {
        "use_repa",
        "repa_mode",
        "repa_weight",
        "repa_layer",
        "repa_encoder",
        "repa_lr_scale",
    }
    missing = must_have - set(NETWORK_KWARGS)
    assert not missing, f"allowlist missing repa kwargs: {missing}"


# ---------------------------------------------------------------------------
# resolve_network_spec precedence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "lora"),
        ({"use_ortho": "true"}, "ortho"),
        ({"use_ortho_init": "true"}, "ortho_init"),
        ({"use_moe_style": "shared_A"}, "hydra"),
        ({"use_moe_style": "shared_A", "use_ortho": "true"}, "ortho_hydra"),
        ({"use_moe_style": "independent_A"}, "stacked_experts_global_fei"),
        # Falsey forms of use_moe_style resolve to plain LoRA.
        ({"use_moe_style": False}, "lora"),
        ({"use_moe_style": "false"}, "lora"),
        ({"use_moe_style": ""}, "lora"),
    ],
)
def test_resolve_precedence(kwargs, expected):
    spec = resolve_network_spec(kwargs)
    assert spec.name == expected


def test_ortho_and_ortho_init_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_network_spec({"use_ortho": "true", "use_ortho_init": "true"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_ortho_init": "true", "use_moe_style": "shared_A"},
        {"use_ortho_init": "true", "use_moe_style": "independent_A"},
    ],
)
def test_ortho_init_rejects_moe(kwargs):
    with pytest.raises(NotImplementedError):
        resolve_network_spec(kwargs)


def test_ortho_init_composes_with_chimera():
    """OrthoInit now rides the chimera_hydra spec (trainable bases threaded via
    cfg.use_ortho_init); it resolves rather than raising."""
    spec = resolve_network_spec({"use_ortho_init": "true", "use_chimera_hydra": "true"})
    assert spec.name == "chimera_hydra"


# ---------------------------------------------------------------------------
# save_network_weights round-trips — synthetic state_dicts, one per variant
# ---------------------------------------------------------------------------


def _alpha(value: float) -> torch.Tensor:
    return torch.tensor(float(value))


def _make_std_lora_sd(prefix: str, r: int, in_dim: int, out_dim: int) -> dict:
    """Fake fused-qkv LoRA state_dict entry (runtime form).

    The runtime uses fused self_attn.qkv_proj; save defuses it into q/k/v.
    """
    return {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up.weight": torch.randn(3 * out_dim, r),
        f"{prefix}.alpha": _alpha(r),
    }


def _save_and_reload(
    state_dict: dict,
    tmp_path: Path,
    save_variant: str,
    filename: str = "out.safetensors",
) -> dict[str, torch.Tensor]:
    out = tmp_path / filename
    lora_save.save_network_weights(
        dict(state_dict),  # copy — save mutates
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": save_variant},
        save_variant=save_variant,
    )
    # hydra writes *_moe.safetensors alongside (not the main file)
    if save_variant in ("hydra_moe", "ortho_hydra_to_hydra"):
        moe_path = tmp_path / (out.stem + "_moe.safetensors")
        assert moe_path.exists(), f"expected _moe file at {moe_path}"
        return load_file(str(moe_path))
    assert out.exists()
    return load_file(str(out))


def test_save_standard_lora_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)

    loaded = _save_and_reload(sd, tmp_path, save_variant="standard")

    # qkv_proj should be defused into q/k/v with matching shapes
    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert f"{base}_{suffix}.lora_down.weight" in loaded
        assert f"{base}_{suffix}.lora_up.weight" in loaded
        assert f"{base}_{suffix}.alpha" in loaded
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    # fused key must be gone
    assert f"{prefix}.lora_down.weight" not in loaded


def test_save_ortho_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoLoRA (PSOFT) runtime keys: Cayley params + frozen SVD bases
    sd = {
        f"{prefix}.S_p": torch.randn(r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_to_lora")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    for k in loaded:
        assert not k.endswith(".S_p") and not k.endswith(".S_q")
        assert not k.endswith(".P_basis") and not k.endswith(".Q_basis")


def test_save_ortho_init_roundtrip(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoInit runtime keys: trainable P_init/Q_init + λ (no S_p/S_q, no
    # frozen P_basis/Q_basis). Discriminated by ``.P_init``.
    sd = {
        f"{prefix}.P_init": torch.randn(3 * out_dim, r),
        f"{prefix}.Q_init": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_to_lora")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, r)
    # runtime-only keys must be gone
    for k in loaded:
        assert not k.endswith(".P_init") and not k.endswith(".Q_init")
        assert not k.endswith(".lambda_layer")


def test_save_lokr_roundtrip(tmp_path: Path):
    """Full-factor LoKR (``lokr_w1``/``lokr_w2`` + ``inv_scale``) → standard LoRA.

    Pins the load-bearing invariants of ``_convert_lokr_to_standard_lora``:
    Kronecker orientation (``kron(w1, w2)`` = ``(out, in)``, not transposed),
    ``inv_scale`` column-bake (``delta *= inv_scale.unsqueeze(0)``), the
    SVD-split (``up @ down == delta`` at full rank), and qkv defuse. The fused
    qkv target carries ``3 * out_dim`` output rows so defuse yields q/k/v.
    """
    in_dim, out_dim = 8, 12  # → fused out = 36
    # factor pairs: out_a*out_b == 3*out_dim (36), in_a*in_b == in_dim (8)
    out_a, out_b, in_a, in_b = 6, 6, 2, 4
    rank = min(128, min(3 * out_dim, in_dim))  # SVD rank cap = 8 here
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"

    w1 = torch.randn(out_a, in_a)
    w2 = torch.randn(out_b, in_b)
    inv_scale = torch.rand(in_dim) + 0.5  # strictly positive
    sd = {
        f"{prefix}.lokr_w1": w1,
        f"{prefix}.lokr_w2": w2,
        f"{prefix}.inv_scale": inv_scale,
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="lokr")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (rank, in_dim)
        assert loaded[f"{base}_{suffix}.lora_up.weight"].shape == (out_dim, rank)
        assert f"{base}_{suffix}.alpha" in loaded
    # LoKR factor + inv_scale keys must be purged
    for k in loaded:
        assert not k.endswith(".lokr_w1") and not k.endswith(".lokr_w2")
        assert not k.endswith(".inv_scale")

    # Numeric: at full rank, up @ down reconstructs the materialised delta.
    # defuse chunks lora_up rows per component (q/k/v), clones down for each.
    expected_delta = torch.kron(w1, w2) * inv_scale.unsqueeze(0)
    down = loaded[f"{base}_q_proj.lora_down.weight"]
    for i, suffix in enumerate(("q_proj", "k_proj", "v_proj")):
        up = loaded[f"{base}_{suffix}.lora_up.weight"]
        recon = up.to(torch.float) @ down.to(torch.float)
        rows = expected_delta[i * out_dim : (i + 1) * out_dim, :]
        assert torch.allclose(recon, rows, atol=1e-4), f"delta mismatch in {suffix}"


def test_save_lokr_decomposed_roundtrip(tmp_path: Path):
    """Decomposed LoKR (``w1a``/``w1b``/``w2a``/``w2b``, no ``inv_scale``).

    Verifies the ``w1a @ w1b`` / ``w2a @ w2b`` reconstruction branch and the
    Kronecker orientation on the decomposed factor path. Calls
    ``_convert_lokr_to_standard_lora`` directly (bypassing the save_variant
    dispatch, which routes no-inv_scale decomposed LoKR to the native lokr
    format — covered by ``test_native_lokr_decomposed_raw_factors_restore_scale``
    in test_lokr_channel_scale.py). Pinned at the lossless full-rank boundary.
    """
    from networks.lora_save import _convert_lokr_to_standard_lora

    in_dim, out_dim = 8, 12
    out_a, out_b, in_a, in_b, lora_dim = 6, 6, 2, 4, 4
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"

    w1a = torch.randn(out_a, lora_dim)
    w1b = torch.randn(lora_dim, in_a)
    w2a = torch.randn(out_b, lora_dim)
    w2b = torch.randn(lora_dim, in_b)
    # The kron delta = kron(w1a@w1b, w2a@w2b) has rank ≤ in_a*in_b = 8. Full
    # rank (lora_rank=0) keeps the roundtrip bit-equivalent.
    alpha_val = in_a * in_b  # 8
    sd = {
        f"{prefix}.w1a": w1a,
        f"{prefix}.w1b": w1b,
        f"{prefix}.w2a": w2a,
        f"{prefix}.w2b": w2b,
        f"{prefix}.alpha": torch.tensor(float(alpha_val), dtype=torch.float32),
    }
    rank = min(out_dim, in_dim)  # 8

    _convert_lokr_to_standard_lora(sd, dtype=torch.float32, lora_rank=0)

    up = sd[f"{prefix}.lora_up.weight"]
    down = sd[f"{prefix}.lora_down.weight"]
    # kron delta is (out_a*out_b=36, in_a*in_b=8); full SVD rank = min(36,8) = 8.
    assert up.shape == (out_a * out_b, rank), f"lora_up shape {up.shape}"
    assert down.shape == (rank, in_a * in_b), f"lora_down shape {down.shape}"
    for k in sd:
        assert not any(k.endswith(s) for s in (".w1a", ".w1b", ".w2a", ".w2b")), (
            f"leftover decomposed key: {k}"
        )
        assert not k.endswith(".inv_scale")

    # Numeric: at full rank, up @ down reconstructs the materialised delta.
    # scale = alpha/network_dim; with network_dim=alpha (alpha_val) scale=1.
    expected_delta = torch.kron(w1a @ w1b, w2a @ w2b)
    recon = up.to(torch.float) @ down.to(torch.float)
    saved_alpha = sd[f"{prefix}.alpha"].item()
    # ComfyUI formula: (alpha/rank) * (up @ down). alpha==rank → scale=1.
    recon = recon * (saved_alpha / rank)
    assert torch.allclose(recon, expected_delta, atol=1e-4), (
        "decomposed kron delta mismatch"
    )


def test_save_lokr_falls_back_to_lora_rank_without_alpha(tmp_path: Path):
    """When no ``.alpha`` key is present, the converter must fall back to the
    ``lora_rank`` argument (default 128) as the SVD rank cap.

    This deliberately exercises the fallback branch so it's covered by intent,
    not accidentally — ``test_save_lokr_decomposed_roundtrip`` covers the
    per-module-alpha headline path; this one pins the no-alpha fallback.
    """
    from networks.lora_save import _convert_lokr_to_standard_lora

    # Factors derive from a 12×8 Linear: _factorization(12,-1)=(6,2),
    # _factorization(8,-1)=(4,2) → kron delta (36, 8).
    out_a, out_b, in_a, in_b = 6, 6, 2, 4
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    explicit_rank = 8  # smaller than the default 128 to make the cap observable

    w1 = torch.randn(out_a, in_a)
    w2 = torch.randn(out_b, in_b)
    sd = {
        f"{prefix}.lokr_w1": w1,
        f"{prefix}.lokr_w2": w2,
        # NOTE: no .alpha key → converter must use the lora_rank arg.
    }

    _convert_lokr_to_standard_lora(sd, dtype=torch.float32, lora_rank=explicit_rank)

    # kron delta is (out_a*out_b=36, in_a*in_b=8); cap = min(8, 36) = 8.
    expected_rank = min(explicit_rank, min(out_a * out_b, in_a * in_b))
    up = sd[f"{prefix}.lora_up.weight"]
    down = sd[f"{prefix}.lora_down.weight"]
    assert up.shape == (out_a * out_b, expected_rank), (
        f"lora_up rank should be capped at lora_rank={explicit_rank}, got {up.shape}"
    )
    assert down.shape == (expected_rank, in_a * in_b)
    # When alpha was absent, the converter writes the *capped* rank back as alpha
    # so the saved file stays self-describing.
    assert sd[f"{prefix}.alpha"].item() == expected_rank


def test_save_lokr_uses_lora_rank_for_svd_truncation(tmp_path: Path):
    """LoKR→standard SVD conversion caps rank at the ``lora_rank`` argument.

    Design (post black-image fix): ``alpha`` is the training scale source
    (scale = alpha/network_dim), not a rank cap. The SVD truncation rank is
    controlled solely by ``lora_rank`` (0 = full rank). This test pins that
    contract — passing ``lora_rank=32`` caps the output at rank 32 regardless
    of the per-module alpha value.
    """
    from networks.lora_save import _convert_lokr_to_standard_lora

    # 8×8 factors → 64×64 delta (rank up to 64).
    out_a, in_a, out_b, in_b = 8, 8, 8, 8
    alpha_val = 32  # unrelated to rank now
    prefix = "lora_unet_blocks_0_ffn_proj"

    w1 = torch.randn(out_a, in_a)
    w2 = torch.randn(out_b, in_b)
    sd = {
        f"{prefix}.lokr_w1": w1,
        f"{prefix}.lokr_w2": w2,
        f"{prefix}.alpha": torch.tensor(alpha_val, dtype=torch.float32),
    }
    _convert_lokr_to_standard_lora(sd, dtype=torch.float32, lora_rank=32)

    up = sd[f"{prefix}.lora_up.weight"]
    down = sd[f"{prefix}.lora_down.weight"]
    assert up.shape[1] == 32, (
        f"SVD rank should be capped at lora_rank=32, got {up.shape[1]}"
    )
    assert down.shape[0] == 32
    # alpha is written back as the (capped) rank so ComfyUI's scale = alpha/rank = 1.0.
    assert sd[f"{prefix}.alpha"].item() == 32


def test_ortho_init_module_zero_delta_and_distill_fidelity():
    """ΔW=0 at init (λ=0) and the sqrt-split distill reproduces P·diag(λ)·Q."""
    from networks.lora_modules import OrthoInitLoRAModule

    torch.manual_seed(0)
    in_dim, out_dim, r = 16, 24, 4
    lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    mod = OrthoInitLoRAModule("lora_test", lin, lora_dim=r, alpha=r)

    # ΔW = 0 at init: adapter output equals the base Linear output.
    mod.apply_to()
    x = torch.randn(2, in_dim)
    base_out = lin.weight @ x[0]  # org weight preserved (apply_to deletes ref)
    with torch.no_grad():
        y = mod.forward(x)
    assert torch.allclose(y[0], base_out, atol=1e-5)

    # Give λ a nonzero value, then check distill round-trips the product.
    with torch.no_grad():
        mod.lambda_layer.copy_(torch.randn(1, r))
    P = mod.P_init.detach().float()
    Q = mod.Q_init.detach().float()
    lam = mod.lambda_layer.detach().squeeze(0).float()
    expected_dW = P @ torch.diag(lam) @ Q  # (out, in)

    sd = {
        "m.P_init": mod.P_init.detach().clone(),
        "m.Q_init": mod.Q_init.detach().clone(),
        "m.lambda_layer": mod.lambda_layer.detach().clone(),
        "m.alpha": torch.tensor(float(r)),
    }
    OrthoInitLoRAModule.distill_save_state_dict(sd, torch.float32)
    up = sd["m.lora_up.weight"]
    down = sd["m.lora_down.weight"]
    assert torch.allclose(up @ down, expected_dW, atol=1e-5)


def test_ortho_init_training_grads_flow():
    """OrthoInit's training forward (activation-dtype GEMMs) must deliver
    gradient to the trainable SVD bases, λ, and the input."""
    from networks.lora_modules import OrthoInitLoRAModule

    torch.manual_seed(1)
    in_dim, out_dim, r = 16, 24, 4
    lin = torch.nn.Linear(in_dim, out_dim, bias=False)
    mod = OrthoInitLoRAModule("lora_test", lin, lora_dim=r, alpha=r)
    with torch.no_grad():
        mod.lambda_layer.copy_(torch.randn(1, r))
    mod.apply_to()
    mod.train()

    x = torch.randn(2, 5, in_dim, requires_grad=True)
    y = mod.forward(x)
    y.sum().backward()

    assert mod.Q_init.grad is not None and mod.Q_init.grad.abs().sum() > 0
    assert mod.P_init.grad is not None and mod.P_init.grad.abs().sum() > 0
    assert mod.lambda_layer.grad is not None
    assert x.grad is not None


def test_save_hydra_moe_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{prefix}.lora_up_weight": torch.randn(E, 3 * out_dim, r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
        f"{prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="hydra_moe")

    base = "lora_unet_blocks_0_self_attn"
    # per-expert ups expanded, qkv defused per-expert
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)
        assert loaded[f"{base}_{suffix}.router.weight"].shape == (E, in_dim)
        assert loaded[f"{base}_{suffix}.router.bias"].shape == (E,)
    # fused lora_up_weight must be gone (expanded into per-expert keys)
    for k in loaded:
        assert not k.endswith(".lora_up_weight")


def test_save_hydra_moe_mixed_with_plain_lora_qkv_defuses_up(tmp_path: Path):
    """Regression: when ``router_targets`` filters some fused-qkv modules
    out of MoE, the resulting plain-LoRA leg for those modules must also be
    q/k/v-defused by the hydra save pipeline. Previously only ``lora_down`` /
    ``alpha`` were split; ``lora_up.weight`` stayed fused, producing a
    mismatched checkpoint.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12

    # Hydra-routed module (cross_attn.kv — regex-matched target)
    hydra_prefix = "lora_unet_blocks_0_cross_attn_kv_proj"
    # Plain-LoRA module (self_attn.qkv — regex-excluded by router_targets)
    plain_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"

    sd = {
        # hydra leg — stacked lora_up_weight
        f"{hydra_prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{hydra_prefix}.lora_up_weight": torch.randn(E, 2 * out_dim, r),
        f"{hydra_prefix}.router.weight": torch.randn(E, r),
        f"{hydra_prefix}.router.bias": torch.randn(E),
        f"{hydra_prefix}.alpha": _alpha(r),
        # plain LoRA leg — standard single lora_up.weight, no router
        f"{plain_prefix}.lora_down.weight": torch.randn(r, in_dim),
        f"{plain_prefix}.lora_up.weight": torch.randn(3 * out_dim, r),
        f"{plain_prefix}.alpha": _alpha(r),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="hydra_moe")

    # Hydra leg: split into k/v with per-expert ups
    hydra_base = "lora_unet_blocks_0_cross_attn"
    for suffix in ("k_proj", "v_proj"):
        assert loaded[f"{hydra_base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{hydra_base}_{suffix}.lora_ups.{e}.weight"].shape == (
                out_dim,
                r,
            )

    # Plain leg: must also be defused — lora_up.weight split per q/k/v,
    # fused prefix fully gone.
    plain_base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{plain_base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        assert loaded[f"{plain_base}_{suffix}.lora_up.weight"].shape == (out_dim, r), (
            f"plain-LoRA self_attn_{suffix} lora_up.weight missing or still fused — "
            "hydra save pipeline didn't defuse the plain leg"
        )
        assert f"{plain_base}_{suffix}.alpha" in loaded
        # plain leg must NOT have hydra-only keys
        assert f"{plain_base}_{suffix}.lora_ups.0.weight" not in loaded
        assert f"{plain_base}_{suffix}.router.weight" not in loaded
    # fused prefix must be entirely purged
    for k in loaded:
        assert not k.startswith(plain_prefix), f"fused plain-LoRA key survived: {k}"


def test_save_ortho_hydra_roundtrip(tmp_path: Path):
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    # OrthoHydraLoRA runtime keys: S_p is 3-D (E, r, r); P_bases is (E, out, r)
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_bases": torch.randn(E, 3 * out_dim, r),
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
    }

    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_hydra_to_hydra")

    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        assert loaded[f"{base}_{suffix}.lora_down.weight"].shape == (r, in_dim)
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)
    for k in loaded:
        assert not k.endswith(".S_p") and not k.endswith(".S_q")
        assert not k.endswith(".P_bases") and not k.endswith(".P_basis")


def test_save_ortho_hydra_legacy_P_basis_still_bakes(tmp_path: Path):
    """Legacy OrthoHydra checkpoints (pre-disjoint-bases) used a single
    (out, r) ``P_basis`` shared across experts. The save pipeline must still
    bake these into hydra moe form so old artifacts remain convertible.
    """
    E, r, in_dim, out_dim = 4, 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{prefix}.S_p": torch.randn(E, r, r),
        f"{prefix}.S_q": torch.randn(r, r),
        f"{prefix}.P_basis": torch.randn(3 * out_dim, r),  # legacy 2-D
        f"{prefix}.Q_basis": torch.randn(r, in_dim),
        f"{prefix}.lambda_layer": torch.randn(1, r),
        f"{prefix}.alpha": _alpha(r),
        f"{prefix}.router.weight": torch.randn(E, in_dim),
        f"{prefix}.router.bias": torch.randn(E),
    }
    loaded = _save_and_reload(sd, tmp_path, save_variant="ortho_hydra_to_hydra")
    base = "lora_unet_blocks_0_self_attn"
    for suffix in ("q_proj", "k_proj", "v_proj"):
        for e in range(E):
            assert loaded[f"{base}_{suffix}.lora_ups.{e}.weight"].shape == (out_dim, r)


# ---------------------------------------------------------------------------
# Metadata stamp
# ---------------------------------------------------------------------------


def _load_metadata(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return f.metadata() or {}


def test_metadata_stamps_ss_network_spec(tmp_path: Path):
    r, in_dim, out_dim = 4, 8, 12
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = _make_std_lora_sd(prefix, r, in_dim, out_dim)

    out = tmp_path / "out.safetensors"
    lora_save.save_network_weights(
        dict(sd),
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": "lora"},
        save_variant="standard",
    )
    meta = _load_metadata(out)
    assert meta.get("ss_network_spec") == "lora"
