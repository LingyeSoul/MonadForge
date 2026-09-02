"""Official LyCORIS LoKr backend and compatibility-boundary regressions."""

from __future__ import annotations

import argparse
import copy

import pytest
import torch
import networks.lora_modules.lokr as lokr_impl
from lycoris.modules.lokr import LokrModule as LycorisLokrModule
from safetensors import safe_open
from safetensors.torch import save_file

from networks.lora_anima.factory import create_network, create_network_from_weights
from networks.lora_anima.loading import _normalize_native_lokr_keys
from networks.lora_modules.lokr import LoKRModule


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


class _MultiShapeDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.square = Block(512, 512)
        self.wide = Block(512, 1536)


def _make_module(
    *,
    in_dim: int = 512,
    out_dim: int = 512,
    rank: int = 4,
    alpha: float = 16,
    factor: int = -1,
    multiplier: float = 1.0,
    full_factor: bool = False,
    decompose_both: bool = False,
) -> tuple[torch.nn.Linear, LoKRModule]:
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    base.weight.requires_grad_(False)
    module = LoKRModule(
        "lora_test",
        base,
        multiplier=multiplier,
        lora_dim=rank,
        alpha=alpha,
        lokr_factor=factor,
        full_factor=full_factor,
        decompose_both=decompose_both,
    )
    return base, module


def _make_delta_nonzero(module: LoKRModule) -> None:
    with torch.no_grad():
        if module.use_w2:
            module.lokr_w2.normal_(0, 0.1)
        else:
            module.lokr_w2_b.normal_(0, 0.1)


def _legacy_inv_scale_state() -> dict[str, torch.Tensor]:
    lora_name = "lora_unet_block_proj"
    return {
        f"{lora_name}.lokr_w1": torch.randn(4, 3),
        f"{lora_name}.lokr_w2_a": torch.randn(5, 2),
        f"{lora_name}.lokr_w2_b": torch.randn(2, 4),
        f"{lora_name}.alpha": torch.tensor(2),
        f"{lora_name}.inv_scale": torch.rand(12) + 0.5,
    }


def _legacy_inv_scale_delta(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    prefix = "lora_unet_block_proj"
    w1 = state_dict[f"{prefix}.lokr_w1"]
    w2 = state_dict[f"{prefix}.lokr_w2_a"] @ state_dict[f"{prefix}.lokr_w2_b"]
    return torch.kron(w1, w2) * state_dict[f"{prefix}.inv_scale"].unsqueeze(0)


def test_backend_is_official_lycoris_module():
    assert issubclass(LoKRModule, LycorisLokrModule)


def test_decomposed_layout_and_state_keys_match_lycoris():
    _, module = _make_module(rank=4, alpha=16, factor=-1)

    assert module.use_w1 is True
    assert module.use_w2 is False
    assert module.scale == 4.0
    assert module.lokr_w1.shape == (16, 16)
    assert module.lokr_w2_a.shape == (32, 4)
    assert module.lokr_w2_b.shape == (4, 32)
    assert set(module.state_dict()) == {
        "alpha",
        "lokr_w1",
        "lokr_w2_a",
        "lokr_w2_b",
    }


def test_full_factor_maps_to_lycoris_full_matrix_and_unit_scale():
    _, module = _make_module(
        rank=4,
        alpha=16,
        factor=8,
        full_factor=True,
        decompose_both=True,
    )

    assert module.full_matrix is True
    assert module.use_w1 is True
    assert module.use_w2 is True
    assert module.scale == 1.0
    assert module.alpha.item() == 4
    assert set(module.state_dict()) == {"alpha", "lokr_w1", "lokr_w2"}


def test_legacy_large_dim_uses_official_full_matrix_scale():
    _, module = _make_module(rank=114514, alpha=32, factor=16)

    assert module.use_w1 and module.use_w2
    assert module.scale == 1.0
    assert module.alpha.item() == 114514


def test_channel_scaling_is_rejected_at_direct_module_boundary():
    base = torch.nn.Linear(12, 20, bias=False)
    with pytest.raises(ValueError, match="channel scaling"):
        LoKRModule(
            "lora_test",
            base,
            lora_dim=4,
            alpha=4,
            channel_scale=torch.ones(12),
        )


def test_wrapper_forward_and_gradients_equal_direct_lycoris():
    torch.manual_seed(123)
    wrapped_base = torch.nn.Linear(512, 512, bias=False)
    wrapped_base.weight.requires_grad_(False)
    official_base = copy.deepcopy(wrapped_base)

    torch.manual_seed(456)
    wrapped = LoKRModule(
        "wrapped",
        wrapped_base,
        multiplier=0.75,
        lora_dim=4,
        alpha=16,
        lokr_factor=16,
    )
    official = LycorisLokrModule(
        "official",
        official_base,
        multiplier=0.75,
        lora_dim=4,
        alpha=16,
        factor=16,
        # The official regular path is the numerical reference. MonadForge's
        # wrapper uses the official bypass operations (LyCORIS 4.0 applies
        # the alpha/rank scale inside the bypass itself; pre-4.0 the wrapper
        # had to restore it at the call site).
        bypass_mode=False,
    )
    official.load_state_dict(wrapped.state_dict())
    _make_delta_nonzero(wrapped)
    official.load_state_dict(wrapped.state_dict())

    wrapped.apply_to()
    official.apply_to()
    x = torch.randn(2, 3, 512)
    wrapped_out = wrapped_base(x)
    official_out = official_base(x)

    # Rebuild mode materializes kron(W1, W2), while bypass mode evaluates the
    # equivalent grouped linears. Their FP32 accumulation orders can differ by
    # one or two ulps on this 512-wide case.
    assert torch.allclose(wrapped_out, official_out, atol=2e-6, rtol=1e-6)

    wrapped_out.square().mean().backward()
    official_out.square().mean().backward()
    wrapped_grads = dict(wrapped.named_parameters())
    official_grads = dict(official.named_parameters())
    assert wrapped_grads.keys() == official_grads.keys()
    for name, parameter in wrapped_grads.items():
        assert parameter.grad is not None
        assert torch.allclose(
            parameter.grad,
            official_grads[name].grad,
            atol=1e-6,
            rtol=1e-5,
        )


def test_bypass_scale_applied_once_and_pre40_compensator_double_scales():
    # Locks the 4.0 call-site invariant: the wrapper passes multiplier only
    # (the official bypass applies ``self.scale`` itself), so reviving the
    # pre-4.0 ``multiplier * self.scale`` compensator at the call site would
    # inflate the residual by exactly ``self.scale``.
    base, module = _make_module(rank=4, alpha=16, factor=16, multiplier=0.75)
    _make_delta_nonzero(module)
    module.apply_to()
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        base_out = module.org_forward(x)
        wrapped_out = base(x)
        legacy_form = base_out + module.bypass_forward_diff(
            x, scale=module.multiplier * module.scale
        )

    delta_norm = (wrapped_out - base_out).norm()
    legacy_norm = (legacy_form - base_out).norm()
    assert delta_norm > 0
    assert (legacy_norm / delta_norm).item() == pytest.approx(module.scale, rel=1e-3)


def test_fp32_compute_casts_all_official_bypass_operands(monkeypatch):
    base, module = _make_module(
        in_dim=64,
        out_dim=64,
        rank=4,
        alpha=16,
        factor=8,
    )
    base.half()
    module.half()
    module.fp32_compute = True
    module.train()
    _make_delta_nonzero(module)

    seen_dtypes: list[torch.dtype] = []
    seen_backends: list[object] = []
    original = lokr_impl.lycoris_lokr_bypass_forward_diff

    def spy(h, org_out, *weights, **kwargs):
        seen_dtypes.append(h.dtype)
        seen_dtypes.extend(weight.dtype for weight in weights if weight is not None)
        seen_backends.append(kwargs.get("backend"))
        return original(h, org_out, *weights, **kwargs)

    monkeypatch.setattr(lokr_impl, "lycoris_lokr_bypass_forward_diff", spy)
    module.apply_to()
    x = torch.randn(2, 3, 64, dtype=torch.float16)
    expected = (
        module.org_forward(x).float()
        + torch.nn.functional.linear(x.float(), module.get_weight()).float()
    )
    output = base(x)

    assert output.dtype == torch.float32
    assert torch.allclose(output, expected, atol=1e-3, rtol=1e-3)
    assert seen_dtypes
    assert set(seen_dtypes) == {torch.float32}
    # The fp32 lane must pin the reference backend: fused Triton tiers stay
    # out of the V100/fp16 protection lane (see networks/CLAUDE.md).
    assert seen_backends == ["torch"]


def test_zero_init_trains_the_official_w2_chain_end_first():
    base, module = _make_module(rank=4, alpha=4, factor=16)
    module.apply_to()
    loss = base(torch.randn(2, 3, 512)).square().mean()
    loss.backward()

    assert module.lokr_w2_b.grad is not None
    assert torch.count_nonzero(module.lokr_w2_b.grad) > 0
    assert module.lokr_w2_a.grad is not None
    assert torch.count_nonzero(module.lokr_w2_a.grad) == 0


def test_merge_uses_official_functional_delta():
    base, module = _make_module(rank=4, alpha=16, factor=16, multiplier=0.75)
    _make_delta_nonzero(module)
    original = base.weight.detach().clone()
    expected = module.get_weight().clone()

    module.merge_to(module.state_dict(), dtype=torch.float32, device="cpu")

    assert torch.allclose(base.weight, original + expected, atol=1e-6, rtol=1e-5)


def test_fuse_unfuse_preserves_forward_and_restores_weight():
    base, module = _make_module(rank=4, alpha=16, factor=16, multiplier=0.75)
    _make_delta_nonzero(module)
    module.apply_to()
    x = torch.randn(2, 3, 512)
    original = base.weight.detach().clone()
    expected = base(x)

    module.fuse_weight()
    assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)

    module.unfuse_weight()
    assert torch.allclose(base.weight, original, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base(x), expected, atol=1e-5, rtol=1e-5)


def test_fuse_unfuse_is_reversible_with_rank_dropout_in_training_mode():
    base = torch.nn.Linear(512, 512, bias=False)
    module = LoKRModule(
        "lora_test",
        base,
        multiplier=0.75,
        lora_dim=4,
        alpha=16,
        lokr_factor=16,
        rank_dropout=0.5,
    )
    _make_delta_nonzero(module)
    module.apply_to()
    module.train()
    original = base.weight.detach().clone()

    module.fuse_weight()
    module.unfuse_weight()

    assert module.training is True
    assert torch.allclose(base.weight, original, atol=1e-6, rtol=1e-6)


def test_legacy_short_factor_keys_normalize_to_official_names():
    old = {
        "P.w1a": torch.randn(8, 4),
        "P.w1b": torch.randn(4, 8),
        "P.w2a": torch.randn(16, 4),
        "P.w2b": torch.randn(4, 16),
    }
    normalized = _normalize_native_lokr_keys(old)

    assert set(normalized) == {
        "P.lokr_w1_a",
        "P.lokr_w1_b",
        "P.lokr_w2_a",
        "P.lokr_w2_b",
    }


def test_native_decomposed_checkpoint_keeps_keys_and_infers_rank():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.lokr_w1": torch.randn(16, 16),
        f"{lora_name}.lokr_w2_a": torch.randn(32, 4),
        f"{lora_name}.lokr_w2_b": torch.randn(4, 32),
        f"{lora_name}.alpha": torch.tensor(16),
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

    assert normalized_sd.keys() == native_sd.keys()
    module = network.unet_loras[0]
    assert module.lora_dim == 4
    assert module.scale == 4.0
    assert module.lokr_w2_a.shape == (32, 4)

    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = network.load_state_dict(normalized_sd, strict=False)
    assert not any("lokr_w2" in key for key in info.missing_keys)
    assert not any("lokr_w2" in key for key in info.unexpected_keys)
    assert torch.equal(module.lokr_w2_a.detach(), native_sd[f"{lora_name}.lokr_w2_a"])
    assert torch.equal(module.lokr_w2_b.detach(), native_sd[f"{lora_name}.lokr_w2_b"])


def test_network_args_metadata_controls_decompose_both_factor():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.lokr_w1_a": torch.randn(8, 2),
        f"{lora_name}.lokr_w1_b": torch.randn(2, 8),
        f"{lora_name}.lokr_w2_a": torch.randn(64, 2),
        f"{lora_name}.lokr_w2_b": torch.randn(2, 64),
        f"{lora_name}.alpha": torch.tensor(2),
    }
    network, _ = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={
            "ss_network_spec": "lokr",
            "ss_network_args": ('{"algo":"lokr","factor":8,"decompose_both":true}'),
        },
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )

    module = network.unet_loras[0]
    assert network.cfg.lokr_factor == 8
    assert network.cfg.decompose_both is True
    assert module.use_w1 is False
    assert module.lokr_w1_a.shape == (8, 2)


def test_unstamped_decompose_both_checkpoint_infers_lycoris_factor():
    lora_name = "lora_unet_block_proj"
    native_sd = {
        f"{lora_name}.lokr_w1_a": torch.randn(8, 2),
        f"{lora_name}.lokr_w1_b": torch.randn(2, 8),
        f"{lora_name}.lokr_w2_a": torch.randn(64, 2),
        f"{lora_name}.lokr_w2_b": torch.randn(2, 64),
        f"{lora_name}.alpha": torch.tensor(2),
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

    module = network.unet_loras[0]
    assert network.cfg.lokr_factor == 8
    assert network.cfg.decompose_both is True
    assert module.lokr_w1_a.shape == (8, 2)


def test_unstamped_multishape_checkpoint_infers_one_consistent_factor():
    source = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=4,
        vae=None,
        text_encoders=[],
        unet=_MultiShapeDiT(),
        use_lokr="true",
        lokr_factor="16",
        channel_scaling_alpha="0",
    )
    source.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    native_sd = source.state_dict()

    restored, normalized_sd = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=native_sd,
        metadata={"ss_network_spec": "lokr"},
        ae=None,
        text_encoders=[],
        unet=_MultiShapeDiT(),
    )

    assert restored.cfg.lokr_factor == 16
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(normalized_sd, strict=False)
    assert not info.missing_keys
    assert not info.unexpected_keys


def test_legacy_inv_scale_checkpoint_converts_to_standard_lora():
    legacy_sd = _legacy_inv_scale_state()
    expected_delta = _legacy_inv_scale_delta(legacy_sd)
    network, normalized_sd = create_network_from_weights(
        multiplier=1.0,
        file=None,
        weights_sd=legacy_sd,
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "2"},
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(12, 20),
    )

    assert network._network_spec.name == "lora"
    assert any(key.endswith(".lora_down.weight") for key in normalized_sd)
    assert any(key.endswith(".lora_up.weight") for key in normalized_sd)
    assert not any("lokr_" in key or key.endswith("inv_scale") for key in normalized_sd)

    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = network.load_state_dict(normalized_sd, strict=False)
    assert not info.missing_keys
    assert not info.unexpected_keys
    assert torch.allclose(
        network.unet_loras[0].get_weight(),
        expected_delta,
        atol=1e-5,
        rtol=1e-5,
    )


def test_runtime_harness_loads_factory_converted_legacy_checkpoint(
    tmp_path, monkeypatch
):
    from library.anima import checkpoint as checkpoint_module
    from library.anima import weights as anima_weights
    from library.anima.checkpoint import AnimaCheckpointLayout
    from library.runtime.harness import build_anima

    legacy_sd = _legacy_inv_scale_state()
    expected_delta = _legacy_inv_scale_delta(legacy_sd)
    adapter = tmp_path / "legacy-lokr.safetensors"
    save_file(
        legacy_sd,
        str(adapter),
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "2"},
    )
    model = _TinyDiT(12, 20)
    monkeypatch.setattr(
        anima_weights,
        "load_anima_model",
        lambda **kwargs: model,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "inspect_anima_checkpoint",
        lambda _path: AnimaCheckpointLayout(
            "anima-2048-28", "anima-2.1b-base-v1", 28, 2048, 16, "net."
        ),
    )
    monkeypatch.setattr(
        checkpoint_module, "anima_checkpoint_sha256", lambda _path: "a" * 64
    )
    args = argparse.Namespace(
        device="cpu",
        dtype="float32",
        attn_mode="torch",
        gradient_checkpointing=False,
        compile=False,
    )

    bundle = build_anima(args, dit_path="unused", adapter=str(adapter))

    assert bundle.network._network_spec.name == "lora"
    assert torch.allclose(
        bundle.network.unet_loras[0].get_weight(),
        expected_delta,
        atol=1e-5,
        rtol=1e-5,
    )


def test_direct_load_rejects_legacy_inv_scale_checkpoint(tmp_path):
    adapter = tmp_path / "legacy-lokr.safetensors"
    save_file(_legacy_inv_scale_state(), str(adapter))
    network = create_network(
        multiplier=1.0,
        network_dim=2,
        network_alpha=2,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(12, 20),
        use_lokr="true",
        channel_scaling_alpha="0",
    )
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)

    with pytest.raises(RuntimeError, match="Use --dim_from_weights"):
        network.load_weights(str(adapter))


@pytest.mark.parametrize("full_factor", [False, True])
def test_checkpoint_save_and_reload_preserves_official_delta(tmp_path, full_factor):
    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=16,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        use_lokr="true",
        lokr_factor="16",
        lokr_full_factor=str(full_factor).lower(),
    )
    network.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    source = network.unet_loras[0]
    _make_delta_nonzero(source)
    expected = source.get_weight().clone()

    out = tmp_path / f"lokr-{full_factor}.safetensors"
    network.save_weights(str(out), torch.float32, metadata={})

    with safe_open(str(out), framework="pt") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata() or {}
        saved_alpha = handle.get_tensor(f"{source.lora_name}.alpha").item()
    assert metadata["ss_network_spec"] == "lokr"
    assert metadata["ss_lokr_full_factor"] == str(full_factor).lower()
    assert not any(key.endswith((".w1a", ".w1b", ".w2a", ".w2b")) for key in keys)
    if full_factor:
        assert saved_alpha == 4
        assert source.scale == 1.0
        assert any(key.endswith(".lokr_w2") for key in keys)
    else:
        assert saved_alpha == 16
        assert source.scale == 4.0
        assert any(key.endswith(".lokr_w2_a") for key in keys)

    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
    )
    restored.apply_to(text_encoders=[], unet=None, apply_text_encoder=False)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert not info.missing_keys
    assert not info.unexpected_keys
    actual = restored.unet_loras[0].get_weight()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
