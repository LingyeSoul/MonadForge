"""Unit tests for apply_convrot_to_lora_network."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.convrot.checks import (
    assert_convrot_block_swap_mutex,
    normalize_base_compute,
    warn_convrot_blocks_to_swap,
)
from library.runtime.convrot.metadata import (
    metadata_indicates_convrot,
    raise_if_merge_with_convrot,
    stamp_convrot_metadata,
)


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear, *, dora: bool = False) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, 4, bias=False)
        self.lora_up = nn.Linear(4, linear.out_features, bias=False)
        if dora:
            self.dora_scale = nn.Parameter(torch.ones(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x) + self.lora_up(self.lora_down(x))


class _FakeDoRA(nn.Module):
    """Name-based DoRA detection."""

    def __init__(self, name: str, linear: nn.Linear) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, 4, bias=False)
        self.magnitude = nn.Parameter(torch.ones(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x)


class _FakeLoRANetwork(nn.Module):
    def __init__(self, loras: list[nn.Module]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def _frozen_linear(in_f: int, out_f: int) -> nn.Linear:
    linear = nn.Linear(in_f, out_f, bias=False)
    linear.weight.requires_grad_(False)
    return linear


def test_apply_convrot_patches_mlp_scope_only() -> None:
    torch.manual_seed(0)
    mlp = _frozen_linear(32, 64)
    attn = _frozen_linear(32, 32)
    network = _FakeLoRANetwork(
        [
            _FakeLoRAModule("blocks.0.mlp.layer1", mlp),
            _FakeLoRAModule("blocks.0.self_attn.output_proj", attn),
        ]
    )
    x = torch.randn(3, 32)
    expected = network.unet_loras[0].org_forward(x)

    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16
    )
    actual = network.unet_loras[0].org_forward(x)

    assert result.patched_count == 1
    assert result.patches[0].name == "blocks.0.mlp.layer1"
    assert hasattr(mlp, "_convrot_quantized_weight")
    assert not hasattr(attn, "_convrot_quantized_weight")
    rel = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
    assert rel.item() < 0.05


def test_apply_convrot_dry_run_does_not_mutate() -> None:
    mlp = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp)])
    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16, dry_run=True
    )
    assert result.patched_count == 1
    assert not hasattr(mlp, "_convrot_quantized_weight")


def test_apply_convrot_skips_adaln_and_trainable() -> None:
    adaln = _frozen_linear(32, 32)
    trainable = nn.Linear(32, 32, bias=False)
    network = _FakeLoRANetwork(
        [
            _FakeLoRAModule("blocks.0.adaln_up_mlp", adaln),
            _FakeLoRAModule("blocks.0.mlp.layer1", trainable),
        ]
    )
    with pytest.raises(RuntimeError, match="patched=0"):
        apply_convrot_to_lora_network(
            network, mode="w8a16", scope="mlp", group_size=16
        )


def test_apply_convrot_skips_missing_org_module_ref() -> None:
    mlp = _frozen_linear(32, 32)
    lora = _FakeLoRAModule("blocks.0.mlp.layer1", mlp)
    del lora.org_module_ref
    lora.org_forward = lambda x: x  # no __self__ Linear
    network = _FakeLoRANetwork([lora])
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        dry_run=True,
        allow_zero_patches=True,
    )
    assert result.patched_count == 0
    assert result.skipped_count == 1
    assert result.skipped[0].skipped_reason == "no_org_module_ref"


def test_apply_convrot_rejects_dora() -> None:
    mlp = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeDoRA("blocks.0.mlp.layer1", mlp)])
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        dry_run=True,
        allow_zero_patches=True,
    )
    assert result.patched_count == 0
    assert result.skipped[0].skipped_reason == "dora_unsupported"


def test_apply_convrot_w8a8_mode() -> None:
    torch.manual_seed(4)
    mlp = _frozen_linear(32, 16)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer2", mlp)])
    result = apply_convrot_to_lora_network(
        network, mode="w8a8", scope="mlp", group_size=16
    )
    assert result.mode == "w8a8"
    # P1.6: W8A8 stores contiguous [K,N] (= weight.T) for torch._int_mm.
    q = mlp._convrot_quantized_weight.weight
    assert tuple(q.shape) == (32, 16)
    assert getattr(network.unet_loras[0], "_convrot_weight_layout") == "kn"
    x = torch.randn(2, 32)
    y = network.unet_loras[0].org_forward(x)
    assert y.shape == (2, 16)
    assert torch.isfinite(y).all()


def test_apply_convrot_buffers_do_not_collide_with_int8_names() -> None:
    mlp = _frozen_linear(32, 32)
    lora = _FakeLoRAModule("blocks.0.mlp.layer1", mlp)
    lora.register_buffer("_int8_base_quantized_weight", torch.zeros(1, dtype=torch.int8))
    network = _FakeLoRANetwork([lora])
    apply_convrot_to_lora_network(network, mode="w8a16", scope="mlp", group_size=16)
    assert hasattr(lora, "_int8_base_quantized_weight")
    assert hasattr(mlp, "_convrot_quantized_weight")
    assert not hasattr(lora, "_convrot_quantized_weight")


def test_apply_convrot_raises_if_compiled_flag() -> None:
    mlp = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp)])
    network._anima_blocks_compiled = True
    with pytest.raises(RuntimeError, match="already compiled"):
        apply_convrot_to_lora_network(
            network, mode="w8a16", scope="mlp", group_size=16
        )


def test_normalize_and_mutex() -> None:
    assert normalize_base_compute("W8A16_CONVROT") == "w8a16_convrot"
    assert normalize_base_compute("bf16") == "bf16"
    with pytest.raises(ValueError):
        normalize_base_compute("int8")
    assert_convrot_block_swap_mutex(
        base_compute="w8a16_convrot", block_swap_transfer_dtype="bf16"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        assert_convrot_block_swap_mutex(
            base_compute="w8a16_convrot", block_swap_transfer_dtype="int8"
        )
    assert warn_convrot_blocks_to_swap(base_compute="bf16", blocks_to_swap=8) is None
    assert warn_convrot_blocks_to_swap(base_compute="w8a16_convrot", blocks_to_swap=0) is None
    msg = warn_convrot_blocks_to_swap(base_compute="w8a16_convrot", blocks_to_swap=8)
    assert msg is not None and "blocks_to_swap=8" in msg
    assert "payloads follow their DiT blocks" in msg
    assert "transfer/step latency" in msg


def test_metadata_stamp_and_merge_reject() -> None:
    meta: dict = {}
    stamp_convrot_metadata(
        meta,
        base_compute="w8a16_convrot",
        group_size=256,
        scope="mlp",
        weight_source="online_from_bf16",
        hadamard="regular",
    )
    assert metadata_indicates_convrot(meta)
    assert meta["ss_convrot_hadamard"] == "regular"
    with pytest.raises(RuntimeError, match="refused for ConvRot"):
        raise_if_merge_with_convrot(meta)
    raise_if_merge_with_convrot({"ss_base_compute": "bf16"})  # no raise


def test_lora_residual_stays_in_original_space() -> None:
    """lora_delta(x) must not be forced through RHT; only org_forward is rotated."""
    torch.manual_seed(5)
    base = _frozen_linear(32, 32)
    lora = _FakeLoRAModule("blocks.0.mlp.layer1", base)
    # Make delta deterministic and large relative to quant noise.
    with torch.no_grad():
        lora.lora_down.weight.zero_()
        lora.lora_up.weight.zero_()
        lora.lora_down.weight[0, 0] = 1.0
        lora.lora_up.weight[0, 0] = 2.0
    network = _FakeLoRANetwork([lora])
    x = torch.zeros(1, 32)
    x[0, 0] = 1.0
    y_before = lora(x)
    apply_convrot_to_lora_network(network, mode="w8a16", scope="mlp", group_size=16)
    y_after = lora(x)
    # Delta component is still lora_up(lora_down(x)) in original space.
    delta = lora.lora_up(lora.lora_down(x))
    base_out = lora.org_forward(x)
    assert torch.allclose(y_after, base_out + delta, atol=1e-5)
    assert y_after.shape == y_before.shape


def test_apply_min_in_features_skips_small() -> None:
    small = _frozen_linear(32, 64)   # in=32
    large = _frozen_linear(128, 64)  # in=128
    network = _FakeLoRANetwork(
        [
            _FakeLoRAModule("blocks.0.mlp.layer1", small),
            _FakeLoRAModule("blocks.0.mlp.layer2", large),
        ]
    )
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        min_in_features=64,
    )
    assert result.patched_count == 1
    assert result.patches[0].name == "blocks.0.mlp.layer2"
    assert any("min_in_features" in (s.skipped_reason or "") for s in result.skipped)
    assert hasattr(large, "_convrot_quantized_weight")
    assert not hasattr(small, "_convrot_quantized_weight")


def test_apply_largest_in_features_only() -> None:
    layer1 = _frozen_linear(32, 128)  # in=32
    layer2 = _frozen_linear(128, 32)  # in=128 → max
    network = _FakeLoRANetwork(
        [
            _FakeLoRAModule("blocks.0.mlp.layer1", layer1),
            _FakeLoRAModule("blocks.0.mlp.layer2", layer2),
        ]
    )
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        largest_in_features_only=True,
    )
    assert result.patched_count == 1
    assert result.patches[0].name == "blocks.0.mlp.layer2"
    assert result.largest_in_features_only is True


def test_apply_large_layer_mode_mixed() -> None:
    layer1 = _frozen_linear(32, 128)
    layer2 = _frozen_linear(128, 32)
    network = _FakeLoRANetwork(
        [
            _FakeLoRAModule("blocks.0.mlp.layer1", layer1),
            _FakeLoRAModule("blocks.0.mlp.layer2", layer2),
        ]
    )
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        large_layer_mode="w8a8",
        large_min_in_features=64,
    )
    assert result.patched_count == 2
    by_name = {p.name: p.mode for p in result.patches}
    assert by_name["blocks.0.mlp.layer1"] == "w8a16"
    assert by_name["blocks.0.mlp.layer2"] == "w8a8"
    assert network.unet_loras[0]._convrot_mode == "w8a16"
    assert network.unet_loras[1]._convrot_mode == "w8a8"


def test_apply_attaches_precomputed_hadamard_buffer() -> None:
    linear = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    apply_convrot_to_lora_network(network, mode="w8a16", scope="mlp", group_size=16)
    lora = network.unet_loras[0]
    assert hasattr(lora, "_convrot_hadamard")
    h = lora._convrot_hadamard
    assert tuple(h.shape) == (16, 16)
    x = torch.randn(2, 32)
    y = lora.org_forward(x)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scale storage path")
def test_apply_stores_scale_bf16_on_cuda_w8a16_only() -> None:
    """P1.10/11: W8A16 CUDA scale is bf16; W8A8 keeps float32 for grad gate."""
    device = torch.device("cuda")
    linear16 = nn.Linear(32, 16, bias=False).to(device)
    linear16.weight.requires_grad_(False)
    lora16 = _FakeLoRAModule("blocks.0.mlp.layer1", linear16)
    apply_convrot_to_lora_network(
        _FakeLoRANetwork([lora16]), mode="w8a16", scope="mlp", group_size=16
    )
    assert linear16._convrot_scale.weight.dtype == torch.bfloat16

    linear8 = nn.Linear(32, 16, bias=False).to(device)
    linear8.weight.requires_grad_(False)
    lora8 = _FakeLoRAModule("blocks.0.mlp.layer1", linear8)
    apply_convrot_to_lora_network(
        _FakeLoRANetwork([lora8]), mode="w8a8", scope="mlp", group_size=16
    )
    assert linear8._convrot_scale.weight.dtype == torch.float32
    x = torch.randn(2, 32, device=device, dtype=torch.bfloat16)
    assert torch.isfinite(lora16.org_forward(x)).all()
    assert torch.isfinite(lora8.org_forward(x)).all()


def test_apply_large_mode_requires_threshold() -> None:
    linear = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    with pytest.raises(ValueError, match="large_min_in_features"):
        apply_convrot_to_lora_network(
            network,
            mode="w8a16",
            scope="mlp",
            group_size=16,
            large_layer_mode="w8a8",
        )
