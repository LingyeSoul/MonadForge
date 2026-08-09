"""Tests for ConvRot prequant checkpoint load/save and apply path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.convrot.free_base import is_base_weight_freed
from library.runtime.convrot.prequant import (
    FORMAT_V1,
    build_prequant_layers_from_modules,
    load_prequant_checkpoint,
    save_prequant_checkpoint,
)
from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.linear_w8a16 import w8a16_forward_from_buffers


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, 4, bias=False)
        self.lora_up = nn.Linear(4, linear.out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x) + self.lora_up(self.lora_down(x))


class _FakeLoRANetwork(nn.Module):
    def __init__(self, loras: list[nn.Module]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def _frozen_linear(in_f: int, out_f: int, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=False)
    linear.weight.requires_grad_(False)
    return linear


def test_save_load_roundtrip_native_v1(tmp_path: Path) -> None:
    linear = _frozen_linear(32, 16, seed=1)
    layers = build_prequant_layers_from_modules(
        {"blocks.0.mlp.layer1": linear}, group_size=16
    )
    path = save_prequant_checkpoint(
        tmp_path / "pq.safetensors",
        layers,
        group_size=16,
        mode="w8a16",
    )
    ckpt = load_prequant_checkpoint(path)
    assert ckpt.format == FORMAT_V1
    assert ckpt.group_size == 16
    assert ckpt.layer_count == 1
    layer = ckpt.get("blocks.0.mlp.layer1")
    assert layer is not None
    q_ref, s_ref = layers["blocks.0.mlp.layer1"]
    assert torch.equal(layer.quantized_weight, q_ref)
    assert torch.allclose(layer.scale, s_ref)


def test_load_accepts_weight_scale_alias(tmp_path: Path) -> None:
    linear = _frozen_linear(32, 8, seed=2)
    q, scale = rotate_and_quantize_weight(linear.weight.detach(), 16)
    # Manual safetensors with Comfy-style scale key.
    from safetensors.torch import save_file

    path = tmp_path / "alias.safetensors"
    save_file(
        {
            "blocks.0.mlp.layer1.weight": q.cpu(),
            "blocks.0.mlp.layer1.weight_scale": scale.cpu().float(),
        },
        str(path),
        metadata={"group_size": "16", "format": "comfy_like"},
    )
    ckpt = load_prequant_checkpoint(path)
    assert ckpt.get("blocks.0.mlp.layer1") is not None
    assert ckpt.group_size == 16


def test_apply_prequant_matches_online_and_frees_base(tmp_path: Path) -> None:
    torch.manual_seed(3)
    mlp = _frozen_linear(32, 64, seed=3)
    # Snapshot online quant for reference before free.
    q_online, s_online = rotate_and_quantize_weight(mlp.weight.detach(), 16)
    layers = {"blocks.0.mlp.layer1": (q_online, s_online)}
    path = save_prequant_checkpoint(
        tmp_path / "pq2.safetensors", layers, group_size=16, mode="w8a16"
    )

    # Fresh identical linear for prequant apply.
    mlp2 = _frozen_linear(32, 64, seed=3)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp2)])
    x = torch.randn(5, 32)
    y_online = w8a16_forward_from_buffers(x, q_online, s_online, group_size=16)

    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        weight_source="prequant_checkpoint",
        prequant_path=str(path),
        free_base_weights=True,
    )
    assert result.patched_count == 1
    assert result.weight_source == "prequant_checkpoint"
    assert result.freed_modules == 1
    assert is_base_weight_freed(mlp2)
    y_pre = network.unet_loras[0].org_forward(x)
    rel = (y_pre - y_online).norm() / y_online.norm().clamp_min(1e-8)
    assert rel.item() < 1e-5


def test_apply_prequant_missing_layer_skips_or_raises(tmp_path: Path) -> None:
    # File only has a different layer name.
    linear = _frozen_linear(32, 32, seed=4)
    layers = build_prequant_layers_from_modules(
        {"blocks.0.mlp.layer2": linear}, group_size=16
    )
    path = save_prequant_checkpoint(tmp_path / "miss.safetensors", layers, group_size=16)

    mlp = _frozen_linear(32, 32, seed=5)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp)])
    with pytest.raises(RuntimeError, match="patched=0"):
        apply_convrot_to_lora_network(
            network,
            mode="w8a16",
            scope="mlp",
            group_size=16,
            weight_source="prequant_checkpoint",
            prequant_path=str(path),
        )


def test_apply_prequant_group_size_mismatch_raises(tmp_path: Path) -> None:
    linear = _frozen_linear(32, 32, seed=6)
    layers = build_prequant_layers_from_modules(
        {"blocks.0.mlp.layer1": linear}, group_size=16
    )
    path = save_prequant_checkpoint(tmp_path / "gs.safetensors", layers, group_size=16)
    mlp = _frozen_linear(32, 32, seed=6)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp)])
    with pytest.raises(ValueError, match="group_size"):
        apply_convrot_to_lora_network(
            network,
            mode="w8a16",
            scope="mlp",
            group_size=32,  # mismatch vs file 16
            weight_source="prequant_checkpoint",
            prequant_path=str(path),
            prequant_group_size_strict=True,
        )


def test_apply_prequant_requires_path() -> None:
    mlp = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp)])
    with pytest.raises(ValueError, match="prequant_path"):
        apply_convrot_to_lora_network(
            network,
            mode="w8a16",
            scope="mlp",
            group_size=16,
            weight_source="prequant_checkpoint",
            prequant_path=None,
        )


def test_online_source_rejects_stray_prequant_path(tmp_path: Path) -> None:
    mlp = _frozen_linear(32, 32)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", mlp)])
    with pytest.raises(ValueError, match="only valid"):
        apply_convrot_to_lora_network(
            network,
            mode="w8a16",
            scope="mlp",
            group_size=16,
            weight_source="online_from_bf16",
            prequant_path=str(tmp_path / "x.safetensors"),
        )
