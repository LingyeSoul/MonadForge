"""Tests for freeing dual-resident bf16 base weights after ConvRot quant."""

from __future__ import annotations

import torch
from torch import nn

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.convrot.free_base import free_linear_weight_storage, is_base_weight_freed


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


def test_free_linear_weight_storage_uses_meta_and_is_idempotent() -> None:
    linear = nn.Linear(32, 16, bias=False)
    linear.weight.requires_grad_(False)
    nbytes = free_linear_weight_storage(linear)
    assert nbytes == 32 * 16 * 4  # fp32 default
    assert is_base_weight_freed(linear)
    assert linear.weight.device.type == "meta"
    assert free_linear_weight_storage(linear) == 0


def test_apply_frees_base_weight_by_default() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    # move to cuda if available to assert storage release intent
    device = torch.device("cpu")
    linear = linear.to(device)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    x = torch.randn(2, 32, device=device)
    y_before = linear(x)

    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16
    )
    assert result.patched_count == 1
    assert result.freed_modules == 1
    assert result.freed_bytes > 0
    assert is_base_weight_freed(linear)
    assert linear.weight.device.type == "meta"

    # org_forward still works via int8 buffers on the LoRA module
    y_after = network.unet_loras[0].org_forward(x)
    rel = (y_after - y_before).norm() / y_before.norm().clamp_min(1e-8)
    assert rel.item() < 0.05


def test_apply_can_keep_base_weights() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        free_base_weights=False,
    )
    assert result.patched_count == 1
    assert result.freed_modules == 0
    assert not is_base_weight_freed(linear)
    assert linear.weight.device.type != "meta"


def test_apply_frees_base_weights_incrementally_across_modules() -> None:
    """Incremental free-base: every patched base is meta as soon as apply returns.

    The apply loop frees each base weight right after installing its int8
    payload (instead of one batch free at the end) to avoid the transient
    "full bf16 base + all payloads" overlap peak. ``freed_modules``/``freed_bytes``
    must still reflect the true totals and the math must stay within tolerance.
    """
    torch.manual_seed(0)
    linears = [nn.Linear(32, 32, bias=False) for _ in range(3)]
    for lin in linears:
        lin.weight.requires_grad_(False)
    loras = [
        _FakeLoRAModule(f"blocks.{i}.mlp.layer1", lin)
        for i, lin in enumerate(linears)
    ]
    network = _FakeLoRANetwork(loras)
    x = torch.randn(2, 32)
    y_before = [lin(x) for lin in linears]

    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16
    )
    assert result.patched_count == 3
    assert result.freed_modules == 3
    expected_bytes = sum(32 * 32 * 4 for _ in linears)
    assert result.freed_bytes == expected_bytes
    for lora, lin, ref in zip(loras, linears, y_before):
        assert is_base_weight_freed(lin)
        assert lin.weight.device.type == "meta"
        rel = (lora.org_forward(x) - ref).norm() / ref.norm().clamp_min(1e-8)
        assert rel.item() < 0.05


def test_dry_run_does_not_free_base_weights() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16, dry_run=True
    )
    assert result.patched_count == 1
    assert result.freed_modules == 0
    assert not is_base_weight_freed(linear)


def test_freed_linear_survives_module_to_and_parent_to() -> None:
    """Regression: free-base meta weights must not break accelerator.prepare.

    Hot-test crash was:
    ``NotImplementedError: Cannot copy out of meta tensor`` inside
    ``model.to(self.device)`` after ConvRot freed 56 mlp weights.
    """
    linear = nn.Linear(32, 16, bias=False)
    linear.weight.requires_grad_(False)
    free_linear_weight_storage(linear)
    assert linear.weight.device.type == "meta"

    # Direct .to on the freed module.
    linear.to("cpu")
    assert linear.weight.device.type == "meta"

    # Parent walk (same path Accelerate uses for unet.to / prepare_model).
    class _UNet(nn.Module):
        def __init__(self, child: nn.Module) -> None:
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.layer1 = child  # type: ignore[attr-defined]

    unet = _UNet(linear)
    unet.to("cpu")
    unet.to(dtype=torch.float16)
    assert linear.weight.device.type == "meta"
    # dtype cast is also skipped for meta; shape preserved
    assert tuple(linear.weight.shape) == (16, 32)

    if torch.cuda.is_available():
        unet.to("cuda")
        assert linear.weight.device.type == "meta"
