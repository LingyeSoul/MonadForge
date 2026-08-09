"""End-to-end contracts for the optional ConvRot frozen-base training path."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from library.anima.merge import NonBakeableError, merge_adapter_into_dit
from library.config.cli_args import add_dit_training_arguments
from library.training.convrot import maybe_apply_convrot_base
from library.training.metadata import build_training_metadata
from networks.lora_anima.merge_guard import raise_if_convrot_active


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, 2, bias=False)
        self.lora_up = nn.Linear(2, linear.out_features, bias=False)


class _FakeNetwork(nn.Module):
    def __init__(self, loras: list[nn.Module]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def _args(**kwargs) -> SimpleNamespace:
    defaults = {
        "base_compute": "bf16",
        "convrot_group_size": 16,
        "convrot_scope": "mlp",
        "convrot_hadamard": "sylvester",
        "convrot_weight_source": "online_from_bf16",
        "convrot_prequant_path": None,
        "convrot_min_in_features": 0,
        "convrot_largest_in_features_only": False,
        "convrot_large_layer_mode": None,
        "convrot_large_min_in_features": None,
        "block_swap_transfer_dtype": "bf16",
        "blocks_to_swap": 0,
        "network_args": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _network_for(linear: nn.Linear, name: str = "blocks.0.mlp.layer1") -> _FakeNetwork:
    return _FakeNetwork([_FakeLoRAModule(name, linear)])


def test_convrot_cli_defaults_and_choices() -> None:
    parser = argparse.ArgumentParser()
    add_dit_training_arguments(parser)

    args = parser.parse_args([])
    assert args.base_compute == "bf16"
    assert args.convrot_group_size == 256
    assert args.convrot_hadamard == "sylvester"
    assert args.convrot_scope == "mlp"
    assert args.convrot_large_layer_mode is None

    compute = next(a for a in parser._actions if a.dest == "base_compute")
    group = next(a for a in parser._actions if a.dest == "convrot_group_size")
    assert set(compute.choices or ()) == {
        "bf16",
        "w8a16_convrot",
        "w8a8_convrot",
    }
    assert set(group.choices or ()) == {64, 256, 1024}
    assert parser.parse_args(["--convrot_large_layer_mode", "none"])


def test_maybe_apply_convrot_base_is_noop_for_bf16() -> None:
    linear = nn.Linear(32, 32, bias=False)
    network = _network_for(linear)

    assert maybe_apply_convrot_base(_args(), network) is False
    assert not hasattr(linear, "_convrot_quantized_weight")


@pytest.mark.parametrize("base_compute", ["w8a16_convrot", "w8a8_convrot"])
def test_maybe_apply_convrot_base_freezes_and_patches(base_compute: str) -> None:
    linear = nn.Linear(32, 32, bias=False)
    unet = nn.Module()
    unet.linear = linear
    args = _args(base_compute=base_compute)

    assert maybe_apply_convrot_base(args, _network_for(linear), unet=unet) is True
    assert linear.weight.requires_grad is False
    assert linear.weight.device.type == "meta"
    assert hasattr(linear, "_convrot_quantized_weight")
    assert args._convrot_apply_result.patched_count == 1


def test_maybe_apply_convrot_base_rejects_dora() -> None:
    linear = nn.Linear(32, 32, bias=False)
    with pytest.raises(ValueError, match="DoRA"):
        maybe_apply_convrot_base(
            _args(base_compute="w8a16_convrot", network_args=["dora_wd=true"]),
            _network_for(linear),
        )


def test_maybe_apply_convrot_base_honors_size_filters() -> None:
    small = nn.Linear(32, 64, bias=False)
    large = nn.Linear(128, 32, bias=False)
    network = _FakeNetwork(
        [
            _FakeLoRAModule("blocks.0.mlp.layer1", small),
            _FakeLoRAModule("blocks.0.mlp.layer2", large),
        ]
    )
    args = _args(
        base_compute="w8a16_convrot",
        convrot_min_in_features=64,
        convrot_largest_in_features_only=True,
    )

    assert maybe_apply_convrot_base(args, network) is True
    assert args._convrot_apply_result.patched_count == 1
    assert args._convrot_apply_result.patches[0].name == "blocks.0.mlp.layer2"


def test_maybe_apply_convrot_base_maps_none_large_mode_to_disabled() -> None:
    linear = nn.Linear(32, 32, bias=False)
    args = _args(
        base_compute="w8a16_convrot",
        convrot_large_layer_mode="none",
        convrot_large_min_in_features=0,
    )

    assert maybe_apply_convrot_base(args, _network_for(linear)) is True
    assert args._convrot_apply_result.large_layer_mode is None


class _MetadataArgs:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)

    def __getattr__(self, _name: str):
        return None


def _metadata(monkeypatch, **values) -> dict:
    monkeypatch.setattr(
        "library.training.hashing.get_git_revision_hash", lambda: "test-revision"
    )
    defaults = {
        "base_compute": "bf16",
        "convrot_group_size": 256,
        "convrot_scope": "mlp",
        "convrot_hadamard": "sylvester",
        "convrot_weight_source": "online_from_bf16",
        "convrot_min_in_features": 0,
        "convrot_largest_in_features_only": False,
        "convrot_large_layer_mode": "none",
        "convrot_large_min_in_features": 0,
    }
    defaults.update(values)
    args = _MetadataArgs(**defaults)
    return build_training_metadata(
        args,
        session_id=1,
        training_started_at=1.0,
        text_encoder_lr=None,
        optimizer_name="AdamW",
        optimizer_args="",
        model_version="anima",
        num_train_images=1,
        num_val_images=0,
        num_reg_images=0,
        num_batches_per_epoch=1,
        num_train_epochs=1,
    )


def test_training_metadata_stamps_only_active_convrot(monkeypatch) -> None:
    baseline = _metadata(monkeypatch)
    assert baseline["ss_base_compute"] == "bf16"
    assert "ss_convrot_mode" not in baseline

    active = _metadata(
        monkeypatch,
        base_compute="w8a8_convrot",
        convrot_group_size=64,
        convrot_hadamard="regular",
    )
    assert active["ss_convrot_mode"] == "w8a8"
    assert active["ss_convrot_group_size"] == "64"
    assert active["ss_convrot_hadamard"] == "regular"
    assert "ss_convrot_large_layer_mode" not in active


def test_convrot_checkpoint_metadata_blocks_plain_bake(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.safetensors"
    save_file(
        {"dummy": torch.zeros(1)},
        str(adapter),
        metadata={"ss_base_compute": "w8a16_convrot"},
    )

    with pytest.raises(NonBakeableError, match="ConvRot base_compute"):
        merge_adapter_into_dit(adapter, tmp_path / "missing-dit.safetensors")


def test_runtime_convrot_network_blocks_fuse() -> None:
    network = SimpleNamespace(
        text_encoder_loras=[],
        unet_loras=[SimpleNamespace(_convrot_mode="w8a16")],
    )

    with pytest.raises(RuntimeError, match="refused for ConvRot"):
        raise_if_convrot_active(network, context="fuse_weights")


def test_train_applies_convrot_after_adapter_and_before_compile() -> None:
    source = (Path(__file__).parents[1] / "train.py").read_text(encoding="utf-8")
    start = source.index("def _create_and_apply_network")
    end = source.index("def _setup_optimizer_and_dataloader", start)
    hook = source[start:end]

    apply_index = hook.index("network.apply_to(")
    convrot_index = hook.index("maybe_apply_convrot_base(")
    compile_index = hook.index("compile_blocks_for_training(")
    assert apply_index < convrot_index < compile_index
