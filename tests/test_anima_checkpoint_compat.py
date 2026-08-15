from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from library.anima.checkpoint import (
    _BLOCK_SHAPES,
    _BLOCK_SUFFIXES,
    AnimaCheckpointLayout,
    inspect_anima_checkpoint,
)
from library.anima.compat import (
    ANIMA29_PREVIEW_V1_SHA256,
    adapter_identity_metadata,
    classify_anima_training,
    compatibility_for_layout,
    preflight_anima_training,
    validate_adapter_compatibility,
    validate_channel_stats_compatibility,
    validate_resume_model_signature,
)


def _shape_for_suffix(suffix: str) -> list[int]:
    return list(_BLOCK_SHAPES[suffix])


def _block_headers(
    blocks: range,
    *,
    prefix: str = "net.",
) -> dict[str, dict[str, object]]:
    return {
        f"{prefix}blocks.{index}.{suffix}": {
            "dtype": "F16",
            "shape": _shape_for_suffix(suffix),
            "data_offsets": [0, 0],
        }
        for index in blocks
        for suffix in _BLOCK_SUFFIXES
    }


def _write_header(path: Path, header: dict[str, dict[str, object]]) -> Path:
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    return path


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
@pytest.mark.parametrize("num_blocks", [28, 40])
def test_inspector_accepts_supported_prefixes_and_layouts(tmp_path, prefix, num_blocks):
    checkpoint = _write_header(
        tmp_path / "model.safetensors",
        _block_headers(range(num_blocks), prefix=prefix),
    )

    layout = inspect_anima_checkpoint(checkpoint)

    assert layout.arch == f"anima-2048-{num_blocks}"
    assert layout.num_blocks == num_blocks
    assert layout.model_channels == 2048
    assert layout.num_heads == 16
    assert layout.key_prefix == prefix


def test_inspector_expands_existing_shard_names(tmp_path):
    first = _write_header(
        tmp_path / "model-00001-of-00002.safetensors",
        _block_headers(range(14)),
    )
    _write_header(
        tmp_path / "model-00002-of-00002.safetensors",
        _block_headers(range(14, 28)),
    )

    assert inspect_anima_checkpoint(first).num_blocks == 28


def test_inspector_rejects_duplicate_keys_across_shards(tmp_path):
    first = _write_header(
        tmp_path / "model-00001-of-00002.safetensors",
        _block_headers(range(14)),
    )
    second_headers = _block_headers(range(14, 28))
    duplicate = next(iter(_block_headers(range(1))))
    second_headers[duplicate] = _block_headers(range(1))[duplicate]
    _write_header(tmp_path / "model-00002-of-00002.safetensors", second_headers)

    with pytest.raises(ValueError, match="Duplicate tensor key"):
        inspect_anima_checkpoint(first)


@pytest.mark.parametrize("num_blocks", [27, 29, 39, 41])
def test_inspector_rejects_unknown_or_incomplete_block_counts(tmp_path, num_blocks):
    checkpoint = _write_header(
        tmp_path / "model.safetensors", _block_headers(range(num_blocks))
    )
    with pytest.raises(ValueError, match="Unsupported or incomplete"):
        inspect_anima_checkpoint(checkpoint)


def test_inspector_rejects_gap_mixed_prefix_and_shape_drift(tmp_path):
    gap = _block_headers(range(28))
    for key in [key for key in gap if key.startswith("net.blocks.7.")]:
        del gap[key]
    with pytest.raises(ValueError, match="Unsupported or incomplete"):
        inspect_anima_checkpoint(_write_header(tmp_path / "gap.safetensors", gap))

    mixed = _block_headers(range(28))
    mixed.update(_block_headers(range(1), prefix="model.diffusion_model."))
    with pytest.raises(ValueError, match="mixes Anima key prefixes"):
        inspect_anima_checkpoint(_write_header(tmp_path / "mixed.safetensors", mixed))

    drift = _block_headers(range(28))
    drift["net.blocks.9.mlp.layer1.weight"]["shape"] = [4096, 2048]
    with pytest.raises(ValueError, match="shape mismatch"):
        inspect_anima_checkpoint(_write_header(tmp_path / "drift.safetensors", drift))


def test_inspector_never_reads_tensor_payload(tmp_path, monkeypatch):
    from library.anima import checkpoint as checkpoint_module

    checkpoint = _write_header(
        tmp_path / "model.safetensors", _block_headers(range(28))
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("tensor payload was read")

    monkeypatch.setattr(checkpoint_module.MemoryEfficientSafeOpen, "get_tensor", fail)
    assert inspect_anima_checkpoint(checkpoint).num_blocks == 28


LAYOUT_28 = AnimaCheckpointLayout(
    "anima-2048-28", "anima-base-v1.0", 28, 2048, 16, "net."
)
LAYOUT_40 = AnimaCheckpointLayout(
    "anima-2048-40", "anima-2.9b-preview-v1", 40, 2048, 16, "net."
)


def _plain(**overrides):
    config = {
        "method": "lora",
        "network_module": "networks.lora_anima",
        "base_compute": "bf16",
        "down_init": "kaiming",
        "use_timestep_mask": False,
        "router_source": "none",
    }
    config.update(overrides)
    return config


def test_40_block_profile_matrix():
    plain = compatibility_for_layout(_plain(), LAYOUT_40)
    assert plain.supported and plain.profile == "plain_lora"

    fp16 = compatibility_for_layout(_plain(base_compute="fp16"), LAYOUT_40)
    assert fp16.supported and fp16.profile == "plain_lora"

    tlora = compatibility_for_layout(
        _plain(use_timestep_mask=True, down_init="weight_svd"), LAYOUT_40
    )
    assert tlora.supported and tlora.profile == "tlora_ortho"

    explicit_ortho = compatibility_for_layout(
        _plain(use_timestep_mask=True, use_ortho=True), LAYOUT_40
    )
    assert explicit_ortho.supported and explicit_ortho.profile == "tlora_ortho"

    blocked = classify_anima_training(
        _plain(use_repa=True, use_lokr=True, vr_loss_weight=0.1)
    )
    assert not blocked.supported
    assert blocked.profile == "unsupported"
    assert {"REPA", "LoKr", "VR loss"}.issubset(blocked.blockers)


def test_28_block_keeps_existing_feature_matrix():
    result = compatibility_for_layout(
        _plain(method="easycontrol", use_repa=True, use_lokr=True), LAYOUT_28
    )
    assert result.supported


def test_preflight_only_certifies_the_verified_40_block_checkpoint(
    tmp_path, monkeypatch
):
    from library.anima import compat

    checkpoint = _write_header(
        tmp_path / "model.safetensors", _block_headers(range(40))
    )
    monkeypatch.setattr(
        compat, "anima_checkpoint_sha256", lambda _path: ANIMA29_PREVIEW_V1_SHA256
    )
    _layout, _base_sha256, certified = preflight_anima_training(
        _plain(channel_scaling_alpha=0.0), checkpoint
    )
    assert certified.supported

    monkeypatch.setattr(compat, "anima_checkpoint_sha256", lambda _path: "f" * 64)
    _layout, _base_sha256, derivative = preflight_anima_training(
        _plain(channel_scaling_alpha=0.0),
        checkpoint,
        raise_on_blockers=False,
    )
    assert not derivative.supported
    assert derivative.profile == "unsupported"
    assert any(
        "uncertified 40-block checkpoint" in item for item in derivative.blockers
    )


def _write_adapter(path: Path, metadata: dict[str, str] | None = None) -> Path:
    save_file({"lora_unet_x.weight": torch.ones(1)}, str(path), metadata=metadata)
    return path


def test_adapter_identity_matrix(tmp_path):
    hash_28 = "1" * 64
    hash_40 = "2" * 64
    legacy = _write_adapter(tmp_path / "legacy.safetensors")
    assert validate_adapter_compatibility(legacy, LAYOUT_28, hash_28) == {}
    with pytest.raises(ValueError, match="requires adapter architecture metadata"):
        validate_adapter_compatibility(legacy, LAYOUT_40, hash_40)

    new_28 = _write_adapter(
        tmp_path / "new28.safetensors", adapter_identity_metadata(LAYOUT_28, hash_28)
    )
    new_40 = _write_adapter(
        tmp_path / "new40.safetensors", adapter_identity_metadata(LAYOUT_40, hash_40)
    )
    validate_adapter_compatibility(new_28, LAYOUT_28, hash_28)
    validate_adapter_compatibility(new_40, LAYOUT_40, hash_40)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_adapter_compatibility(new_28, LAYOUT_40, hash_40)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_adapter_compatibility(new_40, LAYOUT_28, hash_28)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_adapter_compatibility(new_40, LAYOUT_40, "3" * 64)

    partial = _write_adapter(
        tmp_path / "partial.safetensors", {"ss_anima_arch": LAYOUT_40.arch}
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_adapter_compatibility(partial, LAYOUT_40, hash_40)


def test_resume_signature_matrix():
    validate_resume_model_signature({}, expected_signature="base28", num_blocks=28)
    with pytest.raises(ValueError, match="missing"):
        validate_resume_model_signature({}, expected_signature="base40", num_blocks=40)
    with pytest.raises(ValueError, match="mismatch"):
        validate_resume_model_signature(
            {"anima_model_signature": "other"},
            expected_signature="base28",
            num_blocks=28,
        )
    validate_resume_model_signature(
        {"anima_model_signature": "base40"},
        expected_signature="base40",
        num_blocks=40,
    )


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)
        self._modulation = torch.nn.Linear(4, 3, bias=False)


class PatchEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False)


class FinalLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False)


class TinyAnima40(torch.nn.Module):
    num_blocks = 40
    anima_base_sha256 = "a" * 64

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(40))
        self.x_embedder = PatchEmbed()
        self.final_layer = FinalLayer()


def _write_stats40(
    path: Path, *, missing_last: bool = False, nonfinite: bool = False
) -> None:
    count = 39 if missing_last else 40
    tensors = {
        f"lora_unet_blocks_{index}_proj": torch.ones(4) for index in range(count)
    }
    if nonfinite:
        tensors["lora_unet_blocks_39_proj"][0] = float("nan")
    save_file(
        tensors,
        str(path),
        metadata={
            "anima_stats_schema": "1",
            "anima_arch": "anima-2048-40",
            "anima_num_blocks": "40",
            "anima_model_channels": "2048",
            "anima_base_sha256": "a" * 64,
        },
    )


def test_40_block_channel_stats_preflight_and_runtime_target_coverage(
    tmp_path, monkeypatch
):
    from library.anima import compat
    from networks.lora_anima import factory

    stats = tmp_path / "channel_stats_anima40.safetensors"
    _write_stats40(stats)
    monkeypatch.setattr(compat, "_CHANNEL_STATS_40_PATH", stats)
    monkeypatch.setattr(factory, "_CHANNEL_STATS_40_PATH", stats)

    validate_channel_stats_compatibility(
        {"channel_scaling_alpha": 0.5}, LAYOUT_40, "a" * 64
    )
    scales = factory._load_channel_scales({"channel_scaling_alpha": 0.5}, TinyAnima40())
    assert scales is not None and len(scales) == 40

    with pytest.raises(ValueError, match="lora_unet_final_layer_linear"):
        factory._load_channel_scales(
            {
                "channel_scaling_alpha": 0.5,
                "include_patterns": [r"final_layer\.linear"],
            },
            TinyAnima40(),
        )

    _write_stats40(stats, missing_last=True)
    with pytest.raises(ValueError, match="blocks 0..39|exactly 0..39"):
        validate_channel_stats_compatibility(
            {"channel_scaling_alpha": 0.5}, LAYOUT_40, "a" * 64
        )
    with pytest.raises(ValueError, match="exactly blocks 0..39"):
        factory._load_channel_scales({"channel_scaling_alpha": 0.5}, TinyAnima40())

    _write_stats40(stats, nonfinite=True)
    with pytest.raises(ValueError, match="non-finite"):
        validate_channel_stats_compatibility(
            {"channel_scaling_alpha": 0.5}, LAYOUT_40, "a" * 64
        )
    with pytest.raises(ValueError, match="malformed"):
        factory._load_channel_scales({"channel_scaling_alpha": 0.5}, TinyAnima40())


def test_channel_stats_disabled_or_28_block_keeps_legacy_behavior(
    tmp_path, monkeypatch
):
    from library.anima import compat

    monkeypatch.setattr(compat, "_CHANNEL_STATS_40_PATH", tmp_path / "missing")
    validate_channel_stats_compatibility(
        {"channel_scaling_alpha": 0.5}, LAYOUT_28, "a" * 64
    )
    validate_channel_stats_compatibility(
        {"channel_scaling_alpha": 0.0}, LAYOUT_40, "a" * 64
    )
