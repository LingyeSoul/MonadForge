"""LoRA merging preserves and validates the identity required by 40-block bases."""

import sys

import pytest
import torch
from safetensors.torch import save_file

from library.anima.checkpoint import AnimaCheckpointLayout
from library.anima.compat import (
    adapter_identity_metadata,
    validate_adapter_compatibility,
)
from scripts.toolkits.merge_loras import main, merged_identity


def _adapter(path, metadata):
    stem = "lora_unet_blocks_0_self_attn_qkv_proj"
    save_file(
        {
            stem + ".lora_down.weight": torch.ones(1, 2),
            stem + ".lora_up.weight": torch.ones(2, 1),
            stem + ".alpha": torch.tensor(1.0),
        },
        str(path),
        metadata=metadata,
    )
    return str(path)


def test_merge_roundtrip_retains_required_40_block_identity(tmp_path, monkeypatch):
    layout = AnimaCheckpointLayout(
        "anima-2048-40", "anima-2.9b-preview-v1", 40, 2048, 16, "net."
    )
    metadata = adapter_identity_metadata(layout, "a" * 64)
    paths = [_adapter(tmp_path / f"{i}.safetensors", metadata) for i in range(2)]
    output = tmp_path / "merged.safetensors"
    monkeypatch.setattr(
        sys, "argv", ["merge_loras", *paths, "--out", str(output), "--dtype", "fp32"]
    )
    assert main() == 0
    restored = validate_adapter_compatibility(output, layout, "a" * 64)
    assert restored["ss_anima_num_blocks"] == "40"
    assert restored["ss_new_sd_model_hash"] == "a" * 64


@pytest.mark.parametrize(
    "change", [{"ss_new_sd_model_hash": "b" * 64}, {"ss_anima_num_blocks": "28"}, None]
)
def test_merge_rejects_mismatched_or_missing_identity(tmp_path, change):
    metadata = {
        "ss_anima_arch": "anima-2048-40",
        "ss_anima_num_blocks": "40",
        "ss_anima_model_channels": "2048",
        "ss_new_sd_model_hash": "a" * 64,
    }
    first = _adapter(tmp_path / "first.safetensors", metadata)
    other = _adapter(
        tmp_path / "other.safetensors", {} if change is None else {**metadata, **change}
    )
    with pytest.raises(ValueError, match="identity|incomplete"):
        merged_identity([first, other])


def test_legacy_merge_does_not_fabricate_identity(tmp_path):
    files = [
        _adapter(tmp_path / f"{i}.safetensors", {"ss_new_sd_model_hash": "a" * 64})
        for i in range(2)
    ]
    assert merged_identity(files) == {}
