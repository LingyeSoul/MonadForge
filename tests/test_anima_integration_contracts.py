from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from safetensors import safe_open

from library.anima.checkpoint import AnimaCheckpointLayout
from library.anima.compat import AnimaCompatibility, adapter_identity_metadata

LAYOUT_28 = AnimaCheckpointLayout(
    "anima-2048-28", "anima-base-v1.0", 28, 2048, 16, "net."
)
LAYOUT_40 = AnimaCheckpointLayout(
    "anima-2048-40", "anima-2.9b-preview-v1", 40, 2048, 16, "net."
)


@pytest.mark.parametrize(
    "relative_path,dry_run_owner",
    [
        ("scripts/distill_mod/distill.py", "cfg"),
        ("scripts/distill_spd.py", "args"),
    ],
)
def test_distillation_dry_run_returns_before_checkpoint_preflight(
    relative_path, dry_run_owner
):
    source = (Path(__file__).resolve().parents[1] / relative_path).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    preflight = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "preflight_anima_training"
    )
    dry_run_if = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == dry_run_owner
        and node.test.attr == "dry_run"
    )
    dry_run_return = next(
        node for node in ast.walk(dry_run_if) if isinstance(node, ast.Return)
    )

    assert preflight.lineno > dry_run_return.lineno


def _prelaunch_config(tmp_path) -> dict[str, object]:
    return {
        "pretrained_model_name_or_path": str(tmp_path / "base.safetensors"),
        "lora_cache_dir": str(tmp_path / "cache"),
        "network_module": "networks.lora_anima",
        "method": "lora",
        "base_compute": "bf16",
        "down_init": "kaiming",
        "router_source": "none",
    }


@pytest.mark.parametrize("layout", [LAYOUT_28, LAYOUT_40])
def test_webui_prelaunch_returns_model_layout_and_supported_profile(
    tmp_path, monkeypatch, layout
):
    from library.anima import compat
    from webui.services import config_service

    merged = _prelaunch_config(tmp_path)
    monkeypatch.setattr(
        config_service,
        "merged_gui_variant_preset",
        lambda _variant, _preset: (dict(merged), {}),
    )
    monkeypatch.setattr(config_service, "validate_config", lambda _merged: [])
    monkeypatch.setattr(
        config_service, "_read_variant_metadata", lambda _path: {"family": "lora"}
    )
    monkeypatch.setattr(
        config_service, "find_resumable_checkpoint", lambda _merged: None
    )
    monkeypatch.setattr(
        compat,
        "preflight_anima_training",
        lambda *_args, **_kwargs: (
            layout,
            "a" * 64,
            AnimaCompatibility(True, "plain_lora"),
        ),
    )

    result = config_service.prelaunch_check("lora", "default")

    assert result["model_layout"]["arch"] == layout.arch
    assert result["model_layout"]["num_blocks"] == layout.num_blocks
    assert result["compatibility"] == {
        "supported": True,
        "profile": "plain_lora",
        "blockers": [],
    }


def test_webui_prelaunch_returns_all_40_blockers_without_raising(tmp_path, monkeypatch):
    from library.anima import compat
    from webui.services import config_service

    merged = _prelaunch_config(tmp_path)
    monkeypatch.setattr(
        config_service,
        "merged_gui_variant_preset",
        lambda _variant, _preset: (dict(merged), {}),
    )
    monkeypatch.setattr(config_service, "validate_config", lambda _merged: [])
    monkeypatch.setattr(
        config_service, "_read_variant_metadata", lambda _path: {"family": "lora"}
    )
    monkeypatch.setattr(
        config_service, "find_resumable_checkpoint", lambda _merged: None
    )
    monkeypatch.setattr(
        compat,
        "preflight_anima_training",
        lambda *_args, **_kwargs: (
            LAYOUT_40,
            "a" * 64,
            AnimaCompatibility(False, "unsupported", ("REPA", "LoKr")),
        ),
    )

    result = config_service.prelaunch_check("lora", "default")

    assert result["compatibility"]["supported"] is False
    assert result["compatibility"]["blockers"] == ["REPA", "LoKr"]


def test_webui_prelaunch_maps_corrupt_checkpoint_to_http_400(tmp_path, monkeypatch):
    from library.anima import compat
    from webui.api import config as config_api
    from webui.services import config_service

    merged = _prelaunch_config(tmp_path)
    monkeypatch.setattr(
        config_service,
        "merged_gui_variant_preset",
        lambda _variant, _preset: (dict(merged), {}),
    )
    monkeypatch.setattr(config_service, "validate_config", lambda _merged: [])

    def fail_preflight(*_args, **_kwargs):
        raise ValueError("broken safetensors header")

    monkeypatch.setattr(compat, "preflight_anima_training", fail_preflight)
    with pytest.raises(ValueError, match="invalid Anima checkpoint"):
        config_service.prelaunch_check("lora", "default")

    monkeypatch.setattr(
        config_api.svc,
        "prelaunch_check",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid Anima checkpoint: broken safetensors header")
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        config_api.prelaunch_check("lora", "default")
    assert exc_info.value.status_code == 400


def test_training_preflight_collects_every_adapter_and_refreshes_snapshot(
    tmp_path, monkeypatch
):
    import train

    base = tmp_path / "base.safetensors"
    adapters = [
        tmp_path / name
        for name in ("network.safetensors", "frozen.safetensors", "base.safetensors")
    ]
    captured: dict[str, object] = {}

    def fake_preflight(config, checkpoint_path, *, adapter_paths, **_kwargs):
        captured["config"] = config
        captured["checkpoint_path"] = checkpoint_path
        captured["adapter_paths"] = adapter_paths
        return LAYOUT_40, "b" * 64, AnimaCompatibility(True, "plain_lora")

    monkeypatch.setattr(train, "preflight_anima_training", fake_preflight)
    monkeypatch.setattr(
        train,
        "refresh_config_snapshot",
        lambda args: captured.setdefault("snapshot_args", args),
    )
    args = SimpleNamespace(
        pretrained_model_name_or_path=str(base),
        network_weights=str(adapters[0]),
        lora_path=str(adapters[1]),
        base_weights=[str(adapters[2])],
        _config_snapshot_provenance={},
    )

    train._prepare_anima_checkpoint_identity(args)

    assert captured["checkpoint_path"] == str(base)
    assert captured["adapter_paths"] == tuple(adapters)
    assert args.anima_arch == LAYOUT_40.arch
    assert args.anima_training_profile == "plain_lora"
    assert args._config_snapshot_provenance["anima_arch"] == "runtime/checkpoint"
    assert captured["snapshot_args"] is args


def test_anima_identity_is_kept_in_minimum_metadata_and_manifest():
    import train
    from library.training.metadata import finalize_metadata

    identity = adapter_identity_metadata(LAYOUT_40, "c" * 64)
    _full, minimum = finalize_metadata(dict(identity))
    assert minimum == identity

    args = SimpleNamespace(
        anima_arch=LAYOUT_40.arch,
        anima_variant=LAYOUT_40.variant,
        anima_num_blocks=40,
        anima_model_channels=2048,
        anima_base_sha256="c" * 64,
        anima_model_signature="signature",
        anima_training_profile="plain_lora",
    )
    assert train._anima_manifest_fields(args) == {
        "anima_arch": LAYOUT_40.arch,
        "anima_variant": LAYOUT_40.variant,
        "anima_num_blocks": 40,
        "anima_model_channels": 2048,
        "anima_base_sha256": "c" * 64,
        "anima_model_signature": "signature",
        "anima_training_profile": "plain_lora",
    }


def test_inference_rejects_adapter_before_loading_tensor_payload(monkeypatch):
    from library.inference import models

    loaded: list[str] = []
    monkeypatch.setattr(models, "inspect_anima_checkpoint", lambda _path: LAYOUT_40)
    monkeypatch.setattr(models, "anima_checkpoint_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(
        models,
        "validate_adapter_compatibility",
        lambda *_args: (_ for _ in ()).throw(ValueError("adapter identity mismatch")),
    )
    monkeypatch.setattr(models, "load_file", lambda path: loaded.append(path))
    args = argparse.Namespace(
        dit="base.safetensors", lora_weight=["adapter.safetensors"]
    )

    with pytest.raises(ValueError, match="adapter identity mismatch"):
        models.load_dit_model(args, torch.device("cpu"))
    assert loaded == []


def test_inference_identity_cache_revalidates_replaced_adapter(tmp_path, monkeypatch):
    from library.inference import models

    base = tmp_path / "base.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    base.write_bytes(b"base")
    adapter.write_bytes(b"adapter-v1")
    validated: list[Path] = []

    monkeypatch.setattr(models, "inspect_anima_checkpoint", lambda _path: LAYOUT_28)
    monkeypatch.setattr(models, "anima_checkpoint_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(
        models,
        "validate_adapter_compatibility",
        lambda path, *_args: validated.append(Path(path)),
    )
    args = argparse.Namespace(dit=str(base), lora_weight=[str(adapter)])

    models._prepare_inference_anima_identity(args)
    models._prepare_inference_anima_identity(args)
    assert validated == [adapter]

    original_stat = adapter.stat()
    replacement = tmp_path / "adapter-replacement.safetensors"
    replacement.write_bytes(b"adapter-v2")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(adapter)
    os.utime(adapter, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert adapter.stat().st_size == original_stat.st_size
    assert adapter.stat().st_mtime_ns == original_stat.st_mtime_ns
    models._prepare_inference_anima_identity(args)
    assert validated == [adapter, adapter]


def test_merge_rejects_adapter_before_loading_tensor_payload(tmp_path, monkeypatch):
    from library.anima import merge

    base = tmp_path / "base.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    base.touch()
    adapter.touch()
    loaded: list[str] = []
    monkeypatch.setattr(merge, "inspect_anima_checkpoint", lambda _path: LAYOUT_40)
    monkeypatch.setattr(merge, "anima_checkpoint_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(merge, "read_adapter_metadata", lambda _path: {})
    monkeypatch.setattr(
        merge,
        "validate_adapter_compatibility",
        lambda *_args: (_ for _ in ()).throw(ValueError("adapter identity mismatch")),
    )
    monkeypatch.setattr(
        merge, "_load_adapter_state_dict", lambda path: loaded.append(str(path))
    )

    with pytest.raises(ValueError, match="adapter identity mismatch"):
        merge.merge_adapter_into_dit(adapter, base, tmp_path / "merged.safetensors")
    assert loaded == []


def test_merge_cli_can_be_launched_by_script_path():
    result = subprocess.run(
        [sys.executable, "scripts/merge_to_dit.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Bake a LoRA adapter into the base DiT" in result.stdout


def test_anima_save_restores_public_unfused_checkpoint_keys(tmp_path):
    from library.anima.weights import save_anima_model

    state = {
        "blocks.0.self_attn.qkv_proj.weight": torch.arange(24).reshape(6, 4),
        "blocks.0.cross_attn.kv_proj.weight": torch.arange(16).reshape(4, 4),
        "blocks.0.adaln_fused_down.1.weight": torch.arange(24).reshape(6, 4),
        "blocks.0.adaln_up_self_attn.weight": torch.arange(8).reshape(2, 4),
    }
    output = tmp_path / "export.safetensors"
    save_anima_model(str(output), state, {}, dtype=torch.float32)

    with safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "net.blocks.0.self_attn.qkv_proj.weight" not in keys
        assert "net.blocks.0.cross_attn.kv_proj.weight" not in keys
        assert "net.blocks.0.adaln_fused_down.1.weight" not in keys
        assert "net.blocks.0.adaln_up_self_attn.weight" not in keys
        assert {
            "net.blocks.0.self_attn.q_proj.weight",
            "net.blocks.0.self_attn.k_proj.weight",
            "net.blocks.0.self_attn.v_proj.weight",
            "net.blocks.0.cross_attn.k_proj.weight",
            "net.blocks.0.cross_attn.v_proj.weight",
            "net.blocks.0.adaln_modulation_self_attn.1.weight",
            "net.blocks.0.adaln_modulation_cross_attn.1.weight",
            "net.blocks.0.adaln_modulation_mlp.1.weight",
            "net.blocks.0.adaln_modulation_self_attn.2.weight",
        } <= keys
        assert torch.equal(
            handle.get_tensor("net.blocks.0.self_attn.q_proj.weight"),
            state["blocks.0.self_attn.qkv_proj.weight"][:2],
        )


def test_harness_attaches_base_identity_before_adapter_construction(tmp_path, monkeypatch):
    from library.runtime import harness

    loaded: list[object] = []
    inspected: list[Path] = []
    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))

    def fake_inspect(path):
        inspected.append(Path(path))
        return LAYOUT_40

    monkeypatch.setattr(
        "library.anima.checkpoint.inspect_anima_checkpoint", fake_inspect
    )
    monkeypatch.setattr(
        "library.anima.checkpoint.anima_checkpoint_sha256", lambda _path: "e" * 64
    )

    class DummyAnima(torch.nn.Module):
        def reset_mod_guidance(self):
            pass

    from library.anima import weights

    monkeypatch.setattr(
        weights,
        "load_anima_model",
        lambda **_kwargs: loaded.append(DummyAnima()) or loaded[-1],
    )
    args = argparse.Namespace(device="cpu", dtype="fp16", attn_mode="torch")
    bundle = harness.build_anima(args, dit_path="models/base.safetensors")

    assert inspected == [tmp_path / "models/base.safetensors"]
    assert bundle.anima.anima_base_sha256 == "e" * 64


def test_channel_stats_writer_rejects_nonfinite_vectors(tmp_path):
    from scripts.calibration.analyze_lora_input_channels import (
        dump_channel_stats_safetensors,
    )

    stats = {
        "blocks.0.mlp.layer1": {
            "count": 1,
            "sum_abs": torch.tensor([1.0, float("nan")], dtype=torch.float64),
        }
    }

    with pytest.raises(ValueError, match="non-finite channel statistics"):
        dump_channel_stats_safetensors(stats, tmp_path / "stats.safetensors")
    assert not (tmp_path / "stats.safetensors").exists()
