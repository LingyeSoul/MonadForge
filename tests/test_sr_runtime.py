"""Focused tests for the SR dual-environment and dtype policy."""

import ast
from pathlib import Path

import pytest

from scripts.tasks import sr


def test_setup_profile_flags_are_removed_from_forwarded_args(monkeypatch):
    monkeypatch.delenv("SR_SETUP_PROFILE", raising=False)
    profile, dry_run, forwarded = sr._parse_setup_args(
        ["--profile", "v100", "--dry-run", "--refresh"]
    )
    assert (profile, dry_run, forwarded) == ("v100", True, ["--refresh"])


def test_auto_detects_v100_stack_when_cuda_is_hidden():
    probe = {
        "torch": sr.V100_TORCH,
        "torch_cuda": sr.V100_CUDA,
        "torchvision": sr.V100_TORCHVISION,
        "cuda_available": False,
        "capability": None,
        "device": None,
    }
    assert sr._is_v100_runtime(probe)


def test_auto_keeps_standard_profile_for_bf16_gpu():
    probe = {
        "torch": "2.12.0+cu132",
        "torch_cuda": "13.2",
        "torchvision": "0.27.0+cu132",
        "cuda_available": True,
        "capability": [8, 9],
        "device": "NVIDIA RTX 5090",
        "flash_v100_module": False,
    }
    assert not sr._is_v100_runtime(probe)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("torch", "2.12.0+cu132"),
        ("onnxruntime", "1.27.0"),
        ("flash_attn_v100", "26.7"),
        ("flash_v100_module", False),
    ],
)
def test_v100_protected_fingerprint_rejects_core_change(field, replacement):
    before = {
        "torch": sr.V100_TORCH,
        "torchvision": sr.V100_TORCHVISION,
        "torch_cuda": sr.V100_CUDA,
        "onnxruntime": sr.V100_ORT,
        "flash_attn_v100": sr.V100_FLASH,
        "flash_v100_module": True,
    }
    changed = {**before, field: replacement}
    with pytest.raises(SystemExit, match="protected runtime"):
        sr._check_v100_after_install(before, changed)


@pytest.mark.parametrize(
    ("profile", "prefix"),
    [("standard", ["uv", "sync", "--group", "sr"]), ("v100", ["uv", "pip", "install"])],
)
def test_setup_profile_selects_install_command(monkeypatch, profile, prefix):
    calls = []
    runtime = {
        "torch": sr.V100_TORCH,
        "torchvision": sr.V100_TORCHVISION,
        "torch_cuda": sr.V100_CUDA,
        "onnxruntime": sr.V100_ORT,
        "flash_attn_v100": sr.V100_FLASH,
        "flash_v100_module": True,
        "cuda_available": False,
        "capability": None,
        "device": None,
    }
    monkeypatch.setenv("SR_SETUP_PROFILE", profile)
    monkeypatch.setattr(sr, "_probe_runtime", lambda: runtime)
    monkeypatch.setattr(sr, "_run", lambda cmd, cwd=sr.ROOT: calls.append((cmd, cwd)))
    sr.cmd_sr_setup(["--dry-run"])
    assert calls
    assert calls[0][0][: len(prefix)] == prefix
    if profile == "v100":
        assert "--python" in calls[0][0]


def test_sr_script_uses_importable_module_name():
    assert sr._module_for_script(sr.SR / "train_sr" / "train.py") == "sr.train_sr.train"
    assert sr._module_for_script(sr.SR / "scripts" / "detect_text_boxes.py") == (
        "sr.scripts.detect_text_boxes"
    )


def test_sr_runtime_verifier_uses_python_not_shell(monkeypatch):
    calls = []
    monkeypatch.setattr(sr, "_run", lambda cmd, cwd=sr.ROOT: calls.append((cmd, cwd)))

    sr._verify_sr_runtime()

    assert calls[0][0][0] == sr._venv_py()
    assert calls[0][0][1] == "-c"
    assert not any(str(arg).endswith(".sh") for arg in calls[0][0])


def test_x2_inference_uses_training_output_directory():
    source = (
        Path(__file__).resolve().parents[1] / "sr" / "scripts" / "sr_infer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    local = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_LOCAL"
            for target in node.targets
        )
    )
    x2_value = next(
        value
        for key, value in zip(local.keys, local.values)
        if isinstance(key, ast.Constant) and key.value == "x2"
    )
    path_parts = {
        node.value for node in ast.walk(x2_value) if isinstance(node, ast.Constant)
    }

    assert "x2_lpips_30k" in path_parts


def test_safe_spatial_pad_falls_back_for_tiny_images():
    torch = pytest.importorskip("torch")
    from sr.padding import safe_spatial_pad

    tiny = torch.tensor([[[[1.0]]]])

    padded = safe_spatial_pad(tiny, (0, 3, 0, 2))

    assert padded.shape == (1, 1, 3, 4)
    assert torch.equal(padded, torch.ones_like(padded))


def test_hr_pool_removes_only_stale_managed_links(tmp_path, monkeypatch):
    from sr.scripts.build_hr_pool import _remove_stale_managed_links

    stale = tmp_path / "stale.png"
    current = tmp_path / "current.png"
    unrelated = tmp_path / "notes.txt"
    for path in (stale, current, unrelated):
        path.write_text(path.name, encoding="utf-8")
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self in {stale, current} or real_is_symlink(self),
    )

    removed = _remove_stale_managed_links(
        tmp_path, {stale.name, current.name}, {current.name}
    )

    assert removed == 1
    assert not stale.exists()
    assert current.exists()
    assert unrelated.exists()


def test_hr_pool_reads_managed_links_from_manifest_only(tmp_path):
    from sr.scripts.build_hr_pool import _load_managed_links

    (tmp_path / "manifest.json").write_text(
        '{"managed_links": ["owned.png", 123]}', encoding="utf-8"
    )
    (tmp_path / "unrelated.png").write_text("keep", encoding="utf-8")

    assert _load_managed_links(tmp_path) == {"owned.png"}


def test_amp_auto_uses_fp16_on_v100(monkeypatch):
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("sr.distill_rsd.rsd_models")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (7, 0))
    assert models.resolve_amp_dtype("auto") == torch.float16
    assert models.resolve_amp_dtype("bf16") == torch.bfloat16


def test_amp_auto_uses_bf16_on_newer_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("sr.distill_rsd.rsd_models")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args: (8, 0))
    assert models.resolve_amp_dtype("auto") == torch.bfloat16
