import tomllib
from pathlib import Path
from types import SimpleNamespace

import library.runtime.backend as backend
from library.runtime.backend import (
    diagnose_cuda_unavailable,
    is_rocm,
    needs_rocm_attention_fallback,
    resolve_attention_mode,
)
from scripts import update

ROOT = Path(__file__).resolve().parents[1]


def _torch(hip):
    return SimpleNamespace(version=SimpleNamespace(hip=hip))


def test_rocm_detection_uses_hip_version():
    assert is_rocm(_torch("7.14.0"))
    assert not is_rocm(_torch(None))


def test_rocm_flash_falls_back_to_torch_sdpa():
    assert resolve_attention_mode("flash", _torch("7.14.0")) == "torch"


def test_rocm_keeps_explicit_non_flash_modes():
    assert resolve_attention_mode("flex", _torch("7.14.0")) == "flex"
    assert resolve_attention_mode("torch", _torch("7.14.0")) == "torch"


def test_cuda_keeps_flash():
    assert resolve_attention_mode("flash", _torch(None)) == "flash"


def test_none_request_does_not_report_rocm_fallback():
    assert resolve_attention_mode(None, _torch(None)) == "torch"
    assert not needs_rocm_attention_fallback(None, _torch(None))
    assert not needs_rocm_attention_fallback(None, _torch("7.14.0"))


def test_only_explicit_rocm_flash_request_reports_fallback():
    assert needs_rocm_attention_fallback("flash", _torch("7.14.0"))
    assert not needs_rocm_attention_fallback("torch", _torch("7.14.0"))
    assert not needs_rocm_attention_fallback("flash", _torch(None))












def _torch_full(*, available, hip=None, cuda=None):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available),
        version=SimpleNamespace(hip=hip, cuda=cuda),
    )


def test_diagnose_silent_when_cuda_works(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    assert diagnose_cuda_unavailable(_torch_full(available=True, cuda="13.2")) is None


def test_diagnose_silent_without_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: None)
    assert diagnose_cuda_unavailable(_torch_full(available=False, hip="7.14.0")) is None


def test_diagnose_flags_rocm_build_on_nvidia(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    message = diagnose_cuda_unavailable(_torch_full(available=False, hip="7.14.0"))
    assert message is not None and "ROCm build" in message


def test_diagnose_flags_cpu_build_on_nvidia(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    message = diagnose_cuda_unavailable(_torch_full(available=False))
    assert message is not None and "CPU-only build" in message


def test_diagnose_flags_driver_problem(monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    message = diagnose_cuda_unavailable(_torch_full(available=False, cuda="13.2"))
    assert message is not None and "driver" in message


