"""_resolve_vae_dtype regression.

Locks in the pre-Ampere (sm<8) + fp16 → fp32 VAE auto-protection that prevents
preview/sample 花图/糊图 (fp16 VAE decode artifacts: the decoder conv/group-norm
stack runs naked in fp16 ±65504 dynamic range; only QwenImageUpsample
self-protects). Mirrors the test pattern in tests/test_mixed_precision_resolver.py:
module-scoped ``import train`` fixture + monkeypatch of torch.cuda capability
probes.

Priority order pinned here: --no_half_vae wins unconditionally → --half_vae is
the explicit opt-out → otherwise pre-Ampere+fp16 auto-forces fp32 → Ampere /
bf16 / fp32 / no-CUDA / probe-failure all keep weight_dtype.
"""

from __future__ import annotations

import types

import pytest
import torch


@pytest.fixture(scope="module")
def train_mod():
    import train

    return train


def _fake_args(mp="fp16", no_half_vae=False, half_vae=False):
    return types.SimpleNamespace(
        mixed_precision=mp,
        no_half_vae=no_half_vae,
        half_vae=half_vae,
    )


def _patch_cuda(monkeypatch, available=True, capability=(8, 0)):
    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    if available:
        monkeypatch.setattr(
            "torch.cuda.get_device_capability", lambda *a, **k: capability
        )


# --- --no_half_vae: unconditional fp32, beats everything ---------------------


def test_no_half_vae_forces_fp32_on_ampere(train_mod, monkeypatch):
    # no_half_vae wins regardless of GPU / mixed_precision.
    _patch_cuda(monkeypatch, capability=(8, 0))
    args = _fake_args(mp="bf16", no_half_vae=True)
    assert train_mod._resolve_vae_dtype(args, torch.bfloat16) == torch.float32


def test_no_half_vae_beats_half_vae(train_mod, monkeypatch):
    # Conflict: both flags set. no_half_vae is the safe direction, it wins.
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args(mp="fp16", no_half_vae=True, half_vae=True)
    assert train_mod._resolve_vae_dtype(args, torch.float16) == torch.float32


# --- --half_vae: explicit opt-out -------------------------------------------


def test_half_vae_overrides_auto_fp32_on_v100(train_mod, monkeypatch):
    # User explicitly accepts fp16 artifacts → respect their choice, keep weight_dtype.
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args(mp="fp16", half_vae=True)
    assert train_mod._resolve_vae_dtype(args, torch.float16) == torch.float16


def test_half_vae_inert_when_not_fp16(train_mod, monkeypatch):
    # half_vae only matters under fp16 on pre-Ampere. On bf16 it's a no-op
    # (bf16 VAE is already safe).
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args(mp="bf16", half_vae=True)
    assert train_mod._resolve_vae_dtype(args, torch.bfloat16) == torch.bfloat16


# --- auto fp32 on pre-Ampere + fp16 -----------------------------------------


def test_auto_fp32_on_v100_fp16(train_mod, monkeypatch, caplog):
    _patch_cuda(monkeypatch, capability=(7, 0))  # V100 sm_70
    args = _fake_args(mp="fp16")
    with caplog.at_level("INFO"):
        dtype = train_mod._resolve_vae_dtype(args, torch.float16)
    assert dtype == torch.float32
    assert any("sm_70" in r.getMessage() for r in caplog.records)


def test_auto_fp32_on_t4_fp16(train_mod, monkeypatch):
    # T4 is sm_75 → major 7 < 8, covered by the same guard as V100.
    _patch_cuda(monkeypatch, capability=(7, 5))
    args = _fake_args(mp="fp16")
    assert train_mod._resolve_vae_dtype(args, torch.float16) == torch.float32


# --- no auto-forcing --------------------------------------------------------


def test_no_force_on_ampere_fp16(train_mod, monkeypatch):
    # Ampere sm_80 under fp16: fp16 VAE decode is still sub-optimal in theory,
    # but the guard is scoped to pre-Ampere (matches _resolve_mixed_precision).
    _patch_cuda(monkeypatch, capability=(8, 0))
    args = _fake_args(mp="fp16")
    assert train_mod._resolve_vae_dtype(args, torch.float16) == torch.float16


def test_no_force_on_bf16(train_mod, monkeypatch):
    # bf16 is safe (exponent matches fp32) even on pre-Ampere — though in
    # practice _resolve_mixed_precision would have flipped bf16→fp16 first.
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args(mp="bf16")
    assert train_mod._resolve_vae_dtype(args, torch.bfloat16) == torch.bfloat16


def test_no_force_when_no_cuda(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, available=False)
    args = _fake_args(mp="fp16")
    assert train_mod._resolve_vae_dtype(args, torch.float16) == torch.float16


def test_capability_probe_failure_is_safe(train_mod, monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def _raise(*a, **k):
        raise RuntimeError("cuda init failed")

    monkeypatch.setattr("torch.cuda.get_device_capability", _raise)
    args = _fake_args(mp="fp16")
    with caplog.at_level("WARNING"):
        dtype = train_mod._resolve_vae_dtype(args, torch.float16)
    # Never force fp32 on a probe failure — keep weight_dtype, log so it's
    # diagnosable (mirrors _resolve_mixed_precision's safe fallback).
    assert dtype == torch.float16
    assert any("could not read GPU compute capability" in r.getMessage() for r in caplog.records)
