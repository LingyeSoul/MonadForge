"""_resolve_mixed_precision regression.

Mirrors the test pattern in tests/test_v100_flash_stability.py: a module-scoped
``import train`` fixture + monkeypatch of torch.cuda capability probes. Locks
in the pre-Ampere bf16→fp16 auto-switch + the args back-write + the safe
fallbacks (capability-probe failure, no CUDA, Ampere GPU).

Note: argparse defaults mixed_precision to "bf16" (library/config/cli_args.py),
so the resolver cannot distinguish "user omitted the flag" from "user passed
--mixed_precision bf16 explicitly" — both look like bf16 and both switch on a
pre-Ampere GPU. test_explicit_bf16_on_pre_ampere_still_switches pins this
known limitation.
"""

from __future__ import annotations

import types

import pytest


@pytest.fixture(scope="module")
def train_mod():
    import train

    return train


def _fake_args(mp="bf16"):
    return types.SimpleNamespace(mixed_precision=mp)


def _fake_accelerator(device="cuda:0"):
    return types.SimpleNamespace(device=device)


def _patch_cuda(monkeypatch, available=True, capability=(8, 0)):
    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    if available:
        monkeypatch.setattr(
            "torch.cuda.get_device_capability", lambda *a, **k: capability
        )


def test_no_switch_on_ampere(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, capability=(8, 0))
    args = _fake_args()
    train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_switch_on_pre_ampere_back_writes_args(train_mod, monkeypatch, caplog):
    _patch_cuda(monkeypatch, capability=(7, 0))  # V100
    args = _fake_args()
    with caplog.at_level("WARNING"):
        train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"  # back-write is the core fix
    assert any("sm_70" in r.getMessage() for r in caplog.records)


def test_switch_on_t4(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, capability=(7, 5))  # T4 is sm_75 → major 7 < 8
    args = _fake_args()
    train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_explicit_fp16_left_alone(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args("fp16")
    train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_explicit_bf16_on_pre_ampere_still_switches(train_mod, monkeypatch):
    # Known limitation: can't tell default-bf16 from explicit-bf16, so both
    # switch. Pin the current behavior so a future change is intentional.
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args("bf16")
    train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_capability_probe_failure_is_safe(train_mod, monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def _raise(*a, **k):
        raise RuntimeError("cuda init failed")

    monkeypatch.setattr("torch.cuda.get_device_capability", _raise)
    args = _fake_args()
    with caplog.at_level("WARNING"):
        train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"  # never switch on a probe failure
    assert any("keeping" in r.getMessage() for r in caplog.records)


def test_no_cuda_is_noop(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, available=False)
    args = _fake_args()
    train_mod._resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_no_mixed_precision_attr_is_safe(train_mod, monkeypatch):
    # Defensive: a malformed args without mixed_precision must not crash.
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = types.SimpleNamespace()  # no mixed_precision attribute
    train_mod._resolve_mixed_precision(args)  # must not raise
    assert not hasattr(args, "mixed_precision")


def test_auto_lora_fp32_compute_on_v100_fp16(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args("fp16")
    assert train_mod._should_auto_enable_lora_fp32_compute(
        args, _fake_accelerator(), {}
    )


def test_auto_lora_fp32_compute_respects_explicit_false(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, capability=(7, 0))
    args = _fake_args("fp16")
    assert not train_mod._should_auto_enable_lora_fp32_compute(
        args, _fake_accelerator(), {"lora_fp32_compute": "false"}
    )


def test_auto_lora_fp32_compute_only_v100_fp16(train_mod, monkeypatch):
    _patch_cuda(monkeypatch, capability=(8, 0))
    assert not train_mod._should_auto_enable_lora_fp32_compute(
        _fake_args("fp16"), _fake_accelerator(), {}
    )
    _patch_cuda(monkeypatch, capability=(7, 0))
    assert not train_mod._should_auto_enable_lora_fp32_compute(
        _fake_args("bf16"), _fake_accelerator(), {}
    )
    _patch_cuda(monkeypatch, available=False)
    assert not train_mod._should_auto_enable_lora_fp32_compute(
        _fake_args("fp16"), _fake_accelerator("cpu"), {}
    )
