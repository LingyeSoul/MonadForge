"""Block-swap compatibility and Volta reliable-mode policy tests."""

from __future__ import annotations

import types

import pytest


@pytest.fixture(scope="module")
def train_mod():
    import train

    return train


def _args(**overrides):
    values = {
        "blocks_to_swap": 1,
        "network_module": "networks.lora_anima",
        "method": None,
        "use_byg": False,
        "use_easycontrol": False,
        "unsloth_offload_checkpointing": False,
        "torch_compile": False,
        "compile_inductor_mode": None,
        "dynamo_backend": "inductor",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "label"),
    [
        ({"network_module": "networks.methods.soft_tokens"}, "Soft Tokens"),
        ({"network_module": "networks.methods.easycontrol"}, "EasyControl"),
        ({"use_easycontrol": True}, "EasyControl"),
        ({"use_byg": True}, "BYG"),
        ({"unsloth_offload_checkpointing": True}, "unsloth_offload_checkpointing"),
    ],
)
def test_block_swap_rejects_unsupported_paths(train_mod, overrides, label):
    args = _args(**overrides)
    with pytest.raises(ValueError, match=label):
        train_mod._validate_block_swap_config(args)


def test_block_swap_zero_keeps_method_paths_usable(train_mod):
    args = _args(
        blocks_to_swap=0,
        network_module="networks.methods.soft_tokens",
        use_byg=True,
        use_easycontrol=True,
        unsloth_offload_checkpointing=True,
    )
    train_mod._validate_block_swap_config(args)


def test_sm70_block_swap_disables_compile_before_accelerator(
    train_mod, monkeypatch, caplog
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _args(torch_compile=True, compile_inductor_mode="max-autotune")

    with caplog.at_level("WARNING"):
        assert train_mod._resolve_block_swap_reliable_mode(args) is True

    assert args.block_swap_reliable_mode is True
    assert args.block_swap_gpu_sm == "sm_70"
    assert args.block_swap_requested_torch_compile is True
    assert args.block_swap_compile_disabled is True
    assert args.torch_compile is False
    assert args.compile_inductor_mode is None
    assert args.dynamo_backend == "eager"
    assert any("reliable mode enabled" in rec.getMessage() for rec in caplog.records)


def test_non_sm70_block_swap_preserves_compile_request(train_mod, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _args(torch_compile=True, compile_inductor_mode="default")

    assert train_mod._resolve_block_swap_reliable_mode(args) is False
    assert args.block_swap_reliable_mode is False
    assert args.block_swap_gpu_sm == "sm_80"
    assert args.torch_compile is True
    assert args.compile_inductor_mode == "default"
    assert args.dynamo_backend == "inductor"


def test_reliable_mode_overrides_explicit_eager_policy_opt_out(
    train_mod, monkeypatch
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _args(
        mixed_precision="fp16",
        block_swap_reliable_mode=True,
        torch_compile=False,
    )
    accelerator = types.SimpleNamespace(device="cuda:0")
    net_kwargs = {
        "lora_fp32_compute": "false",
        "use_custom_down_autograd": "false",
    }

    effective = train_mod._apply_v100_adapter_runtime_policy(
        args, accelerator, net_kwargs
    )
    assert effective["lora_fp32_compute"] == "true"
    assert effective["use_custom_down_autograd"] == "true"
    assert args.block_swap_effective_lora_fp32_compute is True
    assert args.block_swap_effective_use_custom_down_autograd is True
    assert args.block_swap_effective_network_spec == "lora"


def test_manifest_fields_report_effective_policy(train_mod):
    args = _args(
        blocks_to_swap=2,
        block_swap_reliable_mode=True,
        block_swap_gpu_sm="sm_70",
        block_swap_requested_torch_compile=True,
        block_swap_compile_disabled=True,
        torch_compile=False,
        block_swap_effective_lora_fp32_compute=True,
        block_swap_effective_use_custom_down_autograd=True,
        block_swap_effective_network_spec="lora",
    )
    fields = train_mod._block_swap_manifest_fields(args)
    assert fields == {
        "blocks_to_swap": 2,
        "block_swap_reliable_mode": True,
        "block_swap_gpu_sm": "sm_70",
        "block_swap_requested_torch_compile": True,
        "block_swap_compile_disabled": True,
        "block_swap_effective_torch_compile": False,
        "block_swap_effective_lora_fp32_compute": True,
        "block_swap_effective_use_custom_down_autograd": True,
        "block_swap_effective_network_spec": "lora",
    }


def test_manifest_fields_include_measured_budget(train_mod):
    args = _args(block_swap_budget={"max_tokens": 4200, "free_bytes": 123})
    fields = train_mod._block_swap_manifest_fields(args)
    assert fields["block_swap_budget"] == {
        "max_tokens": 4200,
        "free_bytes": 123,
    }
