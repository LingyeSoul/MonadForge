"""Real CUDA coverage for LoKr backend selection and autocast boundaries."""

from __future__ import annotations

import copy

import pytest
import torch
import toml

from library.config.lokr import validate_lokr_triton_shape
from library.anima.compat import validate_lokr_training_options
from library.config.schema import ConfigSchemaError, validate_entry
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_modules.lokr import LoKRModule
from webui.services import config_service


def test_triton_rejects_attention_and_mlp_factor_sizes():
    with pytest.raises(ValueError, match="256, 256"):
        validate_lokr_triton_shape("self_attn.q_proj", 2048, 2048, 8)
    with pytest.raises(ValueError, match="mlp.layer1"):
        validate_lokr_triton_shape("mlp.layer1", 2048, 8192, 16)
    validate_lokr_triton_shape("mlp.layer1", 2048, 8192, 64)
    validate_lokr_triton_shape("mlp.layer2", 8192, 2048, 64)


def test_preflight_respects_target_exclusions_and_rank_zero():
    options = {"use_lokr": True, "lokr_backend": "triton", "lokr_factor": 16}
    with pytest.raises(ValueError, match="mlp.layer1"):
        validate_lokr_training_options(options, 28)
    validate_lokr_training_options({**options, "exclude_patterns": [r".*mlp.*"]}, 28)
    validate_lokr_training_options({**options, "network_reg_dims": ".*mlp.*=0"}, 28)
    with pytest.raises(ValueError, match="mlp.layer1"):
        validate_lokr_training_options(
            {
                **options,
                "exclude_patterns": [r".*mlp.*"],
                "include_patterns": [r".*mlp.layer1"],
            },
            28,
        )


@pytest.mark.parametrize("key", ["bypass", "ypass", "use_triton", "apply_bypass"])
def test_obsolete_keys_fail_with_migration_instructions(key):
    with pytest.raises(ConfigSchemaError, match="lokr_backend"):
        validate_entry(key, False)
    assert "lokr_backend" in config_service.validate_config({key: False})[0]


def test_backend_is_validated_and_forwarded_to_network_config():
    cfg = LoRANetworkCfg.from_kwargs(
        {"use_lokr": "true", "lokr_backend": "triton"},
        network_dim=8,
        network_alpha=16,
        neuron_dropout=None,
        module_class=LoKRModule,
    )
    assert cfg.lokr_backend == "triton"
    with pytest.raises(ValueError, match="lokr_backend"):
        LoRANetworkCfg.from_kwargs(
            {"use_lokr": "true", "lokr_backend": "typo"},
            network_dim=8,
            network_alpha=16,
            neuron_dropout=None,
            module_class=LoKRModule,
        )


def test_webui_exposes_backend_and_rejects_oversized_triton_config():
    response = config_service.build_merged_config("lokr", "default", lang="en")
    field = next(item for item in response["fields"] if item["key"] == "lokr_backend")
    assert field["field_type"] == "select"
    assert field["options"] == ["torch", "triton"]
    assert field["read_only"] is False
    assert field["group"] == "Architecture"
    assert config_service.validate_config(
        {"use_lokr": True, "lokr_backend": "triton", "lokr_factor": 8}
    )
    assert not config_service.validate_config(
        {"use_lokr": True, "lokr_backend": "triton", "lokr_factor": 64}
    )


def test_backend_round_trip_through_webui_toml_and_trainer(monkeypatch, tmp_path):
    from library.config.io import load_method_preset
    from train import resolve_network_kwargs
    from types import SimpleNamespace

    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    (configs / "base.toml").write_text("", encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (methods / "lokr.toml").write_text(
        'use_lokr = true\nlokr_factor = 8\nlokr_backend = "torch"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_VARIANTS_DIR", overlays)
    config_service.save_variant_config(
        "lokr", {"lokr_backend": "triton", "lokr_factor": 64}
    )
    saved = toml.loads((overlays / "lokr.toml").read_text(encoding="utf-8"))
    assert saved["lokr_backend"] == "triton"
    loaded = load_method_preset("lokr", "default", str(configs), "gui-methods")
    kwargs = resolve_network_kwargs(SimpleNamespace(**loaded))
    assert kwargs["lokr_backend"] == "triton"
    with pytest.raises(ValueError, match="exceeding"):
        config_service.save_variant_config("lokr", {"lokr_factor": 8})
    assert toml.loads((overlays / "lokr.toml").read_text(encoding="utf-8")) == saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("full_factor", [False, True])
@pytest.mark.parametrize("out_features", [2048, 8192])
def test_triton_autocast_matches_torch_forward_and_backward(
    dtype, full_factor, out_features
):
    torch.manual_seed(217)
    base = torch.nn.Linear(2048, out_features, bias=False, device="cuda", dtype=dtype)
    base.requires_grad_(False)
    module = LoKRModule(
        "self_attn.q_proj",
        base,
        lora_dim=8,
        alpha=16,
        lokr_factor=64,
        full_factor=full_factor,
    ).cuda()
    module.lokr_backend = "triton"
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.05, 0.05)
    reference = copy.deepcopy(module)
    reference.lokr_backend = "torch"
    x = torch.randn(2, 17, 2048, device="cuda", requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=dtype):
        result = module(x)
        expected = reference(ref_x)
    assert result.dtype == expected.dtype == dtype
    torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
    result.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-5, rtol=0.05)
    for (name, parameter), (_, ref_parameter) in zip(
        module.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, ref_parameter.grad, atol=2e-5, rtol=0.08
        )
