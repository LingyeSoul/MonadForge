"""Anima form metadata follows the adapter and enabled training features."""

from __future__ import annotations

import pytest
import toml

from webui.services import config_service as config


def _visible(key, values):
    result = config.describe_config(values, {}, "custom/example", "default", "en")
    field = next(field for field in result["fields"] if field["key"] == key)
    conditions = field["visible_when"]
    return conditions is None or any(
        all(values.get(parent) in accepted for parent, accepted in clause.items())
        for clause in conditions
    )


@pytest.mark.parametrize(
    ("key", "parent", "enabled"),
    [
        ("lokr_factor", "use_lokr", True),
        ("lokr_backend", "use_lokr", True),
        ("lokr_full_factor", "use_lokr", True),
        ("glokr_factor", "use_glokr", True),
        ("glokr_bora", "use_glokr", True),
        ("dylora_unit", "use_dylora", True),
        ("vera_seed", "use_ve", True),
        ("min_rank", "use_timestep_mask", True),
        ("alpha_rank_scale", "use_timestep_mask", True),
        ("num_experts", "use_moe_style", "shared_A"),
        ("router_source", "use_moe_style", "shared_A"),
        ("num_experts_content", "use_chimera_hydra", True),
        ("freq_router_mode", "use_chimera_hydra", True),
        ("repa_weight", "use_repa", True),
        ("sigma_lowres_route", "sigma_lowres", True),
        ("adaln_rank", "train_adaln", True),
        ("compile_dynamic_seq", "torch_compile", True),
        ("validation_split_num", "use_valid", True),
    ],
)
def test_feature_fields_follow_parent_without_losing_values(key, parent, enabled):
    values = {"network_module": "networks.lora_anima", key: 7, parent: False}
    assert not _visible(key, values)
    values[parent] = enabled
    assert _visible(key, values)
    assert values[key] == 7


def test_kronecker_shared_field_and_plain_lora_initialization():
    values = {"decompose_both": False, "down_init": "weight_svd"}
    assert not _visible("decompose_both", values)
    assert _visible("down_init", values)
    for flag in ("use_lokr", "use_glokr"):
        configured = {**values, flag: True}
        assert _visible("decompose_both", configured)
        assert not _visible("down_init", configured)
    for flag in ("use_loha", "use_ve", "use_dylora", "use_ortho_init"):
        assert not _visible("down_init", {**values, flag: True})


def test_chimera_uses_its_pools_instead_of_generic_moe_expert_controls():
    values = {
        "use_chimera_hydra": True,
        "use_moe_style": "shared_A",
        "num_experts": 4,
        "num_experts_content": 3,
        "router_targets": "mlp",
    }
    assert not _visible("num_experts", values)
    assert _visible("num_experts_content", values)
    assert _visible("router_targets", values)


def test_router_controls_follow_the_router_that_actually_consumes_them():
    values = {
        "use_chimera_hydra": True,
        "freq_router_mode": "fei",
        "router_hidden_dim": 64,
        "freq_router_tau": 1.0,
        "freq_router_layer_norm": True,
        "balance_w_freq": 0.1,
    }
    assert _visible("router_hidden_dim", values)  # ContentRouter is still learned.
    assert _visible("freq_router_tau", values)
    assert not _visible("freq_router_layer_norm", values)
    assert not _visible("balance_w_freq", values)
    values["freq_router_mode"] = "learned"
    assert not _visible("freq_router_tau", values)
    assert _visible("freq_router_layer_norm", values)
    assert _visible("balance_w_freq", values)


def test_conditioning_modules_do_not_expose_unused_lora_or_repa_controls():
    values = {
        "network_module": "networks.methods.controlnet_lllite",
        "network_dim": 1,
        "network_alpha": 1,
        "use_repa": True,
        "train_adaln": True,
        "adaln_rank": 16,
        "channel_scaling_alpha": 0.5,
    }
    for key in values.keys() - {"network_module"}:
        assert not _visible(key, values), key
    easycontrol = {**values, "network_module": "networks.methods.easycontrol"}
    assert _visible("network_dim", easycontrol)
    assert _visible("use_repa", easycontrol)
    assert _visible("adaln_rank", easycontrol)


def test_nested_feature_dependencies_hide_inactive_subsettings():
    values = {
        "base_compute": "w8a16_convrot",
        "convrot_prequant_path": "cache.pt",
        "convrot_weight_source": "online_from_bf16",
    }
    assert not _visible("convrot_prequant_path", values)
    values["convrot_weight_source"] = "prequant_checkpoint"
    assert _visible("convrot_prequant_path", values)
    values["base_compute"] = "bf16"
    assert not _visible("convrot_prequant_path", values)


def test_config_api_preserves_visibility_metadata(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from webui.api.config import router

    merged = config.describe_config(
        {"use_lokr": False, "lokr_factor": 8}, {}, "lora", "default", "en"
    )
    monkeypatch.setattr(config, "build_merged_config", lambda *a, **kw: merged)
    app = FastAPI()
    app.include_router(router, prefix="/api/config")
    with TestClient(app) as client:
        response = client.get("/api/config/merged?variant=lora")
    assert response.status_code == 200
    fields = {field["key"]: field for field in response.json()["fields"]}
    assert fields["lokr_factor"]["visible_when"] == config._field_visibility(
        "lokr_factor"
    )
    assert fields["lokr_factor"]["value"] == 8


def test_hidden_settings_survive_overlay_save_and_reenable(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    (configs / "base.toml").write_text('network_module = "networks.lora_anima"\n')
    (configs / "presets.toml").write_text('[default]\n[default.gui]\nlabel="Default"\n')
    (methods / "lora.toml").write_text("network_dim=32\nnetwork_alpha=32\n")
    (overlays / "lora.toml").write_text("use_lokr=false\nlokr_factor=16\n")
    monkeypatch.setattr(config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", overlays)
    monkeypatch.setattr(config, "CUSTOM_PRESETS_DIR", configs / "custom" / "presets")

    response = config.build_merged_config("lora", "default")
    fields = {field["key"]: field for field in response["fields"]}
    assert fields["lokr_factor"]["value"] == 16
    assert fields["lokr_factor"]["visible_when"]
    assert not ({"gui", "label", "group", "order", "description"} & fields.keys())
    config.save_variant_config("lora", {"learning_rate": 0.0002})
    saved = toml.loads((overlays / "lora.toml").read_text())
    assert saved["lokr_factor"] == 16
    assert saved["use_lokr"] is False
    config.save_variant_config("lora", {"use_lokr": True})
    merged, _ = config.merged_gui_variant_preset("lora", "default")
    assert merged["lokr_factor"] == 16
    assert _visible("lokr_factor", merged)
