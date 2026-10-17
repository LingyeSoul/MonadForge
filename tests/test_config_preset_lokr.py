"""GUI variant selection must not inherit LoKr activation from old presets."""

from __future__ import annotations

import argparse

import pytest
import toml

from library.config.io import load_method_preset, load_path_overrides
from webui.services import config_service


@pytest.fixture
def lokr_preset_tree(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    presets = configs / "custom" / "presets"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    presets.mkdir(parents=True)
    for path in config_service.GUI_METHODS_DIR.glob("*.toml"):
        (methods / path.name).write_bytes(path.read_bytes())
    (configs / "base.toml").write_text(
        'network_module = "networks.lora_anima"\n', encoding="utf-8"
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    legacy = {
        "use_lokr": True,
        "lokr_factor": 64,
        "lokr_backend": "triton",
        "lokr_full_factor": True,
        "lokr_allow_legacy_dim": True,
        "decompose_both": True,
        "network_args": ["use_lokr=true", "lokr_factor=64", "decompose_both=true"],
        "blocks_to_swap": 12,
        "gradient_accumulation_steps": 3,
    }
    preset_path = presets / "from-lokr.toml"
    preset_path.write_text(toml.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_VARIANTS_DIR", overlays)
    monkeypatch.setattr(config_service, "CUSTOM_PRESETS_DIR", presets)
    return configs, preset_path


def _merge_both(configs, variant):
    ui, origin = config_service.merged_gui_variant_preset(variant, "from-lokr")
    # lora-gui unwraps custom/<stem> before passing the method to train.py.
    method = variant.removeprefix("custom/")
    training, provenance = load_method_preset(
        method,
        "from-lokr",
        configs_dir=str(configs),
        methods_subdir="gui-methods",
        return_provenance=True,
    )
    return ui, origin, training, provenance


@pytest.mark.parametrize(
    "variant",
    [
        "lora",
        "lora-8gb",
        "lora-convrot-vram",
        "tlora",
        "tlora-8gb",
        "hydralora",
        "hydralora-8gb",
        "chimera_hydra",
        "loha",
        "dylora",
        "vera",
        "glokr",
        "easycontrol",
        "controlnet",
        "soft_tokens",
    ],
)
def test_legacy_lokr_preset_does_not_change_selected_variant(lokr_preset_tree, variant):
    from train import resolve_network_kwargs

    configs, preset_path = lokr_preset_tree
    before = preset_path.read_bytes()
    ui, origin, training, provenance = _merge_both(configs, variant)
    for merged, sources in ((ui, origin), (training, provenance)):
        assert "use_lokr" not in merged
        assert "use_lokr" not in sources
        assert not any(key.startswith("lokr_") for key in merged)
        assert not any(key.startswith("lokr_") for key in sources)
        assert not any(
            "use_lokr=" in item or "lokr_" in item
            for item in merged.get("network_args", [])
        )
        if variant != "glokr":
            assert "decompose_both" not in merged
        assert merged["gradient_accumulation_steps"] == 3
    kwargs = resolve_network_kwargs(argparse.Namespace(**training))
    assert "use_lokr" not in kwargs
    assert not any(key.startswith("lokr_") for key in kwargs)
    paths = load_path_overrides(
        "from-lokr", str(configs), method=variant, methods_subdir="gui-methods"
    )
    assert "use_lokr" not in paths
    assert "lokr_backend" not in paths
    assert preset_path.read_bytes() == before


@pytest.mark.parametrize("variant", ["lokr", "custom/my-lokr", "lora"])
def test_lokr_method_and_overlay_keep_compatible_preset_tuning(
    lokr_preset_tree, variant
):
    configs, preset_path = lokr_preset_tree
    before = preset_path.read_bytes()
    overlays = configs / "custom" / "variants"
    if variant == "custom/my-lokr":
        (overlays / "my-lokr.toml").write_text(
            "use_lokr = true\nlokr_factor = 16\n", encoding="utf-8"
        )
    elif variant == "lora":
        (overlays / "lora.toml").write_text(
            "use_lokr = true\nlokr_factor = 16\n", encoding="utf-8"
        )
    else:
        (overlays / "lokr.toml").write_text(
            'lokr_backend = "torch"\n', encoding="utf-8"
        )
    ui, origin, training, provenance = _merge_both(configs, variant)
    for merged in (ui, training):
        assert merged["use_lokr"] is True
        assert merged["lokr_allow_legacy_dim"] is True
        assert merged["lokr_factor"] == (8 if variant == "lokr" else 16)
        assert "use_lokr=true" not in merged.get("network_args", [])
        assert merged["gradient_accumulation_steps"] == 3
    from train import resolve_network_kwargs

    kwargs = resolve_network_kwargs(argparse.Namespace(**training))
    assert kwargs["use_lokr"].lower() == "true"
    assert int(kwargs["lokr_factor"]) == ui["lokr_factor"]
    assert origin["use_lokr"] == "method"
    assert "variants" in provenance["use_lokr"]
    assert preset_path.read_bytes() == before


def test_disabled_lokr_overlay_filters_old_preset(lokr_preset_tree):
    configs, _preset_path = lokr_preset_tree
    (configs / "custom" / "variants" / "lokr.toml").write_text(
        "use_lokr = false\n", encoding="utf-8"
    )
    ui, origin, training, _provenance = _merge_both(configs, "lokr")
    for merged in (ui, training):
        assert merged["use_lokr"] is False
        # Explicit method parameters remain, but preset-only tuning is gone.
        assert merged["lokr_factor"] == 8
        assert "lokr_allow_legacy_dim" not in merged
        assert "network_args" not in merged
    assert origin["use_lokr"] == "method"


def test_preset_toggle_cannot_disable_explicit_lokr_method(lokr_preset_tree):
    from train import resolve_network_kwargs

    configs, preset_path = lokr_preset_tree
    preset_path.write_text('network_args = ["use_lokr=false"]\n', encoding="utf-8")
    _ui, _origin, training, _provenance = _merge_both(configs, "lokr")
    kwargs = resolve_network_kwargs(argparse.Namespace(**training))
    assert kwargs["use_lokr"].lower() == "true"


def test_toggle_based_cli_method_preserves_existing_preset_contract(lokr_preset_tree):
    configs, _preset_path = lokr_preset_tree
    methods = configs / "methods"
    methods.mkdir()
    (methods / "lora.toml").write_text("network_dim = 16\n", encoding="utf-8")
    training = load_method_preset("lora", "from-lokr", configs_dir=str(configs))
    assert training["use_lokr"] is True
    assert training["lokr_factor"] == 64
    assert training["network_args"][0] == "use_lokr=true"
