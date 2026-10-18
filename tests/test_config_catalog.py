"""Only runnable template families belong in the configuration selectors."""

from __future__ import annotations

import pytest
import toml

from webui.services import config_service as config


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    builtin = configs / "gui-methods"
    custom = configs / "custom/variants"
    builtin.mkdir(parents=True)
    custom.mkdir(parents=True)
    (configs / "base.toml").write_text('network_module = "networks.lora_anima"\n')
    (configs / "presets.toml").write_text("[default]\n")
    for name in ("lora", "tlora"):
        (builtin / f"{name}.toml").write_text(
            toml.dumps({"network_dim": 16, "variant": {"family": name, "label": name}})
        )
    monkeypatch.setattr(config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config, "GUI_METHODS_DIR", builtin)
    monkeypatch.setattr(config, "CUSTOM_VARIANTS_DIR", custom)
    monkeypatch.setattr(config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config, "CUSTOM_PRESETS_DIR", configs / "custom/presets")
    return builtin, custom


def test_families_without_templates_are_not_advertised(catalog):
    assert config.list_methods() == ["lora", "tlora"]
    assert not {"reft", "postfix", "fera", "ip_adapter"} & set(config.list_methods())


def test_builtin_overlay_is_not_a_custom_template_in_other_families(catalog):
    _, custom = catalog
    path = custom / "lora.toml"
    path.write_text('network_dim = 32\n[variant]\nfamily = "reft"\n')
    before = path.read_bytes()

    assert config.list_methods() == ["lora", "tlora"]
    assert config.list_gui_variants("lora") == ["lora"]
    assert config.list_gui_variants("tlora") == ["tlora"]
    assert config.variant_labels("lora") == {"lora": "lora"}
    assert config.variant_labels("tlora") == {"tlora": "tlora"}
    assert config.list_gui_variants("reft") == []
    # Existing bookmarks and saved selections keep their readable alias.
    original, _ = config.merged_gui_variant_preset("lora", "default")
    alias, _ = config.merged_gui_variant_preset("custom/lora", "default")
    assert alias == original and alias["network_dim"] == 32
    assert path.read_bytes() == before


def test_explicit_custom_family_is_listed_only_in_its_declared_family(catalog):
    _, custom = catalog
    (custom / "my-control.toml").write_text(
        'network_module = "networks.methods.controlnet_lllite"\n'
        '[variant]\nfamily = "controlnet"\nlabel = "My control"\n'
    )

    assert config.list_methods() == ["lora", "tlora", "controlnet"]
    assert config.list_gui_variants("controlnet") == ["custom/my-control"]
    assert config.variant_labels("controlnet") == {"custom/my-control": "My control"}
    for family in ("lora", "tlora"):
        assert "custom/my-control" not in config.list_gui_variants(family)
        assert "custom/my-control" not in config.variant_labels(family)


def test_unclassified_legacy_custom_keeps_its_existing_availability(catalog):
    _, custom = catalog
    path = custom / "legacy.toml"
    path.write_text("network_dim = 8\n")
    before = path.read_bytes()

    assert config.list_methods() == ["lora", "tlora"]
    for family in config.list_methods():
        assert config.list_gui_variants(family) == [family, "custom/legacy"]
        assert config.variant_labels(family)["custom/legacy"] == "legacy"
    assert path.read_bytes() == before


def test_explicit_custom_template_can_supply_a_previously_unavailable_family(catalog):
    _, custom = catalog
    (custom / "my-reft.toml").write_text(
        'network_module = "custom_network"\n[variant]\nfamily = "reft"\n'
    )
    assert "reft" in config.list_methods()
    assert config.list_gui_variants("reft") == ["custom/my-reft"]
    assert config.list_gui_variants("lora") == ["lora"]
