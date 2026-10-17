"""Regression coverage for shipped preprocessing defaults and sparse overlays."""

import pytest

from library.config.io import load_method_preset, load_path_overrides


@pytest.fixture
def configs(tmp_path):
    (tmp_path / "custom").mkdir()
    (tmp_path / "methods").mkdir()
    (tmp_path / "base.toml").write_text("", encoding="utf-8")
    (tmp_path / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (tmp_path / "methods" / "lora.toml").write_text("", encoding="utf-8")
    (tmp_path / "preprocess.toml").write_text(
        "drop_lowres_images = true\nmin_pixels = 250000\n"
        "target_res = [1024]\nmultires_per_image = false\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("custom", [None, "target_res = [768]\n"])
def test_defaults_survive_missing_or_sparse_custom(configs, custom, monkeypatch):
    if custom is not None:
        (configs / "custom" / "preprocess.toml").write_text(custom, encoding="utf-8")
    values = load_path_overrides(configs_dir=str(configs))
    assert values["min_pixels"] == 250000
    assert values["drop_lowres_images"] is True
    assert values["target_res"] == ([1024] if custom is None else [768])

    from scripts.tasks import _common, preprocess

    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", values)
    monkeypatch.delenv("MIN_PIXELS", raising=False)
    monkeypatch.delenv("DROP_LOWRES_IMAGES", raising=False)
    assert preprocess._min_pixels_args() == ["--min_pixels", "250000"]
    config = preprocess._preprocess_config([])
    assert config["min_pixels"] == 250000


@pytest.mark.parametrize(
    "custom, expected",
    [("min_pixels = 100000\n", 100000), ("drop_lowres_images = false\n", 250000)],
)
def test_custom_filter_values_override_defaults(configs, custom, expected, monkeypatch):
    (configs / "custom" / "preprocess.toml").write_text(custom, encoding="utf-8")
    values = load_path_overrides(configs_dir=str(configs))
    assert values["min_pixels"] == expected
    from scripts.tasks import _common, preprocess

    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", values)
    monkeypatch.delenv("MIN_PIXELS", raising=False)
    monkeypatch.delenv("DROP_LOWRES_IMAGES", raising=False)
    effective = expected if values["drop_lowres_images"] else 0
    assert preprocess._min_pixels_args() == ["--min_pixels", str(effective)]


def test_legacy_base_filter_override_keeps_precedence(configs):
    (configs / "base.toml").write_text(
        "min_pixels = 123456\ntarget_res = [512]\n", encoding="utf-8"
    )
    values = load_path_overrides(configs_dir=str(configs))
    assert values["min_pixels"] == 123456
    assert values["target_res"] == [1024]


def test_training_tiers_match_preprocessing_with_sparse_overlay(configs):
    (configs / "custom" / "preprocess.toml").write_text(
        "multires_per_image = true\n", encoding="utf-8"
    )
    merged, provenance = load_method_preset(
        "lora", "default", configs_dir=str(configs), return_provenance=True
    )
    values = load_path_overrides(configs_dir=str(configs))
    for key in ("target_res", "multires_per_image"):
        assert merged[key] == values[key]
    assert provenance["target_res"].endswith("/preprocess.toml")
    assert "custom/preprocess.toml" in provenance["multires_per_image"]


def test_shared_defaults_keep_selected_source(configs, monkeypatch):
    from library.config import io
    from scripts.tasks import _common

    (configs / "preprocess.toml").write_text(
        'source_image_dir = "image_dataset"\nmin_pixels = 250000\n',
        encoding="utf-8",
    )
    (configs / "methods" / "lora.toml").write_text(
        'source_image_dir = "selected-dataset"\nmin_pixels = 999999\n',
        encoding="utf-8",
    )
    original = io.load_path_overrides
    monkeypatch.setattr(
        io, "load_path_overrides",
        lambda **kwargs: original(configs_dir=str(configs), **kwargs),
    )
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", None)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.setenv("METHOD", "lora")
    monkeypatch.setenv("METHODS_SUBDIR", "methods")
    monkeypatch.setenv("PRESET", "default")
    values = _common._path_overrides()
    assert values["source_image_dir"] == "selected-dataset"
    assert values["min_pixels"] == 250000

