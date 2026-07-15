"""Tests for the M3 config schema: validation, provenance, print-config.

Covers:

* schema population (known keys present, aliases resolved)
* typo detection (unknown key → warning with file:line; strict → raises)
* off-list ``choices`` rejection
* soft type coercion (TOML ``1`` → ``float`` when schema says float)
* every ``methods × presets`` combination round-trips without warnings
* ``_render_merged_toml`` output re-parses as valid TOML whose keys are
  a subset of the populated schema
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
import toml

from library.config import schema as config_schema
from library.config.io import (
    _flatten_toml,
    _render_merged_toml,
    load_dataset_config_from_base,
    load_method_preset,
    load_preset_section,
    list_presets,
)
from tests.conftest import iter_method_names


# ---------------------------------------------------------------------------
# Schema population
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_parser():
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())
    return parser


def test_schema_has_known_keys(populated_parser):
    schema = config_schema.get_schema()
    # a handful of must-have keys that come from different argparse layers
    for k in (
        "network_dim",
        "network_alpha",
        "optimizer_type",
        "learning_rate",
        "max_train_epochs",
        "attn_mode",
        "v100_flash_stability",
        "debug_finite_checks",
        "base_config",  # manual extra
        "use_moe_style",  # network-module allowlist (three-axis routing)
    ):
        assert k in schema, f"expected {k!r} in populated schema"


def test_choices_preserved(populated_parser):
    lw = config_schema.get_schema()["log_with"]
    assert "tensorboard" in lw.choices
    assert "wandb" in lw.choices
    assert "mem_efficient" in config_schema.get_schema()["attn_mode"].choices


# ---------------------------------------------------------------------------
# Typo / choice detection
# ---------------------------------------------------------------------------


def test_unknown_key_warns(populated_parser, tmp_path: Path, caplog):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with caplog.at_level(logging.WARNING):
        out = _flatten_toml({"a": {"network_ditm": 16}}, source=str(bogus))
    assert out == {"network_ditm": 16}
    assert any(
        "unknown key 'network_ditm'" in rec.getMessage() for rec in caplog.records
    )
    # line locator should include the line number
    assert any(":1:" in rec.getMessage() for rec in caplog.records)


def test_unknown_key_strict_raises(populated_parser, tmp_path: Path):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with pytest.raises(config_schema.ConfigSchemaError):
        _flatten_toml({"a": {"network_ditm": 16}}, source=str(bogus), strict=True)


def test_off_list_choice_warns(populated_parser, caplog):
    with caplog.at_level(logging.WARNING):
        _flatten_toml({"a": {"log_with": "carrierpigeon"}}, source="x.toml")
    assert any(
        "log_with" in rec.getMessage() and "not in choices" in rec.getMessage()
        for rec in caplog.records
    )


def test_int_to_float_coerced(populated_parser):
    # schema says network_alpha is float; TOML ``1`` comes in as int.
    out = _flatten_toml({"a": {"network_alpha": 64}}, source="x.toml")
    assert isinstance(out["network_alpha"], float)
    assert out["network_alpha"] == 64.0


# ---------------------------------------------------------------------------
# Round-trip: all methods × presets produce no warnings
# ---------------------------------------------------------------------------


METHODS = list(iter_method_names())


def _load_preset_names() -> list[str]:
    return list(toml.load("configs/presets.toml").keys())


def test_list_presets_includes_new_and_legacy_custom_layouts(tmp_path: Path):
    configs = tmp_path / "configs"
    (configs / "custom" / "presets").mkdir(parents=True)
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "custom" / "presets" / "V100.toml").write_text(
        'attn_mode = "torch"\n', encoding="utf-8"
    )
    (configs / "custom" / "V100.toml").write_text(
        'attn_mode = "flash"\n', encoding="utf-8"
    )
    (configs / "custom" / "legacy.toml").write_text(
        'attn_mode = "torch"\n', encoding="utf-8"
    )

    assert list_presets(str(configs)) == ["V100", "default", "legacy"]
    assert load_preset_section("V100", str(configs)) == {"attn_mode": "torch"}
    assert load_preset_section("legacy", str(configs)) == {"attn_mode": "torch"}


@pytest.mark.parametrize("method", METHODS)
def test_method_configs_clean(populated_parser, method: str, caplog):
    presets = _load_preset_names()
    for preset in presets:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            load_method_preset(method, preset)
        offenders = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
            and rec.name.startswith("library.train_util")
        ]
        assert not offenders, f"{method} × {preset} warnings: {offenders}"


# ---------------------------------------------------------------------------
# Provenance + render
# ---------------------------------------------------------------------------


def test_provenance_returned():
    merged, provenance = load_method_preset("lora", "default", return_provenance=True)
    # base key
    assert provenance["network_module"] == "configs/base.toml"
    # method key
    assert provenance["network_dim"] == "configs/methods/lora.toml"
    assert set(provenance) == set(merged)


def _reparse_without_comments(text: str) -> dict:
    # toml.loads ignores comments natively, but our output has `# --- from ... ---`
    # headers that are valid TOML comments, so it round-trips directly.
    return toml.loads(text)


def test_render_roundtrips_to_valid_toml(populated_parser):
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = load_method_preset("lora", "default", return_provenance=True)
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(["--method", "lora", "--preset", "default"], namespace=ns)

    rendered = _render_merged_toml(args, parser, provenance)
    parsed = _reparse_without_comments(rendered)

    schema = config_schema.get_schema()
    for key in parsed:
        assert key in schema, f"rendered key {key!r} not in schema"


def test_render_header_includes_method_and_preset(populated_parser):
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = load_method_preset("lora", "low_vram", return_provenance=True)
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(["--method", "lora", "--preset", "low_vram"], namespace=ns)
    rendered = _render_merged_toml(args, parser, provenance)
    assert "Method: lora" in rendered
    assert "Preset: low_vram" in rendered
    # section ordering: base → preset → method
    base_idx = rendered.index("configs/base.toml")
    preset_idx = rendered.index("configs/presets.toml[low_vram]")
    method_idx = rendered.index("configs/methods/lora.toml")
    assert base_idx < preset_idx < method_idx


# ---------------------------------------------------------------------------
# gui-methods overlay merge contract
# ---------------------------------------------------------------------------


def _build_gui_configs_tree(root: Path) -> str:
    """Materialize a minimal configs/ tree under *root* for overlay-merge tests.

    Mirrors the real layout: base.toml + gui-methods/<m>.toml builtin +
    custom/variants/<m>.toml overlay. The builtin ships ``max_train_epochs``
    (and the network_module needed by the trainer); the overlay is deliberately
    sparse — just a path override — to reproduce the "1600 steps" regression
    where a path-only overlay had silently dropped the builtin's epoch knob.
    """
    configs = root / "configs"
    (configs / "gui-methods").mkdir(parents=True)
    (configs / "custom" / "variants").mkdir(parents=True)

    (configs / "base.toml").write_text(
        'network_module = "networks.lora_anima"\noutput_name = "anima"\n',
        encoding="utf-8",
    )
    # presets.toml: load_method_preset(require_files=True) needs the preset.
    (configs / "presets.toml").write_text(
        "[default]\n",
        encoding="utf-8",
    )
    (configs / "gui-methods" / "lora.toml").write_text(
        "network_dim = 32\n"
        "network_alpha = 32\n"
        "learning_rate = 2e-5\n"
        "max_train_epochs = 4\n"
        'output_name = "anima"\n'
        "[variant]\n"
        'family = "lora"\n',
        encoding="utf-8",
    )
    # Sparse overlay: only a path override, no training knobs.
    (configs / "custom" / "variants" / "lora.toml").write_text(
        'output_name = "test"\n',
        encoding="utf-8",
    )
    return str(configs)


def test_sparse_overlay_inherits_builtin_knobs(tmp_path: Path):
    """A sparse gui-methods overlay must inherit the builtin's knobs.

    Regression guard for the contract mismatch where the training merge chain
    treated a user overlay as a wholesale replacement (dropping
    ``max_train_epochs`` → fallback to ``--max_train_steps`` default 1600),
    while the WebUI merged ``{**builtin, **overlay}``. The two paths must now
    agree: builtin fills gaps, overlay wins on conflict.
    """
    configs_dir = _build_gui_configs_tree(tmp_path)
    merged = load_method_preset(
        "lora", "default", configs_dir=configs_dir, methods_subdir="gui-methods"
    )
    # Inherited from builtin — would be absent (→ argparse default 1600) under
    # the old wholesale-replace behavior.
    assert merged.get("max_train_epochs") == 4
    assert merged.get("network_dim") == 32
    assert merged.get("learning_rate") == 2e-5
    # Overlay wins on conflict.
    assert merged.get("output_name") == "test"


def test_blank_network_weights_in_existing_custom_preset_is_unset(tmp_path: Path):
    """Legacy WebUI presets may contain an empty warm-start path."""
    configs_dir = Path(_build_gui_configs_tree(tmp_path))
    custom_presets = configs_dir / "custom" / "presets"
    custom_presets.mkdir(parents=True)
    (custom_presets / "legacy.toml").write_text(
        'network_weights = ""\n', encoding="utf-8"
    )

    merged, provenance = load_method_preset(
        "lora",
        "legacy",
        configs_dir=str(configs_dir),
        methods_subdir="gui-methods",
        return_provenance=True,
    )

    assert merged.get("network_weights") is None
    assert "network_weights" not in provenance


def test_overlay_provenance_points_at_user_file(tmp_path: Path):
    """Provenance for an inherited builtin key tags the overlay path (the
    user owns the effective config), matching the WebUI's origin model."""
    configs_dir = _build_gui_configs_tree(tmp_path)
    _merged, provenance = load_method_preset(
        "lora",
        "default",
        configs_dir=configs_dir,
        methods_subdir="gui-methods",
        return_provenance=True,
    )
    tag = provenance.get("max_train_epochs", "")
    assert "custom/variants/lora.toml" in tag


# ---------------------------------------------------------------------------
# gui-methods overlay with sparse [[datasets.subsets]] (regression: the
# full-blueprint replace path had treated a user variant declaring just
# num_repeats as a wholesale replacement of base.toml's dataset definition,
# dropping image_dir/cache_dir and crashing voluptuous with
# "required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']").
# A variant overlay is a sparse override, not a self-contained blueprint —
# subset-level keys must shallow-merge over base, just like flat keys.
# ---------------------------------------------------------------------------


def _build_gui_dataset_tree(root: Path) -> str:
    """Like ``_build_gui_configs_tree`` but base.toml carries the full dataset
    blueprint (mirrors the real repo), so we can assert the blueprint survives a
    sparse subset-level overlay."""
    configs = root / "configs"
    (configs / "gui-methods").mkdir(parents=True)
    (configs / "custom" / "variants").mkdir(parents=True)

    (configs / "base.toml").write_text(
        'resized_image_dir = "post_image_dataset/resized"\n'
        'lora_cache_dir = "post_image_dataset/lora"\n'
        "[[datasets]]\n"
        "batch_size = 1\n"
        "validation_split_num = 16\n"
        "[[datasets.subsets]]\n"
        'image_dir = "{resized_image_dir}"\n'
        'cache_dir = "{lora_cache_dir}"\n'
        "num_repeats = 1\n"
        "recursive = true\n"
        "[general]\n"
        'caption_extension = ".txt"\n',
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "gui-methods" / "lora.toml").write_text(
        "network_dim = 32\nlearning_rate = 2e-5\nmax_train_epochs = 4\n"
        '[variant]\nfamily = "lora"\n',
        encoding="utf-8",
    )
    return str(configs)


def test_sparse_subset_overlay_keeps_base_image_dir(tmp_path: Path):
    """A gui-methods variant that declares only ``num_repeats`` in
    ``[[datasets.subsets]]`` must NOT drop base.toml's ``image_dir`` /
    ``cache_dir``. The overlay is a sparse override; the blueprint owner is
    base.toml. Regression guard for the voluptuous ``required key not
    provided @ ...['image_dir']`` crash (io.py full-blueprint replace)."""
    configs_dir = _build_gui_dataset_tree(tmp_path)
    (Path(configs_dir) / "custom" / "variants" / "lora.toml").write_text(
        "[[datasets]]\nvalidation_split_num = 0\n"
        "[[datasets.subsets]]\nnum_repeats = 4\n",
        encoding="utf-8",
    )
    bp = load_dataset_config_from_base(
        configs_dir, method="lora", methods_subdir="gui-methods"
    )
    subset = bp["datasets"][0]["subsets"][0]
    # Overlay wins on declared keys.
    assert subset["num_repeats"] == 4
    # Base-provided required keys survive the merge.
    assert subset["image_dir"] == "post_image_dataset/resized"
    assert subset["cache_dir"] == "post_image_dataset/lora"
    assert subset["recursive"] is True
    # Dataset-level override too.
    assert bp["datasets"][0]["validation_split_num"] == 0
    # general untouched.
    assert bp["general"] == {"caption_extension": ".txt"}


def test_webui_num_repeats_writer_round_trips(tmp_path: Path):
    """The WebUI's ``_apply_num_repeats`` produces exactly the sparse subset
    structure that triggered the bug (a subset with ONLY ``num_repeats``, no
    ``image_dir``). Feed it through the training load chain to prove the
    save→load path is now crash-free end to end."""
    from webui.services.config_service import _apply_num_repeats

    configs_dir = _build_gui_dataset_tree(tmp_path)
    # Simulate what save_variant_config writes: start from an empty overlay
    # and apply a num_repeats override the way the GUI does.
    overlay: dict = {}
    _apply_num_repeats(overlay, value=4, base_value=1)
    # The GUI writes only the override — no image_dir / cache_dir.
    assert overlay == {"datasets": [{"subsets": [{"num_repeats": 4}]}]}, (
        "sanity: this is the exact shape that used to crash"
    )
    (Path(configs_dir) / "custom" / "variants" / "lora.toml").write_text(
        toml.dumps(overlay), encoding="utf-8"
    )
    bp = load_dataset_config_from_base(
        configs_dir, method="lora", methods_subdir="gui-methods"
    )
    subset = bp["datasets"][0]["subsets"][0]
    assert subset["num_repeats"] == 4
    assert subset["image_dir"] == "post_image_dataset/resized"
    assert subset["cache_dir"] == "post_image_dataset/lora"


# ---------------------------------------------------------------------------
# Symmetric guard: flat ``methods`` subdir (NOT gui-methods) with a full
# inline blueprint MUST still wholesale-replace base's dataset definition.
# This is the other half of the gui-methods shallow-merge exception — if the
# ``methods_subdir != GUI_METHODS_SUBDIR`` guard were ever inverted (``==``
# instead of ``!=``), a self-contained method like easycontrol would silently
# degrade to shallow-merge and drop its own ``image_dir``.
# ---------------------------------------------------------------------------


def test_flat_method_full_blueprint_replaces_base(tmp_path: Path):
    """A ``methods_subdir="methods"`` method file carrying its own
    ``[[datasets]]`` + ``[[datasets.subsets]]`` (image_dir included) replaces
    base.toml's blueprint wholesale — base's dataset/subset keys do NOT
    survive. Pairs with test_sparse_subset_overlay_keeps_base_image_dir to
    pin both branches of the gui-methods guard."""
    configs = tmp_path / "configs"
    (configs / "methods").mkdir(parents=True)

    # base.toml carries a blueprint with a DISTINCT image_dir we can detect.
    (configs / "base.toml").write_text(
        "[[datasets]]\nbatch_size = 1\n"
        "[[datasets.subsets]]\n"
        'image_dir = "BASE_IMAGE_DIR"\n'
        'cache_dir = "BASE_CACHE_DIR"\n'
        "num_repeats = 1\n",
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    # The method file is a self-contained blueprint (its own image_dir, no
    # base reference) — exactly the easycontrol-style layout.
    (configs / "methods" / "lora.toml").write_text(
        "[[datasets]]\nbatch_size = 4\n"
        "[[datasets.subsets]]\n"
        'image_dir = "METHOD_IMAGE_DIR"\n'
        "num_repeats = 8\n",
        encoding="utf-8",
    )

    bp = load_dataset_config_from_base(
        str(configs), method="lora", methods_subdir="methods"
    )
    subset = bp["datasets"][0]["subsets"][0]
    # Method's full blueprint wins — base's values are gone (wholesale replace).
    assert subset["image_dir"] == "METHOD_IMAGE_DIR"
    assert subset["num_repeats"] == 8
    # Base-only keys (cache_dir) are NOT inherited — full replace, not merge.
    assert "cache_dir" not in subset
    # Dataset-level too.
    assert bp["datasets"][0]["batch_size"] == 4
