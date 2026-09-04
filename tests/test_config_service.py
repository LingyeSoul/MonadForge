"""Drift guards for the WebUI's curated optimizer / scheduler option lists.

The WebUI ships a hand-curated ``_SELECT_OPTIONS`` for ``optimizer_type`` and
``lr_scheduler`` (``webui/services/config_service.py``). It must mirror exactly
what the trainer accepts: ``library/training/optimizers.py::get_optimizer`` and
``library/training/schedulers.py::get_scheduler_fix``. A prior drift after the
training-knobs prune (commit 772dda7) left the WebUI offering ``Adafactor`` and
five LR-scheduler names that the trainer had just dropped — selecting them
either crashed (Adafactor → AttributeError; inverse_sqrt → TypeError;
cosine_with_min_lr → ValueError) or silently dropped their parameters
(cosine_with_restarts / polynomial / warmup_stable_decay).

These tests lock the contract: every name the WebUI offers is accepted by the
trainer, and the names the WebUI must not offer stay off the list.
"""

from __future__ import annotations

import argparse

import pytest
import toml

from library.anima.training import add_anima_training_arguments
from library.config.io import load_method_preset
from library.inference.args import build_parser as build_inference_parser
from networks import NETWORK_KWARGS
from webui.services import config_service
from webui.services.config_service import _SELECT_OPTIONS, validate_config


def test_glokr_training_fields_are_complete_and_grouped_as_architecture():
    """Keep the WebUI aligned with GLoKr's documented training surface."""
    expected = {
        "network_dim",
        "network_alpha",
        "use_glokr",
        "decompose_both",
    } | {key for key in NETWORK_KWARGS if key.startswith("glokr_")}

    canonical = toml.loads(
        (config_service.METHODS_DIR / "glokr.toml").read_text(encoding="utf-8")
    )
    gui = toml.loads(
        (config_service.GUI_METHODS_DIR / "glokr.toml").read_text(encoding="utf-8")
    )

    result = config_service.build_merged_config("glokr", "default", lang="en")
    fields = {field["key"]: field for field in result["fields"]}

    assert expected <= canonical.keys()
    assert expected <= gui.keys()
    assert {key: gui[key] for key in expected} == {
        key: canonical[key] for key in expected
    }
    assert expected <= fields.keys()
    assert {key: fields[key]["group"] for key in expected} == {
        key: "Architecture" for key in expected
    }
    assert fields["channel_scaling_alpha"]["group"] == "Performance"


def test_sample_decode_inline_is_editable_and_round_trips(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    (configs / "base.toml").write_text(
        '[preview]\nsample_decode_inline = "auto"\n', encoding="utf-8"
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (methods / "lora.toml").write_text("network_dim = 16\n", encoding="utf-8")

    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_VARIANTS_DIR", overlays)

    result = config_service.build_merged_config("lora", "default", lang="en")
    field = next(f for f in result["fields"] if f["key"] == "sample_decode_inline")
    assert field["value"] == "auto"
    assert field["origin"] == "base"
    assert field["field_type"] == "select"
    assert field["group"] == "Preview Sampling"
    assert field["read_only"] is False
    assert field["options"] == ["auto", "true", "false"]

    config_service.save_variant_config("lora", {"sample_decode_inline": "false"})
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert saved["sample_decode_inline"] is False

    merged, origin = config_service.merged_gui_variant_preset("lora", "default")
    assert merged["sample_decode_inline"] is False
    assert origin["sample_decode_inline"] == "method"

    config_service.save_variant_config("lora", {"sample_decode_inline": "auto"})
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert "sample_decode_inline" not in saved


def test_sample_decode_inline_validation_rejects_unknown_mode():
    assert validate_config({"sample_decode_inline": "sometimes"}) == [
        "sample_decode_inline must be auto, true, or false"
    ]


def test_lora_fp32_compute_is_editable_tri_state_and_round_trips(
    monkeypatch, tmp_path
):
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    (configs / "base.toml").write_text(
        'network_module = "networks.lora_anima"\n', encoding="utf-8"
    )
    (configs / "model.toml").write_text("", encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (methods / "lora.toml").write_text("network_dim = 16\n", encoding="utf-8")

    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_VARIANTS_DIR", overlays)

    result = config_service.build_merged_config("lora", "default", lang="en")
    field = next(f for f in result["fields"] if f["key"] == "lora_fp32_compute")
    assert field["value"] == "auto"
    assert field["origin"] == "base"
    assert field["field_type"] == "select"
    assert field["group"] == "Performance"
    assert field["read_only"] is False
    assert field["options"] == ["auto", "true", "false"]
    assert field["description"]

    config_service.save_variant_config("lora", {"lora_fp32_compute": "true"})
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert saved["lora_fp32_compute"] is True

    merged, origin = config_service.merged_gui_variant_preset("lora", "default")
    assert merged["lora_fp32_compute"] is True
    assert origin["lora_fp32_compute"] == "method"

    config_service.save_variant_config("lora", {"lora_fp32_compute": "false"})
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert saved["lora_fp32_compute"] is False

    config_service.save_variant_config("lora", {"lora_fp32_compute": "auto"})
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert saved["lora_fp32_compute"] == "auto"

    training = load_method_preset(
        "lora", "default", configs_dir=str(configs), methods_subdir="gui-methods"
    )
    assert "lora_fp32_compute" not in training


def test_lora_fp32_auto_clears_preset_and_legacy_network_args(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    custom_presets = configs / "custom" / "presets"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    custom_presets.mkdir(parents=True)
    (configs / "base.toml").write_text(
        'network_module = "networks.lora_anima"\n', encoding="utf-8"
    )
    (configs / "model.toml").write_text("", encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (custom_presets / "forced.toml").write_text(
        'network_args = ["lora_fp32_compute=true"]\n', encoding="utf-8"
    )
    (methods / "lora.toml").write_text(
        'network_args = ["use_timestep_mask=true", "lora_fp32_compute=false"]\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_PRESETS_DIR", custom_presets)
    monkeypatch.setattr(config_service, "CUSTOM_VARIANTS_DIR", overlays)

    result = config_service.build_merged_config("lora", "forced", lang="en")
    field = next(f for f in result["fields"] if f["key"] == "lora_fp32_compute")
    assert field["value"] == "false"
    assert field["origin"] == "method"

    config_service.save_variant_config("lora", {"lora_fp32_compute": "true"})
    training = load_method_preset(
        "lora", "forced", configs_dir=str(configs), methods_subdir="gui-methods"
    )
    assert str(training["lora_fp32_compute"]).lower() == "true"
    assert training["network_args"] == ["use_timestep_mask=true"]

    config_service.save_variant_config("lora", {"lora_fp32_compute": "auto"})
    merged, origin = config_service.merged_gui_variant_preset("lora", "forced")
    assert merged["lora_fp32_compute"] == "auto"
    assert origin["lora_fp32_compute"] == "method"

    training = load_method_preset(
        "lora", "forced", configs_dir=str(configs), methods_subdir="gui-methods"
    )
    assert "lora_fp32_compute" not in training
    assert training["network_args"] == ["use_timestep_mask=true"]


def test_lora_fp32_compute_not_shown_for_non_lora_network(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    custom_presets = configs / "custom" / "presets"
    methods.mkdir(parents=True)
    custom_presets.mkdir(parents=True)
    (configs / "base.toml").write_text(
        'network_module = "networks.lora_anima"\n', encoding="utf-8"
    )
    (configs / "model.toml").write_text("", encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (custom_presets / "forced.toml").write_text(
        "lora_fp32_compute = true\n", encoding="utf-8"
    )
    (methods / "easycontrol.toml").write_text(
        'network_module = "networks.methods.easycontrol"\n', encoding="utf-8"
    )

    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_PRESETS_DIR", custom_presets)
    monkeypatch.setattr(
        config_service, "CUSTOM_VARIANTS_DIR", configs / "custom" / "variants"
    )

    result = config_service.build_merged_config(
        "easycontrol", "forced", lang="en"
    )
    assert "lora_fp32_compute" not in {field["key"] for field in result["fields"]}


def test_lora_fp32_compute_validation_rejects_unknown_mode():
    assert validate_config({"lora_fp32_compute": "sometimes"}) == [
        "lora_fp32_compute must be auto, true, or false"
    ]


def test_lora_fp32_compute_help_documents_reliable_mode_override():
    description = config_service._field_desc("lora_fp32_compute", "en")
    assert description is not None
    assert "reliable block-swap mode overrides false" in description


def test_create_custom_preset_normalizes_lora_fp32_compute(monkeypatch, tmp_path):
    configs = tmp_path / "configs"
    custom_presets = configs / "custom" / "presets"
    custom_presets.mkdir(parents=True)
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")

    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_PRESETS_DIR", custom_presets)

    config_service.create_custom_preset(
        "auto-protection", {"lora_fp32_compute": "auto"}
    )
    auto_saved = toml.loads(
        (custom_presets / "auto-protection.toml").read_text(encoding="utf-8")
    )
    assert "lora_fp32_compute" not in auto_saved

    config_service.create_custom_preset(
        "forced-protection", {"lora_fp32_compute": "true"}
    )
    forced_saved = toml.loads(
        (custom_presets / "forced-protection.toml").read_text(encoding="utf-8")
    )
    assert forced_saved["lora_fp32_compute"] is True


def test_create_custom_preset_strips_neutral_resume_defaults(monkeypatch, tmp_path):
    """A fresh WebUI preset must not turn an empty warm-start into a path."""
    configs = tmp_path / "configs"
    custom_presets = configs / "custom" / "presets"
    methods = configs / "gui-methods"
    custom_presets.mkdir(parents=True)
    methods.mkdir(parents=True)
    (configs / "base.toml").write_text("network_dim = 16\n", encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (methods / "lora.toml").write_text("network_alpha = 16\n", encoding="utf-8")

    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_PRESETS_DIR", custom_presets)

    config_service.create_custom_preset(
        "fresh",
        {
            "network_weights": "",
            "dim_from_weights": False,
            "save_state_on_train_end": False,
        },
    )

    saved = toml.loads((custom_presets / "fresh.toml").read_text(encoding="utf-8"))
    assert not (set(config_service._RESUME_DEFAULTS) & saved.keys())

    merged = load_method_preset(
        "lora", "fresh", str(configs), methods_subdir="gui-methods"
    )
    assert merged.get("network_weights") is None

    config_service.create_custom_preset(
        "warm-start",
        {
            "network_weights": "weights/model.safetensors",
            "dim_from_weights": True,
            "save_state_on_train_end": True,
        },
    )
    warm_start = toml.loads(
        (custom_presets / "warm-start.toml").read_text(encoding="utf-8")
    )
    assert warm_start == {
        "network_weights": "weights/model.safetensors",
        "dim_from_weights": True,
        "save_state_on_train_end": True,
    }
    warm_merged = load_method_preset(
        "lora", "warm-start", str(configs), methods_subdir="gui-methods"
    )
    assert warm_merged["network_weights"] == "weights/model.safetensors"
    assert warm_merged["dim_from_weights"] is True


def test_lokr_legacy_dim_validation_accepts_official_full_matrix_semantics():
    errors = validate_config(
        {
            "use_lokr": True,
            "network_dim": 114514,
            "network_alpha": 32,
            "lokr_full_factor": False,
        }
    )
    assert errors == []


def test_lokr_recommended_full_factor_config_is_valid():
    errors = validate_config(
        {
            "use_lokr": True,
            "network_dim": 32,
            "network_alpha": 32,
            "lokr_full_factor": True,
            "decompose_both": False,
        }
    )
    assert errors == []


def test_lokr_full_factor_with_decompose_both_is_valid():
    errors = validate_config(
        {
            "use_lokr": True,
            "network_dim": 32,
            "network_alpha": 16,
            "lokr_full_factor": True,
            "decompose_both": True,
        }
    )
    assert errors == []


def test_lokr_with_timestep_mask_is_rejected():
    errors = validate_config(
        {
            "use_lokr": True,
            "network_dim": 32,
            "network_alpha": 32,
            "lokr_full_factor": True,
            "use_timestep_mask": True,
        }
    )
    assert errors == [
        "use_timestep_mask is not supported by LoKr; set it to false"
    ]


def test_lokr_legacy_dim_validation_allows_explicit_resume_compatibility():
    errors = validate_config(
        {
            "use_lokr": True,
            "network_dim": 114514,
            "network_alpha": 32,
            "lokr_allow_legacy_dim": True,
        }
    )
    assert errors == []


# Names pruned from the trainer in commit 772dda7. They must never reappear in
# the WebUI's curated lists — each one either crashes the trainer or silently
# drops its parameters (see module docstring).
_PRUNED_OPTIMIZERS = {"Adafactor"}
_PRUNED_SCHEDULERS = {
    "cosine_with_restarts",
    "polynomial",
    "inverse_sqrt",
    "cosine_with_min_lr",
    "warmup_stable_decay",
}


def _choices(parser: argparse.ArgumentParser, dest: str) -> set[str]:
    action = next(action for action in parser._actions if action.dest == dest)
    return set(action.choices or ())


def test_attention_modes_match_training_and_inference_parsers():
    training_parser = argparse.ArgumentParser()
    add_anima_training_arguments(training_parser)
    training_modes = _choices(training_parser, "attn_mode") - {"sdpa"}
    inference_modes = _choices(build_inference_parser(), "attn_mode") - {"sdpa"}
    webui_modes = set(_SELECT_OPTIONS["attn_mode"]) - {"xformers"}

    assert webui_modes == training_modes == inference_modes


def test_v100_flash_fields_are_grouped_and_typed_for_webui():
    result = config_service.build_merged_config("lora", "default", lang="en")
    fields = {field["key"]: field for field in result["fields"]}

    assert fields["compile_dynamic_seq"]["group"] == "Performance"
    assert config_service._K2G["v100_flash_stability"] == "Performance"
    assert config_service._K2G["debug_finite_checks"] == "Performance"
    assert _SELECT_OPTIONS["v100_flash_stability"] == ["off", "hybrid", "safe"]


def test_no_pruned_optimizers_offered():
    offered = set(_SELECT_OPTIONS["optimizer_type"])
    leaked = offered & _PRUNED_OPTIMIZERS
    assert not leaked, (
        f"WebUI offers optimizer(s) the trainer no longer supports: {leaked}. "
        f"Remove them from _SELECT_OPTIONS['optimizer_type'] in "
        f"webui/services/config_service.py."
    )


def test_came_is_offered_in_webui():
    assert "CAME" in _SELECT_OPTIONS["optimizer_type"]


def test_no_pruned_schedulers_offered():
    offered = set(_SELECT_OPTIONS["lr_scheduler"])
    leaked = offered & _PRUNED_SCHEDULERS
    assert not leaked, (
        f"WebUI offers lr_scheduler(s) the trainer no longer supports: {leaked}. "
        f"Remove them from _SELECT_OPTIONS['lr_scheduler'] in "
        f"webui/services/config_service.py."
    )


def test_every_offered_optimizer_is_accepted_by_trainer():
    """Every optimizer in the WebUI list must resolve in ``get_optimizer``
    without raising. Guards against both pruned names and typos."""
    import argparse

    from library.training.optimizers import get_optimizer

    for name in _SELECT_OPTIONS["optimizer_type"]:
        args = argparse.Namespace(
            optimizer_type=name,
            optimizer_args=None,
            learning_rate=1e-4,
            max_grad_norm=0.0,
        )
        # get_optimizer only needs a single trivial trainable param; it builds
        # the optimizer class. We don't step it, so the heavy imports (bnb /
        # lion / schedulefree) only fire for the names that need them.
        param = __import__("torch").nn.Parameter(__import__("torch").zeros(1))
        try:
            get_optimizer(args, [{"params": [param]}])
        except ImportError:
            # An optional dependency (bitsandbytes / lion_pytorch / schedulefree)
            # not installed in the test env is fine — the *name* is still valid,
            # the trainer would just ask the user to install it. The drift bug
            # raised AttributeError/KeyError, not ImportError.
            continue


def test_every_offered_scheduler_is_accepted_by_trainer():
    """Every scheduler in the WebUI list must resolve in ``get_scheduler_fix``
    without raising. Guards against both pruned names and typos."""
    import argparse

    import torch
    from library.training.schedulers import get_scheduler_fix

    for name in _SELECT_OPTIONS["lr_scheduler"]:
        # ``constant`` rejects num_warmup_steps; the rest require it. Match
        # each name's contract so the test exercises the real build path.
        # ``optimizer_type`` is read by ``is_schedulefree_optimizer``; keep a
        # non-schedulefree type so we test the scheduler side directly.
        needs_warmup = name != "constant"
        # ``piecewise_constant`` is parametric — it requires a ``step_rules``
        # value passed through ``--lr_scheduler_args value="..."`` (the trainer
        # surfaces this as a required knob, not a free name). Exercising it
        # here would test the step_rules contract, not the name-acceptance
        # contract this guard is about, so skip it (it's still in the offered
        # list and is exercised by the round-trip tests in test_config.py).
        if name == "piecewise_constant":
            continue
        args = argparse.Namespace(
            lr_scheduler=name,
            lr_scheduler_type="",
            lr_scheduler_args=None,
            lr_warmup_steps=10 if needs_warmup else None,
            max_train_steps=100,
            optimizer_type="AdamW",
        )
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1e-3)
        # Should not raise. (The pruned names raised TypeError / ValueError
        # here — that's exactly what this test catches.)
        get_scheduler_fix(args, optimizer, num_processes=1)


def test_piecewise_constant_still_offered():
    """``piecewise_constant`` is parametric (``step_rules`` is fed via
    ``--lr_scheduler_args step_rules="..."``), so its parameter contract is
    exercised by the round-trip tests in test_config.py rather than the
    name-acceptance loop above. This just pins that it remains offered (so
    the WebUI dropdown stays in sync with what the trainer accepts)."""
    assert "piecewise_constant" in _SELECT_OPTIONS["lr_scheduler"]


def test_convrot_fields_are_editable_and_grouped_for_webui():
    result = config_service.build_merged_config("lora", "default", lang="en")
    fields = {field["key"]: field for field in result["fields"]}
    expected = {
        "base_compute",
        "convrot_group_size",
        "convrot_hadamard",
        "convrot_scope",
        "convrot_weight_source",
        "convrot_prequant_path",
        "convrot_min_in_features",
        "convrot_largest_in_features_only",
        "convrot_large_layer_mode",
        "convrot_large_min_in_features",
    }

    assert expected <= fields.keys()
    assert {fields[key]["group"] for key in expected} == {"Performance"}
    assert all(fields[key]["read_only"] is False for key in expected)
    assert fields["base_compute"]["options"] == [
        "bf16",
        "w8a16_convrot",
        "w8a8_convrot",
    ]
    assert fields["convrot_group_size"]["field_type"] == "int"
    assert fields["convrot_large_layer_mode"]["options"] == [
        "none",
        "w8a16",
        "w8a8",
    ]
    assert all(fields[key]["description"] for key in expected)


def test_convrot_vram_profile_is_discoverable_and_resolves() -> None:
    assert "lora-convrot-vram" in config_service.list_gui_variants("lora")

    result = config_service.build_merged_config(
        "lora-convrot-vram", "default", lang="en"
    )
    fields = {field["key"]: field for field in result["fields"]}
    assert fields["base_compute"]["value"] == "w8a16_convrot"
    assert fields["convrot_scope"]["value"] == "all"
    assert fields["convrot_group_size"]["value"] == 256
    assert fields["base_compute"]["origin"] == "method"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"base_compute": "int8"}, "base_compute must be"),
        ({"convrot_group_size": 128}, "convrot_group_size must be"),
        ({"convrot_scope": "decoder"}, "unknown ConvRot scope"),
        (
            {"convrot_weight_source": "prequant_checkpoint"},
            "convrot_prequant_path is required",
        ),
        (
            {"convrot_prequant_path": "weights.safetensors"},
            "convrot_prequant_path is only valid",
        ),
        (
            {
                "convrot_large_layer_mode": "w8a8",
                "convrot_large_min_in_features": 0,
            },
            "convrot_large_layer_mode requires",
        ),
        (
            {
                "convrot_large_layer_mode": "none",
                "convrot_large_min_in_features": 4096,
            },
            "convrot_large_min_in_features requires",
        ),
    ],
)
def test_convrot_config_validation_rejects_invalid_combinations(config, message):
    assert any(message in error for error in validate_config(config))


def test_convrot_config_validation_accepts_supported_profile():
    assert validate_config(
        {
            "base_compute": "w8a16_convrot",
            "convrot_group_size": 64,
            "convrot_hadamard": "regular",
            "convrot_scope": "mlp,attention_out",
            "convrot_weight_source": "online_from_bf16",
            "convrot_prequant_path": "",
            "convrot_min_in_features": 0,
            "convrot_large_layer_mode": "none",
            "convrot_large_min_in_features": 0,
        }
    ) == []


# ── network_reg_dims / network_reg_lrs (regex sets) ─────────────────────────


def _patch_regex_set_config_dirs(monkeypatch, tmp_path):
    """Isolate the config layers the same way the sample_decode_inline tests do."""
    configs = tmp_path / "configs"
    methods = configs / "gui-methods"
    overlays = configs / "custom" / "variants"
    methods.mkdir(parents=True)
    overlays.mkdir(parents=True)
    (configs / "base.toml").write_text(
        'network_module = "networks.lora_anima"\n', encoding="utf-8"
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (methods / "lora.toml").write_text("network_dim = 16\n", encoding="utf-8")

    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", methods)
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "CUSTOM_VARIANTS_DIR", overlays)
    return overlays


def test_regex_set_fields_render_in_architecture_group(monkeypatch, tmp_path):
    """The regex sets must surface as editable Architecture fields by default."""
    _patch_regex_set_config_dirs(monkeypatch, tmp_path)

    result = config_service.build_merged_config("lora", "default", lang="en")
    fields = {f["key"]: f for f in result["fields"]}
    for key in ("network_reg_dims", "network_reg_lrs"):
        assert key in fields, f"{key} missing from merged config"
        field = fields[key]
        assert field["field_type"] == "regex_set"
        assert field["group"] == "Architecture"
        assert field["read_only"] is False
        assert field["origin"] == "method"
        assert field["value"] == ""
        assert field["description"], f"{key} has no field help"


def test_regex_set_field_help_present_in_all_languages():
    from webui.explanations import _read_fields

    for lang in ("en", "cn", "ja", "ko"):
        help_texts = _read_fields(lang)
        assert help_texts.get("network_reg_dims"), f"missing dims help: {lang}"
        assert help_texts.get("network_reg_lrs"), f"missing lrs help: {lang}"


def test_regex_set_value_round_trips_through_variant_toml(monkeypatch, tmp_path):
    overlays = _patch_regex_set_config_dirs(monkeypatch, tmp_path)
    dims = r"blocks\.0\..*=8, blocks\.[12].*=16"
    lrs = r"blocks\.[01]\..*=1e-4, .*cross_attn.*=5e-5"

    config_service.save_variant_config(
        "lora", {"network_reg_dims": dims, "network_reg_lrs": lrs}
    )
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert saved["network_reg_dims"] == dims
    assert saved["network_reg_lrs"] == lrs

    merged, origin = config_service.merged_gui_variant_preset("lora", "default")
    assert merged["network_reg_dims"] == dims
    assert merged["network_reg_lrs"] == lrs
    assert origin["network_reg_dims"] == "method"

    # Saving empty strings clears the keys again — an empty value means the
    # feature is off and must not land in the TOML.
    config_service.save_variant_config(
        "lora", {"network_reg_dims": "", "network_reg_lrs": ""}
    )
    saved = toml.loads((overlays / "lora.toml").read_text(encoding="utf-8"))
    assert "network_reg_dims" not in saved
    assert "network_reg_lrs" not in saved


def test_regex_set_save_rejects_invalid_payloads():
    bad_dims = [
        "blocks.*",              # no '=' segment
        "blocks.*=abc",          # non-integer rank
        "blocks.*=-4",           # negative rank
        "([unclosed=8",          # regex does not compile
        "=8",                    # empty pattern
    ]
    for value in bad_dims:
        errors = validate_config({"network_reg_dims": value})
        assert errors, f"network_reg_dims={value!r} should fail validation"

    bad_lrs = [
        "blocks.*=1e-4x",        # not a number
        "blocks.*=-1e-4",        # negative LR
        "blocks.*",              # no '=' segment
    ]
    for value in bad_lrs:
        errors = validate_config({"network_reg_lrs": value})
        assert errors, f"network_reg_lrs={value!r} should fail validation"


def test_regex_set_validation_accepts_trainer_parseable_values():
    assert validate_config({"network_reg_dims": ""}) == []
    assert validate_config({"network_reg_lrs": ""}) == []
    assert validate_config(
        {
            "network_reg_dims": r"blocks\.0\..*=8, blocks\.[12].*=16",
            "network_reg_lrs": r"blocks\.[01]\..*=1e-4",
        }
    ) == []


def test_regex_set_reaches_trainer_network_kwargs(monkeypatch, tmp_path):
    """Full exposure seam: WebUI variant TOML → trainer network kwargs.

    Mirrors the GUI launch path: `make lora-gui` / the daemon merge
    ``gui-methods/<variant>.toml`` through ``load_method_preset``, the values
    land on the training args namespace, and ``resolve_network_kwargs``
    forwards allowlisted top-level keys into ``create_network``. If the WebUI
    saves a regex set, the trainer must see exactly that string.
    """
    from library.config.io import load_method_preset
    from train import resolve_network_kwargs
    from networks.lora_anima.config import LoRANetworkCfg
    from networks.lora_anima.network import LoRAModule

    overlays = _patch_regex_set_config_dirs(monkeypatch, tmp_path)
    dims = r"blocks\.0\..*=8"
    lrs = r"blocks\.[01]\..*=1e-4"
    config_service.save_variant_config(
        "lora", {"network_reg_dims": dims, "network_reg_lrs": lrs}
    )

    merged = load_method_preset(
        "lora",
        "default",
        configs_dir=str(overlays.parent.parent),
        methods_subdir="gui-methods",
    )
    assert merged["network_reg_dims"] == dims
    assert merged["network_reg_lrs"] == lrs

    args = argparse.Namespace(**merged)
    kwargs = resolve_network_kwargs(args)
    assert kwargs["network_reg_dims"] == dims
    assert kwargs["network_reg_lrs"] == lrs

    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=16,
        network_alpha=1.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    assert cfg.reg_dims == {r"blocks\.0\..*": 8}
    assert cfg.reg_lrs == {r"blocks\.[01]\..*": 1e-4}
