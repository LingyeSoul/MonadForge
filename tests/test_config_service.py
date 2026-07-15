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

import toml

from library.anima.training import add_anima_training_arguments
from library.config.io import load_method_preset
from library.inference.args import build_parser as build_inference_parser
from webui.services import config_service
from webui.services.config_service import _SELECT_OPTIONS, validate_config


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
