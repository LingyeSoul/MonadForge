from __future__ import annotations

import argparse

from library.config.io import _mark_training_budget_provenance, _render_merged_toml


def test_cli_help_describes_explicit_steps_as_authoritative():
    from library.config.cli_args import add_training_arguments

    parser = argparse.ArgumentParser()
    add_training_arguments(parser, support_dreambooth=True)
    action = next(
        action for action in parser._actions if action.dest == "max_train_epochs"
    )
    assert "only when max_train_steps was not explicitly configured" in action.help


def _args(**overrides):
    values = {
        "max_train_steps": 1600,
        "max_train_epochs": None,
        "gradient_accumulation_steps": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_epoch_only_config_does_not_treat_argparse_default_as_explicit():
    import train

    args = _args(max_train_epochs=4)
    _mark_training_budget_provenance(args, {"max_train_epochs": "configs/method.toml"}, [])
    train._finalize_training_budget(args, dataloader_length=5, num_processes=1)
    assert args.max_train_steps == 20
    assert args.training_budget_source == "max_train_epochs"


def test_explicit_step_wins_over_conflicting_epoch_config():
    import train

    args = _args(max_train_steps=7, max_train_epochs=4)
    _mark_training_budget_provenance(
        args,
        {
            "max_train_steps": "configs/job.toml",
            "max_train_epochs": "configs/job.toml",
        },
        [],
    )
    train._finalize_training_budget(args, dataloader_length=5, num_processes=1)
    assert args.max_train_steps == 7
    assert args.training_budget_source == "max_train_steps"
    assert args.training_budget_origin == "configs/job.toml"


def test_cli_step_wins_over_epoch_config():
    import train

    args = _args(max_train_steps=3, max_train_epochs=9)
    _mark_training_budget_provenance(
        args, {"max_train_epochs": "configs/job.toml"}, ["--max_train_steps=3"]
    )
    train._finalize_training_budget(args, dataloader_length=5, num_processes=1)
    assert args.max_train_steps == 3
    assert args.training_budget_source == "max_train_steps"
    assert args.training_budget_origin == "CLI"


def test_snapshot_header_records_resolved_budget_source_and_steps():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_steps", type=int, default=1600)
    args = argparse.Namespace(
        method="lora",
        preset="default",
        max_train_steps=7,
        effective_max_train_steps=7,
        training_budget_source="max_train_steps",
        training_budget_origin="CLI",
    )

    rendered = _render_merged_toml(args, parser, {"max_train_steps": "CLI"})

    assert (
        "# Training budget: source=max_train_steps origin=CLI "
        "effective_max_train_steps=7" in rendered
    )
