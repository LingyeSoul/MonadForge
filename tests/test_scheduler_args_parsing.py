import ast
import argparse
from unittest.mock import MagicMock

import pytest


def _make_args(lr_scheduler="constant_with_warmup", lr_scheduler_args=None):
    args = argparse.Namespace(
        lr_scheduler=lr_scheduler,
        lr_scheduler_type=None,
        lr_scheduler_args=lr_scheduler_args,
        lr_warmup_steps=0.0,
        max_train_steps=1000,
    )
    return args


def test_scheduler_args_value_with_equals():
    """lr_scheduler_args values containing '=' should parse correctly."""
    torch = pytest.importorskip("torch")
    from library.training.schedulers import get_scheduler_fix

    optimizer = MagicMock()
    args = _make_args(
        lr_scheduler="constant",
        lr_scheduler_args=["scale=1+1=2"],
    )
    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)
    assert scheduler is not None


def test_parse_args_value_with_equals_standalone():
    """Verify that split('=', 1) correctly parses values containing '='."""
    test_cases = [
        ("scale=1+1=2", "scale", "1+1=2"),
        ("key=val=ue", "key", "val=ue"),
        ("simple=value", "simple", "value"),
    ]
    for arg, expected_key, expected_value in test_cases:
        key, value = arg.split("=", 1)
        assert key == expected_key, f"key mismatch for {arg!r}"
        assert value == expected_value, f"value mismatch for {arg!r}"


def test_parse_args_value_with_equals_old_behavior_fails():
    """Demonstrate that the old split('=') would fail with multiple '='."""
    arg = "scale=1+1=2"
    with pytest.raises(ValueError):
        _key, _value = arg.split("=")


def test_adafactor_scheduler_missing_colon_raises():
    """--lr_scheduler 'adafactor' without ':lr' should raise ValueError, not IndexError."""
    pytest.importorskip("torch")
    from library.training.schedulers import get_scheduler_fix

    optimizer = MagicMock()
    args = _make_args(lr_scheduler="adafactor")
    with pytest.raises(ValueError, match="adafactor"):
        get_scheduler_fix(args, optimizer, num_processes=1)


def test_adafactor_scheduler_wrong_optimizer_raises():
    """adafactor scheduler with non-Adafactor optimizer should raise ValueError."""
    pytest.importorskip("torch")
    from library.training.schedulers import get_scheduler_fix

    optimizer = MagicMock()  # not an Adafactor instance
    args = _make_args(lr_scheduler="adafactor:0.001")
    with pytest.raises(ValueError, match="Adafactor"):
        get_scheduler_fix(args, optimizer, num_processes=1)
