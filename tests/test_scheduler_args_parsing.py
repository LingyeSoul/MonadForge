import argparse
from unittest.mock import MagicMock

import pytest


def _make_args(lr_scheduler="constant_with_warmup", lr_scheduler_args=None):
    args = argparse.Namespace(
        optimizer_type="AdamW",
        lr_scheduler=lr_scheduler,
        lr_scheduler_type=None,
        lr_scheduler_args=lr_scheduler_args,
        lr_warmup_steps=0.0,
        max_train_steps=1000,
    )
    return args


def test_scheduler_args_value_with_equals():
    """lr_scheduler_args values containing '=' should parse correctly."""
    import ast

    lr_scheduler_args = ['foo="a=b"']
    lr_scheduler_kwargs = {}
    for arg in lr_scheduler_args:
        key, value = arg.split("=", 1)
        value = ast.literal_eval(value)
        lr_scheduler_kwargs[key] = value
    assert lr_scheduler_kwargs == {"foo": "a=b"}


def test_parse_args_value_with_equals_standalone():
    """Verify that split('=', 1) correctly parses values containing '='."""
    test_cases = [
        ('foo="a=b"', "foo", '"a=b"'),
        ("key=val=ue", "key", "val=ue"),
        ("simple=value", "simple", "value"),
    ]
    for arg, expected_key, expected_value in test_cases:
        key, value = arg.split("=", 1)
        assert key == expected_key, f"key mismatch for {arg!r}"
        assert value == expected_value, f"value mismatch for {arg!r}"


def test_parse_args_value_with_equals_old_behavior_fails():
    """Demonstrate that the old split('=') would fail with multiple '='."""
    arg = 'foo="a=b"'
    with pytest.raises(ValueError):
        _key, _value = arg.split("=")


def test_scheduler_fix_parses_value_with_equals():
    """End-to-end: a value containing '=' must round-trip through get_scheduler_fix.

    Regression guard for the ``split("=", 1)`` parse loop inside
    ``get_scheduler_fix`` (schedulers.py). The ``constant`` scheduler accepts no
    extra kwargs, so forwarding ``foo="a=b"`` raises ``TypeError`` mentioning the
    *parsed* key ``foo`` — which only happens if the value survived the parse
    intact (``split("=")`` would have raised ``ValueError`` before reaching the
    scheduler). The isolated ``test_scheduler_args_value_with_equals`` above
    covers the algorithm in isolation; this test pins the integration path.
    """
    pytest.importorskip("torch")
    from library.training.schedulers import get_scheduler_fix

    optimizer = MagicMock()
    args = _make_args(lr_scheduler="constant", lr_scheduler_args=['foo="a=b"'])
    with pytest.raises(TypeError, match="foo"):
        get_scheduler_fix(args, optimizer, num_processes=1)


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
