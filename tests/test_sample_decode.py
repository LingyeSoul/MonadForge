"""Training preview decode timing regression guards."""

from __future__ import annotations

import argparse

import pytest

from library.anima.training import _should_decode_inline


@pytest.mark.parametrize(
    ("explicit", "blocks_to_swap", "expected"),
    [
        (None, 0, False),
        (None, 20, False),
        ("auto", 0, False),
        ("auto", 20, False),
        (True, 20, True),
        (False, 0, False),
        ("true", 20, True),
        ("false", 0, False),
    ],
)
def test_sample_decode_inline_tri_state(explicit, blocks_to_swap, expected):
    args = argparse.Namespace(
        sample_decode_inline=explicit,
        blocks_to_swap=blocks_to_swap,
    )
    assert _should_decode_inline(args) is expected
