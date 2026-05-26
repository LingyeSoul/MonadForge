"""Tests for ``ensure_text_strategies`` — the lazy installer for the global
tokenize/encode strategy singletons (suggestions.md #8).

The singletons are process-global with set-once semantics, so each test saves
and restores them around its mutation.

Covers:

* no-op when both singletons are already installed (composes with the CLI path)
* clear ``ValueError`` when a strategy is missing and no path can build it
* the encode-only branch installs ``AnimaTextEncodingStrategy`` without a path
  (it needs no model) while leaving an already-set tokenize strategy untouched
"""

from __future__ import annotations

import pytest

from library.anima import text_strategies
from library.inference.text import ensure_text_strategies


@pytest.fixture(autouse=True)
def restore_singletons():
    """Snapshot and restore the process-global strategy singletons."""
    tok = text_strategies.TokenizeStrategy._strategy
    enc = text_strategies.TextEncodingStrategy._strategy
    try:
        yield
    finally:
        text_strategies.TokenizeStrategy._strategy = tok
        text_strategies.TextEncodingStrategy._strategy = enc


def test_noop_when_both_already_set():
    tok_sentinel = object()
    enc_sentinel = object()
    text_strategies.TokenizeStrategy._strategy = tok_sentinel
    text_strategies.TextEncodingStrategy._strategy = enc_sentinel

    # No path needed, must not raise, must not replace the installed strategies.
    ensure_text_strategies(None)

    assert text_strategies.TokenizeStrategy._strategy is tok_sentinel
    assert text_strategies.TextEncodingStrategy._strategy is enc_sentinel


def test_raises_when_tokenize_missing_and_no_path():
    text_strategies.TokenizeStrategy._strategy = None
    text_strategies.TextEncodingStrategy._strategy = object()

    with pytest.raises(ValueError, match="no text-encoder path"):
        ensure_text_strategies(None)


def test_installs_encoding_strategy_without_path():
    # Tokenize already set → no path required; only the encode singleton is built.
    text_strategies.TokenizeStrategy._strategy = object()
    text_strategies.TextEncodingStrategy._strategy = None

    ensure_text_strategies(None)

    enc = text_strategies.TextEncodingStrategy.get_strategy()
    assert isinstance(enc, text_strategies.TextEncodingStrategy)
    assert type(enc).__name__ == "AnimaTextEncodingStrategy"
