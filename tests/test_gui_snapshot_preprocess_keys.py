"""Regression test: the GUI training snapshot must not carry preprocess-only keys.

Bug: ``_load_base()`` folds ``configs/preprocess.toml`` into the Config-tab
baseline so preprocess-owned fields surface in the form. That baseline also fed
``_queue_config_snapshot``, so the submitted training config inherited
preprocess knobs. Most were harmless ("unknown key" warnings), but
``caption_tag_dropout_rate`` collides with a real ``train.py`` argparse arg whose
meaning is *live dataloader tag dropout* — the dataset blueprint generator
pushed it onto every subset and tripped ``assert_extra_args``:

    AssertionError: when caching Text Encoder output, token_warmup_step or
    caption_tag_dropout_rate cannot be used

The snapshot builder now strips the whole preprocess-only key set.
"""

from __future__ import annotations

import os


def _make_config_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from gui.tabs.config_tab import ConfigTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    return ConfigTab()


def test_snapshot_drops_preprocess_only_keys():
    from gui.tabs.preprocess_tab import _GUI_PREPROCESS_KEYS

    tab = _make_config_tab()
    try:
        snapshot = tab._queue_config_snapshot("lora")
    finally:
        tab.deleteLater()

    # The specific key that crashed training must be gone.
    assert "caption_tag_dropout_rate" not in snapshot
    assert "caption_tag_randomize_rate" not in snapshot

    # And no preprocess-owned knob should ride into the training config at all.
    leaked = _GUI_PREPROCESS_KEYS & snapshot.keys()
    assert not leaked, f"preprocess-only keys leaked into training snapshot: {leaked}"

    # Sanity: a genuine training knob still survives (caption_dropout_rate is a
    # legit train arg — it's TE-cache-compatible via cache_supports_dropout).
    assert "caption_dropout_rate" in snapshot
