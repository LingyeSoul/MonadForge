"""Line-panel layout: the Anima-aligned structure of the MOSS / Qwen forms.

Both lines share ``gui.lines.window``, so every assertion runs against both specs:
a grouped form (Paths / Training / … boxes), the Advanced fold, the queue split
buttons, the extra-args escape hatch and the log copy button.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    import gui.core.paths as paths

    path = tmp_path / "gui_settings.json"
    monkeypatch.setattr(paths, "GUI_SETTINGS_FILE", path)
    return path


def _group_title(widget):
    """Title of the nearest ``QGroupBox`` ancestor, or ``None``."""
    from PySide6.QtWidgets import QGroupBox

    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent.title()
        parent = parent.parentWidget()
    return None


@pytest.mark.parametrize("spec_module", ["gui.moss.app", "gui.qwen21.app"])
def test_line_form_is_grouped_like_anima(qapp, settings_file, spec_module):
    from gui.lines import window

    spec = importlib.import_module(spec_module).SPEC
    panel = window.build_panel(spec, "en")
    try:
        form = panel.cache_form
        # A dataset path is a Paths field, drawn inside a Paths group box rather
        # than a flat form row.
        assert _group_title(form.widgets["src"]) == "Paths"
        # The model folder is advanced: it lives in the collapsed Advanced fold.
        advanced = form.widgets["model_dir"]
        assert _group_title(advanced) == "Paths"
        assert not advanced.isVisibleTo(form)
        # An everyday cache knob stays outside the fold.
        assert form.widgets["overwrite"].isVisibleTo(form)
        # Opening the fold brings the advanced row back.
        advanced_box = advanced
        while (
            advanced_box is not None
            and not getattr(advanced_box, "isCheckable", lambda: False)()
        ):
            advanced_box = advanced_box.parentWidget()
        assert advanced_box is not None
        advanced_box.setChecked(True)
        assert advanced.isVisibleTo(form)
    finally:
        panel.cleanup_subprocess()


@pytest.mark.parametrize("spec_module", ["gui.moss.app", "gui.qwen21.app"])
def test_line_panel_has_queue_extra_args_and_log_copy(qapp, settings_file, spec_module):
    from PySide6.QtWidgets import QToolButton

    from gui.lines import window

    spec = importlib.import_module(spec_module).SPEC
    panel = window.build_panel(spec, "en")
    try:
        # Run buttons are split buttons: main action + an "enqueue" menu.
        for button in (panel.btn_cache, panel.btn_train):
            assert isinstance(button, QToolButton)
            assert button.menu() is not None
            assert button.menu().actions()
        # One extra-args escape hatch per request form, empty by default.
        assert set(panel._extra) == {"cache", "train", "test"}
        assert all(box.text() == "" for box in panel._extra.values())
        # The log has the same floating copy button as the Anima pages.
        assert isinstance(panel._log_copy_btn, QToolButton)
        assert panel._log_copy_btn.text()
    finally:
        panel.cleanup_subprocess()
