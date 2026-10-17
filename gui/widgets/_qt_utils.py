"""Tiny Qt-only helpers shared by sibling widget modules.

Kept separate from ``fields.py`` so it has no ``gui.widgets`` dependents —
``fields.py`` and ``sample_prompts.py`` both use it, and neither may import
the other (see ``gui/widgets/__init__.py``), so the shared bit has to sit
below both.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


def _no_wheel(w: QWidget) -> QWidget:
    """Stop a hovered combo/spin from changing value (and stealing focus) on
    mouse-wheel scroll — otherwise scrolling the form silently edits whichever
    dropdown the cursor passes over. The widget still works via click + keys."""
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.wheelEvent = lambda e: e.ignore()
    return w
