"""MethodsTab — one Method dropdown over every trainable method.

The experimental section used to spread trainable methods across several tabs:
a ConfigTab for the LoRA-family research variants (FeRA / ChimeraHydra) plus
dedicated SPD and Turbo tabs. They split apart only because the editors differ
under the hood — the LoRA family edits a *flat* ``gui-methods/<variant>.toml``
and submits a ``train.py --method`` training job, while SPD / Turbo edit a
*sectioned* ``configs/methods/{spd,turbo}.toml`` and submit a bespoke
``tasks.py exp-*`` distill command job (no ``train.py`` path).

This wrapper hides that split behind a single Method picker. A ``QStackedWidget``
holds the two editor kinds; selecting a flat method drives the embedded
ConfigTab (its own method picker is hidden so the two don't compete), while
selecting a distill method swaps in its editor. soft_tokens — a normal flat
``train.py --method`` method that was previously only reachable from the CLI —
rides the same ConfigTab page.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.tabs.config_tab import ConfigTab
from gui.tabs.distill_tab import SPDTrainTab, TurboTrainTab


class MethodsTab(QWidget):
    """Unified experimental method picker (flat-config + distill editors)."""

    # Flat methods routed through the embedded ConfigTab (train.py --method,
    # gui-methods/<variant>.toml). soft_tokens is a normal flat method;
    # ChimeraHydra is the LoRA-family research variant kept behind the
    # experimental gate (FeRA was retired from the picker — ChimeraHydra
    # superseded it; the fera.toml / network module still exist for the CLI).
    _FLAT_METHODS = ("chimera", "soft_tokens")

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Embedded editors. The distill tabs are LazyTabMixin — their TOML scan
        # is deferred to the first time the stack shows them (i.e. when picked).
        self._config = ConfigTab(methods=list(self._FLAT_METHODS))
        # The wrapper owns method selection, so hide ConfigTab's own method
        # picker; its inline variant picker stays (it switches the bound
        # gui-methods/<variant>.toml within the selected family).
        self._config._method_label.setVisible(False)
        self._config.method_combo.setVisible(False)
        self._spd = SPDTrainTab()
        self._turbo = TurboTrainTab()
        # Distill method key → editor widget.
        self._distill: dict[str, QWidget] = {"spd": self._spd, "turbo": self._turbo}

        # Method selector lives in a defined header band (padding + bottom
        # border) so it reads as an intentional selector strip instead of a
        # control jammed into the top-left corner above the editor's own row.
        header = QWidget()
        header.setObjectName("methodHeader")
        header.setStyleSheet(
            "#methodHeader { background:#323232; border-bottom:1px solid #454545; }"
        )
        top = QHBoxLayout(header)
        top.setContentsMargins(12, 8, 12, 8)
        top.setSpacing(8)
        # "Method" matches ConfigTab's own (hardcoded) picker label.
        method_lbl = QLabel("Method")
        method_lbl.setStyleSheet("font-weight:bold;")
        top.addWidget(method_lbl)
        self._combo = QComboBox()
        self._combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._combo.setMinimumContentsLength(
            max(len(m) for m in (*self._FLAT_METHODS, *self._distill))
        )
        self._combo.addItems([*self._FLAT_METHODS, *self._distill])
        self._combo.currentTextChanged.connect(self._on_method)
        top.addWidget(self._combo)
        top.addStretch()
        lay.addWidget(header)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._config)  # one page serves all flat methods
        for w in self._distill.values():
            self._stack.addWidget(w)
        lay.addWidget(self._stack)

        # Sync the stack + embedded ConfigTab to the initial selection.
        self._on_method(self._combo.currentText())

    def _on_method(self, method: str) -> None:
        editor = self._distill.get(method)
        if editor is not None:
            self._stack.setCurrentWidget(editor)
            return
        self._stack.setCurrentWidget(self._config)
        # Drive the embedded ConfigTab to the chosen flat method (its hidden
        # combo still emits currentTextChanged → _reload).
        if self._config.method_combo.currentText() != method:
            self._config.method_combo.setCurrentText(method)

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook — forward to every embedded editor (each leaves its
        detached daemon job alive; this only stops the observers)."""
        for w in (self._config, self._spd, self._turbo):
            cleanup = getattr(w, "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()
