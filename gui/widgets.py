"""Reusable Qt widgets + the config-form field factory.

Holds the lazy-tab mixin, the multi-scale ``target_res`` checkbox row, the
``_widget``/``_read`` pair that maps a config value to/from an editor widget,
and the aspect-preserving image label. Pulled out of the package root so the
widget code lives apart from the Qt-free config logic.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
    QWidgetAction,
    QVBoxLayout,
)

from gui.i18n import t
from gui.theme import tok

# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]


class LazyTabMixin:
    """Defer a tab's first expensive scan until the tab is actually opened.

    Several tabs walk dataset/checkpoint directories (and the Merge tab reads
    safetensors keys) during construction. Doing that for *every* tab up front
    is what made the window slow to appear, even though only the first tab is
    visible at launch. Mixing this in lets construction stay cheap: the heavy
    work runs on the first ``showEvent`` — i.e. when the user selects the tab —
    and exactly once thereafter. Subclasses override ``_lazy_init``.

    Mix in BEFORE ``QWidget`` so ``super().showEvent`` resolves to Qt's.
    """

    _lazy_done = False

    def showEvent(self, event):  # noqa: N802 — Qt event handler name
        super().showEvent(event)
        if not self._lazy_done:
            self._lazy_done = True
            self._lazy_init()

    def _lazy_init(self) -> None:
        """Run the tab's first directory scan / classification. Override."""


@functools.cache
def _target_res_tiers() -> tuple[tuple[int, ...], dict[int, int]]:
    """``(allowed tiers, {edge: max token count})`` sourced from
    ``library.datasets.buckets`` (the single source of truth) instead of a
    hardcoded mirror. Lazy + cached so importing ``widgets`` stays cheap; the
    bucket module is torch-free so this never drags the training stack in.

    A tier is "dangerous" (extra compiled block graph + VRAM) when its per-image
    token count exceeds the canonical 1024 tier — which reproduces the previous
    ``{1280: 6300, 1536: 8640}`` flag set, but recomputed from the tables.
    """
    from library.datasets.buckets import (
        ALLOWED_TARGET_RES,
        CONSTANT_TOKEN_BUCKETS_BY_EDGE,
    )

    max_tok = {
        edge: max((w // 16) * (h // 16) for w, h in buckets)
        for edge, buckets in CONSTANT_TOKEN_BUCKETS_BY_EDGE.items()
    }
    canonical = max_tok.get(1024, 4200)
    danger = {edge: tok for edge, tok in max_tok.items() if tok > canonical}
    return tuple(ALLOWED_TARGET_RES), danger


@functools.cache
def _target_res_buckets() -> dict[int, tuple[tuple[int, int], ...]]:
    from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS_BY_EDGE

    return {
        edge: tuple(
            sorted(
                CONSTANT_TOKEN_BUCKETS_BY_EDGE[edge], key=lambda size: size[0] / size[1]
            )
        )
        for edge in sorted(CONSTANT_TOKEN_BUCKETS_BY_EDGE)
    }


class _BucketMenuPanel(QWidget):
    """Keep a bucket popup open when the user misses a checkbox row slightly."""

    def mousePressEvent(self, event) -> None:  # noqa: N802
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        event.accept()


class _TargetResWidget(QWidget):
    """Horizontal row of tier checkboxes for the multi-scale ``target_res`` knob.

    Reads/writes a list of edge ints (e.g. ``[1024, 1536]``). Never returns an
    empty list — unchecking everything falls back to ``[1024]`` (the legacy
    single ~1MP tier) so preprocess/train always have a valid tier.

    The optional bucket checklist is a narrow preprocess-only filter. Empty
    selection means "all buckets in the selected tiers".

    The 1280/1536 tiers are visually flagged as "dangerous" (high token count
    + extra compile graph / VRAM) via colour + an i18n tooltip.
    """

    changed = Signal()

    def __init__(self, selected) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        sel = {int(e) for e in selected} if selected else set()
        tiers, danger = _target_res_tiers()
        self._boxes: dict[int, QCheckBox] = {}
        self._bucket_boxes: dict[tuple[int, int], QCheckBox] = {}
        self._explicit_bucket_selection = False
        for edge in tiers:
            edge_box = QWidget()
            edge_lay = QHBoxLayout(edge_box)
            edge_lay.setContentsMargins(0, 0, 0, 0)
            edge_lay.setSpacing(2)
            cb = QCheckBox()
            cb.setChecked(edge in sel)
            cb.toggled.connect(self._on_tier_toggled)
            if edge in danger:
                cb.setStyleSheet("QCheckBox { color: #d9822b; font-weight: bold; }")
                cb.setToolTip(
                    t(
                        "target_res_danger_tooltip",
                        edge=edge,
                        tokens=danger[edge],
                    )
                )
            edge_lay.addWidget(cb)
            toggle = QToolButton()
            toggle.setText(str(edge))
            toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            toggle.setPopupMode(QToolButton.InstantPopup)
            popup = QMenu(toggle)
            panel = _BucketMenuPanel(popup)
            panel_lay = QVBoxLayout(panel)
            panel_lay.setContentsMargins(8, 4, 8, 4)
            panel_lay.setSpacing(0)
            for width, height in _target_res_buckets().get(edge, ()):
                ratio = width / height
                cb_bucket = QCheckBox(f"{width}x{height} ({ratio:.2f})")
                cb_bucket.setMinimumHeight(22)
                cb_bucket.setToolTip(t("target_res_bucket_tooltip", edge=edge))
                cb_bucket.toggled.connect(self._on_bucket_toggled)
                panel_lay.addWidget(cb_bucket)
                self._bucket_boxes[(width, height)] = cb_bucket
            action = QWidgetAction(popup)
            action.setDefaultWidget(panel)
            popup.addAction(action)
            toggle.setMenu(popup)
            if edge in danger:
                toggle.setStyleSheet(
                    "QToolButton { color: #d9822b; font-weight: bold; }"
                )
                toggle.setToolTip(
                    t(
                        "target_res_danger_tooltip",
                        edge=edge,
                        tokens=danger[edge],
                    )
                )
            edge_lay.addWidget(toggle)
            lay.addWidget(edge_box)
            self._boxes[edge] = cb
        lay.addStretch(1)
        self.set_bucket_resos([])

    def value(self) -> list[int]:
        out = [e for e, cb in self._boxes.items() if cb.isChecked()]
        return out or [1024]

    def bucket_resos(self) -> list[str]:
        if not self._explicit_bucket_selection:
            return []
        selected_edges = set(self.value())
        allowed = set()
        for edge in selected_edges:
            allowed.update(_target_res_buckets().get(edge, ()))
        out = [
            f"{width}x{height}"
            for (width, height), cb in self._bucket_boxes.items()
            if cb.isChecked() and (width, height) in allowed
        ]
        return out

    def set_bucket_resos(self, values) -> None:
        selected = set()
        if isinstance(values, str):
            selected = {
                part.strip().lower().replace("×", "x")
                for part in values.split(",")
                if part.strip()
            }
        elif values:
            selected = {
                str(value).strip().lower().replace("×", "x") for value in values
            }
        self._explicit_bucket_selection = bool(selected)
        for (width, height), cb in self._bucket_boxes.items():
            cb.blockSignals(True)
            cb.setChecked((not selected) or f"{width}x{height}" in selected)
            cb.blockSignals(False)
        self._refresh_bucket_enabled()

    def _refresh_bucket_enabled(self) -> None:
        selected_edges = set(self.value())
        allowed = set()
        for edge in selected_edges:
            allowed.update(_target_res_buckets().get(edge, ()))
        for bucket, cb in self._bucket_boxes.items():
            cb.setEnabled(bucket in allowed)

    def refresh_bucket_enabled(self) -> None:
        self._refresh_bucket_enabled()

    def _on_tier_toggled(self) -> None:
        self._refresh_bucket_enabled()
        self.changed.emit()

    def _on_bucket_toggled(self) -> None:
        checked = [cb.isChecked() for cb in self._bucket_boxes.values()]
        if not any(checked):
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
            return
        self._explicit_bucket_selection = not all(checked)
        self.changed.emit()


class _SamplePromptRow(QFrame):
    """One editable sample prompt while preserving train.py's one-line syntax."""

    changed = Signal()

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ border:1px solid {tok('border_dim')}; border-radius:4px; }} "
            "QLabel, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, "
            "QCheckBox { border:0; }"
        )

        lay = QGridLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(6)

        self.select_box = QCheckBox(t("sample_prompt_select"))
        lay.addWidget(self.select_box, 0, 0, 1, 2)

        prompt_stack = QVBoxLayout()
        prompt_stack.setContentsMargins(0, 0, 0, 0)
        prompt_stack.setSpacing(4)
        lay.addLayout(prompt_stack, 1, 0)

        prompt_label = QLabel(t("sample_prompt_col_prompt"))
        prompt_stack.addWidget(prompt_label)
        self.prompt_edit = QPlainTextEdit(str(data.get("prompt", "")))
        self.prompt_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.prompt_edit.setMinimumHeight(118)
        self.prompt_edit.setPlaceholderText(t("sample_prompt_prompt_placeholder"))
        self.prompt_edit.textChanged.connect(self.changed.emit)
        prompt_stack.addWidget(self.prompt_edit)

        negative_label = QLabel(t("sample_prompt_col_negative"))
        negative_label.setToolTip(t("sample_prompt_tip_negative"))
        prompt_stack.addWidget(negative_label)
        self.negative = QPlainTextEdit(str(data.get("negative", "")))
        self.negative.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.negative.setMinimumHeight(72)
        self.negative.setMaximumHeight(96)
        self.negative.setPlaceholderText(t("sample_prompt_default_negative"))
        self.negative.setToolTip(t("sample_prompt_tip_negative"))
        self.negative.textChanged.connect(self.changed.emit)
        prompt_stack.addWidget(self.negative)

        opts = QGridLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        opts.setHorizontalSpacing(8)
        opts.setVerticalSpacing(4)
        lay.addLayout(opts, 1, 1)

        self.width = self._int_spin(
            data.get("width", 0),
            default_label=t("sample_prompt_default_width"),
            tooltip=t("sample_prompt_tip_width"),
        )
        self.height = self._int_spin(
            data.get("height", 0),
            default_label=t("sample_prompt_default_height"),
            tooltip=t("sample_prompt_tip_height"),
        )
        self.steps = self._int_spin(
            data.get("steps", 0),
            maximum=1000,
            default_label=t("sample_prompt_default_steps"),
            tooltip=t("sample_prompt_tip_steps"),
        )
        self.seed = self._seed_spin(data.get("seed"))
        self.scale = self._float_spin(
            data.get("scale", 0.0),
            default_label=t("sample_prompt_default_cfg"),
            tooltip=t("sample_prompt_tip_cfg"),
        )
        self.guidance = self._float_spin(
            data.get("guidance", 0.0),
            default_label=t("sample_prompt_default_guidance"),
            tooltip=t("sample_prompt_tip_guidance"),
        )
        self.flow_shift = self._float_spin(
            data.get("flow_shift", 0.0),
            default_label=t("sample_prompt_default_shift"),
            tooltip=t("sample_prompt_tip_shift"),
        )
        self.extra = QLineEdit(str(data.get("extra", "")))
        self.extra.setToolTip(t("sample_prompt_tip_extra"))
        self.extra.textChanged.connect(lambda *_: self.changed.emit())

        for row, (label_text, widget, tooltip) in enumerate(
            (
                (
                    t("sample_prompt_col_width"),
                    self.width,
                    t("sample_prompt_tip_width"),
                ),
                (
                    t("sample_prompt_col_height"),
                    self.height,
                    t("sample_prompt_tip_height"),
                ),
                (
                    t("sample_prompt_col_steps"),
                    self.steps,
                    t("sample_prompt_tip_steps"),
                ),
                (t("sample_prompt_col_seed"), self.seed, t("sample_prompt_tip_seed")),
                (t("sample_prompt_col_cfg"), self.scale, t("sample_prompt_tip_cfg")),
                (
                    t("sample_prompt_col_guidance"),
                    self.guidance,
                    t("sample_prompt_tip_guidance"),
                ),
                (
                    t("sample_prompt_col_shift"),
                    self.flow_shift,
                    t("sample_prompt_tip_shift"),
                ),
                (
                    t("sample_prompt_col_extra"),
                    self.extra,
                    t("sample_prompt_tip_extra"),
                ),
            )
        ):
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            opts.addWidget(label, row, 0)
            opts.addWidget(widget, row, 1)

        lay.setColumnStretch(0, 4)
        lay.setColumnStretch(1, 2)
        self.setMinimumHeight(300)

    def _int_spin(
        self,
        value: int = 0,
        *,
        maximum: int = 8192,
        default_label: str,
        tooltip: str = "",
        unset: int = 0,
    ) -> QSpinBox:
        w = QSpinBox()
        w.setRange(unset, maximum)
        w.setSpecialValueText(default_label)
        w.setValue(int(value or unset))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        if tooltip:
            w.setToolTip(tooltip)
        return _no_wheel(w)

    def _seed_spin(self, value: int | None = None) -> QSpinBox:
        w = QSpinBox()
        w.setRange(-1, 2_147_483_647)
        w.setSpecialValueText(t("sample_prompt_default_seed"))
        w.setValue(-1 if value is None else int(value))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        w.setToolTip(t("sample_prompt_tip_seed"))
        return _no_wheel(w)

    def _float_spin(
        self, value: float = 0.0, *, default_label: str, tooltip: str = ""
    ) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(0.0, 100.0)
        w.setDecimals(2)
        w.setSingleStep(0.5)
        w.setSpecialValueText(default_label)
        w.setValue(float(value or 0.0))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        if tooltip:
            w.setToolTip(tooltip)
        return _no_wheel(w)

    @staticmethod
    def _single_line(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def value(self) -> str | None:
        prompt = self._single_line(self.prompt_edit.toPlainText())
        if not prompt:
            return None
        parts = [prompt]
        for widget, flag in (
            (self.width, "w"),
            (self.height, "h"),
            (self.steps, "s"),
        ):
            val = widget.value()
            if val > 0:
                parts.append(f"--{flag} {val}")
        seed = self.seed.value()
        if seed >= 0:
            parts.append(f"--d {seed}")
        for widget, flag in (
            (self.scale, "l"),
            (self.guidance, "g"),
            (self.flow_shift, "fs"),
        ):
            val = widget.value()
            if val > 0:
                parts.append(f"--{flag} {val:g}")
        negative = self._single_line(self.negative.toPlainText())
        if negative:
            parts.append(f"--n {negative}")
        extra = self._single_line(self.extra.text())
        if extra:
            parts.append(extra if extra.startswith("--") else "--" + extra)
        return " ".join(parts)


class _SamplePromptsWidget(QWidget):
    """Structured editor for train.py's one-line sample prompt syntax."""

    changed = Signal()

    _COLLAPSED_HEIGHT = 300
    _EXPANDED_HEIGHT = 650

    def __init__(self, prompts, fill: bool = False) -> None:
        super().__init__()
        # ``fill`` = expand to fill the host (used inside the editor dialog,
        # which is resizable). When false the widget self-clamps to a fixed
        # height and offers an expand/collapse toggle for inline embedding.
        self._fill = fill
        self._expanded = False
        self._rows: list[_SamplePromptRow] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        hint = QLabel(t("sample_prompt_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{tok('text_dim')};")
        lay.addWidget(hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding if fill else QSizePolicy.Fixed,
        )
        self.scroll.setStyleSheet(
            f"""
            QScrollBar:vertical {{
                background: {tok("scroll_bg")};
                width: 14px;
                margin: 0;
            }}
            QScrollBar:horizontal {{
                background: {tok("scroll_bg")};
                height: 14px;
                margin: 0;
            }}
            QScrollBar::handle {{
                background: {tok("scroll_handle")};
                border-radius: 5px;
                min-height: 24px;
                min-width: 24px;
            }}
            QScrollBar::handle:hover {{
                background: {tok("scroll_handle_hover")};
            }}
            QScrollBar::add-line,
            QScrollBar::sub-line {{
                width: 0;
                height: 0;
            }}
            """
        )
        self._rows_widget = QWidget()
        self._row_layout = QVBoxLayout(self._rows_widget)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(8)
        self._row_layout.addStretch(1)
        self.scroll.setWidget(self._rows_widget)
        lay.addWidget(self.scroll)

        row_lay = QHBoxLayout()
        row_lay.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton(t("sample_prompt_add"))
        add_btn.clicked.connect(lambda: self._add_row({}))
        row_lay.addWidget(add_btn)
        select_all_btn = QPushButton(t("sample_prompt_select_all"))
        select_all_btn.clicked.connect(self._select_all)
        row_lay.addWidget(select_all_btn)
        remove_btn = QPushButton(t("sample_prompt_remove"))
        remove_btn.clicked.connect(self._remove_selected)
        row_lay.addWidget(remove_btn)
        self.expand_btn: QPushButton | None = None
        if not fill:
            self.expand_btn = QPushButton(t("sample_prompt_expand"))
            self.expand_btn.clicked.connect(self._toggle_expanded)
            row_lay.addWidget(self.expand_btn)
        row_lay.addStretch(1)
        lay.addLayout(row_lay)

        rows = self._parse_prompts(prompts)
        if rows:
            for row in rows:
                self._add_row(row)
        else:
            self._add_row({})
        self._apply_height()

    def _apply_height(self) -> None:
        if self._fill or self.expand_btn is None:
            return
        height = self._EXPANDED_HEIGHT if self._expanded else self._COLLAPSED_HEIGHT
        self.scroll.setMinimumHeight(height)
        self.scroll.setMaximumHeight(height)
        self.expand_btn.setText(
            t("sample_prompt_collapse") if self._expanded else t("sample_prompt_expand")
        )

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._apply_height()

    @staticmethod
    def _parse_prompts(prompts) -> list[dict[str, Any]]:
        if isinstance(prompts, (list, tuple)):
            lines = [str(p).strip() for p in prompts]
        elif prompts is None:
            lines = []
        else:
            lines = [ln.strip() for ln in str(prompts).splitlines()]
        return [
            _SamplePromptsWidget._parse_line(ln)
            for ln in lines
            if ln and not ln.startswith("#")
        ]

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any]:
        parts = line.split(" --")
        out: dict[str, Any] = {"prompt": parts[0].strip()}
        extra: list[str] = []
        for part in parts[1:]:
            try:
                if m := re.match(r"w (\d+)$", part, re.IGNORECASE):
                    out["width"] = int(m.group(1))
                elif m := re.match(r"h (\d+)$", part, re.IGNORECASE):
                    out["height"] = int(m.group(1))
                elif m := re.match(r"s (\d+)$", part, re.IGNORECASE):
                    out["steps"] = int(m.group(1))
                elif m := re.match(r"d (\d+)$", part, re.IGNORECASE):
                    out["seed"] = int(m.group(1))
                elif m := re.match(r"l ([\d.]+)$", part, re.IGNORECASE):
                    out["scale"] = float(m.group(1))
                elif m := re.match(r"g ([\d.]+)$", part, re.IGNORECASE):
                    out["guidance"] = float(m.group(1))
                elif m := re.match(r"fs ([\d.]+)$", part, re.IGNORECASE):
                    out["flow_shift"] = float(m.group(1))
                elif m := re.match(r"n (.+)$", part, re.IGNORECASE):
                    out["negative"] = m.group(1).strip()
                else:
                    extra.append("--" + part.strip())
            except ValueError:
                extra.append("--" + part.strip())
        if extra:
            out["extra"] = " ".join(extra)
        return out

    def _add_row(self, data: dict[str, Any]) -> None:
        row = _SamplePromptRow(data)
        row.changed.connect(self.changed.emit)
        self._rows.append(row)
        self._row_layout.insertWidget(max(0, self._row_layout.count() - 1), row)
        self.changed.emit()

    def _select_all(self) -> None:
        for row in self._rows:
            row.select_box.setChecked(True)

    def _remove_selected(self) -> None:
        rows = [row for row in self._rows if row.select_box.isChecked()]
        if not rows:
            return
        answer = QMessageBox.question(
            self,
            t("sample_prompt_remove_confirm_title"),
            t("sample_prompt_remove_confirm_body", n=len(rows)),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        for row in rows:
            if row in self._rows:
                self._rows.remove(row)
                self._row_layout.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
        if not self._rows:
            self._add_row({})
        self.changed.emit()

    def value(self) -> list[str]:
        lines: list[str] = []
        for row in self._rows:
            value = row.value()
            if value:
                lines.append(value)
        return lines


def _normalize_prompt_lines(prompts) -> list[str]:
    """Coerce a stored sample_prompts value (list or multiline string) into a
    clean list of non-empty, non-comment one-line prompts."""
    if isinstance(prompts, (list, tuple)):
        lines = [str(p).strip() for p in prompts]
    elif prompts is None:
        lines = []
    else:
        lines = [ln.strip() for ln in str(prompts).splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


class SamplePromptsDialog(QDialog):
    """Popup hosting the structured sample-prompt editor at full size."""

    def __init__(self, prompts, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("sample_prompt_dialog_title"))
        self.setModal(True)
        self.resize(960, 720)
        lay = QVBoxLayout(self)
        self._editor = _SamplePromptsWidget(prompts, fill=True)
        lay.addWidget(self._editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def value(self) -> list[str]:
        return self._editor.value()


class _SamplePromptsLauncher(QWidget):
    """Compact field stand-in: a button + summary that opens the editor popup.

    Replaces the tall inline editor in the config form. Holds the current
    prompt list and round-trips it through :class:`SamplePromptsDialog`.
    """

    changed = Signal()

    def __init__(self, prompts) -> None:
        super().__init__()
        self._prompts = _normalize_prompt_lines(prompts)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._edit_btn = QPushButton(t("sample_prompt_edit_button"))
        self._edit_btn.clicked.connect(self._open_dialog)
        lay.addWidget(self._edit_btn)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color:{tok('text_dim')};")
        lay.addWidget(self._summary, 1)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n = len(self._prompts)
        if not n:
            self._summary.setText(t("sample_prompt_summary_none"))
            self._summary.setToolTip("")
            return
        first = self._prompts[0].split(" --", 1)[0].strip()
        if len(first) > 60:
            first = first[:57] + "…"
        self._summary.setText(t("sample_prompt_summary_count", n=n, first=first))
        self._summary.setToolTip("\n".join(self._prompts))

    def _open_dialog(self) -> None:
        dlg = SamplePromptsDialog(self._prompts, self)
        if dlg.exec():
            self._prompts = dlg.value()
            self._refresh_summary()
            self.changed.emit()

    def value(self) -> list[str]:
        return list(self._prompts)


def _no_wheel(w: QWidget) -> QWidget:
    """Stop a hovered combo/spin from changing value (and stealing focus) on
    mouse-wheel scroll — otherwise scrolling the form silently edits whichever
    dropdown the cursor passes over. The widget still works via click + keys."""
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.wheelEvent = lambda e: e.ignore()
    return w


def _widget(v: Any, key: str = "") -> QWidget:
    if key == "target_res":
        sel = v if isinstance(v, (list, tuple)) else ([v] if v else [1024])
        return _TargetResWidget(sel)
    if key == "sample_prompts":
        return _SamplePromptsLauncher(v)
    if key == "attn_mode":
        w = QComboBox()
        w.addItems(_ATTN_MODES)
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if key == "sample_decode_inline":
        # Tri-state: stored as the literal string "auto" / "true" / "false"
        # (all three are accepted by library.config.cli_args._optional_bool).
        # Must precede the bool branch below so a bool value gets the combo,
        # not a plain checkbox that can't express "auto".
        w = QComboBox()
        w.addItems(["auto", "true", "false"])
        if v is None:
            cur = "auto"
        elif isinstance(v, bool):
            cur = "true" if v else "false"
        else:
            cur = str(v).strip().lower()
            if cur in ("", "none"):
                cur = "auto"
        idx = w.findText(cur)
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if isinstance(v, bool):
        w = QCheckBox()
        w.setChecked(v)
        return w
    if isinstance(v, int):
        w = QSpinBox()
        # 10k default cap guards against typos; per-key overrides for fields that legitimately exceed it.
        if key == "min_pixels":
            w.setRange(0, 100_000_000)  # 100MP — covers any real image
        else:
            w.setRange(0, 10000)
        w.setValue(v)
        return _no_wheel(w)
    if isinstance(v, float):
        return QLineEdit(f"{v:g}")
    if isinstance(v, list):
        return QLineEdit(json.dumps(v))
    return QLineEdit(str(v))


def _read(w: QWidget, orig: Any = None) -> Any:
    if isinstance(w, _TargetResWidget):
        return w.value()
    if isinstance(w, _SamplePromptsLauncher):
        return w.value()
    if isinstance(w, QPlainTextEdit):
        # sample_prompts box → list of non-empty, non-comment lines.
        return [
            ln.strip()
            for ln in w.toPlainText().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    txt = w.text()
    if isinstance(orig, float):
        try:
            return float(txt)
        except ValueError:
            pass
    if isinstance(orig, list):
        try:
            return json.loads(txt)
        except (json.JSONDecodeError, ValueError):
            pass
    # Normalize pasted Windows backslashes — forward slashes work everywhere and dodge TOML escape errors (e.g. "C:\Users" → \U is invalid).
    if "\\" in txt:
        txt = txt.replace("\\", "/")
    return txt


class ClickableLabel(QLabel):
    """QLabel that emits `clicked` on left-click.

    The config-style tabs use it for field labels: clicking the label routes
    that field's help text into the explanation panel.
    """

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):  # noqa: N802 — Qt event handler name
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


def make_field_label(
    text: str,
    *,
    style: str | None = None,
    tooltip: str | None = None,
    on_click=None,
) -> ClickableLabel:
    """Build a form field's :class:`ClickableLabel` (style + tooltip + click wire).

    Centralises the label half of the per-row form boilerplate the tabs share;
    each caller still builds its own editor widget and supplies its own click
    handler (a zero-arg callable, usually a lambda closing over the field key).
    """
    lbl = ClickableLabel(text)
    if style:
        lbl.setStyleSheet(style)
    if tooltip:
        lbl.setToolTip(tooltip)
    if on_click is not None:
        lbl.clicked.connect(on_click)
    return lbl


def apply_variant(w: QWidget, variant: str | None) -> None:
    """(Re)assign an action-button style ``variant`` and repolish so it repaints.

    The saturated action colors live in the global stylesheet as
    ``[variant="…"]`` rules (see :data:`gui.theme.ACTION_COLORS`). Setting the
    dynamic property after a widget is realized does **not** restyle it until the
    style is re-polished — hence the unpolish/polish dance. This is what a state
    flip uses (idle→busy swaps the variant in one line instead of swapping a
    whole stylesheet string). Pass ``None``/"" to clear back to the neutral
    default button look.
    """
    w.setProperty("variant", variant or "")
    style = w.style()
    style.unpolish(w)
    style.polish(w)


def action_button(
    text: str = "",
    *,
    variant: str = "primary",
    tooltip: str | None = None,
    on_click=None,
    tool_button: bool = False,
) -> QPushButton | QToolButton:
    """Build a saturated action button styled by its global-stylesheet *variant*.

    Folds the repeated ``QPushButton(text)/QToolButton()`` + ``setStyleSheet``
    (hardcoded hex) + ``setToolTip`` + ``clicked.connect`` sequence into one
    call. The color lives in :data:`gui.theme.ACTION_COLORS` (one source of
    truth), not here, so it can't drift between call sites and flips correctly on
    a state change via :func:`apply_variant`.

    Variants: ``primary`` (green go), ``secondary`` (purple alt), ``info``
    (blue), ``danger`` (red stop), ``warning`` (orange/dirty), ``busy`` (gray).
    Pass ``tool_button=True`` for a :class:`QToolButton` (the action buttons that
    flip busy state are tool buttons); default is a :class:`QPushButton`.
    """
    btn: QPushButton | QToolButton = QToolButton() if tool_button else QPushButton()
    if text:
        btn.setText(text)
    apply_variant(btn, variant)
    if tooltip:
        btn.setToolTip(tooltip)
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn


def hint_label(text: str = "") -> QLabel:
    """A secondary/hint :class:`QLabel` in the theme's dim text color.

    Replaces the recurring ``QLabel(...)`` + ``setStyleSheet(f"color:{tok('text_dim')}")``
    pair scattered across the tabs. Reads the token at build time, so it follows
    the active theme (call sites are rebuilt on a theme switch)."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{tok('text_dim')};")
    return lbl


class DirtyTrackingMixin:
    """Save-button dirty-state tracking shared by the config-style tabs.

    A tab is *dirty* when its form has edits not yet written back to the config
    file on disk. ConfigTab (+ its EasyControl subclass), the distill editors,
    and PreprocessingTab all carried a near-identical copy of this wiring; this
    mixin holds the one copy.

    Host requirements:
      * ``self._dirty: bool`` initialised in ``__init__``;
      * a Save button exposed as ``self._save_btn`` (override
        :meth:`_dirty_save_button` if it lives under a different name);
      * optionally ``self._save_btn_dirty_variant`` — the action-button variant
        flipped on while dirty (defaults to ``"warning"``, the orange look);
      * optionally ``self._loading_variant`` — when truthy ``_mark_dirty`` is a
        no-op so a bulk reload doesn't trip the flag.

    Override :meth:`_update_save_button` when the button's text/tooltip differ
    from the default (PreprocessingTab does — its label is the localized
    "Save settings").

    Mix in BEFORE ``QWidget`` (and any other event-handling mixin).
    """

    _dirty: bool = False

    def _dirty_save_button(self) -> QPushButton | None:
        return getattr(self, "_save_btn", None)

    def _connect_dirty_signal(self, w: QWidget) -> None:
        """Wire a form widget's change signal to :meth:`_mark_dirty`.

        Connect AFTER the widget's value has been seeded, so the initial
        setValue/addItems calls don't trip the flag. The branch set is a
        superset of what any single tab needs — widget types a given tab never
        builds simply never match.
        """
        if isinstance(
            w, (_TargetResWidget, _SamplePromptsWidget, _SamplePromptsLauncher)
        ):
            w.changed.connect(self._mark_dirty)
        elif isinstance(w, QComboBox):
            w.currentTextChanged.connect(self._mark_dirty)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._mark_dirty)
        elif isinstance(w, QDoubleSpinBox):
            w.valueChanged.connect(self._mark_dirty)
        elif isinstance(w, QSpinBox):
            w.valueChanged.connect(self._mark_dirty)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(self._mark_dirty)
        elif isinstance(w, QPlainTextEdit):
            w.textChanged.connect(self._mark_dirty)

    def _mark_dirty(self, *_) -> None:
        if getattr(self, "_loading_variant", False) or self._dirty:
            return
        self._dirty = True
        self._update_save_button()

    def _clear_dirty(self) -> None:
        self._dirty = False
        self._update_save_button()

    # The dirty look is the centralized "warning" action variant (orange),
    # flipped on/off via apply_variant — no per-tab style strings.
    _save_btn_dirty_variant: str = "warning"

    def _update_save_button(self) -> None:
        btn = self._dirty_save_button()
        if btn is None:
            return
        if self._dirty:
            btn.setText(t("save") + " *")
            btn.setToolTip(t("save_dirty_tooltip"))
            apply_variant(btn, self._save_btn_dirty_variant)
        else:
            btn.setText(t("save"))
            btn.setToolTip("")
            apply_variant(btn, None)


class ScaledImageLabel(QLabel):
    """Aspect-preserving image view with Ctrl+wheel magnify.

    At zoom 1.0 the pixmap is fit to the widget (centered, KeepAspectRatio).
    Ctrl+scroll up/down zooms in/out **centered on the cursor**; once zoomed in,
    drag with the left button to pan. Rendering is done in ``paintEvent`` (rather
    than ``setPixmap``) so the zoomed image can overflow + be panned with the
    view clipping to the widget rect. Zoom resets whenever the source changes.
    """

    MIN_ZOOM = 1.0
    MAX_ZOOM = 8.0
    _ZOOM_STEP = 1.25

    def __init__(self):
        super().__init__()
        self._src: QPixmap | None = None
        self._zoom = 1.0
        # Top-left of the drawn image in widget coords; None = auto-centered.
        self._tl: QPointF | None = None
        self._pan_last: QPointF | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

    def set_source(self, pm: QPixmap):
        self._src = pm
        self._reset_zoom()
        self.update()

    def clear(self):
        self._src = None
        self._reset_zoom()
        super().clear()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._tl = None
        self._pan_last = None

    def _fit_scale(self) -> float:
        sw, sh = self._src.width(), self._src.height()
        if sw <= 0 or sh <= 0:
            return 1.0
        return min(self.width() / sw, self.height() / sh)

    def _display_size(self) -> QSize:
        scale = self._fit_scale() * self._zoom
        return QSize(
            max(1, round(self._src.width() * scale)),
            max(1, round(self._src.height() * scale)),
        )

    def _current_tl(self) -> QPointF:
        if self._tl is not None:
            return self._tl
        d = self._display_size()
        return QPointF((self.width() - d.width()) / 2, (self.height() - d.height()) / 2)

    def _clamp_tl(self) -> None:
        if self._tl is None:
            return
        d = self._display_size()
        x, y = self._tl.x(), self._tl.y()
        if d.width() <= self.width():
            x = (self.width() - d.width()) / 2
        else:
            x = min(0.0, max(float(self.width() - d.width()), x))
        if d.height() <= self.height():
            y = (self.height() - d.height()) / 2
        else:
            y = min(0.0, max(float(self.height() - d.height()), y))
        self._tl = QPointF(x, y)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp_tl()
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._src is None or self._src.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        tl = self._current_tl()
        d = self._display_size()
        painter.drawPixmap(
            int(round(tl.x())), int(round(tl.y())), d.width(), d.height(), self._src
        )

    def wheelEvent(self, ev):
        if not (ev.modifiers() & Qt.ControlModifier):
            super().wheelEvent(ev)
            return
        if self._src is None or self._src.isNull():
            return
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        new_zoom = self._zoom * (self._ZOOM_STEP if delta > 0 else 1 / self._ZOOM_STEP)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if new_zoom == self._zoom:
            ev.accept()
            return
        # Keep the image point under the cursor fixed across the zoom.
        cursor = ev.position()
        tl = self._current_tl()
        old_scale = self._fit_scale() * self._zoom
        img_x = (cursor.x() - tl.x()) / old_scale
        img_y = (cursor.y() - tl.y()) / old_scale
        self._zoom = new_zoom
        new_scale = self._fit_scale() * self._zoom
        self._tl = QPointF(
            cursor.x() - img_x * new_scale, cursor.y() - img_y * new_scale
        )
        self._clamp_tl()
        self.update()
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._zoom > 1.0:
            self._pan_last = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pan_last is not None:
            pos = ev.position()
            delta = pos - self._pan_last
            self._pan_last = pos
            tl = self._current_tl()
            self._tl = QPointF(tl.x() + delta.x(), tl.y() + delta.y())
            self._clamp_tl()
            self.update()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pan_last is not None:
            self._pan_last = None
            self.unsetCursor()
        else:
            super().mouseReleaseEvent(ev)


class ImageViewerDialog(QDialog):
    """Magnified view of a single gallery image (the 🔍 link in the sample /
    test-output galleries).

    Opens at the image's native size clamped to ~90% of the screen, rescales
    live with the dialog via ScaledImageLabel; Esc closes (QDialog default).
    Non-modal — open it with show().
    """

    def __init__(self, path, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(getattr(path, "name", None) or str(path))
        self.setAttribute(Qt.WA_DeleteOnClose)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        pm = QPixmap(str(path))
        if pm.isNull():
            err = QLabel(str(path))
            err.setAlignment(Qt.AlignCenter)
            err.setMargin(20)
            lay.addWidget(err)
            return
        img = ScaledImageLabel()
        img.set_source(pm)
        lay.addWidget(img)
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry()
        scale = min(
            1.0,
            avail.width() * 0.9 / pm.width(),
            avail.height() * 0.9 / pm.height(),
        )
        self.resize(
            max(1, round(pm.width() * scale)), max(1, round(pm.height() * scale))
        )
