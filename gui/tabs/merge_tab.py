"""MergeTab — bake a LoRA adapter into the base DiT.

Layout mirrors ImageViewerTab: top directory combo, left file list, right
details panel (file stats + bakeability scan + merge options + log).

Runs ``scripts/merge_to_dit.py`` via ``QProcess`` and streams stdout/stderr
into the log pane, same pattern as ``ConfigTab`` training.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import ROOT, LazyTabMixin, _adapter_dirs, _safetensors_in
from gui.i18n import t
from gui.theme import tok
from gui.widgets import action_button, apply_variant
from gui.process import kill_process_tree, setup_kill_safe

_DEFAULT_DIT = "models/diffusion_models/anima-base-v1.0.safetensors"


class PickerLineEdit(QLineEdit):
    """Read-only line edit that opens a file/dir picker on click."""

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setReadOnly(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"QLineEdit {{ background: {tok('input_bg')}; color: {tok('text')}; "
            f"border: 1px solid {tok('border')}; border-radius: 3px; padding: 2px 6px; }}"
            f"QLineEdit:hover {{ border-color: #3c78c8; background: {tok('input_hover')}; }}"
            f"QLineEdit:disabled {{ color: {tok('text_dim')}; background: {tok('base')}; }}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(ev)


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _scan_adapter(path: Path) -> dict:
    """Read safetensors keys and classify the adapter's bakeability.

    Returns a dict with verdict (str), severity ('ok'|'partial'|'block'|'unknown'),
    details (list of short strings), and counts per key family.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return {
            "verdict": "safetensors not installed",
            "severity": "unknown",
            "details": [],
            "counts": {},
        }

    counts = {
        "lora_down": 0,
        "lora_up_weight": 0,  # hydra stacked
        "lora_ups": 0,  # hydra split
        "postfix": 0,
        "ortho_sp": 0,  # .S_p keys (OrthoLoRA / OrthoHydraLoRA)
        "other": 0,
    }
    # Postfix mode lives in metadata — its key names share no greppable prefix.
    metadata_mode: str | None = None
    try:
        # framework="np" — keys + metadata only; "pt" would drag torch in (~1.4s).
        with safe_open(str(path), framework="np") as f:
            keys = list(f.keys())
            try:
                meta = f.metadata() or {}
                metadata_mode = meta.get("ss_mode")
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 — surface any read error in UI
        return {
            "verdict": f"unreadable: {exc!s}",
            "severity": "unknown",
            "details": [],
            "counts": {},
        }

    is_postfix = metadata_mode in {
        "postfix",
        "postfix_exp",
        "postfix_func",
        "cond",
    }

    for k in keys:
        if k.endswith(".lora_up_weight"):
            counts["lora_up_weight"] += 1
        elif ".lora_ups." in k:
            counts["lora_ups"] += 1
        elif k.endswith(".lora_down.weight"):
            counts["lora_down"] += 1
        elif k.endswith(".S_p"):
            counts["ortho_sp"] += 1
        elif is_postfix:
            counts["postfix"] += 1
        else:
            counts["other"] += 1

    details = []
    if counts["lora_down"]:
        details.append(f"{counts['lora_down']} LoRA keys")
    if counts["ortho_sp"]:
        details.append(f"{counts['ortho_sp']} OrthoLoRA keys")
    if counts["lora_up_weight"] or counts["lora_ups"]:
        n = counts["lora_up_weight"] + counts["lora_ups"]
        details.append(f"{n} HydraLoRA keys")
    if counts["postfix"]:
        details.append(f"{counts['postfix']} {metadata_mode} keys")

    is_hydra = bool(counts["lora_up_weight"] or counts["lora_ups"])
    is_postfix_only = is_postfix and not (counts["lora_down"] or counts["ortho_sp"])
    has_lora_like = bool(counts["lora_down"] or counts["ortho_sp"])

    if is_hydra:
        verdict = t("merge_verdict_hydra")
        severity = "block"
    elif is_postfix_only:
        verdict = t("merge_verdict_postfix_only")
        severity = "block"
    elif has_lora_like:
        verdict = t("merge_verdict_ready")
        severity = "ok"
    else:
        verdict = t("merge_verdict_unknown")
        severity = "unknown"

    return {
        "verdict": verdict,
        "severity": severity,
        "details": details,
        "counts": counts,
    }


class MergeTab(LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()

        self._dirs = _adapter_dirs()
        self._files: list[Path] = []
        self._current_scan: dict | None = None

        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(t("directory")))
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(list(self._dirs))
        self.dir_combo.currentTextChanged.connect(self._load_dir)
        top.addWidget(self.dir_combo, 1)
        refresh_btn = QPushButton(t("refresh"))
        refresh_btn.clicked.connect(self._refresh_dirs)
        top.addWidget(refresh_btn)
        self.count_label = QLabel()
        top.addWidget(self.count_label)
        lay.addLayout(top)

        sp = QSplitter(Qt.Horizontal)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._show_file)
        sp.addWidget(self.file_list)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(8, 0, 0, 0)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-weight: bold;")
        rlay.addWidget(self.path_label)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(f"color:{tok('text_dim')};")
        rlay.addWidget(self.stats_label)

        self.verdict_label = QLabel()
        self.verdict_label.setWordWrap(True)
        self.verdict_label.setStyleSheet(
            f"padding:6px; border-radius:3px; background:{tok('panel')};"
        )
        rlay.addWidget(self.verdict_label)

        opt_box = QGroupBox(t("merge_options"))
        opt_lay = QFormLayout(opt_box)

        self.dit_edit = PickerLineEdit(_DEFAULT_DIT)
        self.dit_edit.clicked.connect(self._browse_dit)
        opt_lay.addRow(t("merge_base_dit"), self.dit_edit)

        self.mult_spin = QDoubleSpinBox()
        self.mult_spin.setRange(-2.0, 2.0)
        self.mult_spin.setSingleStep(0.1)
        self.mult_spin.setDecimals(3)
        self.mult_spin.setValue(1.0)
        self.mult_spin.setToolTip(t("merge_multiplier_tip"))
        opt_lay.addRow(t("merge_multiplier"), self.mult_spin)

        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["bf16", "fp16", "fp32"])
        opt_lay.addRow(t("merge_dtype"), self.dtype_combo)

        self.out_edit = PickerLineEdit()
        self.out_edit.setPlaceholderText(t("merge_out_placeholder"))
        self.out_edit.clicked.connect(self._browse_out)
        opt_lay.addRow(t("merge_out"), self.out_edit)

        self.allow_partial = QCheckBox(t("merge_allow_partial"))
        self.allow_partial.setToolTip(t("merge_allow_partial_tip"))
        opt_lay.addRow("", self.allow_partial)

        rlay.addWidget(opt_box)

        bar = QHBoxLayout()
        self.merge_btn = action_button(
            t("merge_button"), variant="success", on_click=self._start_merge
        )
        bar.addWidget(self.merge_btn)

        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_merge)
        bar.addWidget(self.stop_btn)
        bar.addStretch()
        rlay.addLayout(bar)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("merge_log_placeholder"))
        rlay.addWidget(self.log, 1)

        sp.addWidget(right)
        sp.setSizes([260, 760])
        lay.addWidget(sp, 1)

        # Kill-safe setup so Stop kills the whole subtree (frees the loaded DiT weights).
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        setup_kill_safe(self._proc)
        self._proc.readyReadStandardOutput.connect(self._read_stdout)
        self._proc.readyReadStandardError.connect(self._read_stderr)
        self._proc.finished.connect(self._on_finished)

        self._clear_details()

    def _lazy_init(self) -> None:
        # Deferred to first show: the dir scan + adapter classify used to block window startup.
        if self._dirs:
            self._load_dir(self.dir_combo.currentText())
        else:
            self.count_label.setText(t("merge_no_adapter"))

    def _refresh_dirs(self):
        previous = self.dir_combo.currentText()
        self._dirs = _adapter_dirs()
        self.dir_combo.blockSignals(True)
        self.dir_combo.clear()
        self.dir_combo.addItems(list(self._dirs))
        if previous in self._dirs:
            self.dir_combo.setCurrentText(previous)
        self.dir_combo.blockSignals(False)
        if self._dirs:
            self._load_dir(self.dir_combo.currentText())
        else:
            self.file_list.clear()
            self._files = []
            self._clear_details()
            self.count_label.setText(t("merge_no_adapter"))

    def _load_dir(self, name: str):
        d = self._dirs.get(name)
        if not d:
            self.file_list.clear()
            self._files = []
            self._clear_details()
            return
        self._files = _safetensors_in(d)
        self.file_list.clear()
        for p in self._files:
            self.file_list.addItem(p.name)
        self.count_label.setText(t("n_files", n=len(self._files)))
        if self._files:
            self.file_list.setCurrentRow(0)
        else:
            self._clear_details()

    def _current_file(self) -> Path | None:
        r = self.file_list.currentRow()
        if 0 <= r < len(self._files):
            return self._files[r]
        return None

    def _show_file(self, row: int):
        if not 0 <= row < len(self._files):
            self._clear_details()
            return
        p = self._files[row]
        try:
            rel = p.resolve().relative_to(ROOT)
            shown_path = str(rel)
        except ValueError:
            shown_path = str(p)
        self.path_label.setText(shown_path)

        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        self.stats_label.setText(f"{_format_size(st.st_size)} · {mtime}")

        scan = _scan_adapter(p)
        self._current_scan = scan
        colors = {
            "ok": ("#0a3d2a", tok("ok")),  # bg (darkened tint, kept), text
            "partial": ("#3d2e0a", tok("warn")),
            "block": ("#3d0a0a", tok("err")),
            "unknown": (tok("panel"), tok("text_dim")),
        }
        bg, fg = colors.get(scan["severity"], colors["unknown"])
        bits = scan["verdict"]
        if scan["details"]:
            bits += "\n" + " · ".join(scan["details"])
        self.verdict_label.setText(bits)
        self.verdict_label.setStyleSheet(
            f"padding:8px; border-radius:3px; background:{bg}; color:{fg};"
        )
        # Match allow-partial to this file so the previous selection's state can't linger.
        self.allow_partial.setChecked(scan["severity"] == "partial")
        # Only "ok" / "partial" are mergeable.
        self.merge_btn.setEnabled(
            self._proc.state() == QProcess.NotRunning
            and scan["severity"] in ("ok", "partial")
        )

    def _clear_details(self):
        self.path_label.setText("")
        self.stats_label.setText("")
        self.verdict_label.setText(t("merge_no_selection"))
        self.verdict_label.setStyleSheet(
            f"padding:8px; border-radius:3px; background:{tok('panel')}; color:{tok('text_dim')};"
        )
        self._current_scan = None
        self.merge_btn.setEnabled(False)

    def _browse_dit(self):
        start = str(ROOT / "models" / "diffusion_models")
        f, _ = QFileDialog.getOpenFileName(
            self, t("merge_pick_dit"), start, "Safetensors (*.safetensors)"
        )
        if f:
            self.dit_edit.setText(f)

    def _browse_out(self):
        start = str(ROOT / "output" / "ckpt")
        f, _ = QFileDialog.getSaveFileName(
            self, t("merge_pick_out"), start, "Safetensors (*.safetensors)"
        )
        if f:
            self.out_edit.setText(f)

    def _start_merge(self):
        import sys as _sys

        if self._proc.state() != QProcess.NotRunning:
            return

        adapter = self._current_file()
        if adapter is None or not adapter.exists():
            QMessageBox.warning(self, t("error"), t("merge_no_adapter_msg"))
            return

        args = [
            "scripts/merge_to_dit.py",
            "--adapter",
            str(adapter),
            "--dit",
            self.dit_edit.text().strip() or _DEFAULT_DIT,
            "--multiplier",
            f"{self.mult_spin.value():g}",
            "--dtype",
            self.dtype_combo.currentText(),
        ]
        out = self.out_edit.text().strip()
        if out:
            args += ["--out", out]
        if self.allow_partial.isChecked():
            args.append("--allow-partial")

        self.log.clear()
        self._log(f"> {_sys.executable} {' '.join(args)}\n")

        self.merge_btn.setEnabled(False)
        apply_variant(self.merge_btn, "busy")
        self.merge_btn.setText(t("merge_button") + " ...")
        self.stop_btn.setEnabled(True)
        self.dir_combo.setEnabled(False)

        self._proc.start(_sys.executable, args)

    def _stop_merge(self):
        kill_process_tree(self._proc)

    def cleanup_subprocess(self):
        """Hook for app shutdown — kill any running launcher + descendants."""
        kill_process_tree(self._proc)

    def _read_stdout(self):
        data = self._proc.readAllStandardOutput().data().decode(errors="replace")
        self._log(data)

    def _read_stderr(self):
        data = self._proc.readAllStandardError().data().decode(errors="replace")
        self._log(data)

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus):
        self._log(f"\n{t('finished', code=exit_code)}\n")
        apply_variant(self.merge_btn, "success")
        self.merge_btn.setText(t("merge_button"))
        self.merge_btn.setEnabled(
            self._current_scan is not None
            and self._current_scan.get("severity") in ("ok", "partial")
        )
        self.stop_btn.setEnabled(False)
        self.dir_combo.setEnabled(True)
        # A merged file may have been created — refresh so it shows up.
        if exit_code == 0:
            self._refresh_dirs()

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
