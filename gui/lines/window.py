"""Sidecar-line window — Preprocess + Train + Test over the daemon.

One window class for the non-Anima lines (``make gui-qwen``, ``make gui-moss``).
A ``LineSpec`` names the line's torch-free request classes, scan helpers and
strings; the forms are rendered from the requests (one widget per field,
advanced ones folded), with a shared log and progress bar, and every action is
a daemon command job running the line's sidecar (``req.SCRIPT``). No torch here,
and nothing line-specific: ``gui/<line>/app.py`` builds the spec.

Chains run on the GUI side: "Preprocess, then train" submits the train job when
the cache job succeeds, so a failed cache pass never trains on a stale folder,
and a Test with several passes (``LineSpec.test_runs``) submits them one after
another. Closing the window mid-chain drops the pending jobs (the running one
keeps going — the daemon outlives the GUI).
"""

from __future__ import annotations

import dataclasses
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.jobs import daemon as gui_daemon
from gui import i18n as main_i18n
from gui import theme as gui_theme
from gui.jobs.mixin import DaemonJobMixin
from gui.core.paths import ROOT, get_setting, set_setting
from gui.jobs.progress import TQDM_RE
from gui.lines.strings import LANGUAGES, UI
from gui.widgets import _no_wheel, action_button, apply_variant
from library.env import resolve_under_home

TEST_THUMB_PX = 360
# Loading a multi-GB text encoder can be silent for a while; the default
# command-job watchdog (120 s) would kill it.
STALL_TIMEOUT = 900.0


@dataclass(frozen=True)
class LineSpec:
    """Everything line-specific the window needs. All callables are torch-free."""

    key: str  # settings keys ``<key>_gui`` / ``<key>_language``, job labels ``<key>-cache|train|test``
    cache_cls: type
    train_cls: type
    test_cls: type
    # (model_dir field text, variant) → the ``model_index.json`` that must exist.
    model_index: Callable[[str | None, str], Path]
    # Folder shown in the header for a model_dir field text.
    resolve_model_dir: Callable[[str | None], Path]
    # (src, out) → object with images / pairs / text_cached / latents_cached /
    # stale_text / duplicates; cache folder → object with text / latents / samples.
    scan: Callable[[Path | None, Path | None], Any]
    cache_counts: Callable[[Path], Any]
    # Test request (lora and out_dir already resolved) → the jobs to submit, in order.
    test_runs: Callable[[Any], list]
    # Run folder → [(image, multiplier, prompt)] to show; [] while it holds none.
    test_images: Callable[[Path], list[tuple[Path, float, str]]]
    ui: dict[str, dict[str, str]]  # line keys over gui.lines.strings.UI
    fields_cn: dict[str, tuple[str, str]] = dataclasses.field(default_factory=dict)
    # Test-request fields whose name means something else on another request.
    fields_cn_test: dict[str, tuple[str, str]] = dataclasses.field(default_factory=dict)

    @property
    def labels(self) -> tuple[str, str, str]:
        return (f"{self.key}-cache", f"{self.key}-train", f"{self.key}-test")


_DIR_FIELDS = {"src", "out", "cache", "model_dir", "out_dir"}
_FILE_FIELDS = {"output"}  # save dialog
_OPEN_FIELDS = {"lora", "prompts_file"}  # open dialog
_PROGRESS_RE = re.compile(r"^\s+(progress|step|text|latents|image) (\d+)/(\d+)(.*)")

_lang = "en"
_spec: LineSpec | None = None


def tr(key: str, **kwargs) -> str:
    tables = [_spec.ui.get(_lang, {}), _spec.ui.get("en", {})] if _spec else []
    tables += [UI.get(_lang, {}), UI["en"]]
    s = next((t[key] for t in tables if key in t), key)
    return s.format(**kwargs) if kwargs else s


def field_text(f: dataclasses.Field, cls) -> tuple[str, str]:
    if _lang == "cn" and _spec is not None:
        if cls is _spec.test_cls and f.name in _spec.fields_cn_test:
            return _spec.fields_cn_test[f.name]
        if f.name in _spec.fields_cn:
            return _spec.fields_cn[f.name]
    return f.name, f.metadata.get("help") or ""


def _base_type(f: dataclasses.Field) -> type:
    annotation = str(f.type)
    for kind in (bool, int, float):
        if kind.__name__ in annotation:
            return kind
    return str


class RequestForm(QWidget):
    """One widget per request field; ``request()`` reads them back typed."""

    def __init__(self, cls, parent=None):
        super().__init__(parent)
        self.cls = cls
        self.widgets: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        basic = QFormLayout()
        advanced_box = QGroupBox(tr("advanced"))
        advanced_box.setCheckable(True)
        advanced_box.setChecked(False)
        advanced_inner = QWidget()
        advanced = QFormLayout(advanced_inner)
        QVBoxLayout(advanced_box).addWidget(advanced_inner)
        advanced_inner.setVisible(False)
        advanced_box.toggled.connect(advanced_inner.setVisible)

        for f in dataclasses.fields(cls):
            label, help_text = field_text(f, cls)
            widget, row = self._make(f)
            widget.setToolTip(help_text)
            name = QLabel(label)
            name.setToolTip(help_text)
            (advanced if f.metadata.get("advanced") else basic).addRow(name, row)
            self.widgets[f.name] = widget
        layout.addLayout(basic)
        layout.addWidget(advanced_box)

    def _make(self, f: dataclasses.Field) -> tuple[QWidget, QWidget]:
        kind = _base_type(f)
        default = None if f.default is dataclasses.MISSING else f.default
        if kind is bool:
            w = QCheckBox()
            w.setChecked(bool(default))
            return w, w
        if f.metadata.get("choices"):
            w = _no_wheel(QComboBox())
            w.addItems(list(f.metadata["choices"]))
            w.setCurrentText(str(default))
            return w, w
        if kind is int and default is not None:
            w = _no_wheel(QSpinBox())
            # -1 is a sentinel on some fields ("auto", "steps - 3").
            w.setRange(min(-1, default), 1_000_000)
            w.setValue(default)
            return w, w
        if f.metadata.get("multiline"):
            w = QPlainTextEdit(str(default or ""))
            w.setFixedHeight(96)
            return w, w
        w = QLineEdit("" if default is None else str(default))
        if default is None:
            w.setPlaceholderText(tr("auto"))
        if f.name == "model_dir":
            w.setPlaceholderText(str(_spec.resolve_model_dir(None)))
        if f.name not in _DIR_FIELDS | _FILE_FIELDS | _OPEN_FIELDS:
            return w, w
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        browse = QPushButton(tr("browse"))
        browse.clicked.connect(lambda _=False, n=f.name, e=w: self._browse(n, e))
        h.addWidget(browse)
        return w, row

    def _browse(self, name: str, edit: QLineEdit) -> None:
        start = str(resolve_under_home(edit.text() or edit.placeholderText() or "."))
        if name in _FILE_FIELDS:
            path, _ = QFileDialog.getSaveFileName(
                self, name, start, "safetensors (*.safetensors)"
            )
        elif name in _OPEN_FIELDS:
            path, _ = QFileDialog.getOpenFileName(self, name, start)
        else:
            path = QFileDialog.getExistingDirectory(self, name, start)
        if path:
            edit.setText(path)

    def text_of(self, name: str) -> str:
        w = self.widgets[name]
        if isinstance(w, QPlainTextEdit):
            return w.toPlainText().strip()
        return w.text().strip() if isinstance(w, QLineEdit) else ""

    def request(self):
        """The typed request, or ``None`` after warning about a bad number."""
        values = {}
        for f in dataclasses.fields(self.cls):
            w = self.widgets[f.name]
            if isinstance(w, QCheckBox):
                values[f.name] = w.isChecked()
            elif isinstance(w, QComboBox):
                values[f.name] = w.currentText()
            elif isinstance(w, QSpinBox):
                values[f.name] = w.value()
            elif isinstance(w, QPlainTextEdit):
                values[f.name] = w.toPlainText().strip()
            else:
                text = w.text().strip()
                kind = _base_type(f)
                if kind in (int, float):
                    if not text:
                        values[f.name] = None
                        continue
                    try:
                        values[f.name] = kind(text)
                    except ValueError:
                        QMessageBox.warning(
                            self,
                            tr("window_title"),
                            tr("bad_value", field=f.name, value=text),
                        )
                        return None
                else:
                    values[f.name] = text or (None if f.default is None else "")
        return self.cls(**values)

    def set_values(self, values: dict) -> None:
        for name, value in values.items():
            w = self.widgets.get(name)
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText(str(value))
            else:
                w.setText("" if value is None else str(value))

    def changed_values(self) -> dict:
        """Fields that differ from the request defaults — what gets persisted."""
        req = self.request()
        if req is None:
            return {}
        out = {}
        for f in dataclasses.fields(self.cls):
            value = getattr(req, f.name)
            if f.default is dataclasses.MISSING or value != f.default:
                out[f.name] = value
        return out

    def watch(self, callback) -> None:
        for w in self.widgets.values():
            if isinstance(w, (QLineEdit, QPlainTextEdit)):
                w.textChanged.connect(callback)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(callback)


class _LineProgress:
    """Drives the bar from the sidecars' ``  step|text|latents i/n`` lines.

    Per-image cache lines are swallowed (the bar shows them); training step
    lines stay in the log — they carry loss and memory.
    """

    def __init__(self, bar: QProgressBar):
        self.bar = bar

    def feed(self, line: str) -> bool:
        m = _PROGRESS_RE.match(line)
        if not m:
            # tqdm bars (weight loading, the pipeline's denoise loop) drive the
            # bar too, instead of one log line per redraw.
            t = TQDM_RE.search(line)
            if not t:
                return False
            cur, tot = int(t.group("cur")), int(t.group("tot"))
            if tot > 0:
                self.bar.setVisible(True)
                self.bar.setRange(0, tot)
                self.bar.setValue(cur)
                self.bar.setFormat(f"{t.group('label').strip() or 'progress'} %v/%m")
            return True
        kind, cur, tot = m.group(1), int(m.group(2)), int(m.group(3))
        self.bar.setVisible(True)
        self.bar.setRange(0, tot)
        self.bar.setValue(cur)
        if kind == "progress":
            # "  progress i/n epoch e/E loss x eta h:mm:ss" — the tail rides along.
            self.bar.setFormat(f"step %v/%m {m.group(4).strip()}")
            return True
        if kind == "step":
            # The every-20th detail line: keep it in the log, leave the bar to
            # the progress lines.
            return False
        self.bar.setFormat(f"{kind} %v/%m")
        return True

    def starting(self, label: str) -> None:
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.bar.setFormat(label)

    def reset(self) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setVisible(False)


class LinePanel(DaemonJobMixin, QWidget):
    """The line's Preprocess / Train / Test forms, log and job observer.

    ``LineWindow`` wraps it standalone; the main GUI's MOSS mode embeds it
    (``build_panel``). ``on_language`` adds the en/cn picker (standalone only).
    """

    def __init__(
        self, spec: LineSpec, on_language: Callable[[str], None] | None = None
    ):
        super().__init__()
        self.spec = spec
        self.label_cache, self.label_train, self.label_test = spec.labels
        self._init_job_observer()
        self._job_label: str | None = None
        # (request, label) submitted one by one as each job succeeds.
        self._pending: list[tuple[Any, str]] = []

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.model_label = QLabel()
        self.model_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.model_label, 1)
        if on_language is not None:
            top.addWidget(QLabel(tr("language")))
            self.lang_combo = QComboBox()
            for code, name in LANGUAGES.items():
                self.lang_combo.addItem(name, code)
            self.lang_combo.setCurrentIndex(list(LANGUAGES).index(_lang))
            self.lang_combo.currentIndexChanged.connect(
                lambda: on_language(self.lang_combo.currentData())
            )
            top.addWidget(self.lang_combo)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.cache_form = RequestForm(spec.cache_cls)
        self.train_form = RequestForm(spec.train_cls)
        saved = get_setting(self._settings_key, {}) or {}
        self.cache_form.set_values(saved.get("cache", {}))
        self.train_form.set_values(saved.get("train", {}))

        self.cache_summary = QLabel()
        self.cache_summary.setWordWrap(True)
        self.btn_cache = action_button(tr("run_cache"), on_click=self._run_cache)
        self.tabs.addTab(
            self._page(self.cache_form, self.cache_summary, [self.btn_cache]),
            tr("tab_preprocess"),
        )

        self.train_summary = QLabel()
        self.train_summary.setWordWrap(True)
        self.btn_train = action_button(tr("run_train"), on_click=self._run_train)
        self.btn_chain = action_button(
            tr("run_chain"), variant="secondary", on_click=self._run_chain
        )
        self.test_form = RequestForm(spec.test_cls)
        self.test_form.set_values(saved.get("test", {}))
        self.btn_test = action_button(
            tr("run_test"), variant="info", on_click=self._run_test
        )
        self.test_images = QHBoxLayout()
        self._test_out: Path | None = None
        test_box = QGroupBox(tr("test_group"))
        tb = QVBoxLayout(test_box)
        hint = QLabel(tr("test_hint"))
        hint.setWordWrap(True)
        tb.addWidget(hint)
        tb.addWidget(self.test_form)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_test)
        tb.addLayout(row)
        tb.addLayout(self.test_images)
        self.tabs.addTab(
            self._page(
                self.train_form,
                self.train_summary,
                [self.btn_train, self.btn_chain],
                extra=test_box,
            ),
            tr("tab_train"),
        )

        self.status = QLabel(tr("idle"))
        self.btn_stop = action_button(tr("stop"), variant="danger", on_click=self._stop)
        self.btn_stop.setEnabled(False)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.btn_stop)

        bar = QProgressBar()
        bar.setVisible(False)
        self._progress_tracker = _LineProgress(bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addLayout(status_row)
        bl.addWidget(bar)
        bl.addWidget(self.log, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(300)
        self._rescan_timer.timeout.connect(self._rescan)
        self.cache_form.watch(self._rescan_timer.start)
        self.train_form.watch(self._rescan_timer.start)
        self.test_form.watch(self._rescan_timer.start)
        self._set_busy(False)
        self._rescan()
        self._show_test(self._latest_test_dir())
        QTimer.singleShot(0, self._reattach)

    def _page(
        self, form: QWidget, summary: QLabel, buttons: list, extra=None
    ) -> QWidget:
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.addWidget(form)
        v.addWidget(summary)
        row = QHBoxLayout()
        row.addStretch(1)
        for b in buttons:
            row.addWidget(b)
        v.addLayout(row)
        if extra is not None:
            v.addWidget(extra)
        v.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # ── state ────────────────────────────────────────────────────────────

    @property
    def _settings_key(self) -> str:
        return f"{self.spec.key}_gui"

    def _model_dir_text(self) -> str:
        return self.cache_form.text_of("model_dir") or self.train_form.text_of(
            "model_dir"
        )

    def _rescan(self) -> None:
        model = self.spec.resolve_model_dir(self._model_dir_text())
        ok = self.spec.model_index(self._model_dir_text(), "base").exists()
        self.model_label.setText(
            tr("model", path=model) if ok else tr("model_missing", path=model)
        )
        scan, cache_counts = self.spec.scan, self.spec.cache_counts
        src = self.cache_form.text_of("src")
        out = self.cache_form.text_of("out")
        lines = []
        if src:
            s = scan(resolve_under_home(src), resolve_under_home(out) if out else None)
            lines.append(
                tr(
                    "scan",
                    pairs=s.pairs,
                    images=s.images,
                    text=s.text_cached,
                    latents=s.latents_cached,
                )
            )
            if s.stale_text:
                lines.append(tr("scan_stale", stale=s.stale_text))
            if s.duplicates:
                lines.append(tr("scan_dupes", n=s.duplicates))
        else:
            lines.append(tr("scan_no_src"))
        if out:
            c = cache_counts(resolve_under_home(out))
            lines.append(
                tr("cache_counts", samples=c.samples, text=c.text, latents=c.latents)
            )
        self.cache_summary.setText("\n".join(lines))
        cache = self.train_form.text_of("cache")
        n = cache_counts(resolve_under_home(cache)).samples if cache else 0
        self.train_summary.setText(tr("train_ready", n=n) if n else tr("train_empty"))
        self.test_form.widgets["lora"].setPlaceholderText(
            tr("test_lora_placeholder", path=self._train_output())
        )

    def _train_output(self) -> Path:
        return resolve_under_home(
            self.train_form.text_of("output") or self.spec.train_cls().output
        )

    def _persist(self) -> None:
        set_setting(
            self._settings_key,
            {
                "cache": self.cache_form.changed_values(),
                "train": self.train_form.changed_values(),
                "test": self.test_form.changed_values(),
            },
        )

    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_cache, self.btn_train, self.btn_chain, self.btn_test):
            b.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        apply_variant(self.btn_stop, "danger" if busy else "busy")

    # ── jobs ─────────────────────────────────────────────────────────────

    def _model_ok(self, req) -> bool:
        index = self.spec.model_index(req.model_dir, getattr(req, "model", "base"))
        if index.exists():
            return True
        QMessageBox.warning(
            self, tr("window_title"), tr("model_missing", path=index.parent)
        )
        return False

    def _src_ok(self, req) -> bool:
        if req.src and resolve_under_home(req.src).is_dir():
            return True
        QMessageBox.warning(self, tr("window_title"), tr("scan_no_src"))
        return False

    def _submit(self, req, label: str) -> bool:
        argv = [req.SCRIPT, *req.to_argv()]
        job_id = self._submit_job(
            lambda: gui_daemon.ensure_daemon().submit_command(
                label=label, argv=argv, stall_timeout=STALL_TIMEOUT, start=True
            ),
            on_fail=lambda: self._set_busy(False),
        )
        if job_id is None:
            self._pending = []
            return False
        self._job_label = label
        self.log.appendPlainText(f"$ python {' '.join(argv)}")
        self.status.setText(tr("running", label=label, job=job_id))
        self._set_busy(True)
        self._progress_tracker.starting(label)
        self._watch_job(job_id, replay_log=False)
        return True

    def _run_cache(self) -> None:
        req = self.cache_form.request()
        if req is None or not self._src_ok(req) or not self._model_ok(req):
            return
        self._persist()
        self._submit(req, self.label_cache)

    def _run_train(self) -> None:
        req = self.train_form.request()
        if req is None or not self._model_ok(req):
            return
        self._persist()
        self._submit(req, self.label_train)

    def _run_chain(self) -> None:
        cache_req = self.cache_form.request()
        if (
            cache_req is None
            or not self._src_ok(cache_req)
            or not self._model_ok(cache_req)
        ):
            return
        # Training reads what this preprocessing run writes.
        self.train_form.set_values({"cache": cache_req.out})
        train_req = self.train_form.request()
        if train_req is None or not self._model_ok(train_req):
            return
        self._persist()
        self.log.appendPlainText(
            tr("chain_cache", path=resolve_under_home(cache_req.out))
        )
        if self._submit(cache_req, self.label_cache):
            self._pending = [(train_req, self.label_train)]

    def _run_test(self) -> None:
        req = self.test_form.request()
        if req is None or not self._model_ok(req):
            return
        lora = resolve_under_home(req.lora) if req.lora else self._train_output()
        if not lora.is_file():
            QMessageBox.warning(self, tr("window_title"), tr("no_lora", path=lora))
            return
        self._persist()
        # One folder per run, so earlier tests stay on disk.
        out = resolve_under_home(req.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        req = dataclasses.replace(req, lora=str(lora), out_dir=str(out))
        first, *rest = self.spec.test_runs(req)
        if self._submit(first, self.label_test):
            self._pending = [(r, self.label_test) for r in rest]
            self._test_out = out

    def _show_test(self, out: Path | None) -> None:
        while self.test_images.count():
            item = self.test_images.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for path, m, prompt in self.spec.test_images(out) if out else []:
            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            image = QLabel()
            image.setPixmap(
                QPixmap(str(path)).scaled(
                    TEST_THUMB_PX,
                    TEST_THUMB_PX,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            image.setToolTip(f"{path}\n{prompt}")
            caption = QLabel(tr("test_lora", m=f"{m:g}") if m else tr("test_base"))
            caption.setAlignment(Qt.AlignCenter)
            v.addWidget(image)
            v.addWidget(caption)
            self.test_images.addWidget(cell)
        self.test_images.addStretch(1)

    def _latest_test_dir(self) -> Path | None:
        base = resolve_under_home(
            self.test_form.text_of("out_dir") or self.spec.test_cls().out_dir
        )
        runs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        return next((p for p in reversed(runs) if self.spec.test_images(p)), None)

    def _stop(self) -> None:
        self._pending = []
        self._stop_job()

    def _on_job_finished(self, state: str) -> None:
        self._end_job_watch()
        label = self._job_label or "job"
        self.log.appendPlainText(tr("finished", label=label, state=state))
        self._progress_tracker.reset()
        self._set_busy(False)
        self.status.setText(tr("finished", label=label, state=state))
        self._rescan()
        ok = gui_daemon.is_success(state)
        nxt = self._pending.pop(0) if ok and self._pending else None
        if not ok:
            self._pending = []
        if label == self.label_test and ok and nxt is None:
            self._show_test(self._test_out or self._latest_test_dir())
            self._test_out = None
        if nxt is not None:
            req, next_label = nxt
            self.log.appendPlainText(
                tr("test_next" if next_label == self.label_test else "chain_next")
            )
            self._submit(req, next_label)

    def _reattach(self) -> None:
        job_id = gui_daemon.active_job_id()
        if not job_id:
            return
        label = gui_daemon.read_job_label(job_id)
        if label not in self.spec.labels:
            return
        self._job_label = label
        self.status.setText(tr("running", label=label, job=job_id))
        self.log.appendPlainText(tr("reattached", job=job_id))
        self._set_busy(True)
        self._watch_job(job_id, replay_log=True)

    def cleanup_subprocess(self) -> None:
        """Stop observing on close; the job itself keeps running in the daemon."""
        self._job_timer.stop()


class LineWindow(QMainWindow):
    """``make gui-qwen`` / ``make gui-moss``: one ``LinePanel`` in its own window."""

    def __init__(self, spec: LineSpec):
        super().__init__()
        self.spec = spec
        self.setWindowTitle(tr("window_title"))
        self.resize(900, 820)
        self.panel = LinePanel(spec, on_language=self._change_language)
        self.setCentralWidget(self.panel)

    def _change_language(self, code: str) -> None:
        if code == _lang:
            return
        self.panel._persist()
        set_setting(f"{self.spec.key}_language", code)
        _set_language(code)
        # Rebuild in place: every label is set at construction.
        global _WINDOW
        geometry = self.saveGeometry()
        self.panel.cleanup_subprocess()
        _WINDOW = LineWindow(self.spec)
        _WINDOW.panel._pending = self.panel._pending
        _WINDOW.restoreGeometry(geometry)
        _WINDOW.show()
        self.close()


def _set_language(code: str) -> None:
    global _lang
    _lang = code if code in LANGUAGES else "en"
    # The job mixin's warnings use the main GUI's table.
    main_i18n.set_language(_lang)


_WINDOW: LineWindow | None = None


def build_panel(spec: LineSpec, main_lang: str) -> LinePanel:
    """The line's panel for embedding in the main GUI: cn when the main GUI is
    cn, else en. Leaves the main GUI's i18n table alone."""
    global _spec, _lang
    _spec = spec
    _lang = "cn" if main_lang == "cn" else "en"
    return LinePanel(spec)


def run(spec: LineSpec) -> None:
    global _spec
    _spec = spec
    main_lang = main_i18n.load_language()
    _set_language(
        get_setting(f"{spec.key}_language") or ("cn" if main_lang == "cn" else "en")
    )
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    icon = ROOT / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    gui_theme.apply_theme(app)
    global _WINDOW
    _WINDOW = LineWindow(spec)
    _WINDOW.show()
    QTimer.singleShot(0, gui_daemon.ensure_daemon_quietly)
    sys.exit(app.exec())
