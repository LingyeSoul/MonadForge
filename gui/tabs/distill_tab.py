"""SPD / Turbo distillation config tabs.

SPD and Turbo train through bespoke distill loops (``make exp-spd`` /
``make exp-turbo`` → ``scripts/distill_{spd,turbo}.py``), NOT ``train.py``, and
read a *sectioned* TOML (``configs/methods/{spd,turbo}.toml``) — nested
``[network]`` / ``[schedule]`` / ``[optim]`` / … tables — instead of the flat
method+preset config the ConfigTab edits. They also have no dataset of their
own: both reuse the ordinary LoRA cache under ``post_image_dataset/lora``.

So rather than the IP-Adapter / EasyControl dataset-browser launcher (which
exists to manage a *separate* image set), these get a structured config-editor
tab that mirrors the method tabs' look: a per-section form on top, a log on the
bottom, and daemon-backed Preprocess / Train / Stop in the top bar.

Editing uses ``tomlkit`` so the configs' extensive inline comments survive a
Save round-trip (a plain ``toml.dumps`` would strip them); those comments
double as the form's field tooltips.
"""

from __future__ import annotations

import html
import re

import tomlkit
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from tomlkit.items import Table, Whitespace

from gui import ROOT, CONFIGS_DIR, LazyTabMixin, _read, _widget
from gui import daemon as gui_daemon
from gui.i18n import t
from gui.progress import TqdmProgressTracker, make_progress_bar
from gui.tabs.config_tab import ClickableLabel


class _DistillConfigTab(LazyTabMixin, QWidget):
    """Structured editor for a bespoke sectioned distill config.

    Subclasses set the config file, the ``tasks.py`` train target, and a label.
    The form is rebuilt from the TOML on first show; Save writes edits back
    through ``tomlkit`` (comments preserved); Train/Preprocess submit the make
    target to the local daemon and observe it via the per-job ``stdout.log``.
    """

    CONFIG_PATH: str = ""  # repo-relative, e.g. "configs/methods/spd.toml"
    TRAIN_TASK: str = ""  # tasks.py target, e.g. "exp-spd"
    METHOD_LABEL: str = ""

    def __init__(self):
        super().__init__()
        self._config_path = ROOT / self.CONFIG_PATH
        self._doc: tomlkit.TOMLDocument | None = None
        # (section_or_None, key, widget, original_python_value)
        self._fields: list[tuple[str | None, str, QWidget, object]] = []
        self._dirty = False
        # Leading file-header comment block → the right panel's default guide.
        self._guide_lines: list[str] = []

        lay = QVBoxLayout(self)

        # ── Top bar: title + file path + Save / Preprocess / Train / Stop ──
        top = QHBoxLayout()
        title = QLabel(self.METHOD_LABEL)
        title.setStyleSheet("font-weight:bold;font-size:14px;")
        top.addWidget(title)
        path_lbl = QLabel(self.CONFIG_PATH)
        path_lbl.setStyleSheet("color:#888;")
        top.addWidget(path_lbl)
        top.addStretch()

        self._save_btn = QPushButton(t("save"))
        self._save_btn_idle_style = ""
        self._save_btn_dirty_style = (
            "background:#e67e22;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._save_btn.clicked.connect(lambda: self._save())
        top.addWidget(self._save_btn)

        self.train_btn = QPushButton(t("train"))
        self._train_idle_style = (
            "background:#27ae60;color:white;font-weight:bold;padding:4px 16px;"
        )
        self._train_busy_style = (
            "background:#7f8c8d;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.clicked.connect(self._start_train)
        top.addWidget(self.train_btn)

        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        lay.addLayout(top)

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        # ── Body: (form | explanation) on top, log on bottom ──
        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        self._form = QWidget()
        self._fl = QVBoxLayout(self._form)
        self._fl.setContentsMargins(0, 0, 0, 0)
        sc.setWidget(self._form)
        hsplit.addWidget(sc)

        # Right-side explanation panel — mirrors the ConfigTab method tabs.
        # Defaults to the config's file-header block (a method overview); a
        # field label click swaps in that field's full doc comment.
        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 13px; padding: 12px; "
            "background: #2b2b2b; color: #e0e0e0; }"
        )
        self._explain.setMinimumWidth(300)
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([640, 360])
        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)
        self._show_explain_placeholder()

        # ── Daemon job observation (command job → no progress.jsonl, tqdm
        # parsing drives the bar). Mirrors ConfigTab's poll-driven approach. ──
        self._job_id: str | None = None
        self._stdout_buf = ""
        self._stdout_tailer = gui_daemon.FileTailer()
        self._job_timer = QTimer(self)
        self._job_timer.setInterval(400)
        self._job_timer.timeout.connect(self._poll_job)

    def _lazy_init(self) -> None:
        self._load_and_build()
        self._try_reattach()

    # ── Form build ────────────────────────────────────────────────

    def _load_and_build(self) -> None:
        try:
            text = self._config_path.read_text(encoding="utf-8")
        except OSError as e:
            self._fl.addWidget(QLabel(t("distill_config_missing", err=str(e))))
            return
        self._doc = tomlkit.parse(text)
        self._guide_lines = self._header_comment_lines()

        self._fields = []
        while self._fl.count():
            it = self._fl.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        # Top-level scalars first (everything that isn't a [section] table),
        # then one box per table — preserving the file's order via doc.body.
        top_body = [(k, item) for k, item in self._doc.body]
        general = self._build_box(t("distill_general_section"), top_body, None)
        if general is not None:
            self._fl.addWidget(general)
        for key, item in self._doc.body:
            if key is not None and isinstance(item, Table):
                box = self._build_box(str(key).strip(), item.value.body, str(key).strip())
                if box is not None:
                    self._fl.addWidget(box)

        self._fl.addStretch()
        for _s, _k, w, _o in self._fields:
            self._connect_dirty(w)
        self._clear_dirty()
        self._show_explain_placeholder()

    def _build_box(self, title: str, body, section: str | None) -> QGroupBox | None:
        """Build a group box of scalar fields from a tomlkit container body.

        Preceding ``# comment`` lines become the field's tooltip (a blank line
        resets the accumulator so a header block doesn't bleed onto the next
        key). Nested tables are skipped here — the top-level loop renders each
        as its own box. Returns ``None`` when the body has no scalar fields.
        """
        box = QGroupBox(title)
        form = QFormLayout()
        pending: list[str] = []
        added = 0
        for key, item in body:
            if key is None:
                if isinstance(item, Whitespace):
                    pending = []  # blank line separates unrelated comment blocks
                else:  # Comment
                    txt = self._comment_text(item)
                    if txt:
                        pending.append(txt)
                continue
            if isinstance(item, Table):
                pending = []
                continue
            name = str(key).strip()
            orig = item.unwrap() if hasattr(item, "unwrap") else item
            w = _widget(orig, key=name)
            if isinstance(w, QSpinBox):
                # _widget caps ints at 10k; iterations / step counts run higher.
                w.setRange(0, 100_000_000)
                w.setValue(int(orig))
            # Tooltip = preceding comment block + this line's inline trailing
            # comment (turbo.toml documents most knobs inline; spd.toml blocks).
            parts = [" ".join(pending).strip(), self._comment_text(item)]
            tip = "  ".join(p for p in parts if p)
            pending = []
            lbl = ClickableLabel(name)
            lbl.setStyleSheet("text-decoration: underline dotted; color:#ddd;")
            if tip:
                lbl.setToolTip(tip)
                w.setToolTip(tip)
            lbl.clicked.connect(
                lambda _n=name, _t=tip: self._show_explain(_n, _t)
            )
            form.addRow(lbl, w)
            self._fields.append((section, name, w, orig))
            added += 1
        box.setLayout(form)
        return box if added else None

    @staticmethod
    def _comment_text(item) -> str:
        """The text of a tomlkit Comment item, stripped of its leading ``#``."""
        if isinstance(item, Whitespace):
            return ""
        trivia = getattr(item, "trivia", None)
        raw = getattr(trivia, "comment", "") if trivia else ""
        return raw.lstrip("#").strip()

    def _header_comment_lines(self) -> list[str]:
        """The config file's leading ``#`` block (up to the first key / blank
        line) — used as the right panel's default method overview. Bare ``#``
        separator lines come through as ``""`` so they split into paragraphs."""
        lines: list[str] = []
        for key, item in self._doc.body:
            if key is not None or isinstance(item, Whitespace):
                break
            lines.append(self._comment_text(item))
        return lines

    # ── Explanation panel ──

    def _show_explain_placeholder(self) -> None:
        title = html.escape(self.METHOD_LABEL)
        paras: list[str] = []
        cur: list[str] = []
        for ln in self._guide_lines:
            if ln:
                cur.append(ln)
            elif cur:
                paras.append(" ".join(cur))
                cur = []
        if cur:
            paras.append(" ".join(cur))
        if paras:
            body = "".join(
                f"<p style='font-size:13px; line-height:1.6;'>{html.escape(p)}</p>"
                for p in paras
            )
        else:
            body = (
                f"<p style='color:#888; font-style:italic;'>"
                f"{html.escape(t('click_field_for_help'))}</p>"
            )
        self._explain.setHtml(
            f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{title}</h2>{body}"
        )

    def _show_explain(self, field: str, help_text: str) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:18px;'>{html.escape(field)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:14px; line-height:1.6;'>"
                f"{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:#888; font-style:italic;'>"
                f"{html.escape(t('no_help_available'))}</p>"
            )
        self._explain.setHtml("".join(parts))

    def _connect_dirty(self, w: QWidget) -> None:
        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(self._mark_dirty)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._mark_dirty)
        elif isinstance(w, QSpinBox):
            w.valueChanged.connect(self._mark_dirty)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(self._mark_dirty)

    # ── Dirty tracking ──

    def _mark_dirty(self, *_):
        if self._dirty:
            return
        self._dirty = True
        self._save_btn.setText(t("save") + " *")
        self._save_btn.setStyleSheet(self._save_btn_dirty_style)

    def _clear_dirty(self):
        self._dirty = False
        self._save_btn.setText(t("save"))
        self._save_btn.setStyleSheet(self._save_btn_idle_style)

    # ── Save ──

    def _save(self, *, silent: bool = False) -> bool:
        if self._doc is None:
            return False
        for section, name, w, orig in self._fields:
            container = self._doc if section is None else self._doc[section]
            container[name] = _read(w, orig)
        try:
            self._config_path.write_text(tomlkit.dumps(self._doc), encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, t("error"), str(e))
            return False
        self._clear_dirty()
        if not silent:
            try:
                rel = self._config_path.relative_to(CONFIGS_DIR.parent)
            except ValueError:
                rel = self._config_path
            QMessageBox.information(self, t("saved"), f"Saved {rel}")
        return True

    # ── Training (daemon) ──

    def _start_train(self):
        self._launch(self.TRAIN_TASK, ["tasks.py", self.TRAIN_TASK])

    def _launch(self, label: str, argv: list[str]) -> None:
        if self._job_id:
            QMessageBox.information(self, "", t("distill_job_running"))
            return
        # train.py / the distill script re-reads the TOML from disk, so flush
        # any unsaved form edits before launching.
        if self._dirty and not self._save(silent=True):
            return
        self._set_busy()
        self.log.clear()
        self.progress.setVisible(True)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_submitting") + "\n")
        try:
            resp = gui_daemon.submit_command(label=label, argv=argv)
        except Exception as e:  # noqa: BLE001 — daemon down / submit failed
            QMessageBox.warning(self, t("error"), t("daemon_submit_failed", err=str(e)))
            self._restore_idle()
            return
        job_id = resp.get("job_id") if isinstance(resp, dict) else None
        if not job_id:
            QMessageBox.warning(self, t("error"), t("daemon_submit_failed", err=str(resp)))
            self._restore_idle()
            return
        self._log(t("daemon_queued", job_id=job_id))
        self._attach(job_id, replay=False)

    def _try_reattach(self) -> None:
        """Re-bind to our own train job still running from a previous session.

        Only re-claims a *command* job whose label matches this tab's train
        target — a shared ``preprocess`` job (also submittable from other tabs)
        is intentionally left for whoever owns it."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "command":
            return
        if gui_daemon.read_job_label(job_id) != self.TRAIN_TASK:
            return
        self.log.clear()
        self.progress.setVisible(True)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach(job_id, replay=True)

    def _attach(self, job_id: str, *, replay: bool) -> None:
        self._job_id = job_id
        self._stdout_buf = ""
        self._stdout_tailer.watch(gui_daemon.stdout_path(job_id))
        if not replay:
            self._stdout_tailer.read_new()  # discard backlog from a fresh launch
        self._set_busy()
        self.stop_btn.setEnabled(True)
        self._job_timer.start()

    def _poll_job(self) -> None:
        if not self._job_id:
            return
        self._drain_stdout()
        state = gui_daemon.read_job_state(self._job_id)
        if gui_daemon.is_terminal(state):
            self._on_job_finished(state)

    def _drain_stdout(self) -> None:
        chunk = self._stdout_tailer.read_new()
        if not chunk:
            return
        parts = re.split(r"[\r\n]", self._stdout_buf + chunk)
        self._stdout_buf = parts[-1]  # incomplete trailing fragment
        for line in parts[:-1]:
            if self._progress_tracker.feed(line):
                continue
            if line:
                self._log(line + "\n")

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        self._drain_stdout()
        if self._stdout_buf:
            self._log(self._stdout_buf + "\n")
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self.progress.setVisible(False)
        self._log("\n" + gui_daemon.format_finish_banner(job_id, state) + "\n")
        self._restore_idle()

    def _stop(self):
        if self._job_id:
            try:
                gui_daemon.stop_job(self._job_id)
            except Exception as e:  # noqa: BLE001
                self._log(f"stop failed: {e}\n")

    def cleanup_subprocess(self):
        """App-shutdown hook. Stops observing but leaves the daemon job alive —
        it runs detached so training survives the GUI closing (re-attached on
        next launch)."""
        self._job_timer.stop()

    # ── UI state ──

    def _set_busy(self):
        self.train_btn.setText(t("train") + " ...")
        self.train_btn.setStyleSheet(self._train_busy_style)
        self.train_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _restore_idle(self):
        self.train_btn.setText(t("train"))
        self.train_btn.setStyleSheet(self._train_idle_style)
        self.train_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)


class SPDTrainTab(_DistillConfigTab):
    # SPD "Case B": a trajectory LoRA distilled to follow the multi-resolution
    # SPD inference path. Trains on the SAME data + cache as ordinary LoRA
    # (image_dataset/ → post_image_dataset/lora/, VAE + TE only — no PE) via the
    # bespoke loop `make exp-spd` (scripts/distill_spd.py, NOT train.py), so it
    # has no dataset of its own. Output anima_spd*.safetensors is a normal LoRA —
    # infer with the SPD sampler at the trained schedule (`make exp-test-spd`).
    # See docs/experimental/spd.md.
    CONFIG_PATH = "configs/methods/spd.toml"
    TRAIN_TASK = "exp-spd"
    METHOD_LABEL = "SPD"


class TurboTrainTab(_DistillConfigTab):
    # Turbo Anima: a 4-step LoRA student distilled from the 28-step CFG=4 teacher
    # via Decoupled-Hybrid DMD2. Like SPD it trains on the SAME data + cache as
    # ordinary LoRA training, through the bespoke loop `make exp-turbo`
    # (scripts/distill_turbo.py, NOT train.py). Output anima_turbo.safetensors is
    # a normal LoRA — infer at 4 steps cfg=1.0 (`make exp-test-turbo`). The
    # turbo.toml schema is bespoke/sectioned. See
    # docs/experimental/dmd2-decoupled.md.
    CONFIG_PATH = "configs/methods/turbo.toml"
    TRAIN_TASK = "exp-turbo"
    METHOD_LABEL = "Turbo"
