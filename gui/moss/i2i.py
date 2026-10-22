"""The MOSS window's i2i tab: stage → preprocess → train → test for one i2i task.

Tasks are the ``library.moss.requests.I2I_CACHE`` rows whose
``configs/moss/i2i/<task>/cache.toml`` exists. Each form starts from the task's
``cache.toml`` / ``train.toml`` / ``generate.toml``, and only what differs from
that seed is saved (``moss_gui_i2i`` in ``gui_settings.json``). Stage is
``tasks.py i2i-stage`` with ``TASK`` in the job's env, a daemon command job like
the rest. Test renders with the LoRA only and shows each reference beside its
output.
"""

from __future__ import annotations

import dataclasses
import html
import json
import time
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.core.paths import get_setting, set_setting
from gui.lines.window import LineJob, LinePage, RequestForm, tr
from gui.theme import rich_text_pt, tok
from gui.widgets import ExplainPanel, action_button
from library.env import resolve_under_home
from library.moss.requests import (
    I2I_CACHE,
    ColorizeCacheRequest,
    GenerateRequest,
    TrainRequest,
    i2i_config,
)
from library.moss.scan import cache_counts

STAGE, CACHE, TRAIN, TEST = LABELS = (
    "moss-i2i-stage",
    "moss-i2i-cache",
    "moss-i2i-train",
    "moss-i2i-test",
)
SETTINGS_KEY = "moss_gui_i2i"
THUMB_PX = 300


@dataclasses.dataclass
class StageRequest:
    """``make i2i-stage`` as a daemon job; ``TASK`` rides in the env."""

    SCRIPT: ClassVar[str] = "tasks.py"

    def to_argv(self) -> list[str]:
        return ["i2i-stage"]


def tasks() -> list[str]:
    return [t for t in I2I_CACHE if i2i_config(t, "cache").is_file()]


def _seed(cls, task: str, kind: str) -> dict:
    path = i2i_config(task, kind)
    return cls.load_config(path) if path.is_file() else {}


def _staged(req) -> tuple[int | None, Path]:
    """Pairs the cache request would read, ``None`` while its source is missing."""
    if getattr(req, "pairs", ""):
        path = resolve_under_home(req.pairs)
        if not path.is_file():
            return None, path
        rows = path.read_text(encoding="utf-8").splitlines()
        return sum(1 for r in rows if r.strip()), path
    path = resolve_under_home(getattr(req, "staging", "") or req.src)
    if not path.is_dir():
        return None, path
    return sum(1 for _ in path.rglob("*.png")), path


def _records(run_dir: Path) -> list[dict]:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))["images"]


def _thumb(path: Path, caption: str) -> QWidget:
    cell = QWidget()
    v = QVBoxLayout(cell)
    v.setContentsMargins(0, 0, 0, 0)
    image = QLabel()
    image.setPixmap(
        QPixmap(str(path)).scaled(
            THUMB_PX, THUMB_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )
    image.setToolTip(str(path))
    label = QLabel(caption)
    label.setAlignment(Qt.AlignCenter)
    v.addWidget(image)
    v.addWidget(label)
    return cell


class _TaskView(QWidget):
    """One task's four steps; the page wires the buttons."""

    def __init__(self, task: str, help_cb, saved: dict):
        super().__init__()
        self.task = task
        self.cache_cls = I2I_CACHE[task]
        self.seeds = {
            "cache": _seed(self.cache_cls, task, "cache"),
            "train": _seed(TrainRequest, task, "train"),
            "test": _seed(GenerateRequest, task, "generate"),
        }
        self.forms = {
            "cache": RequestForm(self.cache_cls, help_cb),
            "train": RequestForm(TrainRequest, help_cb),
            "test": RequestForm(GenerateRequest, help_cb),
        }
        for kind, form in self.forms.items():
            form.set_values(self.seeds[kind])
            form.set_values(saved.get(kind, {}))

        v = QVBoxLayout(self)
        self.btn_stage = action_button(tr("run_stage"))
        v.addWidget(
            self._group(
                "i2i_stage_group",
                tr("i2i_stage_hint", task=task),
                None,
                None,
                [self.btn_stage],
            )
        )
        self.cache_summary = QLabel()
        self.btn_cache = action_button(tr("run_cache"))
        v.addWidget(
            self._group(
                "i2i_cache_group",
                None,
                self.forms["cache"],
                self.cache_summary,
                [self.btn_cache],
            )
        )
        self.train_summary = QLabel()
        self.btn_train = action_button(tr("run_train"))
        self.btn_chain = action_button(tr("i2i_chain"), variant="secondary")
        v.addWidget(
            self._group(
                "i2i_train_group",
                None,
                self.forms["train"],
                self.train_summary,
                [self.btn_train, self.btn_chain],
            )
        )
        self.btn_test = action_button(tr("run_test"), variant="info")
        test_box = self._group(
            "i2i_test_group",
            tr("i2i_test_hint"),
            self.forms["test"],
            None,
            [self.btn_test],
        )
        self.images = QGridLayout()
        test_box.layout().addLayout(self.images)
        v.addWidget(test_box)
        v.addStretch(1)

    def _group(self, title, hint, form, summary, buttons) -> QGroupBox:
        box = QGroupBox(tr(title))
        v = QVBoxLayout(box)
        if hint:
            label = QLabel(hint)
            label.setWordWrap(True)
            v.addWidget(label)
        if form is not None:
            v.addWidget(form)
        if summary is not None:
            summary.setWordWrap(True)
            v.addWidget(summary)
        row = QHBoxLayout()
        row.addStretch(1)
        for b in buttons:
            row.addWidget(b)
        v.addLayout(row)
        return box

    @property
    def buttons(self):
        return (
            self.btn_stage,
            self.btn_cache,
            self.btn_train,
            self.btn_chain,
            self.btn_test,
        )

    def state(self) -> dict:
        """Per form, the values that differ from the task's seed."""
        out = {}
        for kind, form in self.forms.items():
            req = form.request()
            if req is None:
                continue
            out[kind] = {
                f.name: getattr(req, f.name)
                for f in dataclasses.fields(req)
                if getattr(req, f.name) != self.seeds[kind].get(f.name, f.default)
            }
        return out

    def train_output(self) -> Path:
        return resolve_under_home(
            self.forms["train"].text_of("output") or TrainRequest().output
        )

    def show_test(self, run_dir: Path | None) -> None:
        while self.images.count():
            item = self.images.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for row, r in enumerate(_records(run_dir) if run_dir else []):
            if r.get("ref"):
                self.images.addWidget(_thumb(Path(r["ref"]), tr("test_ref")), row, 0)
            m = r.get("lora_multiplier")
            caption = tr("test_lora", m=f"{m:g}") if m else tr("test_base")
            self.images.addWidget(_thumb(Path(r["image"]), caption), row, 1)

    def latest_test_dir(self) -> Path | None:
        base = resolve_under_home(
            self.forms["test"].text_of("out_dir") or GenerateRequest().out_dir
        )
        runs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        return next((p for p in reversed(runs) if _records(p)), None)


class I2IPage(LinePage):
    labels = LABELS
    scripts = {ColorizeCacheRequest.SCRIPT: CACHE}

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.title = tr("tab_i2i")
        self.explain = ExplainPanel()
        self._test_view: _TaskView | None = None
        self._test_out: Path | None = None
        saved = get_setting(SETTINGS_KEY, {}) or {}

        self.task_combo = QComboBox()
        self.stack = QStackedWidget()
        self.views: list[_TaskView] = []
        for task in tasks():
            view = _TaskView(task, self.explain.show_field_help, saved.get(task, {}))
            view.btn_stage.clicked.connect(lambda _=False, w=view: self._run_stage(w))
            view.btn_cache.clicked.connect(lambda _=False, w=view: self._run_cache(w))
            view.btn_train.clicked.connect(lambda _=False, w=view: self._run_train(w))
            view.btn_chain.clicked.connect(lambda _=False, w=view: self._run_chain(w))
            view.btn_test.clicked.connect(lambda _=False, w=view: self._run_test(w))
            self.views.append(view)
            self.task_combo.addItem(task)
            self.stack.addWidget(view)
            view.show_test(view.latest_test_dir())

        inner = QWidget()
        v = QVBoxLayout(inner)
        top = QHBoxLayout()
        top.addWidget(QLabel(tr("i2i_task")))
        top.addWidget(self.task_combo, 1)
        v.addLayout(top)
        if not self.views:
            v.addWidget(QLabel(tr("i2i_no_tasks")))
        v.addWidget(self.stack, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll)
        split.addWidget(self.explain)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([620, 380])
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(split)

        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(300)
        self._rescan_timer.timeout.connect(self._rescan)
        for view in self.views:
            for form in view.forms.values():
                form.watch(self._rescan_timer.start)
        names = [w.task for w in self.views]
        if saved.get("task") in names:
            self.task_combo.setCurrentIndex(names.index(saved["task"]))
        self.task_combo.currentIndexChanged.connect(self._task_changed)
        self.stack.setCurrentIndex(self.task_combo.currentIndex())
        self._rescan()

    # ── state ────────────────────────────────────────────────────────────

    @property
    def view(self) -> _TaskView | None:
        return self.views[self.stack.currentIndex()] if self.views else None

    def _task_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.explain.show_guide(None)
        self._rescan()

    def _guide_html(self, task: str) -> str:
        dim = tok("text_dim")
        return (
            f"<h2 style='margin:0 0 10px 0; font-size:{rich_text_pt(18)};'>"
            f"{html.escape(tr('tab_i2i'))} · {html.escape(task)}</h2>"
            f"<p style='font-size:{rich_text_pt(15)}; line-height:1.6;'>"
            f"{html.escape(tr('i2i_task_' + task))}</p>"
            f"<p style='line-height:1.6;'>{html.escape(tr('i2i_guide'))}</p>"
            f"<p style='color:{dim}; font-style:italic;'>"
            f"{html.escape(tr('click_field'))}</p>"
        )

    def _rescan(self) -> None:
        view = self.view
        if view is None:
            return
        if self.explain.mode is None:
            self.explain.show_guide(self._guide_html(view.task))
        lines = []
        req = view.forms["cache"].request()
        if req is not None:
            n, path = _staged(req)
            lines.append(
                tr("i2i_staged", n=n) if n else tr("i2i_not_staged", path=path)
            )
            c = cache_counts(resolve_under_home(req.out))
            lines.append(
                tr("cache_counts", samples=c.samples, text=c.text, latents=c.latents)
            )
        view.cache_summary.setText("\n".join(lines))
        cache = view.forms["train"].text_of("cache")
        n = cache_counts(resolve_under_home(cache)).samples if cache else 0
        view.train_summary.setText(tr("train_ready", n=n) if n else tr("train_empty"))
        view.forms["test"].widgets["lora"].setPlaceholderText(
            tr("test_lora_placeholder", path=view.train_output())
        )

    def persist(self) -> None:
        data = {w.task: w.state() for w in self.views}
        if self.view is not None:
            data["task"] = self.view.task
        set_setting(SETTINGS_KEY, data)

    def set_busy(self, busy: bool) -> None:
        for view in self.views:
            for b in view.buttons:
                b.setEnabled(not busy)
        if not busy:
            self._rescan()

    def job_finished(self, label: str, ok: bool) -> None:
        if label == TEST and ok and self._test_view is not None:
            view = self._test_view
            view.show_test(self._test_out or view.latest_test_dir())
        if label == TEST:
            self._test_view = self._test_out = None

    # ── jobs ─────────────────────────────────────────────────────────────

    def _stage_job(self, view: _TaskView) -> LineJob:
        return LineJob(StageRequest(), STAGE, env={"TASK": view.task})

    def _warn(self, key: str, **kwargs) -> None:
        QMessageBox.warning(self, tr("window_title"), tr(key, **kwargs))

    def _run_stage(self, view: _TaskView) -> None:
        self.persist()
        self.panel.submit([self._stage_job(view)])

    def _cache_req(self, view: _TaskView, staged: bool):
        req = view.forms["cache"].request()
        if req is None or not self.panel._model_ok(req):
            return None
        if staged:
            n, path = _staged(req)
            if not n:
                self._warn("i2i_not_staged", path=path)
                return None
        return req

    def _run_cache(self, view: _TaskView) -> None:
        req = self._cache_req(view, staged=True)
        if req is None:
            return
        self.persist()
        self.panel.submit([LineJob(req, CACHE)])

    def _run_train(self, view: _TaskView) -> None:
        req = view.forms["train"].request()
        if req is None or not self.panel._model_ok(req):
            return
        self.persist()
        self.panel.submit([LineJob(req, TRAIN)])

    def _run_chain(self, view: _TaskView) -> None:
        cache_req = self._cache_req(view, staged=False)
        if cache_req is None:
            return
        # Training reads what this preprocessing run writes.
        view.forms["train"].set_values({"cache": cache_req.out})
        train_req = view.forms["train"].request()
        if train_req is None or not self.panel._model_ok(train_req):
            return
        self.persist()
        self.panel.submit(
            [
                self._stage_job(view),
                LineJob(cache_req, CACHE, note=tr("i2i_stage_next")),
                LineJob(train_req, TRAIN, note=tr("chain_next")),
            ]
        )

    def _run_test(self, view: _TaskView) -> None:
        req = view.forms["test"].request()
        if req is None or not self.panel._model_ok(req):
            return
        if not req.ref_image or not resolve_under_home(req.ref_image).exists():
            self._warn("i2i_need_ref")
            return
        if not req.ref_vision and not (req.prompt or req.prompts_file):
            self._warn("i2i_need_prompt")
            return
        lora = resolve_under_home(req.lora) if req.lora else view.train_output()
        if not lora.is_file():
            self._warn("no_lora", path=lora)
            return
        self.persist()
        # One folder per run, so earlier tests stay on disk.
        out = resolve_under_home(req.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        req = dataclasses.replace(req, lora=str(lora), out_dir=str(out))
        if self.panel.submit([LineJob(req, TEST)]):
            self._test_view, self._test_out = view, out
