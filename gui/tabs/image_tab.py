"""ImageViewerTab — dataset image browser with caption editor + history."""

from __future__ import annotations

import json
import shutil
import sys
import threading
from html import escape
from pathlib import Path

from PySide6.QtCore import (
    QElapsedTimer,
    QProcess,
    QProcessEnvironment,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QDesktopServices,
    QFontDatabase,
    QImageReader,
    QKeySequence,
    QPixmap,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QToolButton,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import (
    DEFAULT_AUTOTAG_CONFIDENCE,
    DEFAULT_GROUP_CELL_MATCH_MIN,
    DEFAULT_GROUP_MATCH_FRAC_MIN,
    ROOT,
    LazyTabMixin,
    ScaledImageLabel,
    _image_dirs,
    _imgs,
    get_setting,
)
from gui import daemon as gui_daemon
from gui._paths import (
    DEFAULT_CAPTION_INSERT_NO_ARTIST,
    DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS,
    IMAGE_EXTS,
)
from gui.config_io import default_resized_dir
from gui._job_mixin import DaemonJobMixin
from gui.i18n import current_language, t
from gui.progress import TqdmProgressTracker, make_progress_bar
from gui.tabs._caption_editor import (
    BoxedCaptionEdit,
    CaptionVersionsDialog,
    _add_format,
    _append_history,
    _diff_spans,
)
from gui.tabs._image_overlays import (
    _DecodeSignals,
    _DecodeTask,
    _compose_mask_overlay,
    _compose_resize_preview_overlay,
    _format_file_size,
    _load_preprocess_toml_data,
    _load_resize_preview_target_res,
    _resolve_mask_path,
)
from gui.theme import tok
from gui.widgets import apply_variant
from library.captioning.correction import (
    CaptionCorrectionOptions,
    TagKnowledgeBase,
    correct_caption,
    default_tag_csv_candidates,
    find_tag_csv,
    load_tag_knowledge_base,
)
from library.datasets.curation_actions import (
    load_curation_decisions,
    move_linked_files,
    rel_key,
    save_curation_decisions,
)
from library.preprocess.caption_variants import (
    read_variants_sidecar,
    variants_sidecar_path,
)
from library.preprocess.resize_preview import (
    DEFAULT_FREEFIT_MAX_RATIO,
    compute_resize_preview,
)

# Stdio protocol sentinels of the resident autotag worker (kept in sync with
# ``scripts/anima_tagger/autotag_server.py``). Hardcoded rather than imported
# because that module pulls in torch, which the GUI must stay free of.
_AUTOTAG_READY = "ANIMA_AUTOTAG_READY"
_AUTOTAG_RESULT_PREFIX = "ANIMA_AUTOTAG_RESULT\t"
_AUTOTAG_ERROR_PREFIX = "ANIMA_AUTOTAG_ERROR\t"

# Free the resident tagger (VRAM) after this many ms with no autotag request.
_AUTOTAG_IDLE_MS = 10 * 60 * 1000
# Poll cadence (ms) for "did some other GPU job start?" while resident.
_AUTOTAG_GPU_WATCH_MS = 700
# Debounce (ms) before loading the selected image. Holding an arrow key
# auto-repeats currentItemChanged many times/sec; each _show() does a full-res
# QPixmap decode + caption/meta refresh, so loading every intermediate leaf
# pegs the UI. Coalesce: only load once selection settles for this long.
_NAV_DEBOUNCE_MS = 90
# How many neighbours on each side of the current leaf to decode ahead of time
# (so up/down lands on an already-decoded pixmap). Window = 2*_PREFETCH_RADIUS.
_PREFETCH_RADIUS = 2
# Max decoded pixmaps held in the prefetch cache (current + neighbours + a little
# history). Full-res manga frames are a few MB each as QPixmap; keep it bounded.
_PM_CACHE_MAX = 12


# Text prefixes for GUI preprocess decisions and images marked for moving.
_USE_MARK_PREFIX = "■ "
_SKIP_MARK_PREFIX = "■ "
_MOVE_MARK_PREFIX = "■ "
_TREE_BASE_TEXT_ROLE = Qt.UserRole + 1


class ImageViewerTab(DaemonJobMixin, LazyTabMixin, QWidget):
    # Carries the (names, kind_lookup) tag-completion payload from the
    # background loader thread back to the GUI thread (queued delivery).
    _completion_ready = Signal(object)

    def __init__(self, preprocess_tab=None):
        super().__init__()
        # Daemon job observer so curate-group's progress bar lives in this tab.
        self._init_job_observer()
        self._preprocess_tab = preprocess_tab
        self._all_images: list[Path] = []  # unfiltered, alphabetical (from _imgs)
        self._images: list[Path] = []  # currently displayed (filter + sort applied)
        self._dirs = _image_dirs()
        self._current_dir: Path | None = (
            None  # base of the loaded directory (for relative labels)
        )
        self._current_caption_path: Path | None = None
        self._disk_text: str = ""  # last value seen on disk (for diff baseline)
        self._suspend_dirty = False  # while we set text programmatically
        # Resident autotag worker: a torch QProcess holding the tagger model so
        # consecutive clicks skip the reload; torn down before any other GPU
        # work frees the card. See _run_autotag / _kill_tagger_worker.
        self._tagger_proc: QProcess | None = None
        self._tagger_ready = False
        self._tagger_buf = ""  # partial-line buffer for the worker's stdout
        self._autotag_inflight_image: Path | None = None  # image awaiting a result
        self._autotag_idle = QElapsedTimer()
        # Polls "did another GPU job start?" only while the worker is resident.
        self._gpu_watch_timer = QTimer(self)
        self._gpu_watch_timer.setInterval(_AUTOTAG_GPU_WATCH_MS)
        self._gpu_watch_timer.timeout.connect(self._autotag_gpu_watch_tick)
        # Coalesces rapid tree selection (arrow-key auto-repeat) into a single
        # image load once navigation settles. See _NAV_DEBOUNCE_MS.
        self._pending_show_idx: int | None = None
        self._nav_timer = QTimer(self)
        self._nav_timer.setSingleShot(True)
        self._nav_timer.setInterval(_NAV_DEBOUNCE_MS)
        self._nav_timer.timeout.connect(self._show_pending)
        # Neighbour prefetch: decode ±_PREFETCH_RADIUS images off-thread into a
        # bounded pixmap cache so the next up/down shows instantly. Insertion
        # order = LRU-ish; oldest evicted past _PM_CACHE_MAX.
        self._pm_cache: dict[str, QPixmap] = {}
        self._decode_inflight: set[str] = set()
        self._decode_signals = _DecodeSignals()
        self._decode_signals.done.connect(self._on_image_decoded)
        self._decode_pool = QThreadPool(self)
        self._decode_pool.setMaxThreadCount(2)
        self._caption_kb: TagKnowledgeBase | None = None
        self._caption_kb_source: Path | None = None
        self._caption_kb_mtime: float | None = None
        # Per-language description KB cache (path -> (kb, mtime)), used by the
        # tag-explanation tooltip so a translated sibling CSV doesn't clobber the
        # base KB feeding autocomplete / caption-correct.
        self._desc_kb_cache: dict[Path, tuple[TagKnowledgeBase, float]] = {}
        # Make sure the resident worker (and its VRAM) dies with the app.
        _app = QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._kill_tagger_worker)
        self._search_text: str = ""
        self._sort_desc: bool = False
        self._group_sort_mode: str = "name"
        self._group_sort_desc: bool = False
        self._image_size_cache: dict[Path, tuple[int, int]] = {}
        # Group-first: float every similarity group to the top, flattened across
        # folders. Off = per-folder tree. See _rebuild_tree_group_first.
        self._group_first: bool = False
        # Similarity-group manifest (make curate-group). Stem-keyed so it works
        # whether this tab views image_dataset/ or post_image_dataset/resized/.
        self._groups: list[dict] = []
        # Images marked for moving (Delete key toggles the current one; the move
        # button moves the whole set to post_image_dataset/moved). Keyed by full
        # path so a mark survives filter/sort/view rebuilds; cleared on dir change.
        self._marked: set[Path] = set()
        # GUI curation decisions consumed by the preprocess resize step. These
        # never move/edit source files; they only write a JSON sidecar that
        # resize_images.py reads when present.
        self._preprocess_decisions: dict[Path, str] = {}
        # _overlay_pm is lazily composed on first toggle and cached so flipping
        # the checkbox doesn't re-run the QPainter pipeline.
        self._source_pm: QPixmap | None = None
        self._mask_path: Path | None = None
        self._overlay_pm: QPixmap | None = None
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(t("directory")))
        self.dc = QComboBox()
        self.dc.addItems(self._dirs)
        self.dc.currentTextChanged.connect(self._load_dir)
        top.addWidget(self.dc, 1)
        self.reload_btn = QPushButton("↻")
        self.reload_btn.setMinimumWidth(32)
        self.reload_btn.setToolTip(t("dataset_reload_tooltip"))
        self.reload_btn.clicked.connect(self._reload_current_dir)
        top.addWidget(self.reload_btn)
        self.open_dir_btn = QPushButton(t("dataset_open_dir"))
        self.open_dir_btn.setToolTip(t("dataset_open_dir_tooltip"))
        self.open_dir_btn.clicked.connect(self._open_current_dir)
        top.addWidget(self.open_dir_btn)
        self.group_btn = QPushButton(t("dataset_group_rebuild"))
        self.group_btn.setToolTip(t("dataset_group_rebuild_tooltip"))
        apply_variant(self.group_btn, "info")
        self.group_btn.clicked.connect(self._rebuild_groups)
        top.addWidget(self.group_btn)
        self.add_dir_btn = QPushButton(t("dataset_add_dir"))
        self.add_dir_btn.setToolTip(t("dataset_add_dir_tooltip"))
        self.add_dir_btn.clicked.connect(self._add_dir)
        top.addWidget(self.add_dir_btn)
        self.cnt = QLabel()
        top.addWidget(self.cnt)
        lay.addLayout(top)

        self.group_progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.group_progress)
        lay.addWidget(self.group_progress)

        sp = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText(t("dataset_search_placeholder"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search, 1)
        self.sort_btn = QPushButton("a-z")
        self.sort_btn.setMinimumWidth(48)
        self.sort_btn.setToolTip(t("dataset_sort_asc_tooltip"))
        self.sort_btn.clicked.connect(self._toggle_sort)
        search_row.addWidget(self.sort_btn)
        self.group_first_btn = QPushButton(t("dataset_view_tree"))
        self.group_first_btn.setMinimumWidth(56)
        self.group_first_btn.setCheckable(True)
        self.group_first_btn.setToolTip(t("dataset_group_first_tooltip"))
        self.group_first_btn.clicked.connect(self._toggle_group_first)
        search_row.addWidget(self.group_first_btn)
        self.group_sort_combo = QComboBox()
        self.group_sort_combo.setToolTip(t("dataset_group_sort_tooltip"))
        self.group_sort_combo.addItem(t("dataset_group_sort_name"), "name")
        self.group_sort_combo.addItem(t("dataset_group_sort_name_desc"), "name_desc")
        self.group_sort_combo.addItem(t("dataset_group_sort_size"), "size")
        self.group_sort_combo.addItem(t("dataset_group_sort_size_desc"), "size_desc")
        self.group_sort_combo.addItem(t("dataset_group_sort_resolution"), "resolution")
        self.group_sort_combo.addItem(
            t("dataset_group_sort_resolution_desc"), "resolution_desc"
        )
        self.group_sort_combo.currentIndexChanged.connect(self._on_group_sort_changed)
        search_row.addWidget(self.group_sort_combo)
        ll.addLayout(search_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._on_tree_item_changed)
        self._tree_item_to_index: dict[QTreeWidgetItem, int] = {}
        ll.addWidget(self.tree, 1)
        sp.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        # Mask-overlay toggle. Checked state persists across navigation as a
        # sticky "show overlay when available" preference.
        img_head = QHBoxLayout()
        img_head.setContentsMargins(0, 0, 0, 0)
        self.overlay_cb = QCheckBox(t("dataset_mask_overlay"))
        self.overlay_cb.setEnabled(False)
        self.overlay_cb.toggled.connect(self._on_overlay_toggled)
        img_head.addWidget(self.overlay_cb)
        self.resize_preview_cb = QCheckBox(t("dataset_resize_preview"))
        self.resize_preview_cb.setToolTip(t("dataset_resize_preview_tooltip"))
        self.resize_preview_cb.setEnabled(False)
        self.resize_preview_cb.toggled.connect(self._on_overlay_toggled)
        img_head.addWidget(self.resize_preview_cb)
        # No explicit "Use" button — an image with no decision is already
        # processed (preprocess only honours skip/move), so "use" was a no-op
        # marker. Skip / Clear / Move cover the real states.
        self.preprocess_skip_btn = QPushButton(t("dataset_preprocess_skip_short"))
        self.preprocess_skip_btn.setToolTip(t("dataset_preprocess_skip_tooltip"))
        self.preprocess_skip_btn.clicked.connect(
            lambda: self._set_current_preprocess_decision("skip", advance=True)
        )
        img_head.addWidget(self.preprocess_skip_btn)
        self.preprocess_clear_btn = self._make_button_with_menu(
            t("dataset_preprocess_clear_short"),
            t("dataset_preprocess_clear_tooltip"),
            self._clear_current_preprocess_decision,
            [(t("dataset_preprocess_clear_all"), self._clear_all_decisions)],
        )
        img_head.addWidget(self.preprocess_clear_btn)
        self.preprocess_save_btn = QPushButton(t("dataset_preprocess_save"))
        self.preprocess_save_btn.setToolTip(t("dataset_preprocess_save_tooltip"))
        self.preprocess_save_btn.clicked.connect(self._save_preprocess_decisions)
        img_head.addWidget(self.preprocess_save_btn)
        # Move button: moves the images marked by the Delete key into
        # post_image_dataset/moved/. This replaces the old trash-delete action.
        self.delete_btn = QPushButton(t("dataset_delete"))
        self.delete_btn.setToolTip(t("dataset_delete_tooltip"))
        apply_variant(self.delete_btn, "info")
        self.delete_btn.clicked.connect(self._delete_marked)
        img_head.addWidget(self.delete_btn)
        img_head.addStretch()
        rl.addLayout(img_head)

        self.img = ScaledImageLabel()
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setMinimumSize(400, 400)
        rl.addWidget(self.img, 1)

        self.image_meta = QLabel(t("dataset_image_meta_empty"))
        self.image_meta.setTextFormat(Qt.RichText)
        self.image_meta.setMinimumWidth(360)
        self.image_meta.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.image_meta.setStyleSheet(
            f"QLabel {{ color:{tok('text_dim')}; padding:2px 0; }}"
        )
        rl.addWidget(self.image_meta)

        cap_head = QHBoxLayout()
        self.cap_label = QLabel(t("caption"))
        cap_head.addWidget(self.cap_label)
        # Resident-tagger status, updated from the worker's stdout sentinels.
        self.autotag_status = QLabel()
        self.autotag_status.setStyleSheet(
            f"QLabel{{color:{tok('link')};font-size:11px;}}"
        )
        self.autotag_status.setVisible(False)
        cap_head.addWidget(self.autotag_status)
        cap_head.addStretch()
        self.save_btn = QPushButton(t("caption_save"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.revert_btn = QPushButton(t("caption_revert"))
        self.revert_btn.setEnabled(False)
        self.revert_btn.clicked.connect(self._revert)
        self.autotag_btn = QPushButton(t("caption_autotag"))
        self.autotag_btn.setToolTip(t("caption_autotag_tooltip"))
        apply_variant(self.autotag_btn, "info")
        self.autotag_btn.clicked.connect(self._run_autotag)
        self.caption_correct_btn = self._make_button_with_menu(
            t("caption_correct"),
            t("caption_correct_tooltip"),
            self._correct_current_caption,
            [(t("caption_correct_visible"), self._correct_visible_captions)],
            variant="info",
        )
        self.versions_btn = QPushButton(t("caption_versions"))
        self.versions_btn.clicked.connect(self._open_versions)
        # Read-only preview of the train-time caption variants written by
        # preprocess into {stem}.variants.txt — selecting one shows it without
        # touching the editable training caption. Hidden unless a sidecar exists.
        self.variant_combo = QComboBox()
        self.variant_combo.setToolTip(t("caption_variants_tooltip"))
        self.variant_combo.setMaximumWidth(220)
        self.variant_combo.setVisible(False)
        self.variant_combo.currentIndexChanged.connect(self._on_variant_selected)
        self._variant_rows: list[tuple[str, str]] = []
        self._previewing_variant = False
        self._preview_stash = ""
        cap_head.addWidget(self.save_btn)
        cap_head.addWidget(self.revert_btn)
        cap_head.addWidget(self.autotag_btn)
        cap_head.addWidget(self.caption_correct_btn)
        cap_head.addWidget(self.versions_btn)
        cap_head.addWidget(self.variant_combo)
        rl.addLayout(cap_head)

        # Caption editor with inline tag-box overlay; @artist and section
        # headers use accent colors so the trainer's split rules
        # (anima_smart_shuffle in library/anima/training.py) stay visible.
        self.cap = BoxedCaptionEdit()
        self.cap.setMaximumHeight(180)
        self.cap.textChanged.connect(self._on_text_changed)
        self.cap.tag_clicked.connect(self._on_tag_clicked)
        self._completion_ready.connect(self._on_completion_ready)
        self._start_tag_completion_preload()
        rl.addWidget(self.cap)

        # One-line grammar reminder, mirrors anima_smart_shuffle's split rules.
        self.guide = QLabel(t("caption_guideline_html"))
        self.guide.setWordWrap(True)
        self.guide.setTextFormat(Qt.RichText)
        self.guide.setStyleSheet(
            f"QLabel {{ color:{tok('text_dim')}; font-size:11px; padding:2px 4px; }}"
        )
        rl.addWidget(self.guide)

        sp.addWidget(right)
        sp.setSizes([340, 700])
        # On window-resize, the file-tree pane keeps its width and the extra
        # space flows to the image/caption pane (where the image is the
        # Expanding element) rather than growing both proportionally.
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        lay.addWidget(sp, 1)

        QShortcut(QKeySequence("Right"), self, lambda: self._nav(1))
        QShortcut(QKeySequence("Left"), self, lambda: self._nav(-1))
        QShortcut(QKeySequence.Save, self, self._save)
        # Delete toggles the move mark on the current image; Esc un-marks it.
        # Both act per-current-image and are scoped to the tree (WidgetShortcut)
        # so they don't hijack the caption editor on focus.
        for target in (self.tree, self.img):
            move_sc = QShortcut(QKeySequence("D"), target, self._mark_current_for_move)
            move_sc.setContext(Qt.WidgetShortcut)
        _del = QShortcut(QKeySequence.Delete, self.tree, self._toggle_mark_current)
        _del.setContext(Qt.WidgetShortcut)
        _esc = QShortcut(QKeySequence(Qt.Key_Escape), self.tree, self._unmark_current)
        _esc.setContext(Qt.WidgetShortcut)
        for target in (self.tree, self.img):
            skip_sc = QShortcut(
                QKeySequence("S"),
                target,
                lambda: self._set_current_preprocess_decision("skip", advance=True),
            )
            skip_sc.setContext(Qt.WidgetShortcut)
            clear_sc = QShortcut(
                QKeySequence("F"),
                target,
                self._clear_current_preprocess_decision,
            )
            clear_sc.setContext(Qt.WidgetShortcut)
        self._refresh_delete_button()
        self._refresh_preprocess_controls()

    def _lazy_init(self) -> None:
        if self._dirs:
            self._load_dir(self.dc.currentText())

    def _make_button_with_menu(
        self,
        text: str,
        tooltip: str,
        clicked_cb,
        actions,
        *,
        variant: str | None = None,
    ) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        main_btn = QPushButton(text)
        main_btn.setToolTip(tooltip)
        if variant is not None:
            apply_variant(main_btn, variant)
        main_btn.clicked.connect(lambda _checked=False: clicked_cb())
        row.addWidget(main_btn)

        menu_btn = QToolButton()
        menu_btn.setToolTip(tooltip)
        menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu_btn.setFixedWidth(24)
        menu_btn.setFixedHeight(main_btn.sizeHint().height())
        menu_btn.setStyleSheet(
            f"""
            QToolButton {{
                background:{tok("surface")};
                color:{tok("text")};
                border:1px solid {tok("border")};
                border-left:none;
                border-top-right-radius:3px;
                border-bottom-right-radius:3px;
                padding:0;
            }}
            QToolButton:hover {{ background:{tok("surface_hover")}; }}
            QToolButton:disabled {{ color:{tok("text_dim")}; }}
            """
        )
        menu = QMenu(menu_btn)
        for label, cb in actions:
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, cb=cb: cb())
        menu_btn.setMenu(menu)
        row.addWidget(menu_btn)
        return host

    def _open_current_dir(self):
        """Open the currently loaded dataset directory in the OS file manager."""
        if self._current_dir is None or not self._current_dir.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_dir)))

    def _groups_manifest_path(self) -> Path:
        return ROOT / "post_image_dataset" / "groups" / "groups.json"

    def _load_groups(self) -> None:
        """Read groups.json (if present) into ``self._groups``.

        Pure JSON — keeps the GUI torch-free. The tree view folds images under
        green per-group nodes built from this list (ordered as written, largest
        first). A missing/unreadable manifest just leaves the plain folder tree.
        """
        self._groups = []
        path = self._groups_manifest_path()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                groups = data.get("groups", [])
                if isinstance(groups, list):
                    self._groups = groups
            except (json.JSONDecodeError, OSError):
                self._groups = []

    def _rebuild_groups(self) -> None:
        """Submit `make curate-group` and observe it in-tab (progress bar here)."""
        if self._job_id:  # a grouping run is already attached
            QMessageBox.information(
                self, "", t("dataset_group_queued", job_id=self._job_id)
            )
            return
        # Grouping keys off PE-Spatial features of the resized images; with none
        # on disk the task exits with an opaque "no images to group". Point the
        # user at the real cause (run Preprocess first) instead.
        resized_dir = default_resized_dir()
        has_resized = resized_dir.is_dir() and any(
            p.suffix.lower() in IMAGE_EXTS for p in resized_dir.rglob("*")
        )
        if not has_resized:
            QMessageBox.warning(self, t("error"), t("preprocess_no_resized_to_process"))
            return
        # Grouping is GPU work — free the resident tagger first so they don't
        # fight over VRAM.
        self._kill_tagger_worker()
        # Busy UI before the submit so a cold-start daemon spin-up feels responsive.
        self.group_btn.setEnabled(False)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("dataset_group_rebuild"))
        # Grouping tightness is tuned from Settings (higher = tighter); pass the
        # persisted thresholds through to `tasks.py curate-group`.
        frac = float(get_setting("group_match_frac_min", DEFAULT_GROUP_MATCH_FRAC_MIN))
        cell = float(get_setting("group_cell_match_min", DEFAULT_GROUP_CELL_MATCH_MIN))
        argv = [
            "tasks.py",
            "curate-group",
            "--match-frac-min",
            f"{frac:g}",
            "--cell-match-min",
            f"{cell:g}",
        ]
        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(
                label="curate-group", argv=argv, start=True
            ),
            on_fail=self._restore_group_idle_ui,
        )
        if not job_id:
            return
        self._watch_job(job_id, replay_log=False)

    def _emit_log_line(self, line: str) -> None:
        """No log widget on this tab — non-progress stdout lines are dropped.

        The progress bar (tqdm) and the finish banner carry the user-facing
        signal; a full log belongs to the Queue tab.
        """

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        self._drain_job_stdout()
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self._progress_tracker.reset()
        self._restore_group_idle_ui()
        if gui_daemon.is_success(state):
            prev = (
                self._current_caption_path.stem
                if self._current_caption_path is not None
                else None
            )
            self._load_groups()
            self._apply_filter_and_sort(prev_stem=prev)
        else:
            QMessageBox.warning(
                self, t("error"), gui_daemon.format_finish_banner(job_id, state)
            )

    def _restore_group_idle_ui(self) -> None:
        self.group_btn.setEnabled(True)

    def _run_autotag(self) -> None:
        """Tag the current image with the resident Anima Tagger.

        First click spawns a torch subprocess that loads the model once; it then
        stays alive so later clicks just stream an image path to it. The
        predicted tags are appended into the editor when the result comes back —
        the user reviews, then Save writes the ``.txt`` (creating it if absent).
        The worker is freed before any other GPU work (see _kill_tagger_worker).
        """
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        # Don't grab the card while a daemon job (train/preprocess/group) holds
        # it — those take priority; tagging can wait until it's idle.
        if gui_daemon.active_job_id():
            QMessageBox.information(self, "", t("caption_autotag_busy"))
            return
        if self._autotag_inflight_image is not None:
            return  # a request is already in flight; ignore the double-click
        image_path = self._images[idx]
        self._autotag_inflight_image = image_path
        self._autotag_idle.restart()
        self.autotag_btn.setEnabled(False)
        if self._tagger_proc is None:
            self._spawn_tagger_worker()
            self._set_autotag_status(t("caption_autotag_loading"))
        elif self._tagger_ready:
            self._set_autotag_status(t("caption_autotag_running"))
            self._send_autotag_request(image_path)
        # else: worker still loading — _on_tagger_stdout sends it on READY.

    def _spawn_tagger_worker(self) -> None:
        """Launch the resident worker subprocess (torch lives here, not the GUI)."""
        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(["-m", "scripts.anima_tagger.autotag_server"])
        proc.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # stream sentinel lines live
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_tagger_stdout)
        proc.finished.connect(self._on_tagger_finished)
        proc.errorOccurred.connect(lambda _e: self._on_tagger_finished(-1, None))
        self._tagger_proc = proc
        self._tagger_ready = False
        self._tagger_buf = ""
        self.autotag_status.setVisible(True)
        self._gpu_watch_timer.start()
        proc.start()

    def _send_autotag_request(self, image_path: Path) -> None:
        if self._tagger_proc is None:
            return
        # Read the confidence floor fresh each request so a settings change
        # applies without respawning the resident worker.
        try:
            conf = float(get_setting("autotag_confidence", DEFAULT_AUTOTAG_CONFIDENCE))
        except (TypeError, ValueError):
            conf = DEFAULT_AUTOTAG_CONFIDENCE
        conf = max(0.0, min(1.0, conf))
        self._tagger_proc.write(f"{conf}\t{image_path}\n".encode("utf-8"))

    def _on_tagger_stdout(self) -> None:
        if self._tagger_proc is None:
            return
        self._tagger_buf += bytes(self._tagger_proc.readAllStandardOutput()).decode(
            "utf-8", "replace"
        )
        *lines, self._tagger_buf = self._tagger_buf.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            if line == _AUTOTAG_READY:
                self._tagger_ready = True
                self._set_autotag_status(t("caption_autotag_ready"))
                # Send the click that spawned the worker, now that it's loaded.
                if self._autotag_inflight_image is not None:
                    self._set_autotag_status(t("caption_autotag_running"))
                    self._send_autotag_request(self._autotag_inflight_image)
            elif line.startswith(_AUTOTAG_RESULT_PREFIX):
                self._apply_autotag_result(line[len(_AUTOTAG_RESULT_PREFIX) :])
            elif line.startswith(_AUTOTAG_ERROR_PREFIX):
                self._finish_autotag_request()
                QMessageBox.warning(
                    self,
                    t("error"),
                    t("caption_autotag_error", err=line[len(_AUTOTAG_ERROR_PREFIX) :]),
                )

    def _apply_autotag_result(self, caption: str) -> None:
        """Append the predicted caption into the editor (dirty → Save lights up)."""
        image = self._autotag_inflight_image
        self._finish_autotag_request()
        caption = caption.strip()
        if not caption:
            QMessageBox.information(self, "", t("caption_autotag_empty"))
            return
        # The user may have navigated away while the worker ran — only apply the
        # result if it still belongs to the caption currently on screen.
        if image is None or self._current_caption_path != image.with_suffix(".txt"):
            return
        existing = self.cap.toPlainText().strip()
        if existing:
            combined = existing.rstrip().rstrip(",").rstrip() + ", " + caption
        else:
            combined = caption
        # Refresh manually: the suspend-dirty guard swallows the textChanged
        # signal, so diff highlight + dirty state wouldn't update otherwise.
        self._set_caption_text(combined)
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _caption_correction_options(self) -> CaptionCorrectionOptions:
        return CaptionCorrectionOptions(
            insert_no_artist=bool(
                get_setting(
                    "caption_insert_no_artist", DEFAULT_CAPTION_INSERT_NO_ARTIST
                )
            ),
            validate_artist_tags=bool(
                get_setting(
                    "caption_validate_artist_tags",
                    DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS,
                )
            ),
        )

    def _load_caption_kb(self, *, warn: bool = True) -> TagKnowledgeBase | None:
        csv_path = find_tag_csv(ROOT)
        if csv_path is None:
            if warn:
                candidates = "\n".join(
                    f"  {path}" for path in default_tag_csv_candidates(ROOT)
                )
                QMessageBox.warning(
                    self,
                    t("error"),
                    t("caption_correct_db_missing", paths=candidates),
                )
            return None
        try:
            mtime = csv_path.stat().st_mtime
        except OSError as exc:
            if warn:
                QMessageBox.warning(
                    self, t("error"), t("caption_correct_db_failed", err=str(exc))
                )
            return None
        if (
            self._caption_kb is not None
            and self._caption_kb_source == csv_path
            and self._caption_kb_mtime == mtime
        ):
            return self._caption_kb
        try:
            self._caption_kb = load_tag_knowledge_base(csv_path)
            self._caption_kb_source = csv_path
            self._caption_kb_mtime = mtime
        except (OSError, ValueError) as exc:
            if warn:
                QMessageBox.warning(
                    self, t("error"), t("caption_correct_db_failed", err=str(exc))
                )
            self._caption_kb = None
            self._caption_kb_source = None
            self._caption_kb_mtime = None
        return self._caption_kb

    def _describe_kb(self, csv_path: Path) -> TagKnowledgeBase | None:
        """Load (and mtime-cache) a description KB for the tag-click tooltip."""
        try:
            mtime = csv_path.stat().st_mtime
        except OSError:
            return None
        cached = self._desc_kb_cache.get(csv_path)
        if cached is not None and cached[1] == mtime:
            return cached[0]
        try:
            kb = load_tag_knowledge_base(csv_path)
        except (OSError, ValueError):
            return None
        self._desc_kb_cache[csv_path] = (kb, mtime)
        return kb

    def _on_tag_clicked(self, tag: str) -> None:
        """Show the clicked tag's KB entry as a rich tooltip at the cursor.

        The base tag KB (``danbooru_tags_classified.csv``) carries Korean
        descriptions. A non-Korean UI resolves to a same-language sibling if one
        exists, else the English ``danbooru_tags_classified.en.csv`` (shipped by
        ``download-danbooru-tags``) — so ja/cn show English rather than
        untranslated Hangul. We only suppress the tooltip when the sole file is
        the Korean base. The autocomplete helper stays on for every language
        (tag names / categories are language-neutral). See CONTRIBUTING.md §5."""
        lang = current_language()
        if lang == "ko":
            kb = self._load_caption_kb(warn=False)
        else:
            csv_path = find_tag_csv(ROOT, lang=lang)
            # Suppress only when the resolved file is the Korean base (no
            # translated / English sibling on disk) — that would show Hangul.
            if csv_path is None or csv_path.name == "danbooru_tags_classified.csv":
                return
            kb = self._describe_kb(csv_path)
        info = kb.describe(tag) if kb is not None else None
        if info is None:
            QToolTip.showText(QCursor.pos(), t("tag_kb_unknown", tag=tag), self.cap)
            return
        head = f"<b>{escape(info.name)}</b> &middot; {escape(info.kind)}"
        if info.post_count:
            head += " &middot; " + escape(t("tag_kb_posts", n=f"{info.post_count:,}"))
        parts = [head]
        if info.category_path:
            parts.append(
                f"<span style='color:#8aa9c0;'>[{escape(info.category_path)}]</span>"
            )
        if info.description:
            parts.append(escape(info.description))
        html = "<div style='max-width:360px;'>" + "<br>".join(parts) + "</div>"
        QToolTip.showText(QCursor.pos(), html, self.cap)

    def _start_tag_completion_preload(self) -> None:
        """Build the tag-autocomplete model on a daemon thread so the ~114k-row
        CSV parse never blocks the first keystroke. Runs at most once per
        successful load (a missing CSV leaves the door open for a retry)."""
        if getattr(self, "_completion_preloading", False):
            return
        self._completion_preloading = True
        threading.Thread(target=self._preload_tag_completion, daemon=True).start()

    def reload_tag_knowledge_base(self) -> None:
        """Re-attempt the tag-autocomplete load after the danbooru CSV may have
        appeared mid-session (e.g. just downloaded via the Models dialog). A
        no-op once tags are loaded, so it's cheap to call on every download."""
        if getattr(self, "_completion_loaded", False):
            return
        self._completion_preloading = False
        self._start_tag_completion_preload()

    def _preload_tag_completion(self) -> None:
        # Worker thread: pure Python only, no Qt object creation and no writes
        # to self — the parsed payload is marshalled back via a queued signal.
        payload = None
        try:
            csv_path = find_tag_csv(ROOT)
            if csv_path is not None:
                mtime = csv_path.stat().st_mtime
                kb = load_tag_knowledge_base(csv_path)
                infos = kb.ranked_infos()  # popular tags first
                names = [info.name for info in infos]
                kind_lookup = {info.name: info.kind for info in infos}
                payload = (kb, csv_path, mtime, names, kind_lookup)
        except (OSError, ValueError):
            payload = None
        self._completion_ready.emit(payload)

    def _on_completion_ready(self, payload) -> None:
        if payload is None:
            # CSV absent — leave _completion_loaded unset so a later download
            # can retry via reload_tag_knowledge_base().
            self._completion_preloading = False
            return
        self._completion_loaded = True
        kb, csv_path, mtime, names, kind_lookup = payload
        # Seed the shared KB cache (main-thread write) so tag-click explanations
        # and caption-correct reuse this parse instead of re-reading the CSV.
        if self._caption_kb is None:
            self._caption_kb = kb
            self._caption_kb_source = csv_path
            self._caption_kb_mtime = mtime
        self.cap.set_completion_data(names, kind_lookup)

    def _correct_current_caption(self) -> None:
        if self._current_caption_path is None:
            return
        kb = self._load_caption_kb()
        if kb is None:
            return
        result = correct_caption(
            self.cap.toPlainText(),
            kb,
            options=self._caption_correction_options(),
        )
        if not result.changed:
            QMessageBox.information(self, "", t("caption_correct_no_change"))
            return
        self._set_caption_text(result.text)
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _correct_visible_captions(self) -> None:
        if not self._images:
            return
        if not self._confirm_discard_if_dirty():
            return
        kb = self._load_caption_kb()
        if kb is None:
            return
        reply = QMessageBox.question(
            self,
            t("caption_correct"),
            t("caption_correct_visible_confirm", n=len(self._images)),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        changed = 0
        failed: list[str] = []
        options = self._caption_correction_options()
        for image_path in self._images:
            caption_path = image_path.with_suffix(".txt")
            if not caption_path.exists():
                continue
            try:
                old_text = caption_path.read_text(encoding="utf-8")
                result = correct_caption(old_text, kb, options=options)
                if not result.changed:
                    continue
                _append_history(caption_path, old_text)
                caption_path.write_text(result.text, encoding="utf-8")
                changed += 1
            except OSError as exc:
                failed.append(f"{caption_path}: {exc}")

        if (
            self._current_caption_path is not None
            and self._current_caption_path.exists()
        ):
            try:
                self._disk_text = self._current_caption_path.read_text(encoding="utf-8")
            except OSError:
                pass
            self._set_caption_text(self._disk_text)
            self._refresh_buttons()
            self._refresh_inline_diff()

        if failed:
            QMessageBox.warning(
                self,
                t("error"),
                t(
                    "caption_correct_visible_failed",
                    n=changed,
                    err="\n".join(failed[:10]),
                ),
            )
        else:
            QMessageBox.information(
                self, "", t("caption_correct_visible_done", n=changed)
            )

    def _finish_autotag_request(self) -> None:
        """Clear the in-flight state and re-arm the button after a reply."""
        self._autotag_inflight_image = None
        self.autotag_btn.setEnabled(True)
        self._autotag_idle.restart()
        if self._tagger_ready:
            self._set_autotag_status(t("caption_autotag_ready"))

    def _set_autotag_status(self, text: str) -> None:
        self.autotag_status.setText(text)
        self.autotag_status.setVisible(bool(text))

    def _autotag_gpu_watch_tick(self) -> None:
        """While the worker is resident: free it on any other GPU job or idle."""
        if self._tagger_proc is None:
            self._gpu_watch_timer.stop()
            return
        if gui_daemon.active_job_id():  # train/preprocess/group grabbed the card
            self._kill_tagger_worker()
            return
        if (
            self._autotag_inflight_image is None
            and self._autotag_idle.isValid()
            and self._autotag_idle.hasExpired(_AUTOTAG_IDLE_MS)
        ):
            self._kill_tagger_worker()

    def _kill_tagger_worker(self) -> None:
        """Tear down the resident worker and free its VRAM. Idempotent."""
        self._gpu_watch_timer.stop()
        proc = self._tagger_proc
        self._tagger_proc = None
        self._tagger_ready = False
        self._tagger_buf = ""
        self._autotag_inflight_image = None
        self.autotag_btn.setEnabled(True)
        self._set_autotag_status("")
        if proc is None:
            return
        try:
            proc.readyReadStandardOutput.disconnect()
            proc.finished.disconnect()
            proc.errorOccurred.disconnect()
        except (RuntimeError, TypeError):
            pass
        if proc.state() != QProcess.NotRunning:
            proc.closeWriteChannel()  # EOF on stdin → worker exits cleanly
            proc.kill()
            proc.waitForFinished(2000)
        proc.deleteLater()

    def _on_tagger_finished(self, _code, _status) -> None:
        """Worker exited (crash, kill, or EOF) — reset to the no-worker state."""
        was_inflight = self._autotag_inflight_image is not None
        self._gpu_watch_timer.stop()
        self._tagger_proc = None
        self._tagger_ready = False
        self._tagger_buf = ""
        self._autotag_inflight_image = None
        self.autotag_btn.setEnabled(True)
        self._set_autotag_status("")
        if was_inflight:
            QMessageBox.warning(
                self, t("error"), t("caption_autotag_error", err="exit")
            )

    def _load_dir(self, name: str, *, preserve_selection: bool = False):
        if not self._confirm_discard_if_dirty():
            return
        d = self._dirs.get(name)
        if not d:
            return
        prev_stem: str | None = None
        if preserve_selection and self._current_caption_path is not None:
            prev_stem = self._current_caption_path.stem
        if d != self._current_dir:
            # Deletion marks are path-scoped to one dir; drop them on a switch.
            self._marked.clear()
            self._preprocess_decisions.clear()
            self._image_size_cache.clear()
            self._refresh_delete_button()
            self._refresh_preprocess_controls()
        self._current_dir = d
        self._load_preprocess_decisions()
        self._load_groups()  # reload the group manifest for the tree folds
        self._image_size_cache.clear()
        # Drop prefetched pixmaps: a reload re-scans for new/changed files, and
        # cached entries are keyed by path so changed content would be stale.
        self._pm_cache.clear()
        self._decode_inflight.clear()
        self._all_images = _imgs(d)
        had_match = self._apply_filter_and_sort(prev_stem=prev_stem)
        if not self._images:
            self._current_caption_path = None
            self._set_caption_text("")
            self._disk_text = ""
            self._set_image_none()
            self._refresh_buttons()
            self._refresh_inline_diff()
        elif not had_match:
            # Fresh dir, no prior selection to restore — pick the first image.
            self._select_tree_index(0)

    def _display_label(self, p: Path) -> str:
        """``stem`` for top-level images, ``parent/stem`` for nested ones.

        Lets users tell apart shards organized by character/series subfolder
        in ``image_dataset/`` (the trainer enforces unique stems across the
        tree, so the stem itself is still a valid unique key — the prefix is
        purely a display affordance).
        """
        if self._current_dir is None:
            return p.stem
        try:
            rel = p.relative_to(self._current_dir)
        except ValueError:
            return p.stem
        if rel.parent == Path("."):
            return p.stem
        return f"{rel.parent.as_posix()}/{p.stem}"

    def _group_sort_key(self, item: tuple[int, Path]):
        idx, path = item
        label = self._display_label(path).lower()
        if self._group_sort_mode == "size":
            try:
                file_size = path.stat().st_size
            except OSError:
                file_size = 0
            return (file_size, label, idx)
        if self._group_sort_mode == "resolution":
            width, height = self._image_size(path)
            return (width * height, width, height, label, idx)
        return (label, idx)

    def _sort_group_members(
        self, members: list[tuple[int, Path]]
    ) -> list[tuple[int, Path]]:
        return sorted(members, key=self._group_sort_key, reverse=self._group_sort_desc)

    def _apply_filter_and_sort(self, *, prev_stem: str | None = None) -> bool:
        """Rebuild the visible tree from ``_all_images`` using the current
        search text and sort direction.

        Returns True if a row matching ``prev_stem`` was selected, False
        otherwise. Block-signals while rebuilding so search keystrokes don't
        trigger ``_on_tree_item_changed`` (which would prompt to save unsaved
        caption edits on every keystroke).
        """
        q = self._search_text.strip().lower()
        if q:
            visible = [
                p for p in self._all_images if q in self._display_label(p).lower()
            ]
        else:
            visible = list(self._all_images)
        if self._sort_desc:
            visible.reverse()
        # Drop any debounced show: its index points into the old _images order
        # and would load the wrong leaf against the rebuilt list.
        self._nav_timer.stop()
        self._pending_show_idx = None
        self._images = visible

        # Keep the current selection visible after refilter/resort; falls back
        # to ``prev_stem`` when called from _load_dir.
        target_stem: str | None = prev_stem
        if target_stem is None and self._current_caption_path is not None:
            target_stem = self._current_caption_path.stem

        target_row = -1
        for i, p in enumerate(visible):
            if p.stem == target_stem:
                target_row = i
                break

        self.tree.blockSignals(True)
        try:
            self._rebuild_tree(visible)
            if target_row >= 0:
                self._select_tree_index(target_row)
            else:
                self.tree.setCurrentItem(None)
        finally:
            self.tree.blockSignals(False)

        self._refresh_mark_styles()

        total = len(self._all_images)
        shown = len(visible)
        if shown != total:  # narrowed by search
            self.cnt.setText(t("n_images_filtered", shown=shown, total=total))
        else:
            self.cnt.setText(t("n_images", n=total))
        return target_row >= 0

    def _rebuild_tree(self, visible: list[Path]) -> None:
        """Rebuild the tree widget from ``visible``.

        The folder structure under ``self._current_dir`` is primary. Within a
        folder, images that belong to a similarity group (from
        ``make curate-group``) nest one level deeper under a green per-group
        node; ungrouped images sit directly in the folder. Group nodes are
        per-folder, so a group spanning folders shows up once under each. Leaves
        are image stems; everything auto-expands so the hierarchy is visible
        without an extra click."""
        self.tree.clear()
        self._tree_item_to_index.clear()
        if not visible:
            return
        stem_to_group: dict[str, int] = {}
        for gi, g in enumerate(self._groups):
            for m in g.get("members", []):
                stem_to_group[Path(m).stem] = gi

        if self._group_first:
            self._rebuild_tree_group_first(visible, stem_to_group)
        else:
            self._rebuild_tree_folders(visible, stem_to_group)
        self.tree.expandAll()

    def _rebuild_tree_folders(
        self, visible: list[Path], stem_to_group: dict[str, int]
    ) -> None:
        """Folder-primary layout (default): groups nest under their folder, then
        float above the ungrouped files at the same level."""
        # Folders keyed by relative parent path; group sub-nodes by (folder,
        # group index) so each folder gets its own green node per group.
        folder_items: dict[Path, QTreeWidgetItem] = {}
        group_nodes: dict[tuple[Path, int], QTreeWidgetItem] = {}
        group_counts: dict[tuple[Path, int], int] = {}
        group_members: dict[tuple[Path, int], list[tuple[int, Path]]] = {}
        for idx, p in enumerate(visible):
            rel: Path
            if self._current_dir is None:
                rel = Path(p.name)
            else:
                try:
                    rel = p.relative_to(self._current_dir)
                except ValueError:
                    rel = Path(p.name)
            folder = self._ensure_tree_folder(rel.parent, folder_items)
            gi = stem_to_group.get(p.stem)
            if gi is not None:
                key = (rel.parent, gi)
                self._ensure_group_node(folder, key, group_nodes)
                group_counts[key] = group_counts.get(key, 0) + 1
                group_members.setdefault(key, []).append((idx, p))
            else:
                leaf = QTreeWidgetItem(folder, [p.stem])
                leaf.setData(0, _TREE_BASE_TEXT_ROLE, p.stem)
                self._tree_item_to_index[leaf] = idx
        for key, members in group_members.items():
            node = group_nodes[key]
            for idx, p in self._sort_group_members(members):
                leaf = QTreeWidgetItem(node, [p.stem])
                leaf.setData(0, _TREE_BASE_TEXT_ROLE, p.stem)
                self._tree_item_to_index[leaf] = idx
        # Label group nodes once their per-folder visible member count is known.
        for key, node in group_nodes.items():
            node.setText(
                0, t("dataset_group_label", n=key[1] + 1, size=group_counts[key])
            )
        self._float_groups_to_top(folder_items, group_nodes)

    def _rebuild_tree_group_first(
        self, visible: list[Path], stem_to_group: dict[str, int]
    ) -> None:
        """Group-first layout: every similarity group becomes a single
        root-level green node holding all its visible members (across folders,
        labelled with their folder prefix), sorted by group index. The ungrouped
        images follow below in the normal folder tree."""
        group_members: dict[int, list[tuple[int, Path]]] = {}
        ungrouped: list[tuple[int, Path]] = []
        for idx, p in enumerate(visible):
            gi = stem_to_group.get(p.stem)
            if gi is None:
                ungrouped.append((idx, p))
            else:
                group_members.setdefault(gi, []).append((idx, p))

        # Members shown with their folder prefix so cross-folder groups stay
        # legible (stems alone would be ambiguous).
        for gi in sorted(group_members):
            members = group_members[gi]
            node = QTreeWidgetItem(
                self.tree,
                [t("dataset_group_label", n=gi + 1, size=len(members))],
            )
            node.setForeground(0, QColor("#27ae60"))
            font = node.font(0)
            font.setBold(True)
            node.setFont(0, font)
            for idx, p in self._sort_group_members(members):
                label = self._display_label(p)
                leaf = QTreeWidgetItem(node, [label])
                leaf.setData(0, _TREE_BASE_TEXT_ROLE, label)
                self._tree_item_to_index[leaf] = idx

        # Divider only when both sections exist, so it never dangles.
        if group_members and ungrouped:
            self._add_tree_separator()

        folder_items: dict[Path, QTreeWidgetItem] = {}
        for idx, p in ungrouped:
            if self._current_dir is None:
                rel = Path(p.name)
            else:
                try:
                    rel = p.relative_to(self._current_dir)
                except ValueError:
                    rel = Path(p.name)
            folder = self._ensure_tree_folder(rel.parent, folder_items)
            leaf = QTreeWidgetItem(folder, [p.stem])
            leaf.setData(0, _TREE_BASE_TEXT_ROLE, p.stem)
            self._tree_item_to_index[leaf] = idx

    def _add_tree_separator(self) -> None:
        """Append a non-selectable horizontal divider row at the tree root.

        A real 2px QFrame line (set as the row's item widget) reads clearly on
        the tree's light background, unlike dash glyphs which wash out."""
        sep = QTreeWidgetItem(self.tree, [""])
        sep.setFlags(Qt.NoItemFlags)
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet("background:#8a8a8a;")
        self.tree.setItemWidget(sep, 0, line)

    def _float_groups_to_top(
        self,
        folder_items: dict[Path, QTreeWidgetItem],
        group_nodes: dict[tuple[Path, int], QTreeWidgetItem],
    ) -> None:
        """Reorder each folder's children so green group nodes sit above the
        ungrouped files at the *same* level (groups never cross into another
        folder — they stay children of their own folder, so they can't rise
        above a higher tree level). Within the group block and within the rest,
        the original filename order is preserved.
        """
        group_set = set(group_nodes.values())
        # Each folder node + the invisible root (top-level images).
        parents = [*folder_items.values(), self.tree.invisibleRootItem()]
        for parent in parents:
            children = parent.takeChildren()
            grouped = [c for c in children if c in group_set]
            rest = [c for c in children if c not in group_set]
            if grouped:  # only touch folders that actually hold a group node
                parent.addChildren(grouped + rest)
            else:
                parent.addChildren(children)

    def _ensure_group_node(
        self,
        folder: QTreeWidget | QTreeWidgetItem,
        key: tuple[Path, int],
        group_nodes: dict[tuple[Path, int], QTreeWidgetItem],
    ) -> QTreeWidgetItem:
        """Lazily create the green similarity-group node under ``folder``.

        ``key`` is (folder rel-path, group index). Text is set later (in
        ``_rebuild_tree``) once the per-folder visible member count is known;
        only created for groups with a visible member in that folder."""
        cached = group_nodes.get(key)
        if cached is not None:
            return cached
        node = QTreeWidgetItem(folder, [""])
        node.setForeground(0, QColor("#27ae60"))
        font = node.font(0)
        font.setBold(True)
        node.setFont(0, font)
        group_nodes[key] = node
        return node

    def _ensure_tree_folder(
        self, rel_parent: Path, folder_items: dict[Path, QTreeWidgetItem]
    ) -> QTreeWidget | QTreeWidgetItem:
        """Resolve (and lazily create) the QTreeWidgetItem for ``rel_parent``.

        Returns ``self.tree`` for the root (Path('.')) so callers can pass it
        as the parent of a leaf item directly — QTreeWidgetItem(parent, …)
        accepts either the tree widget or another item.
        """
        if rel_parent in (Path("."), Path("")):
            return self.tree
        cached = folder_items.get(rel_parent)
        if cached is not None:
            return cached
        grandparent = self._ensure_tree_folder(rel_parent.parent, folder_items)
        item = QTreeWidgetItem(grandparent, [rel_parent.name])
        folder_items[rel_parent] = item
        return item

    def _select_tree_index(self, idx: int) -> None:
        """Highlight the tree leaf corresponding to image index ``idx``."""
        for item, i in self._tree_item_to_index.items():
            if i == idx:
                self.tree.setCurrentItem(item)
                return
        self.tree.setCurrentItem(None)

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text
        self._apply_filter_and_sort()

    def _toggle_sort(self) -> None:
        self._sort_desc = not self._sort_desc
        self.sort_btn.setText("z-a" if self._sort_desc else "a-z")
        self.sort_btn.setToolTip(
            t("dataset_sort_desc_tooltip")
            if self._sort_desc
            else t("dataset_sort_asc_tooltip")
        )
        self._apply_filter_and_sort()

    def _toggle_group_first(self) -> None:
        self._group_first = self.group_first_btn.isChecked()
        self.group_first_btn.setText(
            t("dataset_view_group") if self._group_first else t("dataset_view_tree")
        )
        self._apply_filter_and_sort()

    def _on_group_sort_changed(self) -> None:
        raw = str(self.group_sort_combo.currentData() or "name")
        self._group_sort_desc = raw.endswith("_desc")
        self._group_sort_mode = raw.removesuffix("_desc")
        self._apply_filter_and_sort()

    def _reload_current_dir(self) -> None:
        """Re-scan the currently selected directory (for new/changed images)."""
        name = self.dc.currentText()
        if name:
            self._load_dir(name, preserve_selection=True)

    def _add_dir(self) -> None:
        """Pick a directory and add it to the combo for this session."""
        if not self._confirm_discard_if_dirty():
            return
        start = str(self._dirs.get(self.dc.currentText(), Path.home()))
        chosen = QFileDialog.getExistingDirectory(
            self, t("dataset_add_dir_picker"), start
        )
        if not chosen:
            return
        path = Path(chosen)
        # Use the absolute path string as the display key — unambiguous and
        # avoids collisions with the built-in short labels (image_dataset, …).
        label = str(path)
        for existing in self._dirs.values():
            if existing == path:
                QMessageBox.information(
                    self, t("directory"), t("dataset_add_dir_already", name=label)
                )
                # Switch to it so the user lands on the dir they tried to add.
                for k, v in self._dirs.items():
                    if v == path:
                        idx = self.dc.findText(k)
                        if idx >= 0:
                            self.dc.setCurrentIndex(idx)
                        break
                return
        self._dirs[label] = path
        self.dc.addItem(label)
        self.dc.setCurrentText(label)

    def _curation_decisions_path(self) -> Path:
        return ROOT / "post_image_dataset" / "curation_decisions.json"

    def _current_source_label(self) -> str:
        if self._current_dir is None:
            return ""
        try:
            return self._current_dir.relative_to(ROOT).as_posix()
        except ValueError:
            return str(self._current_dir).replace("\\", "/")

    def _load_preprocess_decisions(self) -> None:
        self._preprocess_decisions.clear()
        if self._current_dir is None:
            return
        path = self._curation_decisions_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if data.get("source_dir") != self._current_source_label():
            return
        decisions = load_curation_decisions(path)
        for key, value in decisions.items():
            path = self._current_dir / key
            action = str(value.get("action") or "").strip()
            if action in {"use", "skip"}:
                self._preprocess_decisions[path] = action
            elif action == "move":
                self._marked.add(path)

    def _save_preprocess_decisions(self) -> None:
        if self._current_dir is None:
            return
        images: dict[str, dict] = {}
        for path in sorted(
            set(self._preprocess_decisions) | set(self._marked),
            key=lambda p: rel_key(p, self._current_dir),
        ):
            item: dict = {}
            if path in self._marked:
                item["action"] = "move"
            else:
                action = self._preprocess_decisions.get(path)
                if action in {"use", "skip"}:
                    item["action"] = action
            if item:
                images[rel_key(path, self._current_dir)] = item
        save_curation_decisions(
            self._curation_decisions_path(),
            source_dir=self._current_source_label(),
            images=images,
        )
        self.preprocess_save_btn.setText(t("dataset_preprocess_save"))
        QMessageBox.information(
            self,
            t("dataset_preprocess_save"),
            t("dataset_preprocess_saved", path=str(self._curation_decisions_path())),
        )

    def _mark_preprocess_dirty(self) -> None:
        self.preprocess_save_btn.setText(t("dataset_preprocess_save") + " *")

    def _on_tree_item_changed(self, current, _previous) -> None:
        """Show the image for the newly selected tree leaf.

        Folder rows (no index) are non-selectable in the data sense; only
        leaves correspond to an image. We confirm-discard before switching so
        the unsaved-edit prompt fires on navigation.
        """
        if current is None:
            return
        idx = self._tree_item_to_index.get(current)
        if idx is None:
            return
        if not self._confirm_discard_if_dirty():
            prev = self._row_for_path(self._current_caption_path)
            if prev is not None and prev != idx:
                self.tree.blockSignals(True)
                try:
                    self._select_tree_index(prev)
                finally:
                    self.tree.blockSignals(False)
            return
        # Defer the expensive load (full-res decode + caption/meta refresh) so
        # arrow-key auto-repeat doesn't load every leaf it skips past. The dirty
        # prompt above stays synchronous so a cancel can revert immediately.
        self._pending_show_idx = idx
        self._nav_timer.start()

    def _show_pending(self) -> None:
        idx = self._pending_show_idx
        if idx is None:
            return
        self._pending_show_idx = None
        self._show(idx)
        self._refresh_mark_styles()

    def _cache_pixmap(self, key: str, pm: QPixmap) -> None:
        """Insert into the prefetch cache, evicting the oldest past the cap."""
        self._pm_cache.pop(key, None)  # refresh recency
        self._pm_cache[key] = pm
        while len(self._pm_cache) > _PM_CACHE_MAX:
            oldest = next(iter(self._pm_cache))
            del self._pm_cache[oldest]

    def _prefetch_neighbors(self, row: int) -> None:
        """Decode the ±_PREFETCH_RADIUS images around ``row`` off-thread."""
        for offset in range(1, _PREFETCH_RADIUS + 1):
            for nbr in (row - offset, row + offset):
                if not 0 <= nbr < len(self._images):
                    continue
                key = str(self._images[nbr])
                if key in self._pm_cache or key in self._decode_inflight:
                    continue
                self._decode_inflight.add(key)
                self._decode_pool.start(_DecodeTask(key, self._decode_signals))

    def _on_image_decoded(self, key: str, img) -> None:
        """Background decode landed: convert to QPixmap and cache it. The result
        is only stashed — the visible image is set by _show on the GUI thread."""
        self._decode_inflight.discard(key)
        if img is None or key in self._pm_cache:
            return
        pm = QPixmap.fromImage(img)
        if not pm.isNull():
            self._cache_pixmap(key, pm)

    def _show(self, row: int):
        if not 0 <= row < len(self._images):
            return
        p = self._images[row]
        key = str(p)
        # Prefetched neighbour → instant; otherwise decode now (and cache it so
        # bouncing back is instant too).
        pm = self._pm_cache.get(key)
        if pm is None:
            pm = QPixmap(key)
            if not pm.isNull():
                self._cache_pixmap(key, pm)
        self._set_image(p, pm if not pm.isNull() else None)
        self._prefetch_neighbors(row)
        cp = p.with_suffix(".txt")
        self._current_caption_path = cp
        if cp.exists():
            text = cp.read_text(encoding="utf-8")
        else:
            text = ""
        self._disk_text = text
        self._set_caption_text(text if text else "")
        self._refresh_variant_combo(p)
        self._refresh_image_meta(p)
        self._refresh_preprocess_controls()
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _set_image(self, p: Path, source: QPixmap | None) -> None:
        """Bind a new source pixmap + its (possibly absent) mask, then refresh."""
        self._source_pm = source
        self._mask_path = (
            _resolve_mask_path(p, self._current_dir) if source is not None else None
        )
        self._overlay_pm = None  # compose lazily in _apply_image_view
        self.overlay_cb.setEnabled(self._mask_path is not None)
        self.resize_preview_cb.setEnabled(source is not None)
        self._apply_image_view()

    def _apply_image_view(self) -> None:
        """Push the right pixmap onto ``self.img`` based on overlay state."""
        if self._source_pm is None:
            return
        pm = self._source_pm
        if self.overlay_cb.isChecked() and self._mask_path is not None:
            if self._overlay_pm is None:
                self._overlay_pm = _compose_mask_overlay(
                    self._source_pm, self._mask_path
                )
            pm = self._overlay_pm
        if self.resize_preview_cb.isChecked():
            target_res, crop_anchor, bucket_resos, crop_margins, fit_mode, max_ratio = (
                self._resize_preview_config()
            )
            pm = _compose_resize_preview_overlay(
                pm,
                target_res,
                crop_anchor=crop_anchor,
                bucket_resos=bucket_resos,
                crop_margins=crop_margins,
                fit_mode=fit_mode,
                max_ratio=max_ratio,
            )
        self.img.set_source(pm)

    def _on_overlay_toggled(self, _value=None) -> None:
        self._apply_image_view()
        self._refresh_image_meta(self._current_image_path())

    def _resize_preview_target_res(self):
        tab = self._preprocess_tab
        widget = getattr(tab, "target_res_widget", None)
        if widget is not None:
            try:
                return widget.value()
            except (AttributeError, TypeError, ValueError):
                pass
        return _load_resize_preview_target_res()

    def _resize_preview_config(self):
        target_res = self._resize_preview_target_res()
        crop_anchor = None
        bucket_resos = None
        crop_margins = None
        tab = self._preprocess_tab
        anchor_widget = getattr(tab, "resize_crop_anchor_widget", None)
        if anchor_widget is not None:
            crop_anchor = anchor_widget.value()
        widget = getattr(tab, "target_res_widget", None)
        if widget is not None:
            try:
                bucket_resos = widget.bucket_resos()
            except (AttributeError, TypeError, ValueError):
                bucket_resos = None
        if tab is not None and hasattr(tab, "_resize_crop_margins"):
            crop_margins = tab._resize_crop_margins()
        fit_mode, max_ratio = self._resize_preview_fit_mode()
        return target_res, crop_anchor, bucket_resos, crop_margins, fit_mode, max_ratio

    def _resize_preview_fit_mode(self):
        """(fit_mode, max_ratio) from the live preprocess-tab widgets, falling
        back to configs/preprocess.toml when the tab isn't available. Free-fit is
        the only resize mode, so fit_mode is always ``"freefit"``."""
        spin = getattr(self._preprocess_tab, "freefit_max_ratio_spin", None)
        if spin is not None:
            return "freefit", float(spin.value())
        data = _load_preprocess_toml_data()
        max_ratio = float(data.get("freefit_max_ratio", DEFAULT_FREEFIT_MAX_RATIO))
        return "freefit", max_ratio

    def _current_index(self) -> int:
        """Index into ``self._images`` of the currently selected image.

        Reads the selected tree leaf; -1 if nothing is on an image (e.g. a
        folder/group node is selected)."""
        item = self.tree.currentItem()
        if item is not None:
            idx = self._tree_item_to_index.get(item)
            if idx is not None:
                return idx
        return -1

    def app_context_menu(self, target, global_pos):
        """Right-click hook (called by MainWindow's global filter). Over an
        image — the preview or a tree leaf — offer "open in system viewer"
        instead of the app default; return None elsewhere so the default shows."""
        path = self._image_under_cursor(target, global_pos)
        if path is None:
            return None
        menu = QMenu(self)
        act = menu.addAction(t("open_in_system_viewer"))
        act.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        )
        return menu

    def _image_under_cursor(self, target, global_pos) -> Path | None:
        """The image path the right-click landed on: the tree leaf under the
        cursor, or the currently shown preview. None if neither applies."""
        if target is self.tree or self.tree.isAncestorOf(target):
            pos = self.tree.viewport().mapFromGlobal(global_pos)
            item = self.tree.itemAt(pos)
            idx = self._tree_item_to_index.get(item) if item is not None else None
            if idx is not None and 0 <= idx < len(self._images):
                return self._images[idx]
            return None
        if target is self.img or self.img.isAncestorOf(target):
            idx = self._current_index()
            if 0 <= idx < len(self._images):
                return self._images[idx]
        return None

    def _image_size(self, path: Path) -> tuple[int, int]:
        cached = self._image_size_cache.get(path)
        if cached is not None:
            return cached
        size = QImageReader(str(path)).size()
        result = (max(0, size.width()), max(0, size.height()))
        self._image_size_cache[path] = result
        return result

    def _resize_preview_meta(self, width: int, height: int) -> str:
        if not self.resize_preview_cb.isChecked():
            return ""
        try:
            target_res, crop_anchor, bucket_resos, crop_margins, fit_mode, max_ratio = (
                self._resize_preview_config()
            )
            preview = compute_resize_preview(
                width,
                height,
                target_res,
                crop_anchor=crop_anchor,
                bucket_resos=bucket_resos,
                crop_margins=crop_margins,
                fit_mode=fit_mode,
                max_ratio=max_ratio,
            )
        except (KeyError, TypeError, ValueError):
            return ""
        bucket_w, bucket_h = preview.bucket_size
        return t(
            "dataset_image_meta_resize",
            width=bucket_w,
            height=bucket_h,
            edge=preview.target_edge,
        )

    def _refresh_image_meta(self, path: Path | None) -> None:
        if path is None:
            self.image_meta.setText(t("dataset_image_meta_empty"))
            return
        width, height = self._image_size(path)
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = 0
        fmt = path.suffix.lstrip(".").upper() or "?"
        meta = t(
            "dataset_image_meta",
            width=width,
            height=height,
            size=_format_file_size(file_size),
            fmt=escape(fmt),
        )
        resize_meta = self._resize_preview_meta(width, height)
        if resize_meta:
            meta = f"{meta} · {resize_meta}"
        decision = self._preprocess_decision_text(path)
        if decision:
            meta = f"{meta} · {escape(decision)}"
        self.image_meta.setText(meta)

    def _current_image_path(self) -> Path | None:
        idx = self._current_index()
        if 0 <= idx < len(self._images):
            return self._images[idx]
        return None

    def _set_current_preprocess_decision(
        self, action: str, *, advance: bool = False
    ) -> None:
        path = self._current_image_path()
        if path is None or action not in {"use", "skip"}:
            return
        if path in self._marked:
            self._marked.discard(path)
        self._preprocess_decisions[path] = action
        self._mark_preprocess_dirty()
        self._refresh_mark_styles()
        self._refresh_delete_button()
        self._refresh_preprocess_controls()
        if advance:
            self._nav(1)

    def _clear_current_preprocess_decision(self) -> None:
        path = self._current_image_path()
        if path is None:
            return
        preprocess_changed = False
        if path in self._preprocess_decisions:
            self._preprocess_decisions.pop(path, None)
            preprocess_changed = True
        if path in self._marked:
            self._marked.discard(path)
            preprocess_changed = True
        if preprocess_changed:
            self._mark_preprocess_dirty()
        self._refresh_mark_styles()
        self._refresh_delete_button()
        self._refresh_preprocess_controls()

    def _clear_all_decisions(self) -> None:
        """Clear all use/skip/move decisions."""
        changed = bool(self._preprocess_decisions or self._marked)
        if not changed:
            return
        self._preprocess_decisions.clear()
        self._marked.clear()
        self._mark_preprocess_dirty()
        self._refresh_mark_styles()
        self._refresh_delete_button()
        self._refresh_preprocess_controls()

    def _preprocess_decision_text(self, path: Path | None) -> str:
        # The "no decision" state is already conveyed by the tree-view row styling
        # on the left, so render nothing here to avoid clutter over the image.
        if path is None:
            return ""
        if path in self._marked:
            return t("dataset_preprocess_decision_move")
        action = self._preprocess_decisions.get(path)
        if action == "skip":
            return t("dataset_preprocess_decision_skip")
        if action == "use":
            return t("dataset_preprocess_decision_use")
        return ""

    def _refresh_preprocess_controls(self) -> None:
        path = self._current_image_path()
        enabled = path is not None
        self.preprocess_skip_btn.setEnabled(enabled)
        current_has_decision = (
            path in self._preprocess_decisions or path in self._marked
            if path is not None
            else False
        )
        has_any_decision = bool(self._preprocess_decisions) or bool(self._marked)
        self.preprocess_clear_btn.setEnabled(
            enabled and (current_has_decision or has_any_decision)
        )
        self.preprocess_save_btn.setEnabled(self._current_dir is not None)
        # The decision state is appended to the image-meta line (single row).
        self._refresh_image_meta(path)

    def _toggle_mark_current(self) -> None:
        """Toggle the deletion mark on the currently selected image."""
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        p = self._images[idx]
        if p in self._marked:
            self._marked.discard(p)
            self._mark_preprocess_dirty()
        else:
            if self._preprocess_decisions.pop(p, None) is not None:
                self._mark_preprocess_dirty()
            self._marked.add(p)
            self._mark_preprocess_dirty()
        self._refresh_mark_styles()
        self._refresh_delete_button()
        self._refresh_preprocess_controls()

    def _mark_current_for_move(self) -> None:
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        path = self._images[idx]
        if self._preprocess_decisions.pop(path, None) is not None:
            self._mark_preprocess_dirty()
        if path not in self._marked:
            self._marked.add(path)
            self._mark_preprocess_dirty()
        self._refresh_mark_styles()
        self._refresh_delete_button()
        self._refresh_preprocess_controls()
        self._nav(1)

    def _refresh_mark_styles(self) -> None:
        """Repaint tree leaves by pending source-delete/preprocess state.

        Status markers are text prefixes instead of item icons/backgrounds so
        filenames keep their original alignment in the tree."""
        for leaf, idx in self._tree_item_to_index.items():
            path = self._images[idx] if idx < len(self._images) else None
            base = leaf.data(0, _TREE_BASE_TEXT_ROLE) or leaf.text(0)
            prefix = ""
            color = None
            if path in self._marked:
                prefix = _MOVE_MARK_PREFIX
                color = QColor("#e74c3c")
            elif self._preprocess_decisions.get(path) == "skip":
                prefix = _SKIP_MARK_PREFIX
                color = QColor("#f39c12")
            elif self._preprocess_decisions.get(path) == "use":
                prefix = _USE_MARK_PREFIX
                color = QColor("#3498db")
            leaf.setText(0, f"{prefix}{base}")
            if color is not None:
                leaf.setForeground(0, color)
            else:
                leaf.setData(0, Qt.ForegroundRole, None)

    def _unmark_current(self) -> None:
        """Remove the move mark from the currently selected image."""
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        p = self._images[idx]
        if p not in self._marked:
            return
        self._marked.discard(p)
        self._mark_preprocess_dirty()
        self._refresh_mark_styles()
        self._refresh_delete_button()
        self._refresh_preprocess_controls()

    def _refresh_delete_button(self) -> None:
        n = len(self._marked)
        self.delete_btn.setEnabled(n > 0)
        self.delete_btn.setText(t("dataset_delete") + (f" ({n})" if n else ""))

    def _delete_marked(self) -> None:
        """Move every marked image (+ sidecars) under post_image_dataset/moved."""
        targets = sorted(self._marked)
        if not targets:
            return
        reply = QMessageBox.question(
            self,
            t("dataset_delete_confirm_title"),
            t("dataset_delete_confirm_body", n=len(targets)),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # Remember where the user was so the rebuilt tree doesn't snap to the
        # top; anchor on position to land on the nearest surviving neighbour.
        open_stem = (
            self._current_caption_path.stem
            if self._current_caption_path is not None
            else None
        )
        old_images = list(self._images)
        anchor_row = self._current_index()
        targets_set = set(targets)

        target_root = self._moved_images_dir()
        errors: list[str] = []
        for p in targets:
            try:
                move_linked_files(
                    p,
                    source_root=self._current_dir or p.parent,
                    target_root=target_root,
                )
            except (OSError, shutil.Error) as e:
                errors.append(f"{p.name}: {e}")
        self._marked.clear()
        self._mark_preprocess_dirty()
        self._refresh_delete_button()
        # Drop the editor context so the reload doesn't prompt about a caption
        # whose image we just removed.
        self._current_caption_path = None
        self._disk_text = ""
        self._set_caption_text("")
        if self._current_dir is not None:
            self._image_size_cache.clear()
            self._all_images = _imgs(self._current_dir)
        self._apply_filter_and_sort()
        if self._images:
            self._select_tree_index(
                self._post_delete_row(open_stem, old_images, anchor_row, targets_set)
            )
        else:
            self._set_image_none()
            self._refresh_buttons()
            self._refresh_inline_diff()
        if errors:
            QMessageBox.warning(
                self, t("error"), t("dataset_delete_failed", err="\n".join(errors))
            )

    def _post_delete_row(
        self,
        open_stem: str | None,
        old_images: list[Path],
        anchor_row: int,
        deleted: set[Path],
    ) -> int:
        """Pick which row to reselect after a delete so the view stays put.

        If the image the user had open survived, return its new row. Otherwise
        walk outward from ``anchor_row`` (forward first, then backward) over the
        pre-delete order to find the nearest surviving neighbour, and return its
        new row. Falls back to the top only when nothing else matches.
        """
        new_row = {p.stem: i for i, p in enumerate(self._images)}
        if open_stem is not None and open_stem in new_row:
            return new_row[open_stem]
        if old_images:
            start = anchor_row if anchor_row >= 0 else 0
            order = list(range(start, len(old_images))) + list(range(start - 1, -1, -1))
            for j in order:
                if old_images[j] not in deleted and old_images[j].stem in new_row:
                    return new_row[old_images[j].stem]
        return 0

    def _moved_images_dir(self) -> Path:
        return ROOT / "post_image_dataset" / "moved"

    def _set_image_none(self) -> None:
        """Clear the preview pane (used after deleting the last image)."""
        self._source_pm = None
        self._mask_path = None
        self._overlay_pm = None
        self.overlay_cb.setEnabled(False)
        self.resize_preview_cb.setEnabled(False)
        self.img.clear()
        self._refresh_image_meta(None)
        self._refresh_preprocess_controls()
        # No image → no variants to preview; drop any active read-only preview.
        self._previewing_variant = False
        self.cap.setReadOnly(False)
        self._variant_rows = []
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        self.variant_combo.setVisible(False)
        self.variant_combo.blockSignals(False)

    def _set_caption_text(self, text: str) -> None:
        self._suspend_dirty = True
        try:
            self.cap.setPlainText(text)
        finally:
            self._suspend_dirty = False

    @staticmethod
    def _variant_item_label(label: str, text: str) -> str:
        """Dropdown entry: the variant tag plus a short text preview."""
        short = text if len(text) <= 32 else text[:31] + "…"
        return f"{label}  {short}" if short else label

    def _refresh_variant_combo(self, image_path: Path) -> None:
        """Repopulate the variant-preview dropdown for the current image.

        Reads ``{stem}.variants.txt`` next to the image (only present under
        ``resized/`` after preprocess). Resets any active preview back to the
        editable training caption first, so navigating images never leaves the
        editor stuck read-only.
        """
        # Leaving any prior preview: restore edit mode (the new image's caption
        # was already loaded into the editor by ``_show``).
        self._previewing_variant = False
        self.cap.setReadOnly(False)

        rows: list[tuple[str, str]] = []
        sidecar = variants_sidecar_path(image_path)
        if sidecar.exists():
            try:
                rows = read_variants_sidecar(sidecar)
            except OSError:
                rows = []
        self._variant_rows = rows

        self.variant_combo.blockSignals(True)
        try:
            self.variant_combo.clear()
            if rows:
                self.variant_combo.addItem(t("caption_variant_training"))
                for label, text in rows:
                    self.variant_combo.addItem(self._variant_item_label(label, text))
                self.variant_combo.setCurrentIndex(0)
            self.variant_combo.setVisible(bool(rows))
        finally:
            self.variant_combo.blockSignals(False)

    def _on_variant_selected(self, idx: int) -> None:
        """Preview a variant read-only, or restore the editable training caption.

        Index 0 is the training caption; 1..N map to the sidecar rows. The
        editor buffer (possibly with unsaved edits) is stashed on entering a
        preview and restored on returning, so previewing never discards work.
        """
        if idx <= 0:
            if self._previewing_variant:
                self._previewing_variant = False
                self.cap.setReadOnly(False)
                self._set_caption_text(self._preview_stash)
                self._refresh_buttons()
                self._refresh_inline_diff()
            return
        row = idx - 1
        if not 0 <= row < len(self._variant_rows):
            return
        if not self._previewing_variant:
            self._preview_stash = self.cap.toPlainText()
            self._previewing_variant = True
        self.cap.setReadOnly(True)
        self._set_caption_text(self._variant_rows[row][1])
        # No diff/save in preview — this isn't the editable caption.
        self.cap.setExtraSelections([])
        self.save_btn.setEnabled(False)
        self.revert_btn.setEnabled(False)

    def _on_text_changed(self) -> None:
        if self._suspend_dirty:
            return
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _editable_text(self) -> str:
        """The user's edit buffer.

        While a read-only variant preview is on screen the editor holds the
        variant text, not the user's caption — the real buffer lives in
        ``_preview_stash``. Dirty-detection and Save must read this, never the
        raw editor, or previewing a variant looks like an unsaved edit (and
        Save would clobber the caption with the variant).
        """
        if self._previewing_variant:
            return self._preview_stash
        return self.cap.toPlainText()

    def _is_dirty(self) -> bool:
        if self._current_caption_path is None:
            return False
        return self._editable_text() != self._disk_text

    def _refresh_buttons(self) -> None:
        dirty = self._is_dirty()
        self.save_btn.setEnabled(dirty)
        self.revert_btn.setEnabled(dirty)
        marker = t("caption_dirty_marker") if dirty else ""
        label = t("caption") + marker
        if dirty:
            _, add, rem = _diff_spans(self._disk_text, self.cap.toPlainText())
            if add or rem:
                label += "  " + t("caption_diff_stats", add=add, rem=rem)
        self.cap_label.setText(label)
        # Versions button is enabled whenever there's a caption file context;
        # the dialog itself shows "(no prior versions)" when empty.
        self.versions_btn.setEnabled(self._current_caption_path is not None)

    def _refresh_inline_diff(self) -> None:
        """Highlight inserted spans (vs disk) directly in the editor."""
        if self._current_caption_path is None:
            self.cap.setExtraSelections([])
            return
        spans, _, _ = _diff_spans(self._disk_text, self.cap.toPlainText())
        if not spans:
            self.cap.setExtraSelections([])
            return
        fmt = _add_format()
        sels: list[QTextEdit.ExtraSelection] = []
        doc = self.cap.document()
        for j1, j2 in spans:
            cur = QTextCursor(doc)
            cur.setPosition(j1)
            cur.setPosition(j2, QTextCursor.KeepAnchor)
            es = QTextEdit.ExtraSelection()
            es.cursor = cur
            es.format = fmt
            sels.append(es)
        self.cap.setExtraSelections(sels)

    def _save(self) -> None:
        cp = self._current_caption_path
        if cp is None or not self._is_dirty():
            return
        new_text = self._editable_text()
        try:
            # Snapshot the prior on-disk version into history before overwriting.
            if cp.exists():
                _append_history(cp, self._disk_text)
            cp.write_text(new_text, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, t("error"), t("caption_save_failed", err=str(e)))
            return
        self._disk_text = new_text
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _revert(self) -> None:
        if self._current_caption_path is None:
            return
        self._set_caption_text(self._disk_text)
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _open_versions(self) -> None:
        cp = self._current_caption_path
        if cp is None:
            return
        # Dialog always diffs against the on-disk text; a restored version
        # replaces editor contents (a pending edit until Save).
        dlg = CaptionVersionsDialog(cp, self._disk_text, self)
        if dlg.exec() == QDialog.Accepted:
            restored = dlg.restored_text()
            if restored is not None:
                self._set_caption_text(restored)
                self._refresh_buttons()
                self._refresh_inline_diff()

    def _row_for_path(self, cp: Path | None) -> int | None:
        if cp is None:
            return None
        for i, p in enumerate(self._images):
            if p.with_suffix(".txt") == cp:
                return i
        return None

    def _confirm_discard_if_dirty(self) -> bool:
        """Prompt to save if dirty. Returns False if the user cancels."""
        if not self._is_dirty():
            return True
        reply = QMessageBox.question(
            self,
            t("caption_unsaved_title"),
            t("caption_unsaved_body"),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._save()
            # If the save failed, _is_dirty() is still True — abort the switch.
            return not self._is_dirty()
        # Discard: drop edits silently.
        return True

    def _nav(self, d: int):
        leaves = self._visible_tree_leaves()
        if not leaves:
            return
        current = self.tree.currentItem()
        try:
            pos = leaves.index(current) if current is not None else -1
        except ValueError:
            idx = self._current_index()
            pos = next(
                (
                    i
                    for i, item in enumerate(leaves)
                    if self._tree_item_to_index[item] == idx
                ),
                -1,
            )
        new_pos = pos + d
        if 0 <= new_pos < len(leaves):
            self.tree.setCurrentItem(leaves[new_pos])

    def _visible_tree_leaves(self) -> list[QTreeWidgetItem]:
        """Return image leaves in the same order shown by the left tree."""

        leaves: list[QTreeWidgetItem] = []

        def walk(parent: QTreeWidgetItem) -> None:
            if parent in self._tree_item_to_index and not parent.isHidden():
                leaves.append(parent)
                return
            if parent is not self.tree.invisibleRootItem() and not parent.isExpanded():
                return
            for i in range(parent.childCount()):
                child = parent.child(i)
                if not child.isHidden():
                    walk(child)

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if not item.isHidden():
                walk(item)
        return leaves
