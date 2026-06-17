"""ImageViewerTab — dataset image browser with caption editor + history."""

from __future__ import annotations

import difflib
import json
import shutil
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import toml
from PySide6.QtCore import (
    QElapsedTimer,
    QEvent,
    QProcess,
    QProcessEnvironment,
    QRect,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import (
    DEFAULT_AUTOTAG_CONFIDENCE,
    ROOT,
    LazyTabMixin,
    ScaledImageLabel,
    _image_dirs,
    _imgs,
    get_setting,
)
from gui import daemon as gui_daemon
from gui._job_mixin import DaemonJobMixin
from gui.i18n import t
from gui.progress import TqdmProgressTracker, make_progress_bar
from gui.theme import tok
from gui.widgets import apply_variant
from library.datasets.curation_actions import (
    load_curation_decisions,
    move_linked_files,
    rel_key,
    save_curation_decisions,
)
from library.datasets.buckets import buckets_for_edges
from library.preprocess.resize_preview import (
    compute_resize_preview,
    format_bucket_resos,
    normalize_target_res,
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


# Tint over the *masked-out* (inverted) region; alpha driven from the mask +
# QPainter.setOpacity rather than baked into the color.
_MASK_OVERLAY_COLOR_OPAQUE = QColor(255, 60, 60, 255)
_MASK_OVERLAY_OPACITY = 0.55
_RESIZE_PREVIEW_COLOR = QColor(40, 220, 120, 255)
_RESIZE_MARGIN_COLOR = QColor(255, 70, 60, 255)
_RESIZE_PREVIEW_SHADE = QColor(0, 0, 0, 72)

# Text prefixes for GUI preprocess decisions and images marked for moving.
_USE_MARK_PREFIX = "■ "
_SKIP_MARK_PREFIX = "■ "
_MOVE_MARK_PREFIX = "■ "
_TREE_BASE_TEXT_ROLE = Qt.UserRole + 1


def _format_file_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(max(0, size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024


def _resolve_mask_path(image_path: Path, current_dir: Path | None) -> Path | None:
    """Locate the merged mask PNG for ``image_path``.

    Mirrors the trainer's mask layout: ``post_image_dataset/masks/<rel>/<stem>_mask.png``
    where ``rel`` is the image's parent relative to ``current_dir``. Falls back
    to the legacy ``masks/merged/...`` tree before giving up.
    """
    if current_dir is None:
        return None
    try:
        rel = image_path.relative_to(current_dir)
    except ValueError:
        return None
    rel_parent = rel.parent
    name = f"{image_path.stem}_mask.png"
    for root in (ROOT / "post_image_dataset" / "masks", ROOT / "masks" / "merged"):
        candidate = root / rel_parent / name
        if candidate.is_file():
            return candidate
    return None


def _compose_mask_overlay(source: QPixmap, mask_path: Path) -> QPixmap:
    """Return ``source`` with a red translucent tint over the masked-out region.

    Convention from ``scripts/preprocess/merge_masks.py``: **white = "train here",
    black = ignored (text bubble / artifact)**. We invert so the tint lands
    on the *ignored* region — that's the half users want to see at a glance
    ("did the detector catch every bubble?").

    Implementation note: ``convertToFormat(Alpha8)`` does **not** repurpose a
    grayscale channel as alpha — Qt fills it with the source's actual alpha
    (which is opaque-255 for Grayscale8), giving a uniform tint. Use
    ``setAlphaChannel`` instead: when given a grayscale image, it copies the
    luminance into the alpha channel of an ARGB32 layer.

    Alignment: masks are generated at the **bucket** resolution
    (``post_image_dataset/resized/`` = scale-to-cover + center-crop of the
    original in ``image_dataset/``). A plain ``IgnoreAspectRatio`` rescale
    onto the source would (a) stretch non-uniformly when ARs differ and
    (b) ignore the cropped-out margins — both contribute visible drift on
    the original-image view. Invert the bucket transform: scale the mask
    uniformly to match the appropriate axis, then letterbox the other axis
    so masked features land where the trainer actually saw them.
    """
    mask_img = QImage(str(mask_path))
    if mask_img.isNull():
        return source
    gray = mask_img.convertToFormat(QImage.Format_Grayscale8)
    gray.invertPixels()  # bubble (was 0) → 255, train-here (was 255) → 0

    src_w, src_h = source.width(), source.height()
    mask_w, mask_h = gray.width(), gray.height()
    if (src_w, src_h) == (mask_w, mask_h):
        aligned = gray
    elif src_w * mask_h >= src_h * mask_w:
        # ar_src >= ar_mask: bucket cropped left/right; match height, letterbox width.
        scaled_w = max(1, round(mask_w * src_h / mask_h))
        scaled = gray.scaled(
            scaled_w, src_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        aligned = QImage(src_w, src_h, QImage.Format_Grayscale8)
        aligned.fill(0)  # 0 = no tint on the cropped-out bars
        offset_x = max(0, (src_w - scaled_w) // 2)
        painter = QPainter(aligned)
        try:
            painter.drawImage(offset_x, 0, scaled)
        finally:
            painter.end()
    else:
        # ar_src < ar_mask: bucket cropped top/bottom; match width, letterbox height.
        scaled_h = max(1, round(mask_h * src_w / mask_w))
        scaled = gray.scaled(
            src_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        aligned = QImage(src_w, src_h, QImage.Format_Grayscale8)
        aligned.fill(0)
        offset_y = max(0, (src_h - scaled_h) // 2)
        painter = QPainter(aligned)
        try:
            painter.drawImage(0, offset_y, scaled)
        finally:
            painter.end()

    layer = QImage(source.size(), QImage.Format_ARGB32)
    layer.fill(_MASK_OVERLAY_COLOR_OPAQUE)
    layer.setAlphaChannel(aligned)

    result = QPixmap(source)
    p = QPainter(result)
    try:
        p.setOpacity(_MASK_OVERLAY_OPACITY)
        p.drawImage(0, 0, layer)
    finally:
        p.end()
    return result


def _load_resize_preview_target_res():
    path = ROOT / "configs" / "preprocess.toml"
    if not path.is_file():
        return None
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return None
    return data.get("target_res")


def _compose_resize_preview_overlay(
    source: QPixmap,
    target_res,
    crop_anchor=None,
    bucket_resos=None,
    crop_margins=None,
) -> QPixmap:
    try:
        preview = compute_resize_preview(
            source.width(),
            source.height(),
            target_res,
            crop_anchor=crop_anchor,
            bucket_resos=bucket_resos,
            crop_margins=crop_margins,
        )
    except (KeyError, TypeError, ValueError):
        return source

    rect = preview.kept_rect
    left = max(0, min(source.width(), round(rect.left)))
    top = max(0, min(source.height(), round(rect.top)))
    right = max(left, min(source.width(), round(rect.left + rect.width)))
    bottom = max(top, min(source.height(), round(rect.top + rect.height)))
    margin = preview.margin_rect
    margin_left = max(0, min(source.width(), round(margin.left)))
    margin_top = max(0, min(source.height(), round(margin.top)))
    margin_right = max(
        margin_left,
        min(source.width(), round(margin.left + margin.width)),
    )
    margin_bottom = max(
        margin_top,
        min(source.height(), round(margin.top + margin.height)),
    )

    result = QPixmap(source)
    painter = QPainter(result)
    try:
        painter.setPen(Qt.NoPen)
        painter.fillRect(0, 0, source.width(), top, _RESIZE_PREVIEW_SHADE)
        painter.fillRect(
            0,
            bottom,
            source.width(),
            source.height() - bottom,
            _RESIZE_PREVIEW_SHADE,
        )
        painter.fillRect(0, top, left, bottom - top, _RESIZE_PREVIEW_SHADE)
        painter.fillRect(
            right,
            top,
            source.width() - right,
            bottom - top,
            _RESIZE_PREVIEW_SHADE,
        )

        pen_width = max(2, round(min(source.width(), source.height()) * 0.004))
        if any(value > 0 for value in preview.crop_margins.values()):
            painter.setPen(QPen(_RESIZE_MARGIN_COLOR, pen_width, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                QRect(
                    margin_left,
                    margin_top,
                    margin_right - margin_left,
                    margin_bottom - margin_top,
                ).adjusted(1, 1, -1, -1)
            )

        painter.setPen(QPen(_RESIZE_PREVIEW_COLOR, pen_width, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(
            QRect(left, top, right - left, bottom - top).adjusted(1, 1, -1, -1)
        )

    finally:
        painter.end()
    return result


# Translucent green for inserted spans; deletions aren't rendered inline (they
# surface via the (+X / −Y) summary in the caption header).
_ADD_BG = QColor(60, 130, 70, 120)


def _add_format() -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setBackground(_ADD_BG)
    return fmt


def _diff_spans(old: str, new: str) -> tuple[list[tuple[int, int]], int, int]:
    """Char-level diff between old and new.

    Returns (insert_spans_in_new, total_added_chars, total_removed_chars).
    insert_spans are (j1, j2) ranges in `new` that should be highlighted.
    """
    if old == new:
        return [], 0, 0
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    spans: list[tuple[int, int]] = []
    add_total = 0
    rem_total = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            spans.append((j1, j2))
            add_total += j2 - j1
        elif tag == "replace":
            spans.append((j1, j2))
            add_total += j2 - j1
            rem_total += i2 - i1
        elif tag == "delete":
            rem_total += i2 - i1
    return spans, add_total, rem_total


def _history_path(caption_path: Path) -> Path:
    return caption_path.with_suffix(caption_path.suffix + ".history.jsonl")


def _read_history(caption_path: Path) -> list[dict]:
    """Return history entries (oldest first). Skips malformed lines."""
    hp = _history_path(caption_path)
    if not hp.exists():
        return []
    out: list[dict] = []
    for line in hp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "ts" in entry and "text" in entry:
            out.append(entry)
    return out


def _append_history(caption_path: Path, prev_text: str) -> None:
    """Append the previous on-disk text as a history entry."""
    hp = _history_path(caption_path)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "text": prev_text}
    with hp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Border colors for inline tag boxes. @artist and "On the …" / "In the …"
# section headers keep warm/cool tints so the trainer's split rules
# (anima_smart_shuffle in library/anima/training.py) stay visible.
_BOX_BORDER_PLAIN = QColor("#e0e0e0")
_BOX_BORDER_ARTIST = QColor("#c9a227")
_BOX_BORDER_SECTION = QColor("#5e8eb0")


def _tag_ranges(text: str):
    """Yield ``(start, end, tag_text)`` for each comma-separated, trimmed tag.

    Whitespace around each tag is excluded from the range so the painted box
    hugs the visible characters, not the surrounding spaces.
    """
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\n":
            i += 1
        start = i
        while i < n and text[i] != ",":
            i += 1
        end = i
        while end > start and text[end - 1] in " \t\n":
            end -= 1
        if end > start:
            yield (start, end, text[start:end])
        if i < n and text[i] == ",":
            i += 1


def _tag_border_color(tag: str) -> QColor:
    # Mirror library.anima.training._is_artist_tag: `@<non-space>` is an artist
    # handle, but `@ @` (space-form booru eye-shape) is a general tag and must
    # not steal the artist tint. Inline to keep this module free of library/*.
    if len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace():
        return _BOX_BORDER_ARTIST
    if (
        tag.startswith("On the ")
        or tag.startswith("In the ")
        or ". On the " in tag
        or ". In the " in tag
    ):
        return _BOX_BORDER_SECTION
    return _BOX_BORDER_PLAIN


class BoxedCaptionEdit(QTextEdit):
    """QTextEdit that paints thin border boxes inline around each
    comma-separated tag.

    Uses ``viewportEvent`` rather than ``QTextCharFormat`` because Qt's
    text framework can set per-character backgrounds and foregrounds but
    not borders. We let Qt render the text normally, then overlay
    rectangles on the viewport by walking ``cursorRect()`` across each
    tag's character range. Boxes follow scroll, wrap, and live edits
    automatically because ``cursorRect()`` is always queried in current
    viewport coordinates.

    The font is configured with extra letter spacing and the document with
    a roomier line height so tag boxes have visible breathing room both
    horizontally (the comma+space between tags is wider) and vertically
    (wrapped lines don't crowd their box borders together).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPixelSize(14)
        # 115% letter spacing widens the gap between adjacent boxes instead of
        # manufacturing gaps via per-box geometry.
        font.setLetterSpacing(QFont.PercentageSpacing, 115)
        self.setFont(font)
        self._apply_block_format()

    def setPlainText(self, text: str) -> None:  # noqa: N802 — Qt API
        # setPlainText replaces the document, so the line-height format we
        # applied earlier gets reset. Reapply after every full replacement.
        super().setPlainText(text)
        self._apply_block_format()

    def _apply_block_format(self) -> None:
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        fmt = QTextBlockFormat()
        # 140% ProportionalHeight: vertical separation between wrapped lines.
        fmt.setLineHeight(
            140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )
        cursor.mergeBlockFormat(fmt)

    def viewportEvent(self, event) -> bool:  # noqa: N802 — Qt API
        result = super().viewportEvent(event)
        if event.type() == QEvent.Paint:
            self._paint_boxes()
        return result

    def _paint_boxes(self) -> None:
        text = self.toPlainText()
        if not text.strip():
            return
        painter = QPainter(self.viewport())
        try:
            painter.setBrush(Qt.NoBrush)
            for start, end, tag in _tag_ranges(text):
                pen = QPen(_tag_border_color(tag))
                pen.setWidth(1)
                painter.setPen(pen)
                for r in self._tag_rects(start, end):
                    if r.width() > 0:
                        painter.drawRoundedRect(r, 2, 2)
        finally:
            painter.end()

    def _tag_rects(self, start: int, end: int) -> list[QRect]:
        """Per-line bounding rectangles for char range ``[start, end)``.

        Walks character-by-character so soft wraps (visual line breaks
        without an actual ``\\n``) get their own rectangle. For a typical
        caption (~500 chars) this is a few hundred ``cursorRect`` calls
        per paint — well under the budget for live editing.
        """
        if end <= start:
            return []
        cur = QTextCursor(self.document())
        cur.setPosition(start)
        cr = self.cursorRect(cur)
        line_left = cr.left()
        line_right = cr.left()
        line_top = cr.top()
        line_height = cr.height()
        rects: list[QRect] = []

        # Negative pad → box extends 1px OUTWARD so glyphs sit inside with a
        # halo; small extension leaves the comma+space gap between boxes wide.
        pad_x = -1
        pad_y = -1

        def _emit() -> None:
            w = line_right - line_left - 2 * pad_x
            h = line_height - 2 * pad_y
            if w > 0 and h > 0:
                rects.append(QRect(line_left + pad_x, line_top + pad_y, w, h))

        for pos in range(start + 1, end + 1):
            cur.setPosition(pos)
            cr = self.cursorRect(cur)
            if cr.top() != line_top:
                _emit()
                line_left = cr.left()
                line_right = cr.left()
                line_top = cr.top()
                line_height = cr.height()
            else:
                line_right = cr.left()
        _emit()
        return rects


def _unified_diff_html(old: str, new: str) -> str:
    """Tiny unified diff with red-/green+ coloring; empty string means no changes."""
    if old == new:
        return ""
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm="",
        n=3,
    )
    rows: list[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue  # filenames are noise here
        if line.startswith("@@"):
            rows.append(f'<span style="color:{tok("link")};">{escape(line)}</span>')
        elif line.startswith("+"):
            rows.append(f'<span style="color:#9ad17a;">{escape(line)}</span>')
        elif line.startswith("-"):
            rows.append(f'<span style="color:#e07a7a;">{escape(line)}</span>')
        else:
            rows.append(f'<span style="color:{tok("text_dim")};">{escape(line)}</span>')
    if not rows:
        return ""
    return (
        '<pre style="font-family:monospace;font-size:11px;margin:0;">'
        + "\n".join(rows)
        + "</pre>"
    )


class CaptionVersionsDialog(QDialog):
    """Browse prior versions of a caption and restore one in-place."""

    def __init__(self, caption_path: Path, current_disk_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("caption_versions_title", name=caption_path.stem))
        self.resize(820, 520)
        self._caption_path = caption_path
        self._current = current_disk_text
        self._restored: str | None = None  # set on Restore

        history = _read_history(caption_path)
        # Newest first — that's what users want to see at the top.
        self._history = list(reversed(history))

        lay = QVBoxLayout(self)

        sp = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        if not self._history:
            self.list.addItem(t("caption_versions_empty"))
            self.list.setEnabled(False)
        else:
            for entry in self._history:
                self.list.addItem(entry["ts"])
        self.list.currentRowChanged.connect(self._show_diff)
        sp.addWidget(self.list)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.diff = QTextBrowser()
        self.diff.setStyleSheet(
            f"QTextBrowser {{ background:{tok('base')}; color:{tok('text')}; "
            f"border:1px solid {tok('border_dim')}; padding:6px; }}"
        )
        rl.addWidget(self.diff, 1)
        sp.addWidget(right)
        sp.setSizes([220, 600])
        lay.addWidget(sp, 1)

        btns = QDialogButtonBox()
        self.restore_btn = btns.addButton(
            t("caption_versions_restore"), QDialogButtonBox.AcceptRole
        )
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self._restore)
        close_btn = btns.addButton(
            t("caption_versions_close"), QDialogButtonBox.RejectRole
        )
        close_btn.clicked.connect(self.reject)
        lay.addWidget(btns)

        if self._history:
            self.list.setCurrentRow(0)

    def _show_diff(self, row: int) -> None:
        if not (0 <= row < len(self._history)):
            self.restore_btn.setEnabled(False)
            self.diff.setHtml("")
            return
        prev = self._history[row]["text"]
        html = _unified_diff_html(prev, self._current)
        if not html:
            self.diff.setHtml(
                f'<i style="color:{tok("text_dim")};">{t("caption_diff_clean")}</i>'
            )
        else:
            self.diff.setHtml(html)
        self.restore_btn.setEnabled(True)

    def _restore(self) -> None:
        row = self.list.currentRow()
        if not (0 <= row < len(self._history)):
            return
        self._restored = self._history[row]["text"]
        self.accept()

    def restored_text(self) -> str | None:
        return self._restored


class ImageViewerTab(DaemonJobMixin, LazyTabMixin, QWidget):
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
        self.resize_preview_bucket_combo = QComboBox()
        self.resize_preview_bucket_combo.setToolTip(
            t("dataset_resize_preview_bucket_tooltip")
        )
        self.resize_preview_bucket_combo.setEnabled(False)
        self.resize_preview_bucket_combo.currentIndexChanged.connect(
            self._on_overlay_toggled
        )
        img_head.addWidget(self.resize_preview_bucket_combo)
        self.preprocess_use_btn = QPushButton(t("dataset_preprocess_use_short"))
        self.preprocess_use_btn.setToolTip(t("dataset_preprocess_use_tooltip"))
        self.preprocess_use_btn.clicked.connect(
            lambda: self._set_current_preprocess_decision("use", advance=True)
        )
        img_head.addWidget(self.preprocess_use_btn)
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
        self.preprocess_decision_label = QLabel(t("dataset_preprocess_decision_none"))
        self.preprocess_decision_label.setStyleSheet(
            f"QLabel {{ color:{tok('text_dim')}; padding:2px 0; }}"
        )
        rl.addWidget(self.preprocess_decision_label)

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
        self.versions_btn = QPushButton(t("caption_versions"))
        self.versions_btn.clicked.connect(self._open_versions)
        cap_head.addWidget(self.save_btn)
        cap_head.addWidget(self.revert_btn)
        cap_head.addWidget(self.autotag_btn)
        cap_head.addWidget(self.versions_btn)
        rl.addLayout(cap_head)

        # Caption editor with inline tag-box overlay; @artist and section
        # headers use accent colors so the trainer's split rules
        # (anima_smart_shuffle in library/anima/training.py) stay visible.
        self.cap = BoxedCaptionEdit()
        self.cap.setMaximumHeight(180)
        self.cap.textChanged.connect(self._on_text_changed)
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
        lay.addWidget(sp)

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
            use_sc = QShortcut(
                QKeySequence("A"),
                target,
                lambda: self._set_current_preprocess_decision("use", advance=True),
            )
            use_sc.setContext(Qt.WidgetShortcut)
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
        self, text: str, tooltip: str, clicked_cb, actions
    ) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        main_btn = QPushButton(text)
        main_btn.setToolTip(tooltip)
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
        # Grouping is GPU work — free the resident tagger first so they don't
        # fight over VRAM.
        self._kill_tagger_worker()
        # Busy UI before the submit so a cold-start daemon spin-up feels responsive.
        self.group_btn.setEnabled(False)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("dataset_group_rebuild"))
        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(
                label="curate-group", argv=["tasks.py", "curate-group"], start=True
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
        self._show(idx)
        self._refresh_mark_styles()

    def _show(self, row: int):
        if not 0 <= row < len(self._images):
            return
        p = self._images[row]
        pm = QPixmap(str(p))
        if not pm.isNull():
            self._set_image(p, pm)
        else:
            self._set_image(p, None)
        cp = p.with_suffix(".txt")
        self._current_caption_path = cp
        if cp.exists():
            text = cp.read_text(encoding="utf-8")
        else:
            text = ""
        self._disk_text = text
        self._set_caption_text(text if text else "")
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
        self.resize_preview_bucket_combo.setEnabled(source is not None)
        self._refresh_resize_preview_buckets()
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
            target_res, crop_anchor, bucket_resos, crop_margins = (
                self._resize_preview_config()
            )
            pm = _compose_resize_preview_overlay(
                pm,
                target_res,
                crop_anchor=crop_anchor,
                bucket_resos=bucket_resos,
                crop_margins=crop_margins,
            )
        self.img.set_source(pm)

    def _on_overlay_toggled(self, _value=None) -> None:
        self._refresh_resize_preview_buckets()
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
        selected = self.resize_preview_bucket_combo.currentData()
        if selected:
            bucket_resos = [selected]
        return target_res, crop_anchor, bucket_resos, crop_margins

    def _refresh_resize_preview_buckets(self) -> None:
        combo = self.resize_preview_bucket_combo
        current = combo.currentData()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(t("dataset_resize_preview_bucket_auto"), "")
            try:
                tiers = normalize_target_res(self._resize_preview_target_res())
                buckets = buckets_for_edges(tiers)
            except (TypeError, ValueError, KeyError):
                buckets = []
            for label in dict.fromkeys(format_bucket_resos(buckets)):
                width, height = label.split("x", 1)
                ratio = int(width) / int(height)
                combo.addItem(f"{label} ({ratio:.2f})", label)
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            combo.blockSignals(False)

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
            target_res, crop_anchor, bucket_resos, crop_margins = (
                self._resize_preview_config()
            )
            preview = compute_resize_preview(
                width,
                height,
                target_res,
                crop_anchor=crop_anchor,
                bucket_resos=bucket_resos,
                crop_margins=crop_margins,
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
        if path is None:
            return t("dataset_preprocess_decision_none")
        if path in self._marked:
            return t("dataset_preprocess_decision_move")
        action = self._preprocess_decisions.get(path)
        if action == "skip":
            return t("dataset_preprocess_decision_skip")
        if action == "use":
            return t("dataset_preprocess_decision_use")
        return t("dataset_preprocess_decision_none")

    def _refresh_preprocess_controls(self) -> None:
        path = self._current_image_path()
        enabled = path is not None
        self.preprocess_use_btn.setEnabled(enabled)
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
        self.preprocess_decision_label.setText(self._preprocess_decision_text(path))

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
        self.resize_preview_bucket_combo.setEnabled(False)
        self.img.clear()
        self._refresh_image_meta(None)
        self._refresh_preprocess_controls()

    def _set_caption_text(self, text: str) -> None:
        self._suspend_dirty = True
        try:
            self.cap.setPlainText(text)
        finally:
            self._suspend_dirty = False

    def _on_text_changed(self) -> None:
        if self._suspend_dirty:
            return
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _is_dirty(self) -> bool:
        if self._current_caption_path is None:
            return False
        return self.cap.toPlainText() != self._disk_text

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
        new_text = self.cap.toPlainText()
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
