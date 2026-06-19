"""Anima LoRA GUI — main window, dark theme, and entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import toml
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import (
    DEFAULT_AUTOTAG_CONFIDENCE,
    DEFAULT_GROUP_CELL_MATCH_MIN,
    DEFAULT_GROUP_MATCH_FRAC_MIN,
    get_setting,
    set_setting,
)
from gui import daemon as gui_daemon
from gui import theme as gui_theme
from gui._paths import (
    DEFAULT_CAPTION_INSERT_NO_ARTIST,
    DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS,
)
from gui.gpu_status import GpuStatusBar
from gui.widgets import action_button
from gui.i18n import (
    available_languages,
    current_language,
    load_language,
    save_language,
    t,
)
from gui.tabs.easycontrol_tab import EasyControlTab
from gui.tabs.image_tab import ImageViewerTab
from gui.tabs.merge_tab import MergeTab
from gui.tabs.methods_tab import MethodsTab
from gui.tabs.preprocess_tab import PreprocessingTab
from gui.tabs.queue_tab import QueueTab
from gui.tensorboard import TensorBoardTab
from gui.system_dialog import (
    GITHUB_ISSUES_URL,
    GITHUB_REPO_URL,
    check_for_update_async,
    open_models_dialog,
    open_update_dialog,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GUIDELINES = _REPO_ROOT / "docs" / "guidelines"
_GUIDEBOOK_BY_LANG: dict[str, Path] = {
    "en": _GUIDELINES / "guidebook.md",
    "ko": _GUIDELINES / "가이드북.md",
    "cn": _GUIDELINES / "指南书.md",
    "ja": _GUIDELINES / "ガイドブック.md",
}
_GUIDEBOOK_FALLBACK = _GUIDEBOOK_BY_LANG["en"]
ICON_PATH = Path(__file__).resolve().parent / "icon.ico"


def _guidebook_path() -> Path:
    return _GUIDEBOOK_BY_LANG.get(current_language(), _GUIDEBOOK_FALLBACK)


LANG_NAMES = {"en": "English", "ko": "한국어", "cn": "简体中文", "ja": "日本語"}

# Keeps the live MainWindow alive across the in-place rebuild that applies a
# language change (main() seeds it; MainWindow._reload_ui swaps it).
_WINDOW: MainWindow | None = None


def _dark(app: QApplication):
    """Apply the user's chosen named theme (Dark / Light / Sepia).

    Thin wrapper kept for call-site stability; the palette + stylesheet now live
    in ``gui.theme`` so individual widgets can share the same tokens."""
    gui_theme.apply_theme(app)


class GuidebookDialog(QDialog):
    """In-app markdown viewer for the guidebook."""

    def __init__(self, md_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("guidebook"))
        self.resize(900, 720)
        self._md_path = md_path

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setSearchPaths([str(md_path.parent)])
        self.browser.document().setBaseUrl(
            QUrl.fromLocalFile(str(md_path.parent) + "/")
        )
        # Default anchor color is pure blue — illegible on the dark bg.
        self.browser.document().setDefaultStyleSheet(
            "a { color: #ffb86b; text-decoration: underline; }"
            "a:visited { color: #e6944e; }"
            "code { background:#2a2a2a; padding:1px 4px; border-radius:3px; }"
            "pre { background:#2a2a2a; padding:8px; border-radius:4px; }"
        )
        self.browser.setStyleSheet(
            "QTextBrowser { background:#1e1e1e; color:#dcdcdc; "
            "border:1px solid #444; padding:12px; }"
        )
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as e:
            text = f"# Error\n\nCould not read `{md_path}`:\n\n`{e}`"
        self.browser.setMarkdown(text)
        lay.addWidget(self.browser)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        open_ext = QPushButton(t("guidebook_open_external"))
        open_ext.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._md_path)))
        )
        close = QPushButton(t("guidebook_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(open_ext)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)


def _mcp_paths() -> tuple[Path, Path]:
    """(venv python, MCP bridge script) for THIS checkout — real absolute
    paths, not the <repo> placeholder the docs use (scripts/daemon/README.md)."""
    venv_python = (
        _REPO_ROOT
        / ".venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    return venv_python, _REPO_ROOT / "scripts" / "daemon" / "mcp.py"


def _mcp_add_command() -> str:
    """The `claude mcp add` one-liner for Claude Code."""

    def q(p: Path) -> str:
        return f'"{p}"' if " " in str(p) else str(p)

    venv_python, bridge = _mcp_paths()
    return f"claude mcp add anima-daemon -- {q(venv_python)} {q(bridge)}"


def _mcp_json_config() -> str:
    """The client-agnostic mcpServers JSON block (Claude Desktop, OpenClaw, …).
    json.dumps so Windows backslashes come out escaped and paste-able."""
    import json

    venv_python, bridge = _mcp_paths()
    cfg = {
        "mcpServers": {
            "anima-daemon": {"command": str(venv_python), "args": [str(bridge)]}
        }
    }
    return json.dumps(cfg, indent=2)


class SettingsDialog(QDialog):
    """App settings: language + MCP server registration for agent clients."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(560)
        # Set when the user opts into an immediate reload; MainWindow checks it after exec().
        self.reload_requested = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel(t("language")))
        self.lang_combo = QComboBox()
        for code in available_languages():
            self.lang_combo.addItem(LANG_NAMES.get(code, code), code)
        self.lang_combo.setCurrentIndex(available_languages().index(current_language()))
        self.lang_combo.currentIndexChanged.connect(self._change_lang)
        self.lang_combo.setFixedWidth(120)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        lay.addLayout(lang_row)

        prefs_group = QGroupBox(t("settings_prefs_header"))
        prefs_lay = QVBoxLayout(prefs_group)

        # Autotagger confidence floor (applied on top of the model's per-tag F1
        # thresholds; see AnimaTagger.predict_caption min_confidence).
        conf_row = QHBoxLayout()
        conf_label = QLabel(t("settings_autotag_confidence"))
        conf_label.setToolTip(t("settings_autotag_confidence_tooltip"))
        conf_row.addWidget(conf_label)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip(t("settings_autotag_confidence_tooltip"))
        self.conf_spin.setValue(
            float(get_setting("autotag_confidence", DEFAULT_AUTOTAG_CONFIDENCE))
        )
        self.conf_spin.valueChanged.connect(
            lambda v: set_setting("autotag_confidence", round(float(v), 2))
        )
        self.conf_spin.setFixedWidth(125)
        conf_row.addWidget(self.conf_spin)
        conf_row.addStretch()
        prefs_lay.addLayout(conf_row)

        self.caption_insert_no_artist = QCheckBox(
            t("settings_caption_insert_no_artist")
        )
        self.caption_insert_no_artist.setToolTip(
            t("settings_caption_insert_no_artist_tooltip")
        )
        self.caption_insert_no_artist.setChecked(
            bool(
                get_setting(
                    "caption_insert_no_artist", DEFAULT_CAPTION_INSERT_NO_ARTIST
                )
            )
        )
        self.caption_insert_no_artist.toggled.connect(
            lambda checked: set_setting("caption_insert_no_artist", bool(checked))
        )
        prefs_lay.addWidget(self.caption_insert_no_artist)

        self.caption_validate_artist_tags = QCheckBox(
            t("settings_caption_validate_artist_tags")
        )
        self.caption_validate_artist_tags.setToolTip(
            t("settings_caption_validate_artist_tags_tooltip")
        )
        self.caption_validate_artist_tags.setChecked(
            bool(
                get_setting(
                    "caption_validate_artist_tags",
                    DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS,
                )
            )
        )
        self.caption_validate_artist_tags.toggled.connect(
            lambda checked: set_setting("caption_validate_artist_tags", bool(checked))
        )
        prefs_lay.addWidget(self.caption_validate_artist_tags)

        # Dataset-tab grouping (`curate-group`) tightness. Higher = tighter,
        # cleaner groups. Read at grouping time by ImageViewerTab._rebuild_groups
        # and passed as --match-frac-min / --cell-match-min.
        frac_row = QHBoxLayout()
        frac_label = QLabel(t("settings_group_match_frac"))
        frac_label.setToolTip(t("settings_group_match_frac_tooltip"))
        frac_row.addWidget(frac_label)
        self.group_frac_spin = QDoubleSpinBox()
        self.group_frac_spin.setRange(0.0, 1.0)
        self.group_frac_spin.setSingleStep(0.05)
        self.group_frac_spin.setDecimals(2)
        self.group_frac_spin.setToolTip(t("settings_group_match_frac_tooltip"))
        self.group_frac_spin.setValue(
            float(get_setting("group_match_frac_min", DEFAULT_GROUP_MATCH_FRAC_MIN))
        )
        self.group_frac_spin.valueChanged.connect(
            lambda v: set_setting("group_match_frac_min", round(float(v), 2))
        )
        self.group_frac_spin.setFixedWidth(125)
        frac_row.addWidget(self.group_frac_spin)
        frac_row.addStretch()
        prefs_lay.addLayout(frac_row)

        cell_row = QHBoxLayout()
        cell_label = QLabel(t("settings_group_cell_match"))
        cell_label.setToolTip(t("settings_group_cell_match_tooltip"))
        cell_row.addWidget(cell_label)
        self.group_cell_spin = QDoubleSpinBox()
        self.group_cell_spin.setRange(0.0, 1.0)
        self.group_cell_spin.setSingleStep(0.01)
        self.group_cell_spin.setDecimals(2)
        self.group_cell_spin.setToolTip(t("settings_group_cell_match_tooltip"))
        self.group_cell_spin.setValue(
            float(get_setting("group_cell_match_min", DEFAULT_GROUP_CELL_MATCH_MIN))
        )
        self.group_cell_spin.valueChanged.connect(
            lambda v: set_setting("group_cell_match_min", round(float(v), 2))
        )
        self.group_cell_spin.setFixedWidth(125)
        cell_row.addWidget(self.group_cell_spin)
        cell_row.addStretch()
        prefs_lay.addLayout(cell_row)

        # Closing the dialog rebuilds the window so each tab's per-widget tokens (gui.theme.tok) repaint.
        theme_row = QHBoxLayout()
        theme_label = QLabel(t("settings_theme"))
        theme_label.setToolTip(t("settings_theme_tooltip"))
        theme_row.addWidget(theme_label)
        self.theme_combo = QComboBox()
        for code in gui_theme.THEME_ORDER:
            self.theme_combo.addItem(t(gui_theme.THEME_LABEL_KEYS[code]), code)
        cur = gui_theme.current_theme_name()
        self.theme_combo.setCurrentIndex(gui_theme.THEME_ORDER.index(cur))
        self.theme_combo.setToolTip(t("settings_theme_tooltip"))
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        self.theme_combo.setFixedWidth(140)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        prefs_lay.addLayout(theme_row)

        # Closing the dialog rebuilds the window so widgets sized to the old metrics relayout cleanly.
        font_row = QHBoxLayout()
        font_label = QLabel(t("settings_font_size"))
        font_label.setToolTip(t("settings_font_size_tooltip"))
        font_row.addWidget(font_label)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(gui_theme.FONT_SIZE_MIN, gui_theme.FONT_SIZE_MAX)
        self.font_spin.setSuffix(" pt")
        self.font_spin.setToolTip(t("settings_font_size_tooltip"))
        self.font_spin.setValue(gui_theme.current_font_size())
        self.font_spin.valueChanged.connect(self._change_font_size)
        self.font_spin.setFixedWidth(125)
        font_row.addWidget(self.font_spin)
        font_row.addStretch()
        prefs_lay.addLayout(font_row)

        # Debug mode: turns on DEBUG-level daemon logging (picked up the next time
        # the daemon starts) so a bug report can carry the launch/queue decisions.
        # Pairs with the "Copy debug report" button below.
        self.debug_check = QCheckBox(t("settings_debug_mode"))
        self.debug_check.setToolTip(t("settings_debug_mode_tooltip"))
        self.debug_check.setChecked(bool(get_setting("debug_mode", False)))
        self.debug_check.toggled.connect(self._toggle_debug)
        prefs_lay.addWidget(self.debug_check)

        dbg_desc = QLabel(t("settings_debug_report_desc"))
        dbg_desc.setWordWrap(True)
        prefs_lay.addWidget(dbg_desc)
        dbg_row = QHBoxLayout()
        dbg_row.addStretch()
        self.debug_report_btn = QPushButton(t("settings_debug_copy_report"))
        self.debug_report_btn.clicked.connect(self._copy_debug_report)
        dbg_row.addWidget(self.debug_report_btn)
        prefs_lay.addLayout(dbg_row)

        lay.addWidget(prefs_group)

        mcp_group = QGroupBox(t("settings_mcp_header"))
        mcp_lay = QVBoxLayout(mcp_group)
        self._add_command_block(
            mcp_lay, t("settings_mcp_desc"), _mcp_add_command(), height=64
        )
        self._add_command_block(
            mcp_lay, t("settings_mcp_desc_json"), _mcp_json_config(), height=140
        )
        lay.addWidget(mcp_group)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        close = QPushButton(t("settings_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)

    def _add_command_block(
        self, layout: QVBoxLayout, desc: str, text: str, height: int
    ) -> None:
        """A word-wrapped description, a read-only monospace box, and a copy
        button that flashes confirmation."""
        label = QLabel(desc)
        label.setWordWrap(True)
        layout.addWidget(label)

        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.Monospace)
        edit.setFont(mono)
        edit.setFixedHeight(height)
        layout.addWidget(edit)

        copy_row = QHBoxLayout()
        copy_row.addStretch()
        btn = QPushButton(t("settings_mcp_copy"))
        btn.clicked.connect(lambda: self._copy(text, btn))
        copy_row.addWidget(btn)
        layout.addLayout(copy_row)

    def _copy(self, text: str, btn: QPushButton) -> None:
        QApplication.clipboard().setText(text)
        btn.setText(t("settings_mcp_copied"))
        QTimer.singleShot(1500, lambda: btn.setText(t("settings_mcp_copy")))

    def _toggle_debug(self, checked: bool) -> None:
        """Persist the flag and reflect it into the live process env. A daemon
        already running keeps its old log level until it restarts (the report
        still works); a newly-spawned one inherits ANIMA_DEBUG from here."""
        set_setting("debug_mode", bool(checked))
        if checked:
            os.environ["ANIMA_DEBUG"] = "1"
        else:
            os.environ.pop("ANIMA_DEBUG", None)

    def _copy_debug_report(self) -> None:
        from gui.debug_report import build_debug_report

        try:
            report = build_debug_report()
        except Exception as exc:  # noqa: BLE001 — diagnostics must never crash the dialog
            report = f"failed to build debug report: {exc}"
        QApplication.clipboard().setText(report)
        self.debug_report_btn.setText(t("settings_mcp_copied"))
        QTimer.singleShot(
            1500,
            lambda: self.debug_report_btn.setText(t("settings_debug_copy_report")),
        )

    def _change_lang(self, idx: int):
        lang = self.lang_combo.itemData(idx)
        # save_language also flips the in-process language, so the prompt below already renders in it.
        save_language(lang)
        choice = QMessageBox.question(
            self,
            t("settings_lang_apply_title"),
            t("settings_lang_apply_question"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Yes:
            self.reload_requested = True
            self.accept()

    def _change_theme(self, idx: int) -> None:
        """Persist + apply the chosen theme live, then request a window rebuild.

        ``apply_theme`` restyles app-level chrome immediately; the rebuild on
        close makes per-widget ``tok()`` lookups (log boxes, previews, …) repaint
        too. Unlike a language change there's no confirm prompt — it's cheap and
        reversible."""
        name = self.theme_combo.itemData(idx)
        gui_theme.set_theme(name)
        app = QApplication.instance()
        if app is not None:
            gui_theme.apply_theme(app, name)
        self.reload_requested = True

    def _change_font_size(self, size: int) -> None:
        """Persist + apply the chosen app font size live, then request a rebuild.

        ``apply_theme`` re-reads the size into ``app.setFont``; the rebuild on
        close lets widgets that cached the old font metrics relayout. Like the
        theme, it's cheap and reversible, so there's no confirm prompt."""
        gui_theme.set_font_size(size)
        app = QApplication.instance()
        if app is not None:
            gui_theme.apply_theme(app)
        self.reload_requested = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_title"))
        self.resize(1100, 750)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        # App-wide event filter catches ContextMenu before the target widget so right-click is
        # uniform everywhere (text widgets would otherwise show Qt's copy/select-all menu).
        # Removed in closeEvent so a _reload_ui rebuild doesn't stack filters.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        lang_bar = QHBoxLayout()
        if ICON_PATH.exists():
            icon_label = QLabel()
            pix = QPixmap(str(ICON_PATH))
            if not pix.isNull():
                icon_label.setPixmap(
                    pix.scaled(
                        QSize(28, 28),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            icon_label.setContentsMargins(4, 0, 6, 0)
            lang_bar.addWidget(icon_label)
        self.guide_btn = action_button(
            t("guidebook"),
            variant="success",
            tooltip=t("guidebook_tooltip"),
            on_click=self._open_guidebook,
        )
        lang_bar.addWidget(self.guide_btn)

        self.models_btn = QPushButton(t("models_btn"))
        self.models_btn.setToolTip(t("models_btn_tooltip"))
        self.models_btn.clicked.connect(lambda: open_models_dialog(self))
        lang_bar.addWidget(self.models_btn)

        self.update_btn = QPushButton(t("update_btn"))
        self.update_btn.setToolTip(t("update_btn_tooltip"))
        self.update_btn.clicked.connect(lambda: open_update_dialog(self))
        lang_bar.addWidget(self.update_btn)
        # Background check — paints the button amber + "Update ●" when a newer
        # release exists. Skips silently when .anima_release.json is missing
        # (no baseline → can't tell if user is already current) and reuses the
        # 6h gui_settings.json cache so launches don't hit GitHub each time.
        self._update_check_thread = check_for_update_async(
            self, self._show_update_available
        )

        self.issues_btn = QPushButton(t("report_issue"))
        self.issues_btn.setToolTip(t("report_issue_tooltip"))
        self.issues_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_ISSUES_URL))
        )
        lang_bar.addWidget(self.issues_btn)

        # Top-bar toggle, not a tab: the daemon job queue is global (spans every method); overlays are mutually exclusive.
        self.queue_btn = QPushButton(t("tab_queue"))
        self.queue_btn.setCheckable(True)
        self._queue_idle_style = gui_theme.nav_button_qss("queue")
        self._queue_active_style = gui_theme.nav_button_qss("queue_on")
        self.queue_btn.toggled.connect(self._toggle_queue_view)
        lang_bar.addWidget(self.queue_btn)

        # Top-bar toggle, not a tab: the run list is shared across every method (one global view).
        self.tensorboard_btn = QPushButton(t("tab_tensorboard"))
        self.tensorboard_btn.setCheckable(True)
        self._tensorboard_idle_style = gui_theme.nav_button_qss("tensorboard")
        self._tensorboard_active_style = gui_theme.nav_button_qss("tensorboard_on")
        self.tensorboard_btn.toggled.connect(self._toggle_tensorboard)
        lang_bar.addWidget(self.tensorboard_btn)

        lang_bar.addStretch()
        self.settings_btn = QPushButton(t("settings_btn"))
        self.settings_btn.setToolTip(t("settings_btn_tooltip"))
        self.settings_btn.clicked.connect(self._open_settings)
        lang_bar.addWidget(self.settings_btn)
        main_lay.addLayout(lang_bar)

        # Overlays share a QStackedWidget with the tab set; all widgets stay alive across switches
        # so subprocess state and log buffers survive toggling. ConfigTab and MethodsTab both hold
        # a reference to this shared panel's `.panel` to sync the log dir on variant switch / launch.
        self._tb_tab = TensorBoardTab()

        # Built before ConfigTab so the Train auto-chain can flush this tab's
        # GUI preprocess settings to the selected method before it preprocesses.
        self._preprocess_tab = PreprocessingTab()

        self.tabs = QTabWidget()
        # The Config tab is a MethodsTab picker over the shipped LoRA family plus
        # the (now promoted) Turbo distiller; the Experimental tab is the same
        # wrapper over the research methods + the SPD distiller.
        self.tabs.addTab(
            MethodsTab(
                tb_panel=self._tb_tab.panel,
                flat_methods=("lora", "tlora", "hydralora"),
                distill_methods=("turbo",),
                preprocess_tab=self._preprocess_tab,
            ),
            t("tab_config"),
        )
        self.tabs.addTab(self._preprocess_tab, t("tab_preprocess"))
        self.tabs.addTab(
            ImageViewerTab(preprocess_tab=self._preprocess_tab), t("tab_images")
        )
        self.tabs.addTab(MergeTab(), t("tab_merge"))
        # MethodsTab folds every trainable experimental method behind one dropdown; EasyControl keeps
        # a dedicated tab because it has its own preprocess/dataset lifecycle.
        self.tabs.addTab(
            MethodsTab(
                tb_panel=self._tb_tab.panel,
                flat_methods=("chimera", "soft_tokens"),
                distill_methods=("spd",),
            ),
            t("tab_experimental"),
        )
        self.tabs.addTab(EasyControlTab(), t("tab_easycontrol"))

        self._queue_tab = QueueTab()

        self.tab_stack = QStackedWidget()
        self.tab_stack.addWidget(self.tabs)
        self.tab_stack.addWidget(self._tb_tab)
        self.tab_stack.addWidget(self._queue_tab)
        main_lay.addWidget(self.tab_stack)

        # Live GPU utilisation/VRAM footer, always visible below the tab set
        # (so it sits beneath the Config tab's CLI/log panel).
        self._gpu_bar = GpuStatusBar()
        main_lay.addWidget(self._gpu_bar)
        self.setCentralWidget(central)

        self._update_tensorboard_btn_style(False)
        self._update_queue_btn_style(False)

    def closeEvent(self, event):
        # Drop the app-wide context-menu filter so a _reload_ui rebuild doesn't leave a dead window filtering events.
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # Without this, closing the window leaves training subprocesses
        # (accelerate → train.py) orphaned and still holding VRAM.
        for i in range(self.tabs.count()):
            cleanup = getattr(self.tabs.widget(i), "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()
        # The shared TensorBoard + Queue views live in the stack, not a tab set.
        self._tb_tab.cleanup_subprocess()
        self._queue_tab.cleanup_subprocess()
        self._gpu_bar.cleanup()
        super().closeEvent(event)

    def _show_update_available(self, latest_tag: str) -> None:
        self.update_btn.setText(t("update_btn_available"))
        self.update_btn.setToolTip(t("update_btn_available_tooltip", v=latest_tag))
        self.update_btn.setStyleSheet(gui_theme.nav_button_qss("update"))

    def _clear_overlay_toggle(self, btn: QPushButton, style_fn) -> None:
        """Silently un-check an overlay toggle (TensorBoard / Queue) and repaint
        it, without re-firing its toggled handler."""
        if btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            style_fn(False)

    def _toggle_tensorboard(self, on: bool):
        if on:
            # TensorBoard and Queue are mutually-exclusive overlays.
            self._clear_overlay_toggle(self.queue_btn, self._update_queue_btn_style)
            self.tab_stack.setCurrentWidget(self._tb_tab)
        else:
            self.tab_stack.setCurrentWidget(self.tabs)
        self._update_tensorboard_btn_style(on)

    def _update_tensorboard_btn_style(self, on: bool):
        self.tensorboard_btn.setStyleSheet(
            self._tensorboard_active_style if on else self._tensorboard_idle_style
        )

    def _toggle_queue_view(self, on: bool):
        if on:
            # Queue and TensorBoard are mutually-exclusive overlays.
            self._clear_overlay_toggle(
                self.tensorboard_btn, self._update_tensorboard_btn_style
            )
            self.tab_stack.setCurrentWidget(self._queue_tab)
        else:
            self.tab_stack.setCurrentWidget(self.tabs)
        self._update_queue_btn_style(on)

    def _update_queue_btn_style(self, on: bool):
        self.queue_btn.setStyleSheet(
            self._queue_active_style if on else self._queue_idle_style
        )

    def eventFilter(self, obj, event):  # noqa: N802 — Qt event handler name
        """Intercept every right-click in the app and show our menu instead of
        the target widget's default one (text widgets ship a copy/select-all
        menu we want to override). Returning True consumes the event."""
        if event.type() == QEvent.ContextMenu:
            self._show_context_menu(event.globalPos())
            return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, global_pos):
        """Walk up from the widget under the cursor; the first ancestor exposing
        a callable ``app_context_menu(target, global_pos)`` gets to supply the
        menu (e.g. the dataset image view → open-in-system-viewer). If none does
        — or it declines by returning None — fall back to the app default."""
        target = QApplication.widgetAt(global_pos)
        w = target
        while w is not None:
            provider = getattr(w, "app_context_menu", None)
            if callable(provider):
                menu = provider(target, global_pos)
                if menu is not None:
                    menu.exec(global_pos)
                    return
                break
            w = w.parentWidget()
        self._show_app_menu(global_pos)

    def _show_app_menu(self, global_pos):
        """The default right-click menu — currently just a link to the repo."""
        menu = QMenu(self)
        visit = menu.addAction(t("visit_github"))
        visit.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL)))
        menu.exec(global_pos)

    def _open_guidebook(self):
        path = _guidebook_path()
        if not path.exists():
            QMessageBox.warning(
                self, t("guidebook"), t("guidebook_missing", path=str(path))
            )
            return
        dlg = GuidebookDialog(path, self)
        dlg.show()

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()
        if dlg.reload_requested:
            self._reload_ui()

    def _reload_ui(self):
        """Rebuild the main window in place to apply a language or theme change —
        every string and per-widget theme token is resolved at construction, so a
        fresh window is the cleanest way to re-render. The daemon owns running
        jobs, so only local UI
        state (unsaved edits, overlay subprocesses) resets; closeEvent reaps
        the old window's TensorBoard/Queue children as usual. New window is
        shown before the old closes so quitOnLastWindowClosed never fires."""
        global _WINDOW
        new = MainWindow()
        new.setGeometry(self.geometry())
        new.show()
        _WINDOW = new
        self.close()


def _ensure_source_image_dir() -> None:
    """Create the training source dir on launch so first-time users hit an
    empty folder rather than a confusing "no images found" error from the
    preprocess pipeline. Path comes from `source_image_dir` (now in
    configs/preprocess.toml; a legacy copy in base.toml still wins, mirroring
    load_path_overrides); falls back to `image_dataset/`.
    """
    src = "image_dataset"
    # Read order matches load_path_overrides' precedence: a legacy base.toml key (read second) wins.
    for fname in ("preprocess.toml", "base.toml"):
        cfg_path = _REPO_ROOT / "configs" / fname
        try:
            if cfg_path.exists():
                raw = toml.loads(cfg_path.read_text(encoding="utf-8"))
                cfg_src = raw.get("source_image_dir")
                if isinstance(cfg_src, str) and cfg_src.strip():
                    src = cfg_src
        except (OSError, toml.TomlDecodeError):
            pass
    src_path = Path(src)
    if not src_path.is_absolute():
        src_path = _REPO_ROOT / src_path
    try:
        src_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"warn: could not create {src_path}: {e}", file=sys.stderr)


def _prefer_cleartype_font_engine() -> None:
    """Use Qt's GDI font engine on Windows — but only at integer DPI scaling.

    Qt 6 defaults to the DirectWrite font engine on Windows, which rasterizes
    small UI text with *grayscale* antialiasing — it reads soft/blurry next to
    native apps, and the effect is worse on lightly-hinted modern faces like the
    bundled Pretendard. The GDI engine uses ClearType subpixel rendering (what
    native Windows controls use), which snaps small text crisp.

    But GDI ClearType is tied to the *physical* pixel grid: at the fractional
    display scaling most laptops ship (125% / 150%) it renders fringed and can
    come out undersized, because the GDI engine honors Qt's device-pixel-ratio
    less cleanly than DirectWrite. So we only force GDI when the display is at an
    integer scale (100% / 200%, where ClearType lines up) and let DirectWrite
    handle fractional-scaled screens.

    Reading the real scaling requires the process to be DPI-aware first — an
    unaware process always reports 96 DPI (100%). We set per-monitor-v2 awareness
    up front (the same context Qt 6 sets itself, so this only moves the call
    earlier) and query ``GetDpiForSystem``. The platform option must be set
    *before* ``QApplication`` is constructed. Skipped if the user already pinned
    ``QT_QPA_PLATFORM`` (explicit choice, or offscreen in tests), and falls back
    to DirectWrite on older Windows where the DPI APIs are missing."""
    if sys.platform != "win32" or "QT_QPA_PLATFORM" in os.environ:
        return
    try:
        import ctypes

        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4. Harmless if it fails (awareness already set).
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        dpi = ctypes.windll.user32.GetDpiForSystem()  # 96 == 100%
    except (OSError, AttributeError):
        return  # pre-1607 Windows / no DPI API — leave Qt on DirectWrite
    if not dpi:
        return
    scale = dpi / 96.0
    if abs(scale - round(scale)) < 0.01:  # integer scaling only
        os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=gdi"


def main():
    load_language()
    _ensure_source_image_dir()
    _prefer_cleartype_font_engine()
    # Pass fractional display scaling (125%/150%) through instead of snapping to 100%/200%
    # (tiny text on HiDPI Windows). Must be set before QApplication is constructed.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    _dark(app)
    # Reflect the persisted "Debug mode" setting into the env *before* the daemon
    # is spawned below, so a freshly-started daemon logs at DEBUG (see
    # scripts/daemon/__main__._setup_logging).
    if get_setting("debug_mode", False):
        os.environ["ANIMA_DEBUG"] = "1"
    global _WINDOW
    _WINDOW = MainWindow()
    _WINDOW.show()
    # Bring the local training daemon up at launch (idempotent). Deferred to after
    # the first paint via singleShot(0) so a cold-start daemon boot (process spawn
    # + /health poll) runs in the background instead of blocking the window from
    # appearing. Best-effort: a failure never surfaces here — the Train button's
    # own ensure_daemon() does the real wait-for-health when a job is submitted.
    QTimer.singleShot(0, gui_daemon.ensure_daemon_quietly)
    sys.exit(app.exec())
