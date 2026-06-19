"""Shared path constants for the GUI package.

Foundation module — imported by every other ``gui`` submodule and by
``gui/__init__.py``. Deliberately dependency-free (no Qt, no other ``gui``
imports) so it can be imported first without risking a cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Persistent UI state (language, update-check cache, preprocess knobs, and the
# generic preferences below). Separate from configs/ so it survives a config
# reset. The same file the i18n / system-dialog / preprocess modules read.
GUI_SETTINGS_FILE = Path(__file__).resolve().parent / "gui_settings.json"

# Default autotagger probability floor applied on top of the model's per-tag
# F1 thresholds (see AnimaTagger.predict_caption min_confidence).
DEFAULT_AUTOTAG_CONFIDENCE = 0.5
DEFAULT_CAPTION_INSERT_NO_ARTIST = True
DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS = False
# Defaults for the Dataset-tab grouping run (`curate-group`). Mirror the
# library values (library.datasets.grouping.DEFAULT_*) so the GUI stays
# torch-free; higher = tighter groups. See SettingsDialog + ImageViewerTab.
DEFAULT_GROUP_MATCH_FRAC_MIN = 0.25
DEFAULT_GROUP_CELL_MATCH_MIN = 0.93
# Kept for backward compat; the live accent now comes from the active theme (gui/theme.py).
DEFAULT_THEME_COLOR = "#3c78c8"
# Default named theme (see gui/theme.py THEMES). One of "dark" / "light" / "sepia".
DEFAULT_THEME = "dark"


def _read_gui_settings() -> dict:
    """Whole gui_settings.json as a dict (``{}`` if absent/unparseable)."""
    if not GUI_SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def get_setting(key: str, default=None):
    """Read one preference from gui_settings.json, or ``default`` if missing."""
    return _read_gui_settings().get(key, default)


def set_setting(key: str, value) -> None:
    """Persist one preference into gui_settings.json (merge, don't clobber)."""
    settings = _read_gui_settings()
    settings[key] = value
    try:
        GUI_SETTINGS_FILE.write_text(json.dumps(settings), encoding="utf-8")
    except OSError:
        pass


METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
CUSTOM_DIR = CONFIGS_DIR / "custom"
# User-created variants get their own subdir so they don't pollute the built-in family list.
CUSTOM_VARIANTS_DIR = GUI_METHODS_DIR / "custom"


_METHOD_ORDER = (
    "lora",
    "tlora",
    "hydralora",
    "fera",
    "chimera",
    "soft_tokens",
    "easycontrol",
)
