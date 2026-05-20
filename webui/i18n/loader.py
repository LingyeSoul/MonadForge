"""Standalone translation loader for the WebUI.

Loads STRINGS dicts from ``webui/i18n/strings/*.py`` directly — no PySide6
dependency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_I18N_DIR = Path(__file__).parent / "strings"

_cache: dict[str, dict[str, str]] | None = None
_current_lang = "en"


def _load_strings(lang: str) -> dict[str, str]:
    """Load the STRINGS dict from webui/i18n/strings/<lang>.py."""
    mod_path = _I18N_DIR / f"{lang}.py"
    if not mod_path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location(f"webui_i18n_{lang}", mod_path)
    if spec is None or spec.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "STRINGS", {})


def get_translations() -> dict[str, dict[str, str]]:
    """Return all translation dicts, cached after first call."""
    global _cache
    if _cache is None:
        _cache = {}
        for lang in ("en", "cn", "ko"):
            strings = _load_strings(lang)
            if strings:
                _cache[lang] = strings
    return _cache


def current_language() -> str:
    """Return the active language code."""
    return _current_lang


def set_language(lang: str):
    """Set the active language (no persistence — caller handles that)."""
    global _current_lang
    _current_lang = lang
