"""Standalone translation loader for the WebUI.

Loads STRINGS dicts from ``gui/i18n/*.py`` using importlib file loading
so that ``gui/__init__.py`` (which imports PySide6) is never executed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_I18N_DIR = _ROOT / "gui" / "i18n"

_cache: dict[str, dict[str, str]] | None = None


def _load_strings(lang: str) -> dict[str, str]:
    """Load the STRINGS dict from gui/i18n/<lang>.py without importing gui."""
    mod_path = _I18N_DIR / f"{lang}.py"
    if not mod_path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location(f"gui_i18n_{lang}", mod_path)
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
