"""i18n API endpoint — serves translation strings as JSON."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Lazy import to avoid pulling PySide6 at module level
_translations: dict[str, dict[str, str]] | None = None


def _get_translations() -> dict[str, dict[str, str]]:
    global _translations
    if _translations is None:
        from gui.i18n import cn, en, ko

        _translations = {"en": en.STRINGS, "ko": ko.STRINGS, "cn": cn.STRINGS}
    return _translations


@router.get("/{lang}")
def get_translations(lang: str):
    """Return all translation strings for the given language code."""
    translations = _get_translations()
    if lang not in translations:
        # Fallback to English
        if lang != "en" and "en" in translations:
            return translations["en"]
        raise HTTPException(status_code=404, detail=f"Language '{lang}' not found")
    return translations[lang]


@router.get("")
def list_languages():
    """Return available language codes."""
    return {"languages": list(_get_translations().keys())}
