"""i18n API endpoint — serves translation strings as JSON."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from webui.i18n.loader import get_translations

router = APIRouter()


@router.get("/{lang}")
def get_lang_translations(lang: str):
    """Return all translation strings for the given language code."""
    translations = get_translations()
    if lang not in translations:
        if lang != "en" and "en" in translations:
            return translations["en"]
        raise HTTPException(status_code=404, detail=f"Language '{lang}' not found")
    return translations[lang]


@router.get("")
def list_languages():
    """Return available language codes."""
    return {"languages": list(get_translations().keys())}
