"""Documentation API endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent.parent
_GUIDELINES_DIR = _ROOT / "docs" / "guidelines"

# Language → guidebook filename mapping
_GUIDEBOOK_FILES: dict[str, str] = {
    "ko": "가이드북.md",
    "cn": "guidebook.md",
    "en": "guidebook_en.md",
}


@router.get("/guidebook")
def get_guidebook(lang: str = Query("cn")):
    """Return the guidebook markdown content for the given language."""
    filename = _GUIDEBOOK_FILES.get(lang, _GUIDEBOOK_FILES["en"])
    guidebook_path = _GUIDELINES_DIR / filename
    if not guidebook_path.is_file():
        # Fallback: try English/Chinese, then Korean
        for fallback in ("guidebook.md", "가이드북.md"):
            guidebook_path = _GUIDELINES_DIR / fallback
            if guidebook_path.is_file():
                break
        else:
            raise HTTPException(status_code=404, detail="Guidebook not found")
    content = guidebook_path.read_text(encoding="utf-8")
    return {"content": content, "filename": guidebook_path.name, "lang": lang}


class DocEntry(BaseModel):
    name: str
    filename: str


@router.get("/list", response_model=list[DocEntry])
def list_docs():
    """List available documentation files."""
    if not _GUIDELINES_DIR.is_dir():
        return []
    entries = []
    for p in sorted(_GUIDELINES_DIR.glob("*.md")):
        entries.append(DocEntry(name=p.stem, filename=p.name))
    return entries
