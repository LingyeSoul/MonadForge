"""Bilingual help text for config fields and LoRA variant descriptions.

Per-field tooltips live in ``guides/<lang>/_fields.json`` and
``guides/<lang>/_preprocess_fields.json`` — one JSON file per language,
loaded lazily on first access. Missing keys fall back to English.

Method/variant guide HTML blocks live under ``guides/<lang>/<name>.html``
and are also loaded lazily. Shared snippets (``_apply_note``,
``_not_mergeable``) follow the same convention with an underscore prefix.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from gui.i18n import current_language

_GUIDES_DIR = Path(__file__).parent / "guides"


# ── HTML guide loader ──────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _read_guide(name: str, lang: str) -> str:
    path = _GUIDES_DIR / lang / f"{name}.html"
    if not path.exists():
        path = _GUIDES_DIR / "en" / f"{name}.html"
    return path.read_text(encoding="utf-8")


def _guide(name: str) -> str:
    return _read_guide(name, current_language())


# ── JSON field-help loaders ────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _read_fields(lang: str) -> dict[str, str]:
    path = _GUIDES_DIR / lang / "_fields.json"
    if not path.exists():
        path = _GUIDES_DIR / "en" / "_fields.json"
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _read_preprocess_fields(lang: str) -> dict[str, str]:
    path = _GUIDES_DIR / lang / "_preprocess_fields.json"
    if not path.exists():
        path = _GUIDES_DIR / "en" / "_preprocess_fields.json"
    return json.loads(path.read_text(encoding="utf-8"))


def field_help(key: str) -> str | None:
    """Return the help string for *key* in the current language, or None."""
    lang = current_language()
    value = _read_fields(lang).get(key)
    if value is not None:
        return value
    if lang != "en":
        return _read_fields("en").get(key)
    return None


def preprocess_field_help(key: str) -> str | None:
    """Per-field help for the Preprocessing tab. Falls back to field_help."""
    lang = current_language()
    value = _read_preprocess_fields(lang).get(key)
    if value is not None:
        return value
    if lang != "en":
        en_value = _read_preprocess_fields("en").get(key)
        if en_value is not None:
            return en_value
    return field_help(key)


def preprocess_guide() -> str:
    return _guide("preprocess")


# ── Method guide dispatch ──────────────────────────────────────
# Methods that can't be baked into a plain DiT via scripts/merge_to_dit.py
# (router is layer-local / hook-only / not a weight delta) — render the
# "not mergeable" callout above their guide.
_NOT_MERGEABLE = frozenset({"postfix", "hydralora", "reft", "fera"})
_KNOWN_METHODS = frozenset({"lora", "tlora", "postfix", "hydralora", "reft", "fera"})


def method_guide(method: str) -> str | None:
    """Right-panel default HTML for *method*, or None if no guide is registered."""
    if method not in _KNOWN_METHODS:
        return None
    parts = [_guide("_apply_note")]
    if method in _NOT_MERGEABLE:
        parts.append(_guide("_not_mergeable"))
    parts.append(_guide(method))
    return "".join(parts)
