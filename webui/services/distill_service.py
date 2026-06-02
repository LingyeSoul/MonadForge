"""Service for reading/writing sectioned distill TOML configs (spd, turbo)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit

_ROOT = Path(__file__).resolve().parent.parent.parent
_GUIDES_DIR = Path(__file__).resolve().parent.parent / "explanations" / "guides"

_DISTILL_METHODS: dict[str, dict[str, str]] = {
    "spd": {
        "path": "configs/methods/spd.toml",
        "task": "exp-spd",
        "label": "SPD",
    },
    "turbo": {
        "path": "configs/methods/turbo.toml",
        "task": "exp-turbo",
        "label": "Turbo",
    },
}


def list_distill_methods() -> list[dict[str, str]]:
    """Return available distill methods with metadata."""
    out: list[dict[str, str]] = []
    for key, meta in _DISTILL_METHODS.items():
        p = _ROOT / meta["path"]
        if p.is_file():
            out.append({
                "key": key,
                "label": meta["label"],
                "config_path": meta["path"],
                "task_command": meta["task"],
            })
    return out


def _python_type(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    return "str"


def read_distill_config(method: str) -> dict:
    """Read a sectioned distill TOML and return its structure.

    Returns ``{"sections": [{"name": str, "fields": [...]}]}``.
    """
    meta = _DISTILL_METHODS.get(method)
    if meta is None:
        raise ValueError(f"Unknown distill method: {method!r}")

    path = _ROOT / meta["path"]
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))

    sections: list[dict] = []
    root_fields: list[dict] = []

    for key, item in doc.body:
        if key is None:
            # comment or whitespace — skip at body level
            continue
        if isinstance(item, tomlkit.items.Table):
            fields: list[dict] = []
            for fkey, fval in item.value.body:
                if fkey is None:
                    continue
                val = fval
                comment = ""
                if hasattr(fval, "trivia") and fval.trivia.comment:
                    comment = fval.trivia.comment.strip().lstrip("#").strip()
                fields.append({
                    "key": str(fkey),
                    "value": val,
                    "type": _python_type(val),
                    "comment": comment,
                })
            sections.append({"name": str(key), "fields": fields})
        else:
            comment = ""
            if hasattr(item, "trivia") and item.trivia.comment:
                comment = item.trivia.comment.strip().lstrip("#").strip()
            root_fields.append({
                "key": str(key),
                "value": item,
                "type": _python_type(item),
                "comment": comment,
            })

    if root_fields:
        sections.insert(0, {"name": "root", "fields": root_fields})

    return {"method": method, "sections": sections}


def save_distill_config(
    method: str, updates: dict[str, dict[str, Any]]
) -> None:
    """Save field values back to the sectioned TOML, preserving comments.

    *updates* is ``{"section_name": {"key": new_value, ...}, ...}``.
    """
    meta = _DISTILL_METHODS.get(method)
    if meta is None:
        raise ValueError(f"Unknown distill method: {method!r}")

    path = _ROOT / meta["path"]
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))

    # Apply root-level updates
    root_updates = updates.get("root", {})
    for key, new_val in root_updates.items():
        if key in doc:
            doc[key] = new_val

    # Apply section-level updates
    for section_name, fields in updates.items():
        if section_name == "root":
            continue
        if section_name in doc and isinstance(doc[section_name], tomlkit.items.Table):
            tbl = doc[section_name]
            for key, new_val in fields.items():
                if key in tbl:
                    tbl[key] = new_val

    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def load_guide(method: str, lang: str = "cn") -> str:
    """Load guide HTML for a distill method, with lang → en fallback."""
    for try_lang in (lang, "en"):
        p = _GUIDES_DIR / try_lang / f"{method}.html"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""
