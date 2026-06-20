"""Service for reading/writing sectioned distill TOML configs (spd, turbo)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit

_ROOT = Path(__file__).resolve().parent.parent.parent
_GUIDES_DIR = Path(__file__).resolve().parent.parent / "explanations" / "guides"

_DISTILL_METHODS: dict[str, dict[str, str | bool]] = {
    "spd": {
        "path": "configs/methods/spd.toml",
        "task": "exp-spd",
        "label": "SPD",
        "experimental": True,
    },
    "turbo": {
        "path": "configs/methods/turbo.toml",
        "task": "turbo",
        "label": "Turbo",
        "experimental": True,
    },
}


def list_distill_methods() -> list[dict[str, Any]]:
    """Return available distill methods with metadata."""
    out: list[dict[str, Any]] = []
    for key, meta in _DISTILL_METHODS.items():
        p = _ROOT / meta["path"]
        if p.is_file():
            out.append({
                "key": key,
                "label": meta["label"],
                "config_path": meta["path"],
                "task_command": meta["task"],
                "experimental": bool(meta.get("experimental", False)),
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


def _to_python(v: Any) -> Any:
    """Coerce a tomlkit item to a plain JSON-native Python value.

    tomlkit scalars subclass the builtins, but pydantic's ``Any`` serializer
    rejects the subclasses on ``dump_json`` (and ``tomlkit.items.Bool`` isn't a
    ``bool`` subclass at all), so the raw items can't travel through a FastAPI
    ``response_model`` — they raise ``PydanticSerializationError`` at the
    ``/config`` boundary. Recurse through ``Array``/inline-table too.
    """
    if isinstance(v, tomlkit.items.Bool):
        return v.value
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):  # tomlkit Integer subclasses int
        return int(v)
    if isinstance(v, float):  # tomlkit Float subclasses float
        return float(v)
    if isinstance(v, str):  # tomlkit String subclasses str
        return str(v)
    if isinstance(v, list):  # tomlkit Array subclasses list
        return [_to_python(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_python(x) for k, x in v.items()}
    return v


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
                val = _to_python(fval)
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
            val = _to_python(item)
            comment = ""
            if hasattr(item, "trivia") and item.trivia.comment:
                comment = item.trivia.comment.strip().lstrip("#").strip()
            root_fields.append({
                "key": str(key),
                "value": val,
                "type": _python_type(val),
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
