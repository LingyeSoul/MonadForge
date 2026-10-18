"""Service for reading/writing sectioned distill TOML configs (spd, turbo)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import tomlkit

from webui.services.config_storage import CONFIG_WRITE_LOCK, atomic_write_text

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


class UnknownDistillMethod(ValueError):
    """A method outside the explicitly supported distillation workspaces."""


def list_distill_methods() -> list[dict[str, Any]]:
    """Return available distill methods with metadata."""
    out: list[dict[str, Any]] = []
    for key, meta in _DISTILL_METHODS.items():
        p = _ROOT / meta["path"]
        if p.is_file():
            out.append(
                {
                    "key": key,
                    "label": meta["label"],
                    "config_path": meta["path"],
                    "task_command": meta["task"],
                    "experimental": bool(meta.get("experimental", False)),
                }
            )
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
        if not math.isfinite(v):
            raise ValueError("Distillation configuration numbers must be finite")
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
        raise UnknownDistillMethod(f"Unknown distill method: {method!r}")

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
                fields.append(
                    {
                        "key": fkey.key,
                        "value": val,
                        "type": _python_type(val),
                        "comment": comment,
                    }
                )
            sections.append({"name": key.key, "fields": fields})
        else:
            val = _to_python(item)
            comment = ""
            if hasattr(item, "trivia") and item.trivia.comment:
                comment = item.trivia.comment.strip().lstrip("#").strip()
            root_fields.append(
                {
                    "key": key.key,
                    "value": val,
                    "type": _python_type(val),
                    "comment": comment,
                }
            )

    if root_fields:
        sections.insert(0, {"name": "root", "fields": root_fields})

    return {"method": method, "sections": sections}


def _validated_value(current: Any, value: Any, name: str) -> Any:
    """Keep the existing TOML field type; JSON integers may fill float fields."""
    current = _to_python(current)
    if isinstance(current, float) and type(value) in (int, float):
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number
    if type(current) is not type(value):
        raise ValueError(f"{name} must be {_python_type(current)}")
    if isinstance(current, list):
        # The shipped schedule arrays are homogeneous floats. Match their
        # element type while permitting a changed number of stages.
        if current:
            return [
                _validated_value(current[0], item, f"{name}[{index}]")
                for index, item in enumerate(value)
            ]
        if any(type(item) not in (str, bool, int, float) for item in value):
            raise ValueError(f"{name} must contain scalar values")
        return [_to_python(item) for item in value]
    if type(current) not in (str, bool, int):
        raise ValueError(f"{name} is not an editable scalar or array")
    return value


def save_distill_config(method: str, updates: dict[str, dict[str, Any]]) -> None:
    """Save field values back to the sectioned TOML, preserving comments.

    *updates* is ``{"section_name": {"key": new_value, ...}, ...}``.
    """
    meta = _DISTILL_METHODS.get(method)
    if meta is None:
        raise UnknownDistillMethod(f"Unknown distill method: {method!r}")

    path = _ROOT / meta["path"]
    with CONFIG_WRITE_LOCK:
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        for section_name, fields in updates.items():
            if section_name == "root":
                table = doc
            elif section_name in doc and isinstance(
                doc[section_name], tomlkit.items.Table
            ):
                table = doc[section_name]
            else:
                raise ValueError(f"Unknown distill section: {section_name!r}")
            for key, value in fields.items():
                if key not in table or isinstance(table[key], tomlkit.items.Table):
                    raise ValueError(f"Unknown distill field: {section_name}.{key}")
                table[key] = _validated_value(
                    table[key], value, f"{section_name}.{key}"
                )
        # Finish validation and serialization before replacing the existing file.
        atomic_write_text(path, tomlkit.dumps(doc))


def load_guide(method: str, lang: str = "cn") -> str:
    """Load guide HTML for a distill method, with lang → en fallback."""
    if method not in _DISTILL_METHODS:
        raise UnknownDistillMethod(f"Unknown distill method: {method!r}")
    language = lang if lang in ("cn", "en", "ja", "ko") else "en"
    for try_lang in dict.fromkeys((language, "en")):
        p = _GUIDES_DIR / try_lang / f"{method}.html"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""
