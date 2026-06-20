"""Dataset curation helpers shared by GUI and preprocess."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

SIDECAR_EXTENSIONS = (".txt", ".caption", ".json", ".txt.history.jsonl")


def linked_paths(image_path: Path) -> list[Path]:
    """Return an image and existing caption/metadata sidecars."""

    paths = [image_path]
    for ext in SIDECAR_EXTENSIONS:
        sidecar = image_path.with_suffix(ext)
        if sidecar.exists():
            paths.append(sidecar)
    return paths


def move_linked_files(
    image_path: Path,
    *,
    source_root: Path,
    target_root: Path,
) -> list[Path]:
    """Move an image and sidecars to ``target_root`` preserving layout."""

    moved: list[Path] = []
    for source in linked_paths(image_path):
        if not source.exists():
            continue
        try:
            rel = source.relative_to(source_root)
        except ValueError:
            rel = Path(source.name)
        target = _unique_path(target_root / rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append(target)
    return moved


def _unique_path(path: Path) -> Path:
    """Return a non-existing sibling path without overwriting prior moves."""

    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find a free target path for {path}")


def rel_key(path: Path, root: Path) -> str:
    """Stable JSON key for an image path relative to a dataset root."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def load_curation_decisions(
    path: Path | str | None,
    *,
    source_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load per-image preprocess decisions from JSON.

    The file is intentionally optional. Missing or malformed files behave as an
    empty decision set so normal CLI preprocess remains unchanged unless a GUI
    decision file is present.
    """

    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    images = data.get("images") if isinstance(data, dict) else None
    if not isinstance(images, dict):
        return {}
    strip_prefix = ""
    add_prefix = ""
    if source_dir is not None:
        saved_source = str(data.get("source_dir") or "").replace("\\", "/")
        source = Path(source_dir)
        if saved_source:
            try:
                cwd = Path.cwd().resolve()
                saved_abs = (cwd / saved_source).resolve()
                source_abs = (
                    source.resolve()
                    if source.is_absolute()
                    else (cwd / source).resolve()
                )
                if source_abs == saved_abs:
                    pass
                elif source_abs.is_relative_to(saved_abs):
                    strip_prefix = source_abs.relative_to(saved_abs).as_posix()
                elif saved_abs.is_relative_to(source_abs):
                    add_prefix = saved_abs.relative_to(source_abs).as_posix()
                else:
                    return {}
            except OSError:
                candidates = {str(source).replace("\\", "/"), source.as_posix()}
                if saved_source not in candidates:
                    return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in images.items():
        if isinstance(key, str) and isinstance(value, dict):
            norm_key = key.replace("\\", "/")
            if strip_prefix:
                prefix = strip_prefix.rstrip("/") + "/"
                if not norm_key.startswith(prefix):
                    continue
                norm_key = norm_key[len(prefix) :]
            elif add_prefix:
                norm_key = f"{add_prefix.rstrip('/')}/{norm_key}"
            out[norm_key] = dict(value)
    return out


def save_curation_decisions(
    path: Path,
    *,
    source_dir: str,
    images: dict[str, dict[str, Any]],
) -> None:
    """Write GUI decisions consumed by preprocess resize."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "source_dir": source_dir.replace("\\", "/"),
        "images": images,
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
