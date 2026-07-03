"""Path-resolution helpers — sandboxed against ROOT.

Centralizes the containment checks shared by the file browser and merge
services: reject ``..`` in relative paths, ensure resolved paths stay
inside ROOT (symlinks included via ``Path.resolve``), and verify the
expected type (file / directory / either). Absolute paths are accepted
as-is — the WebUI is a localhost tool and lets users browse arbitrary
absolute locations.
"""

from __future__ import annotations

from pathlib import Path

from webui.services.config_service import ROOT


def resolve_path(path: str, *, expect_file: bool | None = None) -> Path | None:
    """Resolve a path relative to ROOT (or absolute), sandboxed to ROOT.

    Rejects ``..`` components in relative paths and resolved paths that
    escape ROOT (e.g. via symlink). Returns None if the path is missing
    or the wrong type.

    Parameters
    ----------
    path:
        Relative to ROOT or absolute.
    expect_file:
        True → require a file; False → require a directory;
        None → accept either (must exist).
    """
    p = Path(path)
    if p.is_absolute():
        target = p
    else:
        if ".." in p.parts:
            return None
        resolved = (ROOT / p).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            return None
        target = resolved

    if expect_file is True:
        return target if target.is_file() else None
    if expect_file is False:
        return target if target.is_dir() else None
    return target if target.exists() else None
