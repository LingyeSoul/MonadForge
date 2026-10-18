"""Strict path sandbox for request-supplied paths.

``paths.resolve_path`` is the *browser* sandbox: it deliberately accepts any
absolute location so a localhost user can point the file picker at another
disk. This module is the stricter sibling used by handlers that hand a path
straight to file I/O (read / overwrite / delete / extract): an absolute path
must resolve inside the project root, or inside a root the operator listed in
``ANIMA_ALLOW_OUTSIDE_ROOTS``, otherwise the request is refused.

``root`` is always passed in (never imported) so this module stays free of
service imports and can be used by ``config_service`` without an import cycle.
"""

from __future__ import annotations

import os
from pathlib import Path

# Extra roots an install may keep its datasets / prompt files in. Path-separator
# (``:`` on POSIX, ``;`` on Windows) or newline separated absolute paths.
OUTSIDE_ROOTS_ENV = "ANIMA_ALLOW_OUTSIDE_ROOTS"


def allowed_roots() -> tuple[Path, ...]:
    """Roots named by ``ANIMA_ALLOW_OUTSIDE_ROOTS`` (empty when unset)."""
    raw = os.environ.get(OUTSIDE_ROOTS_ENV, "")
    roots: list[Path] = []
    for chunk in raw.replace("\n", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if not chunk:
            continue
        resolved = Path(chunk).expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def resolve_within(
    root: Path,
    value: str | Path,
    *,
    suffixes: tuple[str, ...] = (),
    expect: str | None = None,
) -> Path:
    """Resolve ``value`` and require it to stay inside ``root``.

    Absolute paths are rejected unless they resolve inside ``root`` or a root
    from :func:`allowed_roots`, so a caller can never read, overwrite, delete
    or extract into an arbitrary file. ``..`` components are rejected outright,
    and symlinks are followed by ``Path.resolve`` before the containment test.

    Parameters
    ----------
    root:
        The containment root (normally ``config_service.ROOT``).
    value:
        A ROOT-relative or absolute path.
    suffixes:
        Allowed file extensions (lowercase, with the dot). A name without any
        suffix is always allowed — only a *different* extension is refused.
    expect:
        ``"file"`` / ``"dir"`` to require that type, or ``None`` for no
        existence requirement.

    Raises ``ValueError`` with a user-facing message; handlers map it to a 4xx.
    """
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("path is required")
    candidate = Path(raw).expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"path may not contain '..': {raw}")
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (Path(root) / candidate).resolve()
    )
    permitted = (Path(root).resolve(), *allowed_roots())
    if not any(resolved.is_relative_to(base) for base in permitted):
        raise ValueError(
            f"path is outside the project root: {raw} "
            f"(set {OUTSIDE_ROOTS_ENV} to allow another root)"
        )
    if suffixes and resolved.suffix and resolved.suffix.lower() not in suffixes:
        raise ValueError(f"path must end with {', '.join(suffixes)}: {raw}")
    if expect == "file" and not resolved.is_file():
        raise ValueError(f"file not found: {raw}")
    if expect == "dir" and not resolved.is_dir():
        raise ValueError(f"directory not found: {raw}")
    return resolved
