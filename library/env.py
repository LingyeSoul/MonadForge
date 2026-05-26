"""Minimal ``.env`` loader — no external dependency.

Used by scripts that need user-specific paths and credentials (HF token,
ComfyUI registry token, external corpus directories) without hardcoding
them in the repo.

Format: standard ``KEY=VALUE`` lines, ``#`` for comments, optional surrounding
single or double quotes around the value. No shell interpolation; values are
taken literally. Existing process env wins over file values (so a CLI
``CAPTION_CORPUS_DIR=… make foo`` overrides the file).

Looks for ``.env`` at the project root by default — the directory two levels
up from this file (``anima_lora/``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def anima_home() -> Path:
    """Repo home used to anchor every repo-relative path.

    Defaults to :func:`project_root` (the ``anima_lora/`` checkout that holds
    ``configs/``, ``models/``, ``output/`` …). Set ``ANIMA_HOME`` to override —
    this is what lets ``import anima_lora`` and the CLI run from *any* working
    directory instead of requiring a ``cd`` into the repo first.
    """
    override = os.environ.get("ANIMA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return project_root()


def resolve_under_home(path) -> Path:
    """Resolve a possibly-relative path against :func:`anima_home`.

    Absolute and ``~``-prefixed paths pass through untouched; bare relative
    paths are interpreted relative to the repo home rather than the current
    working directory. Idempotent (absolute in → same path out), so it is safe
    to call at every layer of a call chain without double-anchoring.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return anima_home() / p


def load_dotenv(path: Optional[Path] = None) -> dict[str, str]:
    """Read a ``.env`` file into ``os.environ`` (without overriding existing keys).

    Returns the dict of values that were *added* (useful for logging /
    test introspection). A missing file is a no-op — callers shouldn't
    depend on .env being present.
    """
    if path is None:
        path = anima_home() / ".env"
    added: dict[str, str] = {}
    if not path.exists():
        return added
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            added[key] = val
    return added
