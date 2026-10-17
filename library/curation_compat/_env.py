"""Curation-side environment: home dir, path anchoring, ``.env``, logging.

Home-directory resolution: ``ANIME_TOOLS_HOME``, then ``ANIMA_HOME`` (the
trainer's home, so an in-tree run anchors identically), then the current working
directory. There is no checkout-root fallback — once installed this file lives
in site-packages.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from library.curation_compat import workspace as WS


def curation_home() -> Path:
    for key in ("ANIME_TOOLS_HOME", "ANIMA_HOME"):
        override = os.environ.get(key)
        if override:
            return Path(override).expanduser().resolve()
    return Path.cwd()


def resolve_path(path) -> Path:
    """Anchor a bare relative path under :func:`curation_home`.

    Absolute and ``~`` paths pass through; idempotent.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return curation_home() / p


def workspace_dir() -> Path:
    """Where the tools write (``ANIME_TOOLS_WORKSPACE`` overrides; default
    ``<home>/workspace``). Nothing but Export writes outside it."""
    override = os.environ.get("ANIME_TOOLS_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    return curation_home() / WS.WORKSPACE


def models_dir() -> Path:
    """Where curation model checkpoints live (``ANIME_TOOLS_MODELS`` overrides;
    default ``<home>/models``)."""
    override = os.environ.get("ANIME_TOOLS_MODELS")
    if override:
        return Path(override).expanduser().resolve()
    return curation_home() / "models"


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines into ``os.environ`` without overriding existing
    keys; returns what was added. Missing file is a no-op."""
    if path is None:
        path = curation_home() / ".env"
    added: dict[str, str] = {}
    if not path.exists():
        return added
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            added[key] = val
    return added


def setup_logging(log_level: str = "INFO") -> None:
    """Root logger to stderr (rich when available); no-op if already configured."""
    if logging.root.handlers:
        return
    handler: logging.Handler | None = None
    try:
        from rich.console import Console
        from rich.logging import RichHandler

        handler = RichHandler(console=Console(stderr=True))
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "%(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logging.root.setLevel(getattr(logging, log_level))
    logging.root.addHandler(handler)
