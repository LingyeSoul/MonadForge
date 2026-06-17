"""Tray language preference persistence.

Single-key JSON (``{"language": "cn"}``) under the daemon state dir
(``output/daemon/tray-prefs.json``) so the choice survives tray restarts and
sits beside the daemon's own state. Atomic write (tmp + replace) mirrors the
daemon's ``job.json`` pattern so a crash mid-write never leaves a half file.

Deliberately tiny and dependency-free: the tray reads/writes its own file and
never touches the WebUI's localStorage-only language choice.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from scripts.tray.i18n import DEFAULT_LANGUAGE, LANGUAGES

logger = logging.getLogger(__name__)

# output/daemon/ — same dir as the daemon's daemon.json / jobs/. Resolved the
# same way scripts/daemon/config.py does (parents[2] == repo root).
_ROOT = Path(__file__).resolve().parents[2]
PREFS_DIR = _ROOT / "output" / "daemon"
PREFS_FILE = PREFS_DIR / "tray-prefs.json"


def load_language() -> str:
    """Read the persisted language, or :data:`DEFAULT_LANGUAGE` if absent/corrupt."""
    try:
        raw = PREFS_FILE.read_text(encoding="utf-8")
        lang = json.loads(raw).get("language")
    except (OSError, ValueError, json.JSONDecodeError):
        return DEFAULT_LANGUAGE
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE


def save_language(lang: str) -> None:
    """Atomically persist ``lang``. Best-effort: a read-only dir is logged, not fatal."""
    if lang not in LANGUAGES:
        return
    try:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"language": lang}, indent=2)
        # Write a sibling tmp file then atomically replace (POSIX + Windows).
        tmp = PREFS_FILE.with_name(PREFS_FILE.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, PREFS_FILE)
    except OSError as exc:
        logger.warning("could not persist tray language (%s)", exc)
