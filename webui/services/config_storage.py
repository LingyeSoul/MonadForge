"""Local configuration writes with staged replacement and failure rollback.

The lock covers read/modify/write callers in this WebUI process. Multi-file
writes roll back ordinary I/O failures; they are not a power-loss transaction.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

CONFIG_WRITE_LOCK = threading.RLock()


def read_json_object(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _stage(path: Path, content: bytes, suffix: str, mode: int | None) -> Path:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=suffix, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        return temporary
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_many({path: text})


def atomic_write_many(contents: Mapping[Path, str]) -> None:
    """Prepare every file before changing any, restoring originals on failure.

    Resolve symlinks so an existing custom-config link still points at its
    updated target. If restoring a committed file also fails, keep the backup
    files and report their paths instead of discarding the recovery data.
    """
    with CONFIG_WRITE_LOCK:
        prepared: dict[Path, Path] = {}
        backups: dict[Path, Path | None] = {}
        committed: list[Path] = []
        keep_backups = False
        try:
            for requested, text in contents.items():
                path = Path(requested).resolve()
                if path in prepared:
                    raise ValueError(f"Multiple configuration files resolve to {path}")
                payload = text.encode("utf-8")
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
                backups[path] = (
                    _stage(path, path.read_bytes(), ".bak", mode)
                    if path.exists()
                    else None
                )
                prepared[path] = _stage(path, payload, ".tmp", mode)
            for path, temporary in prepared.items():
                temporary.replace(path)
                committed.append(path)
        except BaseException as original:
            failed_restores = []
            for path in reversed(committed):
                try:
                    backup = backups[path]
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        backup.replace(path)
                except OSError as exc:
                    failed_restores.append(f"{path}: {exc}")
            if failed_restores:
                keep_backups = True
                retained = [
                    str(path)
                    for path in backups.values()
                    if path is not None and path.exists()
                ]
                raise OSError(
                    f"Configuration write failed ({original}); rollback failed: {'; '.join(failed_restores)}. "
                    f"Recovery backups retained: {', '.join(retained)}"
                ) from original
            raise
        finally:
            for temporary in prepared.values():
                temporary.unlink(missing_ok=True)
            if not keep_backups:
                for backup in backups.values():
                    if backup is not None:
                        backup.unlink(missing_ok=True)
