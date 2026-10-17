"""Scoped access to anime-tools' import-light GUI helpers.

No JobManager is constructed here. The shared lock covers every temporary
environment/settings override and is held only during synchronous work.
"""

from __future__ import annotations

import importlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


_LOCK = threading.RLock()
_LOCAL = threading.local()
ENV_KEYS = ("ANIME_TOOLS_HOME", "ANIME_TOOLS_WORKSPACE", "ANIME_TOOLS_MODELS")


class CurationUnavailable(RuntimeError):
    pass


def tools():
    try:
        return SimpleNamespace(**{
            name: importlib.import_module(module)
            for name, module in {
                "D": "anime_tools.gui.dataset", "S": "anime_tools.gui.stages",
                "C": "anime_tools.gui._context", "P": "anime_tools.gui.proposals",
                "settings": "anime_tools.gui.settings", "R": "anime_tools.stages.replay",
                "E": "anime_tools.stages.export_workspace",
                "grammar": "anime_tools.captions.position_clauses",
            }.items()
        })
    except ImportError as exc:
        raise CurationUnavailable(
            f"anime-tools v0.7.15 is required for curation: {exc}. "
            "Use the registered curation runtime; do not replace a V100 training environment."
        ) from exc


@contextmanager
def scoped_tools(home: str | Path, snapshot: dict | None = None, environment: dict | None = None):
    """Use one home/settings snapshot; restore globals even on a failed write."""
    with _LOCK:
        # RPC calls can come from separate interpreters. Serialize their
        # settings read/modify/write operations too; nested calls reuse it.
        lock_file = None
        depth = getattr(_LOCAL, "depth", 0)
        if depth == 0 and snapshot is None:
            lock_path = Path(home) / ".anime_tools_webui.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = lock_path.open("a+b")
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0)
                lock_file.write(b"\0")
                lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _LOCAL.depth = depth + 1
        old_env = {key: os.environ.get(key) for key in ENV_KEYS}
        settings_reader = None
        loaded = None
        try:
            if environment is not None:
                for key in ENV_KEYS:
                    if key in environment:
                        os.environ[key] = str(environment[key])
                    else:
                        os.environ.pop(key, None)
            os.environ["ANIME_TOOLS_HOME"] = str(Path(home).absolute())
            loaded = tools()
            if snapshot is not None:
                # Dataset helpers call their imported reader during reachable().
                # Historical actions must use the job's saved trust boundaries.
                settings_reader = loaded.D.load_settings
                loaded.D.load_settings = lambda: snapshot
            yield loaded
        finally:
            if loaded is not None and settings_reader is not None:
                loaded.D.load_settings = settings_reader
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            _LOCAL.depth = depth
            if lock_file is not None:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()


def run_context(loaded, settings: dict):
    roots = loaded.C.roots_for(settings)
    return loaded.C.RunContext(
        settings=settings, roots=roots, defaults=loaded.C.stage_defaults(settings),
        report_root=loaded.C.report_root(settings, roots),
        mask_root=loaded.C.mask_root(settings, roots),
    )


def checked_report(loaded, context: dict, *, fingerprint: str | None = None):
    """Validate a report and every write path before replay/undo uses it."""
    import hashlib

    path = checked_path(context.get("report_path") or "", context)
    if not context.get("report_path") or not path.is_file():
        raise FileNotFoundError("This job has no readable report")
    if fingerprint and hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("The report changed after this operation was queued; refusing to apply it")
    report = loaded.R.load_report(path)
    stage = context["stage"]
    roots = loaded.D.resolve_roots(context["roots"], bases=loaded.D.dataset_bases(context["settings"]))
    shape = loaded.P.SHAPES.get(stage)
    if shape is not None:
        root = roots.src if shape.target_root == "src" else roots.dst
        for row in loaded.R.report_rows(report, shape):
            relative = str(row.get("caption_path") or "")
            if not relative:
                continue
            # reachable alone accepts a traversal remaining inside the home.
            loaded.D._rel_key(relative)
            checked_path(root / relative, context, bases=[root])
            if shape.before_field not in row and "before" not in row:
                raise ValueError("Report lacks an explicit caption baseline; create a new preview")
            if root == roots.src:
                raise ValueError("Report operations cannot modify source captions")
    elif stage != "export":
        raise ValueError(f"{stage} has no caption proposals")
    return path, report, roots


def checked_path(path: str | Path, context: dict, *, bases=None) -> Path:
    """Resolve symlinks before enforcing the saved registration boundary."""
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise ValueError("Parent traversal is not allowed")
    home = Path(context["home"])
    absolute = raw if raw.is_absolute() else home / raw
    candidate = absolute.resolve()
    if bases is None:
        bases = [*context["roots"].values(), context["workspace"], context["job_dir"]]
    allowed = [Path(base).resolve() for base in bases]
    if not any(candidate.is_relative_to(base) for base in allowed):
        raise ValueError(f"Path is outside the registered curation roots: {path}")
    return candidate


def validate_roots(roots: dict[str, str]) -> None:
    """A workspace output must never alias or contain the read-only source."""
    paths = {name: Path(value).resolve() for name, value in roots.items()}
    for name, path in paths.items():
        if path.exists() and not path.is_dir():
            raise ValueError(f"{name} is not a directory: {path}")
        if path == Path(path.anchor):
            raise ValueError(f"A filesystem root cannot be registered as {name}")
    for name, path in paths.items():
        for other, target in paths.items():
            if name < other and (path.is_relative_to(target) or target.is_relative_to(path)):
                raise ValueError(f"Dataset roots must not overlap: {name} and {other}")


def validate_tree(root: str | Path) -> None:
    """Reject nested symlink escapes before a stage's own recursive walker."""
    base = Path(root).resolve()
    if not base.exists():
        return
    for directory, folders, files in os.walk(base, followlinks=False):
        for name in [*folders, *files]:
            path = Path(directory) / name
            if path.is_symlink() and not path.resolve().is_relative_to(base):
                raise ValueError(f"Dataset symlink escapes its registered root: {path}")
