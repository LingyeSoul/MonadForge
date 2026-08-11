"""Shared checkpoint/output layout helpers.

New runs use ``<output_dir>/<safe output name>/`` as one self-contained output
root.  Older MonadForge runs wrote a mixture of files directly in
``output/ckpt`` and trajectory files below ``output/ckpt/<name>/``; all reader
helpers in this module deliberately understand both shapes.

The module is intentionally free of torch/accelerate imports so it can be used
by the CLI, daemon, WebUI and inference helpers on Windows as well as POSIX.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CHECKPOINT_MARKERS = (
    "-checkpoint",
    "-interrupted",
    "-state",
    ".snapshot",
    ".bak.",
)
_TRAJECTORY_RE = re.compile(r"(?:-step\d{8}|-\d{6})$")
OUTPUT_WEIGHT_EXTENSIONS = frozenset({".safetensors", ".ckpt", ".pt"})


def safe_output_name(value: str | os.PathLike | None, default: str = "last") -> str:
    """Return a stable, filesystem-safe output name.

    Path separators, control characters and whitespace become ``_``.  Keeping
    the transformation deterministic is important because a resumed run must
    resolve to the same directory on Linux and Windows.
    """

    raw = unicodedata.normalize("NFKC", str(value or default)).strip()
    raw = raw.replace("/", "_").replace("\\", "_")
    raw = _SAFE_RE.sub("_", raw).strip("._")
    if raw in {"", ".", ".."}:
        raw = default
    return raw[:160]


def _as_path(value: str | os.PathLike | None, *, cwd: Path | None = None) -> Path:
    path = Path(value or "output/ckpt")
    if not path.is_absolute() and cwd is not None:
        path = cwd / path
    return path


@dataclass(frozen=True)
class OutputLayout:
    """Resolved paths for one training output.

    ``root`` is the directory that owns the final adapter, snapshots and state
    directories.  ``base`` is the caller-supplied output directory and is kept
    for legacy discovery.
    """

    base: Path
    root: Path
    name: str
    canonical: bool = True

    @property
    def final(self) -> Path:
        return self.root / f"{self.name}.safetensors"

    def model_file(self, ext: str = ".safetensors") -> Path:
        return self.root / f"{self.name}{ext}"

    @property
    def snapshot(self) -> Path:
        return self.root / f"{self.name}.snapshot.toml"

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def state(self) -> Path:
        return self.root / f"{self.name}-state"

    @property
    def interrupted_state(self) -> Path:
        return self.root / f"{self.name}-interrupted-state"

    @property
    def rolling_state(self) -> Path:
        return self.root / f"{self.name}-rolling-state"

    @property
    def checkpoint_state(self) -> Path:
        return self.root / f"{self.name}-checkpoint-state"

    @property
    def checkpoint_file(self) -> Path:
        return self.root / f"{self.name}-checkpoint.safetensors"

    def epoch_file(self, epoch: int, ext: str = ".safetensors") -> Path:
        return self.root / f"{self.name}-{epoch:06d}{ext}"

    def step_file(self, step: int, ext: str = ".safetensors") -> Path:
        return self.root / f"{self.name}-step{step:08d}{ext}"

    def legacy_candidates(self) -> tuple[Path, ...]:
        """Directories/files written by pre-layout versions, newest first."""

        return (
            self.root,
            self.base,
            self.base.parent / self.name,
        )


def resolve_output_layout(
    output_dir: str | os.PathLike | None,
    output_name: str | os.PathLike | None,
    *,
    cwd: Path | None = None,
) -> OutputLayout:
    """Resolve the canonical output root without touching the filesystem."""

    base = _as_path(output_dir, cwd=cwd)
    name = safe_output_name(output_name)
    # A caller may already pass ``output/ckpt/<name>`` (for example a resumed
    # job's manifest).  Do not create the historical ``<name>/<name>`` nest.
    root = base if safe_output_name(base.name, "") == name else base / name
    return OutputLayout(base=base, root=root, name=name, canonical=True)


def layout_from_args(args, *, cwd: Path | None = None) -> OutputLayout:
    """Resolve and memoize a layout on an argparse namespace.

    Memoization lets checkpoint/config writers agree even if a caller mutates
    ``args.output_dir`` later while preserving the explicit legacy fields.
    """

    cached = getattr(args, "_output_layout", None)
    if isinstance(cached, OutputLayout):
        return cached
    layout = resolve_output_layout(
        getattr(args, "output_dir", None), getattr(args, "output_name", None), cwd=cwd
    )
    try:
        setattr(args, "_output_layout", layout)
        setattr(args, "_legacy_output_dir", str(layout.base))
        setattr(args, "output_dir", str(layout.root))
        setattr(args, "_output_layout_canonical", True)
    except Exception:
        pass
    return layout


def is_checkpoint_weight(path: Path, *, name: str | None = None) -> bool:
    """Whether *path* is a final adapter weight, excluding resumable tracks."""

    if not path.is_file() or path.suffix.lower() not in OUTPUT_WEIGHT_EXTENSIONS:
        return False
    stem = path.stem
    if ".bak." in path.name or stem.endswith(("-checkpoint", "-interrupted", "_moe")):
        return False
    if any(marker in stem for marker in ("-checkpoint", "-interrupted", "-state")):
        return False
    if _TRAJECTORY_RE.search(stem):
        return False
    if name and not (stem == name or stem.startswith(name + "_")):
        return False
    return True


def _is_managed_nonfinal_path(path: Path, root: Path) -> bool:
    """Return whether a weight lives in an internal state/temp directory."""
    for parent in path.parents:
        directory = parent.name
        if directory.endswith("-state"):
            return True
        if directory.startswith(".") and (
            directory.endswith(".tmp") or ".old-" in directory
        ):
            return True
        if parent == root:
            break
    return False


def discover_weights(
    directory: str | os.PathLike,
    *,
    name: str | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Discover final weights in canonical and legacy layouts."""

    root = Path(directory)
    if not root.exists():
        return []
    if recursive:
        iterator: Iterable[Path] = (
            path for ext in OUTPUT_WEIGHT_EXTENSIONS for path in root.rglob(f"*{ext}")
        )
    else:
        iterator = (
            path for ext in OUTPUT_WEIGHT_EXTENSIONS for path in root.glob(f"*{ext}")
        )
    files = [
        p
        for p in iterator
        if not _is_managed_nonfinal_path(p, root) and is_checkpoint_weight(p, name=name)
    ]

    # A canonical run manifest is authoritative for its directory.  This keeps
    # any unusual method-specific trajectory name from winning merely because
    # it was written after the final adapter.  Legacy directories without a
    # manifest retain filename-based discovery.
    manifest_finals: dict[Path, Path] = {}
    manifest_iter = root.rglob("run_manifest.json") if recursive else root.glob("run_manifest.json")
    for manifest_path in manifest_iter:
        manifest = read_run_manifest(manifest_path)
        if manifest is None:
            continue
        final = resolve_manifest_path(manifest_path, manifest.get("final_weight"))
        if (
            final is not None
            and not _is_managed_nonfinal_path(final, root)
            and is_checkpoint_weight(final, name=name)
        ):
            manifest_finals[manifest_path.parent] = final
    if manifest_finals:
        files = [
            path
            for path in files
            if path.parent not in manifest_finals or path == manifest_finals[path.parent]
        ]
        files.extend(path for path in manifest_finals.values() if path not in files)
    # Do not let state/temporary directories leak into inference selection.
    return sorted(files, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)


def latest_weight(
    directory: str | os.PathLike,
    *,
    name: str | None = None,
    prefix: str | None = None,
) -> Path | None:
    files = discover_weights(directory, name=name)
    if prefix:
        files = [p for p in files if p.stem.startswith(prefix)]
    return files[0] if files else None


def read_run_manifest(path: str | os.PathLike) -> dict | None:
    """Read a canonical run manifest, returning ``None`` when invalid."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "run_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_manifest_path(
    manifest_path: str | os.PathLike, value: object
) -> Path | None:
    """Resolve an artifact path stored as either relative or absolute."""

    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    manifest = Path(manifest_path)
    if manifest.is_dir():
        manifest = manifest / "run_manifest.json"
    if not path.is_absolute():
        path = manifest.parent / path
    return path


def atomic_write_json(path: Path, payload: Mapping, *, indent: int | None = 2) -> None:
    """Write JSON via a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=indent, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def remove_path_with_retry(path: Path, *, retries: int = 8) -> None:
    """Remove a file/directory with bounded retries for Windows scanners/locks."""

    for attempt in range(max(1, retries)):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt + 1 >= retries:
                raise
            time.sleep(min(0.1 * (attempt + 1), 0.75))


def atomic_replace_dir(source: Path, target: Path, *, retries: int = 8) -> None:
    """Replace a directory while preserving the previous complete target.

    Directory replacement is a two-step operation when ``target`` already
    exists: move the old target aside, then publish ``source``.  Keep that
    backup for the entire publication retry window so a transient lock cannot
    discard the last complete checkpoint.  If publication ultimately fails,
    restore the backup before returning the error to the caller.
    """

    attempts = max(1, retries)

    def replace_with_retry(src: Path, dst: Path) -> None:
        for attempt in range(attempts):
            try:
                os.replace(src, dst)
                return
            except OSError:
                if attempt + 1 >= attempts:
                    raise
                time.sleep(min(0.25 * (attempt + 1), 1.0))

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.old-{os.getpid()}-{time.time_ns()}")
    had_target = target.exists()
    if had_target:
        replace_with_retry(target, backup)

    try:
        replace_with_retry(source, target)
    except OSError as publish_error:
        if had_target and backup.exists():
            try:
                replace_with_retry(backup, target)
            except OSError as restore_error:
                restore_error.add_note(
                    f"previous complete directory remains at {backup}"
                )
                raise restore_error from publish_error
        raise

    if backup.exists():
        try:
            remove_path_with_retry(backup, retries=attempts)
        except OSError:
            # Publication already succeeded. A stale backup is preferable to
            # reporting a failed checkpoint after the new target is complete.
            pass


def write_run_manifest(layout: OutputLayout, payload: Mapping) -> Path:
    """Persist a manifest with the layout identity and completion marker."""

    data = dict(payload)
    data.setdefault("schema_version", 2)
    data.setdefault("layout", "output/ckpt/<name>")
    data.setdefault("output_name", layout.name)
    data.setdefault("output_root", str(layout.root))
    # Relative artifact paths keep manifests valid when the repo/output tree is
    # copied between Linux and Windows. Readers also accept historical absolute
    # paths through :func:`resolve_manifest_path`.
    if "final_weight" not in data:
        final_candidates = [
            path
            for ext in OUTPUT_WEIGHT_EXTENSIONS
            for path in layout.root.glob(f"{layout.name}{ext}")
            if path.is_file()
        ]
        data["final_weight"] = (
            sorted(
                final_candidates,
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )[0].name
            if final_candidates
            else layout.final.name
        )
    snapshot = getattr(layout, "snapshot", None)
    if snapshot is not None and snapshot.is_file():
        data.setdefault("snapshot", snapshot.name)

    # Keep the manifest useful to readers without requiring every caller to
    # know the checkpoint naming convention. Paths are relative to the run
    # root so the record survives copying between Linux and Windows.
    if layout.root.is_dir():
        weights = sorted(
            (
                p
                for ext in OUTPUT_WEIGHT_EXTENSIONS
                for p in layout.root.glob(f"*{ext}")
                if p.is_file()
            ),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
            reverse=True,
        )
        trajectories = [
            p.name
            for p in weights
            if p.name != layout.final.name
            and (
                p.stem.endswith(("-checkpoint", "-interrupted"))
                or _TRAJECTORY_RE.search(p.stem)
            )
        ]
        if trajectories:
            data.setdefault("checkpoints", trajectories)
            data.setdefault(
                "rolling_checkpoint",
                next((n for n in trajectories if "-checkpoint" in n), None),
            )
            data.setdefault(
                "epoch_checkpoints",
                [
                    n
                    for n in trajectories
                    if re.search(r"-\d{6}\.(?:safetensors|ckpt|pt)$", n)
                ],
            )
            data.setdefault(
                "step_checkpoints",
                [n for n in trajectories if "-step" in n],
            )
        state_dirs = sorted(
            (
                p.name
                for p in layout.root.iterdir()
                if p.is_dir()
                and p.name.endswith("-state")
                and not p.name.startswith(".")
            )
        )
        if state_dirs:
            data.setdefault("state_dirs", state_dirs)
            data.setdefault(
                "resume_state",
                next(
                    (
                        name
                        for suffix in (
                            "-interrupted-state",
                            "-checkpoint-state",
                            "-state",
                        )
                        for name in state_dirs
                        if name.endswith(suffix)
                    ),
                    None,
                ),
            )
    data.setdefault("updated_at", time.time())
    atomic_write_json(layout.manifest, data)
    return layout.manifest
