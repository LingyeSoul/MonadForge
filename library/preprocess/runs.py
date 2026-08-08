"""Preprocessing run identities, manifests, and legacy cache migration.

The preprocess pipeline historically wrote every cache kind into fixed sibling
directories below ``post_image_dataset``.  That made it impossible to keep two
different resize/caption configurations at the same time.  This module owns
the small, torch-free contract used by callers that want an isolated run::

    post_image_dataset/runs/<source-name>-<path-hash>/<config-hash>/

Only deterministic path/config handling and filesystem primitives live here;
the actual image, VAE, text and conditioning workers remain in their existing
modules.  The API intentionally accepts ordinary mappings and ``pathlib``
paths so it is usable from the CLI, WebUI, daemon, and Windows.
"""

from __future__ import annotations

import dataclasses
import filecmp
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from library.env import resolve_under_home

# Bump this when the meaning or on-disk format of a cache changes.  It is
# deliberately independent from the training-state schema in library.training.
PREPROCESS_CACHE_SCHEMA_VERSION = 1
PREPROCESS_RUN_SCHEMA_VERSION = 1
SOURCE_HASH_LENGTH = 12
CONFIG_HASH_LENGTH = 12

RUN_KINDS = ("resized", "lora", "masks", "multires", "conditioning", "captions")
_LEGACY_KIND_DIRS = {
    "resized": ("resized",),
    "lora": ("lora",),
    "masks": ("masks", "mask"),
    "multires": ("multires",),
    "conditioning": ("conditioning", "conditioning_data", "cond_resized"),
    "captions": ("captions",),
}

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID_COMPONENT = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


class PreprocessRunError(RuntimeError):
    """Raised when a run manifest is malformed or does not match its identity."""


@dataclass(frozen=True)
class LegacyValidation:
    """Result of checking a legacy fixed-directory cache.

    ``metadata_present`` is separate from ``valid`` because old releases did
    not write a manifest at all.  Such a cache remains readable, but callers
    can choose to require metadata before migrating it.
    """

    root: Path
    valid: bool
    metadata_present: bool = False
    metadata_valid: bool = False
    reason: str | None = None
    manifest_path: Path | None = None

    def __bool__(self) -> bool:
        return self.valid


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of a best-effort legacy-to-run migration."""

    source_root: Path
    destination_root: Path
    migrated: tuple[Path, ...] = ()
    skipped: tuple[Path, ...] = ()
    failed: tuple[tuple[Path, str], ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def complete(self) -> bool:
        return self.ok

    @property
    def migrated_count(self) -> int:
        return len(self.migrated)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    @property
    def reason(self) -> str | None:
        return self.failed[0][1] if self.failed else None


@dataclass(frozen=True)
class PreprocessRun:
    """Resolved paths and identity for one source/configuration pair."""

    source_dir: Path
    root: Path
    source_group: str
    source_hash: str
    config_hash: str
    config: Mapping[str, Any] = field(default_factory=dict)
    manifest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "manifest_path", Path(self.root) / "manifest.json")

    @property
    def run_id(self) -> str:
        return f"{self.source_group}/{self.config_hash}"

    @property
    def resized_dir(self) -> Path:
        return self.root / "resized"

    @property
    def resized_image_dir(self) -> Path:
        """Compatibility name used by the existing preprocess CLI."""

        return self.resized_dir

    @property
    def lora_dir(self) -> Path:
        return self.root / "lora"

    @property
    def lora_cache_dir(self) -> Path:
        """Compatibility name used by latent/text cache workers."""

        return self.lora_dir

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def mask_dir(self) -> Path:
        return self.masks_dir

    @property
    def multires_dir(self) -> Path:
        return self.root / "multires"

    @property
    def conditioning_dir(self) -> Path:
        return self.root / "conditioning"

    @property
    def conditioning_data_dir(self) -> Path:
        return self.conditioning_dir / "data"

    @property
    def conditioning_resized_dir(self) -> Path:
        return self.conditioning_dir / "resized"

    @property
    def captions_dir(self) -> Path:
        """Run-local derived caption artifacts."""

        return self.root / "captions"

    @property
    def caption_index_path(self) -> Path:
        return self.captions_dir / "caption_index.json"

    @property
    def directories(self) -> dict[str, Path]:
        return {
            "resized": self.resized_dir,
            "lora": self.lora_dir,
            "masks": self.masks_dir,
            "multires": self.multires_dir,
            "conditioning": self.conditioning_dir,
            "conditioning_data": self.conditioning_data_dir,
            "conditioning_resized": self.conditioning_resized_dir,
            "captions": self.captions_dir,
        }

    @property
    def cache_dirs(self) -> dict[str, Path]:
        """Short cache-kind mapping for preprocess workers."""

        return {key: self.directories[key] for key in RUN_KINDS}

    def path_overrides(self) -> dict[str, str]:
        """Return config-key paths that isolate existing preprocess workers."""

        return {
            "source_image_dir": str(self.source_dir),
            "resized_image_dir": str(self.resized_image_dir),
            "lora_cache_dir": str(self.lora_cache_dir),
            "mask_dir": str(self.mask_dir),
            "multires_dir": str(self.multires_dir),
            "conditioning_data_dir": str(self.conditioning_data_dir),
            "conditioning_resized_dir": str(self.conditioning_resized_dir),
            "caption_index_dir": str(self.captions_dir),
            "caption_index_path": str(self.caption_index_path),
            "preprocess_run": str(self.manifest_path),
        }

    def ensure_directories(self) -> PreprocessRun:
        """Create the run's stable directory skeleton and return ``self``."""

        for path in self.directories.values():
            path.mkdir(parents=True, exist_ok=True)
        return self

    def manifest_payload(self, *, status: str = "ready", **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": PREPROCESS_RUN_SCHEMA_VERSION,
            "kind": "preprocess_run",
            "status": status,
            "source": {
                "path": _portable_path(self.source_dir),
                "name": self.source_dir.name,
                "safe_name": self.source_group.rsplit("-", 1)[0],
                "path_hash": self.source_hash,
            },
            "source_dir": _portable_path(self.source_dir),
            "source_path": _portable_path(self.source_dir),
            "source_group": self.source_group,
            "source_hash": self.source_hash,
            "config_hash": self.config_hash,
            "config_fingerprint": self.config_hash,
            "config": _canonicalize(dict(self.config)),
            "cache_schema_version": int(
                self.config.get("cache_schema_version", PREPROCESS_CACHE_SCHEMA_VERSION)
            ),
            "directories": {
                key: _portable_path(path.relative_to(self.root))
                for key, path in self.directories.items()
            },
            "complete": status == "ready",
        }
        payload.update(extra)
        return payload

    def write_manifest(self, *, status: str = "ready", **extra: Any) -> Path:
        """Atomically publish this run's manifest after ensuring its folders."""

        self.ensure_directories()
        atomic_write_manifest(self.manifest_path, self.manifest_payload(status=status, **extra))
        return self.manifest_path


def _portable_path(path: Path) -> str:
    """Use slash-separated paths in JSON so manifests compare cross-platform."""

    return Path(path).as_posix()


def _normalise_path_value(value: Any) -> Any:
    if isinstance(value, Path):
        return _portable_path(value)
    if isinstance(value, os.PathLike):
        return _portable_path(Path(value))
    return value


def _canonicalize(value: Any) -> Any:
    """Convert common config values to deterministic JSON-compatible values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    value = _normalise_path_value(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        # JSON's representation of -0.0 is stable but surprising in a config
        # fingerprint; both values have identical runtime behaviour here.
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: _json_bytes(item))
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    # Enum-like values and simple scalar wrappers are common in WebUI payloads.
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _canonicalize(enum_value)
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON form used for configuration fingerprints."""

    return _json_bytes(_canonicalize(value)).decode("utf-8")


def _first(config: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return default


def _normalise_target_res(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        values = [part for part in re.split(r"[\s,;]+", value.strip()) if part]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = list(value)
    else:
        values = [value]
    normalised: list[Any] = []
    for item in values:
        try:
            number = int(item)
        except (TypeError, ValueError):
            normalised.append(_canonicalize(item))
        else:
            if number not in normalised:
                normalised.append(number)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in normalised):
        return sorted(normalised)
    return sorted(normalised, key=lambda item: (isinstance(item, str), str(item)))


_SOURCE_KEYS = {
    "source",
    "source_dir",
    "source_image_dir",
    "image_dir",
    "dataset_dir",
    "dataset_root",
    "output_dir",
    "output_root",
    "run_dir",
    "manifest",
}
_RESIZE_KEYS = (
    "crop",
    "resize",
    "adapt",
    "fit",
    "bucket",
    "freefit",
    "max_ratio",
    "target_res",
    "resolution",
)
_FILTER_KEYS = ("filter", "drop", "min_", "path_pattern", "recursive", "pattern")
_CAPTION_KEYS = ("caption", "shuffle", "tag_", "token", "keep_tokens")
_PREPROCESS_EXTRA_KEYS = (
    "preprocess",
    "cache",
    "mask",
    "sam",
    "mit",
    "conditioning",
    "vae",
    "text_encoder",
    "qwen",
    "pe_",
    "encoder",
    "correct",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"0", "false", "no", "off", ""}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def canonical_preprocess_config(
    config: Mapping[str, Any] | None = None,
    *,
    cache_schema_version: int | None = None,
) -> dict[str, Any]:
    """Project a preprocess config into a stable, source-independent payload.

    The explicit sections make the contract visible in ``manifest.json``.
    Additional explicitly preprocessing-scoped keys are retained under
    ``extra`` (except path/output identity keys); unrelated training/runtime
    settings do not fragment the cache namespace.
    """

    raw = dict(config or {})
    if cache_schema_version is None:
        cache_schema_version = int(
            _first(
                raw,
                (
                    "cache_schema_version",
                    "preprocess_cache_schema_version",
                    "schema_version",
                ),
                PREPROCESS_CACHE_SCHEMA_VERSION,
            )
        )

    canonical_keys = {
        "cache_schema_version",
        "target_res",
        "multires_per_image",
        "resize",
        "filter",
        "caption",
        "extra",
    }
    if set(raw) == canonical_keys:
        payload = _canonicalize(raw)
        payload["cache_schema_version"] = int(cache_schema_version)
        payload["target_res"] = _normalise_target_res(payload["target_res"])
        payload["multires_per_image"] = _as_bool(payload["multires_per_image"])
        return payload
    target_res = _normalise_target_res(
        _first(raw, ("target_res", "target_resolution", "resolutions"), [])
    )
    multi = _as_bool(
        _first(raw, ("multires_per_image", "multi_resolution", "multires"), False)
    )
    resize: dict[str, Any] = {}
    filtering: dict[str, Any] = {}
    caption: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key)
        lower = key_text.casefold()
        if lower in _SOURCE_KEYS or lower.endswith(("_dir", "_path")):
            continue
        if lower in {"target_res", "target_resolution", "resolutions", "multires_per_image", "multi_resolution", "multires"}:
            continue
        if any(token in lower for token in _RESIZE_KEYS):
            resize[key_text] = value
        elif any(token in lower for token in _FILTER_KEYS):
            filtering[key_text] = value
        elif any(token in lower for token in _CAPTION_KEYS):
            caption[key_text] = value
        elif lower in {"cache_schema_version", "preprocess_cache_schema_version", "schema_version"}:
            # The explicit keyword below is the one source of truth.
            continue
        elif lower.startswith(_PREPROCESS_EXTRA_KEYS):
            extra[key_text] = value
    return {
        "cache_schema_version": int(cache_schema_version),
        "target_res": target_res,
        "multires_per_image": multi,
        "resize": _canonicalize(resize),
        "filter": _canonicalize(filtering),
        "caption": _canonicalize(caption),
        "extra": _canonicalize(extra),
    }


def config_fingerprint(
    config: Mapping[str, Any] | None = None,
    *,
    cache_schema_version: int | None = None,
) -> str:
    """Return the short SHA-256 identity of a preprocess configuration."""

    payload = canonical_preprocess_config(
        config, cache_schema_version=cache_schema_version
    )
    return hashlib.sha256(_json_bytes(payload)).hexdigest()[:CONFIG_HASH_LENGTH]


def safe_source_name(value: str | os.PathLike | None, default: str = "dataset") -> str:
    """Make a source directory name safe on POSIX and Windows."""

    raw = unicodedata.normalize("NFKC", str(value or default)).strip()
    raw = _INVALID_COMPONENT.sub("_", raw)
    raw = _WHITESPACE.sub("_", raw).strip(" .")
    if not raw:
        raw = default
    if raw.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raw = f"_{raw}"
    # Keep the source label readable while leaving enough room for the hash.
    return raw[:96]


def source_path_hash(source_dir: str | os.PathLike) -> str:
    """Hash a resolved source path, with Windows case folding when applicable."""

    path = resolve_under_home(source_dir).expanduser().resolve(strict=False)
    value = path.as_posix()
    if sys.platform == "win32":
        value = value.casefold()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:SOURCE_HASH_LENGTH]


def source_fingerprint(source_dir: str | os.PathLike) -> str:
    """Alias for :func:`source_path_hash` used by manifest consumers."""

    return source_path_hash(source_dir)


def _resolve_runs_root(
    runs_root: str | os.PathLike | None = None,
    *,
    post_image_dataset: str | os.PathLike | None = None,
) -> Path:
    if runs_root is not None:
        return resolve_under_home(runs_root)
    base = post_image_dataset or "post_image_dataset"
    return resolve_under_home(base) / "runs"


def _new_run(
    source_dir: str | os.PathLike,
    config: Mapping[str, Any] | None,
    *,
    runs_root: str | os.PathLike | None = None,
    post_image_dataset: str | os.PathLike | None = None,
) -> PreprocessRun:
    source = resolve_under_home(source_dir).expanduser().resolve(strict=False)
    normalized_config = canonical_preprocess_config(config)
    source_hash = source_path_hash(source)
    source_group = f"{safe_source_name(source.name)}-{source_hash}"
    cfg_hash = config_fingerprint(normalized_config)
    root = _resolve_runs_root(runs_root, post_image_dataset=post_image_dataset)
    return PreprocessRun(
        source_dir=source,
        root=root / source_group / cfg_hash,
        source_group=source_group,
        source_hash=source_hash,
        config_hash=cfg_hash,
        config=normalized_config,
    )


def load_manifest(path: str | os.PathLike | PreprocessRun) -> dict[str, Any]:
    """Read a run manifest and reject malformed/incomplete JSON."""

    manifest_path = (
        path.manifest_path if isinstance(path, PreprocessRun) else Path(path)
    )
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreprocessRunError(f"Preprocess manifest not found: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessRunError(f"Invalid preprocess manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreprocessRunError(f"Preprocess manifest must be an object: {manifest_path}")
    if data.get("kind", "preprocess_run") != "preprocess_run":
        raise PreprocessRunError(f"Unsupported preprocess manifest kind: {manifest_path}")
    if not bool(data.get("complete", data.get("status") == "ready")):
        raise PreprocessRunError(f"Incomplete preprocess manifest: {manifest_path}")
    return data


def validate_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike,
    *,
    source_dir: str | os.PathLike | None = None,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Check a manifest's completion and optional source/config identity."""

    try:
        data = load_manifest(manifest) if not isinstance(manifest, Mapping) else dict(manifest)
    except PreprocessRunError:
        return False
    if not bool(data.get("complete", data.get("status") == "ready")):
        return False
    if source_dir is not None:
        expected = source_path_hash(source_dir)
        actual = data.get("source_hash") or (data.get("source") or {}).get("path_hash")
        if actual != expected:
            return False
    if config is not None:
        expected = config_fingerprint(config)
        if data.get("config_hash") != expected:
            return False
    return True


def resolve_preprocess_run(
    source_dir: str | os.PathLike,
    config: Mapping[str, Any] | None = None,
    *,
    runs_root: str | os.PathLike | None = None,
    post_image_dataset: str | os.PathLike | None = None,
    create: bool = True,
) -> PreprocessRun:
    """Resolve (and by default create) the canonical run for source/config.

    A valid existing manifest is reused unchanged.  If only the directory
    skeleton exists, missing directories are repaired and the manifest is
    published atomically.  A conflicting manifest is never silently replaced.
    """

    run = _new_run(
        source_dir,
        config,
        runs_root=runs_root,
        post_image_dataset=post_image_dataset,
    )
    if not create:
        return run
    run.ensure_directories()
    if run.manifest_path.exists():
        try:
            existing = load_manifest(run.manifest_path)
        except PreprocessRunError as exc:
            # A daemon interruption can leave a deliberately incomplete
            # ``running``/``failed`` manifest.  It is safe to resume that exact
            # identity, but never to accept a malformed or mismatched document.
            try:
                raw = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise PreprocessRunError(
                    f"Cannot reuse existing preprocess run {run.root}: {exc}"
                ) from exc
            if not isinstance(raw, dict) or raw.get("kind", "preprocess_run") != "preprocess_run":
                raise PreprocessRunError(
                    f"Cannot reuse existing preprocess run {run.root}: {exc}"
                ) from exc
            source_hash = raw.get("source_hash") or (raw.get("source") or {}).get("path_hash")
            config_hash = raw.get("config_hash") or raw.get("config_fingerprint")
            if source_hash != run.source_hash or config_hash != run.config_hash:
                raise PreprocessRunError(
                    f"Cannot reuse existing preprocess run {run.root}: identity mismatch"
                ) from exc
            return run
        if not validate_manifest(existing, source_dir=run.source_dir, config=run.config):
            raise PreprocessRunError(
                f"Preprocess manifest identity mismatch: {run.manifest_path}"
            )
        return run
    run.write_manifest()
    return run


def _legacy_root(path: str | os.PathLike) -> Path:
    root = resolve_under_home(path).expanduser()
    if root.name in RUN_KINDS:
        return root.parent
    return root


def legacy_paths(
    post_image_dataset: str | os.PathLike = "post_image_dataset",
) -> dict[str, Path]:
    """Return the historical fixed-directory cache paths."""

    root = resolve_under_home(post_image_dataset)
    return {
        "root": root,
        "resized": root / "resized",
        "lora": root / "lora",
        "masks": root / "masks",
        "mask": root / "mask",
        "multires": root / "multires",
        "conditioning": root / "conditioning",
        "conditioning_data": root / "conditioning_data",
        "conditioning_resized": root / "cond_resized",
        "cond_resized": root / "cond_resized",
        "captions": root / "captions",
    }


def _find_legacy_manifest(root: Path) -> Path | None:
    for candidate in (
        root / "manifest.json",
        root / "preprocess_manifest.json",
        root / ".preprocess_manifest.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_legacy_manifest(path: Path) -> dict[str, Any]:
    """Read legacy metadata without requiring the new ``complete`` field."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessRunError(f"Invalid legacy preprocess metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreprocessRunError(f"Legacy preprocess metadata must be an object: {path}")
    if data.get("kind") not in (None, "preprocess_run"):
        raise PreprocessRunError(f"Unsupported legacy preprocess metadata kind: {path}")
    if data.get("complete") is False or data.get("status") in {"running", "incomplete"}:
        raise PreprocessRunError(f"Incomplete legacy preprocess metadata: {path}")
    return data


def validate_legacy_cache(
    legacy_root: str | os.PathLike,
    *,
    source_dir: str | os.PathLike | None = None,
    config: Mapping[str, Any] | None = None,
    require_metadata: bool = False,
) -> LegacyValidation:
    """Validate a legacy fixed cache before optional lazy migration.

    Legacy directories without metadata are accepted for read compatibility by
    default.  Set ``require_metadata=True`` when a caller needs a migration
    gate that refuses ambiguous caches.
    """

    root = _legacy_root(legacy_root)
    if not root.exists() or not root.is_dir():
        return LegacyValidation(root, False, reason="legacy cache root does not exist")
    manifest_path = _find_legacy_manifest(root)
    if manifest_path is None:
        present = any(
            (root / candidate).is_dir()
            for kind in RUN_KINDS
            for candidate in _LEGACY_KIND_DIRS[kind]
        )
        if not present:
            return LegacyValidation(root, False, reason="no legacy cache directories found")
        if require_metadata:
            return LegacyValidation(
                root, False, metadata_present=False, reason="legacy cache metadata is missing"
            )
        return LegacyValidation(root, True, metadata_present=False, reason="metadata missing")
    try:
        data = _read_legacy_manifest(manifest_path)
    except PreprocessRunError as exc:
        return LegacyValidation(
            root, False, metadata_present=True, reason=str(exc), manifest_path=manifest_path
        )
    if source_dir is not None:
        actual = data.get("source_hash") or (data.get("source") or {}).get("path_hash")
        if actual is None and require_metadata:
            return LegacyValidation(
                root,
                False,
                metadata_present=True,
                metadata_valid=False,
                reason="legacy source metadata is missing",
                manifest_path=manifest_path,
            )
        if actual is not None and actual != source_path_hash(source_dir):
            return LegacyValidation(
                root,
                False,
                metadata_present=True,
                metadata_valid=False,
                reason="legacy source metadata does not match",
                manifest_path=manifest_path,
            )
    if config is not None:
        actual = data.get("config_hash") or data.get("config_fingerprint")
        if actual is None and require_metadata:
            return LegacyValidation(
                root,
                False,
                metadata_present=True,
                metadata_valid=False,
                reason="legacy config metadata is missing",
                manifest_path=manifest_path,
            )
        if actual is not None and actual != config_fingerprint(config):
            return LegacyValidation(
                root,
                False,
                metadata_present=True,
                metadata_valid=False,
                reason="legacy config metadata does not match",
                manifest_path=manifest_path,
            )
    return LegacyValidation(
        root,
        True,
        metadata_present=True,
        metadata_valid=True,
        manifest_path=manifest_path,
    )


def legacy_cache_is_compatible(*args: Any, **kwargs: Any) -> bool:
    """Boolean convenience wrapper around :func:`validate_legacy_cache`."""

    return bool(validate_legacy_cache(*args, **kwargs))


def _atomic_replace(src: Path, dst: Path, *, retries: int = 8) -> None:
    last_error: OSError | None = None
    for attempt in range(max(1, retries)):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 >= retries:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))
    if last_error is not None:
        raise last_error


def atomic_write_manifest(
    path: str | os.PathLike,
    payload: Mapping[str, Any],
    *,
    retries: int = 8,
) -> Path:
    """Write a manifest using a same-directory temp file and ``os.replace``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _canonicalize(dict(payload))
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, destination, retries=retries)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def _copy_or_link_atomic(source: Path, destination: Path, *, retries: int = 8) -> str:
    """Publish one file atomically, preferring a hard link over ``copy2``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if source.stat().st_size == destination.stat().st_size and filecmp.cmp(
                source, destination, shallow=False
            ):
                return "skipped"
        except OSError:
            pass
        raise FileExistsError(f"destination already exists with different size: {destination}")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
            mode = "hardlink"
        except (OSError, NotImplementedError):
            shutil.copy2(source, temporary)
            mode = "copy"
        _atomic_replace(temporary, destination, retries=retries)
        return mode
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def migrate_legacy_cache(
    legacy_root: str | os.PathLike,
    run: PreprocessRun | str | os.PathLike,
    *,
    source_dir: str | os.PathLike | None = None,
    config: Mapping[str, Any] | None = None,
    kinds: Iterable[str] | None = None,
    require_metadata: bool = False,
    retries: int = 8,
) -> MigrationResult:
    """Lazily mirror legacy cache files into ``run``.

    Files are hard-linked when possible, otherwise copied through a temporary
    same-directory file and atomically published.  The legacy source is never
    removed.  Missing/failed files are reported in ``MigrationResult`` so a
    caller can continue reading the old path or retry later.
    """

    source_root = _legacy_root(legacy_root)
    if not isinstance(run, PreprocessRun):
        manifest_path = Path(run)
        run = run_from_manifest(manifest_path)
    validation = validate_legacy_cache(
        source_root,
        source_dir=source_dir or run.source_dir,
        config=config or run.config,
        require_metadata=require_metadata,
    )
    if not validation.valid:
        reason = validation.reason or "legacy cache metadata is incompatible"
        return MigrationResult(source_root, run.root, failed=((source_root, reason),))

    requested = tuple(kinds) if kinds is not None else RUN_KINDS
    invalid = [kind for kind in requested if kind not in RUN_KINDS]
    if invalid:
        return MigrationResult(
            source_root,
            run.root,
            failed=((source_root, f"unknown cache kind(s): {invalid}"),),
        )

    migrated: list[Path] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for kind in requested:
        for legacy_kind in _LEGACY_KIND_DIRS[kind]:
            src_dir = source_root / legacy_kind
            if not src_dir.is_dir():
                continue
            if kind != "conditioning":
                dst_dir = run.directories[kind]
            elif legacy_kind == "conditioning_data":
                dst_dir = run.conditioning_data_dir
            elif legacy_kind == "cond_resized":
                dst_dir = run.conditioning_resized_dir
            else:
                dst_dir = run.conditioning_dir
            for source in sorted(src_dir.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(src_dir)
                destination = dst_dir / relative
                try:
                    mode = _copy_or_link_atomic(source, destination, retries=retries)
                except OSError as exc:  # best effort: preserve source and report it
                    failed.append((source, str(exc)))
                else:
                    if mode == "skipped":
                        skipped.append(destination)
                    else:
                        migrated.append(destination)

    return MigrationResult(
        source_root,
        run.root,
        migrated=tuple(migrated),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )


def run_from_manifest(path: str | os.PathLike) -> PreprocessRun:
    """Reconstruct a :class:`PreprocessRun` from a complete manifest."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    data = load_manifest(manifest_path)
    source = (
        data.get("source_dir")
        or data.get("source_path")
        or (data.get("source") or {}).get("path")
    )
    source_hash = data.get("source_hash") or (data.get("source") or {}).get("path_hash")
    source_group = data.get("source_group")
    config = data.get("config") or {}
    config_hash = data.get("config_hash") or data.get("config_fingerprint")
    if config_hash is None:
        config_hash = config_fingerprint(config)
    if not source or not source_hash or not source_group:
        raise PreprocessRunError(f"Manifest is missing run identity: {manifest_path}")
    if source_path_hash(source) != source_hash:
        raise PreprocessRunError(f"Manifest source identity mismatch: {manifest_path}")
    if config and config_fingerprint(config) != config_hash:
        raise PreprocessRunError(f"Manifest config identity mismatch: {manifest_path}")
    return PreprocessRun(
        source_dir=Path(source),
        root=manifest_path.parent,
        source_group=str(source_group),
        source_hash=str(source_hash),
        config_hash=str(config_hash),
        config=config,
    )


# Names used by callers that prefer an explicit resolver noun.
PreprocessRunResolver = resolve_preprocess_run
resolve_run = resolve_preprocess_run
resolve_preprocess_run_manifest = run_from_manifest
parse_preprocess_run = run_from_manifest
fingerprint_config = config_fingerprint
write_manifest = atomic_write_manifest
migrate_legacy = migrate_legacy_cache


__all__ = [
    "CONFIG_HASH_LENGTH",
    "PREPROCESS_CACHE_SCHEMA_VERSION",
    "PREPROCESS_RUN_SCHEMA_VERSION",
    "RUN_KINDS",
    "SOURCE_HASH_LENGTH",
    "LegacyValidation",
    "MigrationResult",
    "PreprocessRun",
    "PreprocessRunError",
    "PreprocessRunResolver",
    "atomic_write_manifest",
    "canonical_json",
    "canonical_preprocess_config",
    "config_fingerprint",
    "fingerprint_config",
    "legacy_cache_is_compatible",
    "legacy_paths",
    "load_manifest",
    "migrate_legacy",
    "migrate_legacy_cache",
    "parse_preprocess_run",
    "resolve_preprocess_run",
    "resolve_preprocess_run_manifest",
    "resolve_run",
    "run_from_manifest",
    "safe_source_name",
    "source_fingerprint",
    "source_path_hash",
    "validate_legacy_cache",
    "validate_manifest",
    "write_manifest",
]
