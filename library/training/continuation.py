"""Pinned continuation input and scheduler context shared by daemon and trainer."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

_SNAPSHOT_SKIP = {
    "output_dir",
    "resume",
    "config_file",
    "progress_jsonl",
    "sample_dir",
    "config_snapshot",
    "print_config",
}
_SNAPSHOT_SECTIONS = {"datasets", "general", "variant"}


def overlay_continuation_snapshot(args, context: dict) -> None:
    """Restore the original run's hyperparameters over the current variant file."""
    path = context.get("config_snapshot")
    if path:
        snapshot = Path(path)
    else:
        env = os.environ.get("ANIMA_CONTINUATION_FILE")
        snapshot = Path(env).with_name("config.snapshot.toml") if env else None
    if snapshot is None or not snapshot.is_file():
        raise ValueError("Continuation is missing the original config snapshot")
    import toml
    from library.config.io import _drop_unset_config_values

    raw = toml.load(snapshot)
    _drop_unset_config_values(raw)
    for key in _SNAPSHOT_SECTIONS:
        raw.pop(key, None)
    for key, value in raw.items():
        if key in _SNAPSHOT_SKIP or str(key).startswith("_"):
            continue
        setattr(args, key, value)


def blueprint_from_config(cfg: dict) -> dict | None:
    datasets = cfg.get("datasets")
    if not datasets:
        return None
    out = {"datasets": copy.deepcopy(datasets)}
    if cfg.get("general") is not None:
        out["general"] = copy.deepcopy(cfg["general"])
    return out


def blueprint_is_complete(blueprint: dict | None) -> bool:
    if not blueprint or not blueprint.get("datasets"):
        return False
    for dataset in blueprint["datasets"]:
        subsets = (dataset or {}).get("subsets") or []
        if not subsets:
            return False
        for subset in subsets:
            if not subset.get("image_dir") and not subset.get("metadata_file"):
                return False
    return True


def complete_continuation_blueprint(blueprint: dict | None, args) -> dict | None:
    """GUI snapshots store sparse subset overlays, not a full dataset schema."""
    if blueprint is None:
        return None
    if blueprint_is_complete(blueprint):
        return copy.deepcopy(blueprint)
    from library.config.io import (
        _apply_dataset_overrides,
        load_dataset_config_from_base,
    )

    base = load_dataset_config_from_base(overrides=vars(args), method=None)
    if base is None:
        return copy.deepcopy(blueprint)
    merged = copy.deepcopy(base)
    _apply_dataset_overrides(merged, blueprint)
    return merged


def state_fingerprint(directory: Path, *, content: bool = False) -> str:
    """Bind a publication to every payload, not just its step sidecar."""
    from library.training.state import state_is_complete

    if not state_is_complete(directory, require_marker=True):
        raise ValueError("Training state is incomplete")
    required = (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "random_states_0.pkl",
    )
    for name in required:
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise ValueError(f"Training state is missing {name}")
    from safetensors import safe_open
    import zipfile

    try:
        with safe_open(directory / "model.safetensors", framework="pt") as weights:
            if not list(weights.keys()):
                raise ValueError("Empty model state")
        for name in required[1:]:
            with zipfile.ZipFile(directory / name) as archive:
                if not archive.namelist() or (
                    content and archive.testzip() is not None
                ):
                    raise ValueError(f"Corrupt {name}")
    except Exception as exc:
        raise ValueError(f"Unreadable training state: {exc}") from exc
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Training state contains an unsupported payload")
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        if content or path.name in {
            "train_state.json",
            "complete.marker",
            ".save-token",
        }:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def dataset_blueprint_path(args) -> Path | None:
    """Pin the live dataset blueprint to the daemon job, never a method file."""
    if not os.environ.get("ANIMA_DAEMON_JOB_ID"):
        return None
    progress = getattr(args, "progress_jsonl", None)
    if not progress:
        return None
    return Path(progress).parent / "dataset.snapshot.json"


def copy_state(source: Path, destination: Path) -> None:
    """Copy on write where available; retain an independent immutable input."""

    def clone(src, dst):
        try:
            import fcntl

            with open(src, "rb") as incoming, open(dst, "wb") as outgoing:
                fcntl.ioctl(outgoing.fileno(), 0x40049409, incoming.fileno())
            shutil.copystat(src, dst)
            return dst
        except (ImportError, OSError):
            return shutil.copy2(src, dst)

    shutil.copytree(source, destination, copy_function=clone)


def read_context() -> dict | None:
    filename = os.environ.get("ANIMA_CONTINUATION_FILE")
    if not filename:
        return None
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    if data.get("job_id") != os.environ.get("ANIMA_DAEMON_JOB_ID") or data.get(
        "root_job_id"
    ) != os.environ.get("ANIMA_DAEMON_ROOT_JOB_ID"):
        raise ValueError("Continuation context does not belong to this attempt")
    return data


def setup_continuation(args) -> None:
    """Called after original output canonicalization, before any trainer writes."""
    from library.io.output_layout import layout_from_args

    if getattr(args, "_continuation", None):
        return
    context = read_context()
    if context is None:
        return
    state = Path(args.resume or "").resolve()
    if state != Path(context["state_dir"]).resolve():
        raise ValueError("Continuation input path changed")
    if state_fingerprint(state, content=True) != context["state_digest"]:
        raise ValueError("Continuation payload changed after preparation")
    overlay_continuation_snapshot(args, context)
    args._continuation = context
    args._continuation_signature_output = context["signature_output_dir"]
    args.output_dir = context["output_dir"]
    args._output_layout = None
    layout_from_args(args)
    key = context.get("budget_key")
    if key:
        setattr(args, key, context["target"])
        if key == "max_train_epochs" and not getattr(
            args, "_max_train_steps_explicit", False
        ):
            args.max_train_steps = None


def validate_continuation_signatures(args) -> None:
    context = getattr(args, "_continuation", None)
    if context is None:
        return
    # Live argparse hashes are not comparable to the original run: the GUI
    # variant file may have changed, and method-chain vs snapshot merges
    # materialize different default keys. Identity is the pinned snapshot.
    args.config_signature = context["config_signature"]
    expected = context.get("dataset_signature")
    actual = getattr(args, "dataset_signature", None)
    if actual is not None and expected is not None and actual != expected:
        raise ValueError(
            "Continuation dataset_signature mismatch; original configuration/data changed"
        )


def scheduler_budget(args, num_processes: int) -> tuple[int, int | None]:
    """Use the original scheduler horizon while extending the stop budget."""
    context = getattr(args, "_continuation", None)
    if context:
        if num_processes != 1:
            raise ValueError("Continuation currently requires single-process training")
        return context["scheduler_steps"], context["warmup_steps"]
    total = args.max_train_steps * num_processes
    warmup = args.lr_warmup_steps
    return total, int(warmup * total) if isinstance(warmup, float) else warmup
