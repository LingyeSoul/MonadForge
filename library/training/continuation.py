"""Pinned continuation input and scheduler context shared by daemon and trainer."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

#: Raising the training target past the pinned horizon is only numerically
#: sound for schedulers whose curve does not depend on the total step count.
CONSTANT_CONTINUATION_SCHEDULERS = {"constant", "constant_with_warmup"}

#: Parameters that must not drift between the source run and a continuation,
#: even though the sparse GUI snapshot does not carry them: loader settings
#: realign batch math (mis-replayed steps), network settings reshape or
#: rescale the LoRA being resumed, and the scheduler identity defines which
#: curve the pinned horizon reconstructs.
CRITICAL_CONTINUATION_PARAMS = (
    "train_batch_size",
    "gradient_accumulation_steps",
    "network_module",
    "network_dim",
    "network_alpha",
    "network_args",
    "network_dropout",
    "optimizer_type",
    "lr_scheduler",
    "lr_warmup_steps",
)

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
    snapshot = Path(context["config_snapshot"])
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

    # FICLONE: Linux ioctl for a CoW reflink; on other platforms/filesystems
    # (Windows, cross-device) the clone fails and we fall back to a full copy.
    ficlone = 0x40049409
    reflink = True

    def clone(src, dst):
        nonlocal reflink
        if reflink:
            try:
                import fcntl

                with open(src, "rb") as incoming, open(dst, "wb") as outgoing:
                    fcntl.ioctl(outgoing.fileno(), ficlone, incoming.fileno())
                shutil.copystat(src, dst)
                return dst
            except (ImportError, OSError):
                reflink = False
                logger.info("reflink clone unavailable; falling back to plain copy")
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
    # The scheduler identity is part of the run contract: the pinned horizon
    # only reproduces the original LR curve when the scheduler family itself
    # comes from the source run, not from live method-chain defaults or argv
    # overrides inherited from the source command. (Custom schedulers never
    # reach here — the daemon gate rejects them for every mode.)
    if context.get("lr_scheduler"):
        args.lr_scheduler = context["lr_scheduler"]
    if context.get("lr_warmup_steps") is not None:
        args.lr_warmup_steps = context["lr_warmup_steps"]
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
    # The accepted consequence (proposal §5): keys outside the sparse snapshot
    # follow the live chain — except the critical parameters below, which
    # would silently misalign resumed batches or reshape the network.
    for key, expected in (context.get("critical_params") or {}).items():
        actual = getattr(args, key, None)
        if actual != expected:
            raise ValueError(
                f"续训参数 {key} 与原训练不一致（原值 {expected!r}，当前 {actual!r}），"
                "请恢复原配置或新建任务"
            )
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
        if getattr(args, "max_train_steps", 0) > context["scheduler_steps"] and (
            getattr(args, "lr_scheduler_type", None)
            or getattr(args, "lr_scheduler", "constant")
            not in CONSTANT_CONTINUATION_SCHEDULERS
        ):
            raise ValueError(
                "Extending the training target requires a constant-family lr "
                "scheduler; the original curve cannot be stretched"
            )
        return context["scheduler_steps"], context["warmup_steps"]
    total = args.max_train_steps * num_processes
    warmup = args.lr_warmup_steps
    return total, int(warmup * total) if isinstance(warmup, float) else warmup
