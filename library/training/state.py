"""Versioned, JSON-safe training state records.

Accelerate owns the binary optimizer/model payload; this sidecar owns the
semantic cursor needed to resume a run without guessing whether a number means
an optimizer step or a DataLoader micro-batch.  Readers accept the historical
``current_step`` key and normalize it to ``global_step``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import torch

SCHEMA_VERSION = 2
COMPLETE_MARKER = "complete.marker"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _b64(value.detach().cpu().contiguous().numpy().tobytes()),
        }
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": _b64(bytes(value))}
    if isinstance(value, tuple):
        return {"__tuple__": [_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    # NumPy's RNG protocol contains a uint32 ndarray.  Converting that array
    # with ``str(value)`` silently destroys the state and makes a resumed run
    # non-deterministic.  Keep the representation dependency-light by loading
    # NumPy only when a NumPy value is actually encountered.
    module = type(value).__module__
    if module.startswith("numpy"):
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    return {"__ndarray_list__": _json_value(value.tolist())}
                return {
                    "__ndarray__": True,
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "data": _b64(value.tobytes(order="C")),
                }
            if isinstance(value, np.dtype):
                return {"__dtype__": value.str}
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _restore_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("__bytes__"):
        return _unb64(value["__bytes__"])
    if value.get("__tuple__") is not None:
        return tuple(_restore_value(item) for item in value["__tuple__"])
    if value.get("__ndarray__"):
        try:
            import numpy as np

            dtype = np.dtype(value.get("dtype", "uint32"))
            raw = _unb64(value.get("data", ""))
            shape = tuple(int(dim) for dim in value.get("shape", ()))
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        except Exception:
            return None
    if "__ndarray_list__" in value:
        try:
            import numpy as np

            return np.asarray(_restore_value(value["__ndarray_list__"]))
        except Exception:
            return _restore_value(value["__ndarray_list__"])
    if "__dtype__" in value:
        try:
            import numpy as np

            return np.dtype(value["__dtype__"])
        except Exception:
            return value["__dtype__"]
    if value.get("__tensor__"):
        dtype_name = str(value.get("dtype", "torch.uint8")).removeprefix("torch.")
        dtype = getattr(torch, dtype_name, torch.uint8)
        raw = _unb64(value.get("data", ""))
        # RNG tensors are byte tensors.  For arbitrary tensors this fallback is
        # still deterministic enough for a state sidecar and avoids importing
        # numpy dtypes into the protocol.
        if dtype is torch.uint8:
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        return raw
    return {k: _restore_value(v) for k, v in value.items()}


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture Python/NumPy/Torch RNG state without requiring NumPy at import."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return _json_value(state)


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Best-effort inverse of :func:`capture_rng_state`."""

    if not state:
        return
    decoded = _restore_value(dict(state))
    try:
        if decoded.get("python") is not None:
            random.setstate(decoded["python"])
    except Exception:
        pass
    try:
        if decoded.get("torch_cpu") is not None:
            torch.set_rng_state(decoded["torch_cpu"])
    except Exception:
        pass
    try:
        cuda = decoded.get("torch_cuda")
        if cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda)
    except Exception:
        pass
    try:
        import numpy as np

        if decoded.get("numpy") is not None:
            np.random.set_state(decoded["numpy"])
    except Exception:
        pass


def signature(value: Any) -> str:
    """Stable short SHA-256 for config/dataset identity checks."""

    blob = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def normalize_train_state(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize old and new state records to the explicit schema."""

    raw = dict(data or {})
    global_step = raw.get("global_step", raw.get("current_step", 0))
    try:
        global_step = max(0, int(global_step))
    except (TypeError, ValueError):
        global_step = 0
    epoch = raw.get("current_epoch", raw.get("epoch", 0))
    try:
        epoch = max(0, int(epoch))
    except (TypeError, ValueError):
        epoch = 0
    offset = raw.get("micro_batch_offset", raw.get("batch_offset", 0))
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    out = dict(raw)
    out.update(
        {
            "schema_version": int(raw.get("schema_version", 1)),
            "global_step": global_step,
            "current_step": global_step,  # legacy reader/writer contract
            "current_epoch": epoch,
            "micro_batch_offset": offset,
            "stage_index": int(raw.get("stage_index", -1) or -1),
            "stage_batch_cursor": int(raw.get("stage_batch_cursor", 0) or 0),
            "stage_outer_epoch": int(raw.get("stage_outer_epoch", epoch) or epoch),
        }
    )
    return out


def build_train_state(
    *,
    global_step: int,
    current_epoch: int = 0,
    micro_batch_offset: int = 0,
    stage_index: int = -1,
    stage_batch_cursor: int = 0,
    stage_outer_epoch: int | None = None,
    rng_state: Mapping[str, Any] | None = None,
    config_signature: str | None = None,
    dataset_signature: str | None = None,
    interrupted: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a complete semantic state record."""

    state = {
        "schema_version": SCHEMA_VERSION,
        "global_step": max(0, int(global_step)),
        "current_step": max(0, int(global_step)),
        "current_epoch": max(0, int(current_epoch)),
        "micro_batch_offset": max(0, int(micro_batch_offset)),
        "stage_index": int(stage_index),
        "stage_batch_cursor": max(0, int(stage_batch_cursor)),
        "stage_outer_epoch": max(
            0, int(current_epoch if stage_outer_epoch is None else stage_outer_epoch)
        ),
        "interrupted": bool(interrupted),
        "rng_state": _json_value(rng_state) if rng_state is not None else capture_rng_state(),
    }
    if config_signature is not None:
        state["config_signature"] = str(config_signature)
    if dataset_signature is not None:
        state["dataset_signature"] = str(dataset_signature)
    state.update({key: _json_value(value) for key, value in extra.items()})
    return state


def write_complete_marker(directory: Path) -> Path:
    marker = directory / COMPLETE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    tmp.write_text("ok\n", encoding="ascii")
    os.replace(tmp, marker)
    return marker


def state_is_complete(
    directory: str | os.PathLike, *, require_marker: bool = False
) -> bool:
    """Validate a resumable state directory.

    Legacy ``current_step`` records predate the completion marker and remain
    readable. New schema records can opt into strict marker validation; this is
    what automatic recovery uses so a torn directory cannot win over an older
    complete snapshot.
    """

    path = Path(directory)
    if not path.is_dir() or not (path / "train_state.json").is_file():
        return False
    try:
        raw = json.loads((path / "train_state.json").read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            return False
        # A sidecar with no cursor is not a resumable state, even if the JSON
        # happens to be syntactically valid.  Accept either the new explicit
        # key or the historical ``current_step`` key.
        if "global_step" not in raw and "current_step" not in raw:
            return False
        normalize_train_state(raw)
        if require_marker and int(raw.get("schema_version", 1) or 1) >= SCHEMA_VERSION:
            if not (path / COMPLETE_MARKER).is_file():
                return False
        # The marker is deliberately checked after parsing so a torn write
        # cannot be made valid by a stale marker.
        return True
    except (OSError, ValueError, TypeError):
        return False


def read_train_state(directory: str | os.PathLike) -> dict[str, Any] | None:
    path = Path(directory) / "train_state.json"
    try:
        return normalize_train_state(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None
