"""System API — model group status, model paths, environment info, and hw stats."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from library.config.model_paths import (
    MODEL_CONFIG_DEFAULTS,
    MODEL_CONFIG_KEYS,
    load_model_config,
)
from webui.services.config_service import ROOT

router = APIRouter()

# Model groups — the anima group reads the effective model config; others are fixed.
_ANIMA_PATH_KEYS = [(key, default) for key, default in MODEL_CONFIG_DEFAULTS.items()]
_STATIC_GROUPS: list[dict] = [
    {
        "id": "sam3",
        "files": [
            "models/sam3/sam3.pt",
        ],
    },
    {
        "id": "mit",
        "files": [
            "models/mit/model.pth",
        ],
    },
    {
        "id": "pe",
        "files": [
            "models/pe/PE-Core-L14-336.pt",
        ],
    },
]


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _check_anima_group() -> dict:
    model = load_model_config(ROOT / "configs")
    files_info = []
    for toml_key, default_ in _ANIMA_PATH_KEYS:
        raw = model.get(toml_key, default_)
        resolved = _resolve_path(raw)
        files_info.append({"path": raw, "exists": resolved.is_file()})
    installed = all(f["exists"] for f in files_info)
    return {"id": "anima", "installed": installed, "files": files_info}


def _check_group(group: dict) -> dict:
    paths = [ROOT / f for f in group["files"]]
    installed = all(p.is_file() for p in paths)
    return {
        "id": group["id"],
        "installed": installed,
        "files": [{"path": f, "exists": (ROOT / f).is_file()} for f in group["files"]],
    }


@router.get("/models")
def get_model_groups():
    """Check installation status of each model group."""
    groups = [_check_anima_group()]
    groups.extend(_check_group(g) for g in _STATIC_GROUPS)
    return {"groups": groups}


# ── Model paths (configurable from System page) ────────────────

# Model path keys exposed by the System page.
_MODEL_PATH_KEYS: list[tuple[str, str, str]] = [
    (
        "anima_dit",
        "pretrained_model_name_or_path",
        MODEL_CONFIG_DEFAULTS["pretrained_model_name_or_path"],
    ),
    ("anima_te", "qwen3", MODEL_CONFIG_DEFAULTS["qwen3"]),
    ("anima_vae", "vae", MODEL_CONFIG_DEFAULTS["vae"]),
]


class ModelPathUpdate(BaseModel):
    key: str
    value: str


def _resolve_and_check(path_str: str) -> tuple[str, bool]:
    """Resolve a path (relative to ROOT or absolute) and check existence."""
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return str(p), p.is_file()


@router.get("/model-paths")
def get_model_paths():
    """Return the configured model paths with existence status."""
    model = load_model_config(ROOT / "configs")
    paths = []
    for id_, toml_key, default_ in _MODEL_PATH_KEYS:
        raw = model.get(toml_key, default_)
        resolved, exists = _resolve_and_check(raw)
        paths.append(
            {
                "id": id_,
                "toml_key": toml_key,
                "path": raw,
                "resolved": resolved,
                "exists": exists,
            }
        )
    return {"paths": paths}


@router.put("/model-paths")
def update_model_paths(body: list[ModelPathUpdate]):
    """Update machine-local paths in gitignored custom/model.toml."""
    import toml as _toml

    custom_path = ROOT / "configs" / "custom" / "model.toml"
    custom = (
        _toml.loads(custom_path.read_text(encoding="utf-8"))
        if custom_path.exists()
        else {}
    )
    for item in body:
        if item.key in MODEL_CONFIG_KEYS:
            custom[item.key] = item.value
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    custom_path.write_text(_toml.dumps(custom), encoding="utf-8")
    return {"ok": True}


# ── Hardware stats ─────────────────────────────────────────────


@router.get("/hw-stats")
def get_hw_stats():
    """Return live GPU / CPU / memory statistics."""
    import psutil

    # CPU & system memory
    cpu_percent = psutil.cpu_percent(interval=0)
    vm = psutil.virtual_memory()

    stats: dict = {
        "cpu_percent": cpu_percent,
        "mem_used_gb": round(vm.used / (1024**3), 1),
        "mem_total_gb": round(vm.total / (1024**3), 1),
        "mem_percent": vm.percent,
    }

    # GPU detection: nvidia-smi primary (system-wide GPU memory/util/temp),
    # torch.cuda supplement (device name, per-process reserved memory).
    #
    # nvidia-smi reports the ACTUAL system-wide memory usage — torch.cuda only
    # sees the current process's allocation (0 for the WebUI server process
    # since training runs in a separate subprocess).  So nvidia-smi values
    # always win for memory / util / temp.
    try:
        import subprocess

        from library.runtime.proc import no_window_kwargs

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            **no_window_kwargs(),
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 5:
                stats["gpu_name"] = parts[0]
                stats["gpu_mem_used_gb"] = round(int(parts[1]) / 1024, 1)
                stats["gpu_mem_total_gb"] = round(int(parts[2]) / 1024, 1)
                stats["gpu_util_percent"] = int(parts[3])
                stats["gpu_temp_c"] = int(parts[4])
    except Exception:
        pass

    # torch.cuda: supplement with device name + reserved memory when available
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            stats.setdefault("gpu_name", torch.cuda.get_device_name(idx))
            stats.setdefault(
                "gpu_mem_total_gb",
                round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 1),
            )
            # Per-process allocated + reserved (useful for debugging OOM)
            stats["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved(idx) / (1024**3), 1)
    except Exception:
        pass

    return stats
