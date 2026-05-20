"""System API — model group status, model paths, environment info, and hw stats."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from webui.services.config_service import ROOT, read_layer

router = APIRouter()

# Model groups — the anima group reads actual paths from base.toml; others are fixed.
_ANIMA_PATH_KEYS = [
    ("pretrained_model_name_or_path", "models/diffusion_models/anima-base-v1.0.safetensors"),
    ("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
    ("vae", "models/vae/qwen_image_vae.safetensors"),
]
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
    base = read_layer("base")
    files_info = []
    for toml_key, default_ in _ANIMA_PATH_KEYS:
        raw = base.get(toml_key, default_)
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

# Keys in base.toml that hold model weight paths.
_MODEL_PATH_KEYS: list[tuple[str, str, str]] = [
    ("anima_dit", "pretrained_model_name_or_path", "models/diffusion_models/anima-base-v1.0.safetensors"),
    ("anima_te", "qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
    ("anima_vae", "vae", "models/vae/qwen_image_vae.safetensors"),
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
    base = read_layer("base")
    paths = []
    for id_, toml_key, default_ in _MODEL_PATH_KEYS:
        raw = base.get(toml_key, default_)
        resolved, exists = _resolve_and_check(raw)
        paths.append({
            "id": id_,
            "toml_key": toml_key,
            "path": raw,
            "resolved": resolved,
            "exists": exists,
        })
    return {"paths": paths}


@router.put("/model-paths")
def update_model_paths(body: list[ModelPathUpdate]):
    """Update model paths in base.toml.

    Does a targeted update of only the model path keys to avoid
    destroying ``[[datasets]]`` and ``[general]`` sections.
    """
    import toml as _toml

    base_path = Path(__file__).resolve().parent.parent.parent / "configs" / "base.toml"
    # Read raw to preserve all sections (datasets, general, etc.)
    base = _toml.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    valid_keys = {tk for _, tk, _ in _MODEL_PATH_KEYS}
    for item in body:
        if item.key in valid_keys:
            base[item.key] = item.value
    base_path.write_text(_toml.dumps(base), encoding="utf-8")
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

    # GPU — torch.cuda for memory, nvidia-smi for util & temp
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            stats["gpu_name"] = torch.cuda.get_device_name(idx)
            stats["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated(idx) / (1024**3), 1)
            stats["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 1)
            stats["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved(idx) / (1024**3), 1)

            # nvidia-smi for utilization & temperature
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        stats["gpu_util_percent"] = int(parts[0].strip())
                        stats["gpu_temp_c"] = int(parts[1].strip())
            except Exception:
                pass
    except Exception:
        pass

    return stats
