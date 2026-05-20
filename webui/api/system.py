"""System API — model group status, model paths, and environment info."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from webui.services.config_service import ROOT, read_layer

router = APIRouter()

# Model groups (mirrors gui/system_dialog.py _MODEL_GROUPS)
# The anima group reads actual paths from base.toml; others are fixed.
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
