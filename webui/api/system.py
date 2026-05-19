"""System API — model group status and environment info."""

from __future__ import annotations


from fastapi import APIRouter

from webui.services.config_service import ROOT

router = APIRouter()

# Model groups (mirrors gui/system_dialog.py _MODEL_GROUPS)
_MODEL_GROUPS: list[dict] = [
    {
        "id": "anima",
        "files": [
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "models/vae/qwen_image_vae.safetensors",
        ],
    },
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
    return {"groups": [_check_group(g) for g in _MODEL_GROUPS]}
