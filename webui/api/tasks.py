"""Task management API endpoints."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from webui.services.task_service import task_service

router = APIRouter()


class TaskStartRequest(BaseModel):
    command: str
    args: list[str] = []
    env: dict[str, str] = {}


class TaskCommandListResponse(BaseModel):
    commands: dict[str, str]


# Canonical command list with descriptions
_COMMAND_DESCRIPTIONS = {
    "lora": "LoRA family training",
    "lora-gui": "Train from gui-methods variant",
    "test": "Inference with latest LoRA",
    "test-hydra": "Inference with HydraLoRA",
    "test-merge": "Inference with merged DiT",
    "test-dcw": "Inference + DCW correction",
    "test-dcw-v4": "Inference + DCW v4",
    "test-smc-cfg": "Inference + SMC-CFG",
    "test-spectrum-dcw": "Spectrum + DCW",
    "test-dcw-v4-spectrum": "DCW v4 + Spectrum",
    "preprocess": "Full preprocess (resize + VAE + TE)",
    "preprocess-resize": "Resize images",
    "preprocess-vae": "Cache VAE latents",
    "preprocess-te": "Cache text embeddings",
    "preprocess-pooled": "Cache pooled text embeddings",
    "preprocess-pe": "Cache PE features",
    "mask": "Run SAM + MIT masking",
    "mask-clean": "Remove masks",
    "merge": "Merge LoRA into DiT",
    "dcw": "Calibrate DCW v4",
    "dcw-train": "Train DCW fusion head",
    "distill-prep": "Stage modulation guidance artifacts",
    "distill-mod": "Distill pooled_text_proj MLP",
    "download-models": "Download all models",
    "download-anima": "Download Anima model",
    "download-sam3": "Download SAM3 model",
    "download-mit": "Download MIT model",
    "download-pe": "Download PE encoder",
    "download-pe-spatial": "Download PE-Spatial encoder",
    "test-unit": "Run unit tests",
    "print-config": "Dump merged config",
    "exp-postfix": "Postfix tuning",
    "exp-turbo": "DMD2 turbo distillation",
    "exp-chimera": "ChimeraHydra training",
    "exp-ip-adapter": "IP-Adapter training",
    "exp-ip-adapter-preprocess": "IP-Adapter preprocess",
    "exp-easycontrol": "EasyControl training",
    "exp-easycontrol-preprocess": "EasyControl preprocess",
    "exp-test-postfix": "Postfix inference",
    "exp-test-turbo": "Turbo inference",
    "exp-test-ip": "IP-Adapter inference",
    "exp-test-easycontrol": "EasyControl inference",
    "exp-test-directedit": "DirectEdit inference",
    "exp-test-directedit-dry": "DirectEdit dry run",
}


@router.get("", response_model=list[dict])
def list_tasks():
    """List all active tasks."""
    return task_service.list_tasks()


@router.get("/commands", response_model=TaskCommandListResponse)
def list_commands():
    """Return available task commands with descriptions."""
    return TaskCommandListResponse(commands=_COMMAND_DESCRIPTIONS)


@router.get("/{task_id}")
def get_task(task_id: str):
    """Get task status."""
    info = task_service.get_task_info(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return info


@router.get("/{task_id}/output")
def get_task_output(task_id: str):
    """Get all accumulated output lines for a task."""
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"lines": task.lines, "state": task.state.value, "exit_code": task.exit_code}


@router.get("/{task_id}/metrics")
def get_task_metrics(task_id: str):
    """Get parsed training metrics for a task."""
    metrics = task_service.get_task_metrics(task_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return metrics


@router.post("")
async def start_task(body: TaskStartRequest):
    """Start a new task."""
    if body.command not in _COMMAND_DESCRIPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown command: {body.command}")
    task = await task_service.start_task(body.command, body.args, body.env or None)
    return task.info()


@router.delete("/{task_id}")
async def stop_task(task_id: str):
    """Cancel a running task."""
    ok = await task_service.cancel_task(task_id)
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"Task {task_id} not found or not running"
        )
    return {"status": "cancelled", "task_id": task_id}
