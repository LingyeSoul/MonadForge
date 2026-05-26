"""WD Tagger API — auto-tag images with timm (PyTorch) models."""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from webui.services import wd_tagger_service as svc

router = APIRouter()


# ── Request / response models ───────────────────────────────────


class TagRequest(BaseModel):
    directory: str
    threshold: float | None = None
    skip_existing: bool = True
    model_name: str | None = None
    trigger_word: str | None = None


class TaggerSettings(BaseModel):
    model_name: str = svc.DEFAULT_MODEL
    threshold: float = svc.DEFAULT_THRESHOLD
    trigger_word: str = svc.DEFAULT_TRIGGER_WORD


# ── Endpoints ───────────────────────────────────────────────────


@router.get("/status")
def get_status() -> dict:
    """Return tagger model status."""
    return svc.get_status()


@router.post("/tag")
def tag_directory(body: TagRequest):
    """Tag all images in a directory, streaming progress via SSE."""

    def event_stream():
        for event in svc.tag_directory(
            body.directory,
            body.threshold,
            body.skip_existing,
            body.model_name,
            body.trigger_word,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/settings", response_model=TaggerSettings)
def get_settings() -> TaggerSettings:
    """Return current tagger settings."""
    s = svc.get_settings()
    return TaggerSettings(model_name=s["model_name"], threshold=s["threshold"], trigger_word=s.get("trigger_word", ""))


@router.put("/settings", response_model=TaggerSettings)
def save_settings(body: TaggerSettings) -> TaggerSettings:
    """Persist tagger settings."""
    svc.save_settings(body.model_name, body.threshold, body.trigger_word)
    return body
