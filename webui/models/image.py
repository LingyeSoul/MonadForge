"""Pydantic schemas for image API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ImageInfo(BaseModel):
    path: str
    filename: str
    width: int
    height: int
    caption: Optional[str] = None
    has_mask: bool = False


class CaptionVersion(BaseModel):
    timestamp: str
    content: str
