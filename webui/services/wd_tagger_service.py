"""WD ONNX Tagger service — lightweight image auto-tagging.

Uses ``onnxruntime-gpu`` with SmilingWolf's WD tagger models
(wd-eva02-large-tagger-v3 / wd-swinv2-tagger-v3) from HuggingFace.
Models are auto-downloaded on first use via ``hf_hub_download``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image

from webui.services.config_service import ROOT

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

MODEL_REPO_MAP: dict[str, str] = {
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
}
DEFAULT_MODEL = "wd-eva02-large-tagger-v3"
DEFAULT_THRESHOLD = 0.35
MODEL_DIR = ROOT / "models" / "wd-tagger"
SETTINGS_FILE = ROOT / "configs" / "webui_settings.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ── Singleton state ─────────────────────────────────────────────

_session = None
_tags: list[dict] | None = None
_current_model: str | None = None


# ── Settings ────────────────────────────────────────────────────

def get_settings() -> dict:
    """Read tagger settings from webui_settings.json."""
    defaults = {"model_name": DEFAULT_MODEL, "threshold": DEFAULT_THRESHOLD}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        saved = data.get("wd_tagger", {})
        defaults.update(saved)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults


def save_settings(model_name: str, threshold: float) -> dict:
    """Persist tagger settings to webui_settings.json."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data["wd_tagger"] = {"model_name": model_name, "threshold": threshold}
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data["wd_tagger"]


def get_status() -> dict:
    """Return model status: downloaded, loaded, model_dir."""
    settings = get_settings()
    model_name = settings["model_name"]
    model_dir = MODEL_DIR / model_name
    downloaded = (model_dir / "model.onnx").exists() and (
        model_dir / "selected_tags.csv"
    ).exists()

    return {
        "model_name": model_name,
        "downloaded": downloaded,
        "loaded": _current_model == model_name and _session is not None,
        "model_dir": str(model_dir),
        "available_models": list(MODEL_REPO_MAP.keys()),
    }


# ── Model management ───────────────────────────────────────────

def _download_model(model_name: str) -> Path:
    """Download model.onnx + selected_tags.csv from HuggingFace.

    Skips files that already exist locally.
    Returns the model directory path.
    """
    import os

    from huggingface_hub import hf_hub_download

    # Use HF mirror if HF_ENDPOINT is not set and default HF is unreachable
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    repo_id = MODEL_REPO_MAP.get(model_name)
    if not repo_id:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REPO_MAP)}")

    local_dir = MODEL_DIR / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("model.onnx", "selected_tags.csv"):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )

    return local_dir


def _load_model(model_name: str) -> None:
    """Lazy-load ONNX session + tags CSV. Switches model if different."""
    global _session, _tags, _current_model

    if _current_model == model_name and _session is not None:
        return

    import onnxruntime as ort

    model_dir = MODEL_DIR / model_name
    onnx_path = model_dir / "model.onnx"
    tags_path = model_dir / "selected_tags.csv"

    if not onnx_path.exists() or not tags_path.exists():
        logger.info("Model files not found, downloading %s...", model_name)
        _download_model(model_name)

    # Load tags CSV
    _tags = []
    with open(tags_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            _tags.append({"name": row[0], "category": int(row[2])})

    # Create ONNX session with GPU provider
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info("Loading WD tagger model: %s", model_name)
    _session = ort.InferenceSession(str(onnx_path), providers=providers)
    _current_model = model_name
    logger.info("WD tagger loaded. Providers: %s", _session.get_providers())


# ── Preprocessing ───────────────────────────────────────────────

def _preprocess(image: Image.Image) -> np.ndarray:
    """RGB -> square pad (white) -> resize 448x448 -> uint8 NHWC [1,448,448,3]."""
    image = image.convert("RGB")
    w, h = image.size
    max_dim = max(w, h)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    resized = padded.resize((448, 448), Image.LANCZOS)
    return np.array(resized, dtype=np.uint8)[np.newaxis, ...]


# ── Prediction ──────────────────────────────────────────────────

def predict(image_path: str | Path, threshold: float | None = None) -> dict:
    """Single image -> {rating, general_tags[], character_tags[], all_above_threshold[]}.

    Lazy-loads the model on first call.
    """
    settings = get_settings()
    model_name = settings["model_name"]
    threshold = threshold if threshold is not None else settings["threshold"]

    _load_model(model_name)
    assert _session is not None and _tags is not None

    img = Image.open(image_path)
    input_data = _preprocess(img)
    input_name = _session.get_inputs()[0].name
    outputs = _session.run(None, {input_name: input_data})
    probs = outputs[0][0]

    rating_tags: list[dict] = []
    general_tags: list[dict] = []
    character_tags: list[dict] = []
    all_kept: list[dict] = []

    for i, tag_info in enumerate(_tags):
        prob = float(probs[i])
        if prob < threshold:
            continue
        name = tag_info["name"].replace("_", " ")
        entry = {"name": name, "score": round(prob, 4)}
        all_kept.append(entry)

        if tag_info["category"] == 9:
            rating_tags.append(entry)
        elif tag_info["category"] == 4:
            character_tags.append(entry)
        else:
            general_tags.append(entry)

    # Sort by score descending
    rating_tags.sort(key=lambda t: t["score"], reverse=True)
    general_tags.sort(key=lambda t: t["score"], reverse=True)
    character_tags.sort(key=lambda t: t["score"], reverse=True)

    predicted_rating = rating_tags[0]["name"] if rating_tags else "general"

    return {
        "rating": predicted_rating,
        "general_tags": general_tags,
        "character_tags": character_tags,
        "all_above_threshold": all_kept,
    }


# ── Batch tagging ───────────────────────────────────────────────

def tag_directory(
    directory: str,
    threshold: float | None = None,
    skip_existing: bool = True,
    model_name: str | None = None,
) -> Generator[dict, None, None]:
    """Yield progress dicts while tagging all images in a directory.

    Writes .txt sidecar caption files. Preserves caption history when
    overwriting existing files.
    """
    from webui.services.image_service import (
        IMAGE_EXTS,
        _append_history,
        resolve_directory,
    )

    base = resolve_directory(directory)
    if not base:
        yield {"error": f"Directory not found: {directory}"}
        return

    images = sorted(p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    total = len(images)
    if total == 0:
        yield {"error": "No images found in directory"}
        return

    settings = get_settings()
    threshold = threshold if threshold is not None else settings["threshold"]
    model_name = model_name or settings["model_name"]

    try:
        _load_model(model_name)
    except Exception as e:
        yield {"error": f"Failed to load model: {e}"}
        return

    tagged = 0
    skipped = 0
    errors = 0

    for i, img_path in enumerate(images):
        txt_path = img_path.with_suffix(".txt")

        if skip_existing and txt_path.exists() and txt_path.stat().st_size > 0:
            skipped += 1
            yield {
                "current": i + 1,
                "total": total,
                "file": img_path.name,
                "skipped": True,
            }
            continue

        try:
            result = predict(img_path, threshold)

            # Build caption: rating, then general tags, then character tags
            parts = [result["rating"]]
            parts += [t["name"] for t in result["general_tags"]]
            parts += [t["name"] for t in result["character_tags"]]
            caption = ", ".join(parts)

            # Save with history preservation
            if txt_path.exists():
                _append_history(txt_path, txt_path.read_text(encoding="utf-8"))
            txt_path.write_text(caption, encoding="utf-8")

            tagged += 1
            yield {
                "current": i + 1,
                "total": total,
                "file": img_path.name,
                "tags": parts,
            }
        except Exception as e:
            errors += 1
            yield {
                "current": i + 1,
                "total": total,
                "file": img_path.name,
                "error": str(e),
            }

    yield {
        "done": True,
        "tagged": tagged,
        "skipped": skipped,
        "errors": errors,
        "total": total,
    }
