"""WD timm Tagger service — lightweight image auto-tagging.

Uses ``timm`` (PyTorch) with SmilingWolf's WD tagger models
(wd-eva02-large-tagger-v3 / wd-swinv2-tagger-v3) from HuggingFace.
Models are auto-downloaded as safetensors on first use.

This backend replaces the previous ONNX Runtime path, which required
CUDA 12 cuBLAS DLLs incompatible with this project's CUDA 13.2 stack.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Generator

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from webui.services.config_service import ROOT

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

MODEL_REPO_MAP: dict[str, str] = {
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
}
DEFAULT_MODEL = "wd-eva02-large-tagger-v3"
DEFAULT_THRESHOLD = 0.35
DEFAULT_TRIGGER_WORD = ""
MODEL_DIR = ROOT / "models" / "wd-tagger"
SETTINGS_FILE = ROOT / "configs" / "webui_settings.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Singleton state ─────────────────────────────────────────────

_model = None
_transform = None
_tags: list[dict] | None = None
_current_model: str | None = None


# ── Settings ────────────────────────────────────────────────────

def get_settings() -> dict:
    """Read tagger settings from webui_settings.json."""
    defaults = {"model_name": DEFAULT_MODEL, "threshold": DEFAULT_THRESHOLD, "trigger_word": DEFAULT_TRIGGER_WORD}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        saved = data.get("wd_tagger", {})
        defaults.update(saved)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults


def save_settings(model_name: str, threshold: float, trigger_word: str = "") -> dict:
    """Persist tagger settings to webui_settings.json."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    data["wd_tagger"] = {"model_name": model_name, "threshold": threshold, "trigger_word": trigger_word}
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data["wd_tagger"]


def get_status() -> dict:
    """Return model status: downloaded, loaded, model_dir."""
    settings = get_settings()
    model_name = settings["model_name"]
    model_dir = MODEL_DIR / model_name
    downloaded = (model_dir / "model.safetensors").exists() and (
        model_dir / "selected_tags.csv"
    ).exists()

    return {
        "model_name": model_name,
        "downloaded": downloaded,
        "loaded": _current_model == model_name and _model is not None,
        "model_dir": str(model_dir),
        "available_models": list(MODEL_REPO_MAP.keys()),
    }


# ── Model management ───────────────────────────────────────────

def _ensure_model_files(model_name: str) -> Path:
    """Download model files from HuggingFace mirror if not present locally.

    Downloads config.json, model.safetensors, and selected_tags.csv.
    Returns the model directory path.
    """
    import os
    import urllib.request

    base_url = f"https://hf-mirror.com/{MODEL_REPO_MAP[model_name]}/resolve/main"
    local_dir = MODEL_DIR / model_name
    local_dir.mkdir(parents=True, exist_ok=True)

    files = ("config.json", "model.safetensors", "selected_tags.csv")
    for filename in files:
        dest = local_dir / filename
        if dest.exists():
            continue
        url = f"{base_url}/{filename}"
        logger.info("Downloading %s...", filename)
        req = urllib.request.Request(url, headers={"User-Agent": "anima"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())

    return local_dir


def _load_model(model_name: str) -> None:
    """Lazy-load timm model from local safetensors + tags CSV."""
    global _model, _transform, _tags, _current_model

    if _current_model == model_name and _model is not None:
        return

    import timm
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_data_config

    model_dir = _ensure_model_files(model_name)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    tags_path = model_dir / "selected_tags.csv"

    # Load config to determine architecture
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    arch = config["architecture"]
    model_args = config.get("model_args", {})

    # Create timm model (no pretrained download)
    logger.info("Creating WD tagger: %s (%s) on %s", model_name, arch, DEVICE)
    _model = timm.create_model(arch, pretrained=False, num_classes=config["num_classes"], **model_args)

    # Load local safetensors weights
    state_dict = load_file(str(weights_path))
    _model.load_state_dict(state_dict)
    _model.eval()
    _model.to(DEVICE)

    _transform = create_transform(**resolve_data_config(config["pretrained_cfg"], model=_model))

    # Load tags CSV
    _tags = []
    with open(tags_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            _tags.append({"name": row[1], "category": int(row[2])})

    _current_model = model_name
    logger.info("WD tagger loaded on %s", DEVICE)


# ── Preprocessing ───────────────────────────────────────────────

def _pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def _preprocess(image: Image.Image) -> Tensor:
    """RGB -> square pad -> timm transform -> NCHW tensor [1,C,H,W] RGB->BGR."""
    image = _pil_ensure_rgb(image.convert("RGB"))
    w, h = image.size
    max_dim = max(w, h)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    tensor: Tensor = _transform(padded).unsqueeze(0)  # [1, 3, H, W]
    # RGB -> BGR (matches SmilingWolf's training convention)
    tensor = tensor[:, [2, 1, 0]]
    return tensor


# ── Prediction ──────────────────────────────────────────────────

def predict(image_path: str | Path, threshold: float | None = None) -> dict:
    """Single image -> {rating, general_tags[], character_tags[], all_above_threshold[]}.

    Lazy-loads the model on first call.
    """
    settings = get_settings()
    model_name = settings["model_name"]
    threshold = threshold if threshold is not None else settings["threshold"]

    _load_model(model_name)
    assert _model is not None and _tags is not None

    img = Image.open(image_path)
    inputs = _preprocess(img).to(DEVICE)

    with torch.inference_mode():
        outputs = _model(inputs)
        probs: Tensor = F.sigmoid(outputs).squeeze(0).cpu()

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
    trigger_word: str | None = None,
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
    if trigger_word is None:
        trigger_word = settings.get("trigger_word", DEFAULT_TRIGGER_WORD)

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

            # Build caption: trigger word, rating, general tags, character tags
            parts = []
            if trigger_word:
                parts.append(trigger_word)
            parts.append(result["rating"])
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
