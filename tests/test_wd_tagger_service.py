"""Unit tests for WD tagger service — preprocessing logic (no model weights needed)."""

import torch
from PIL import Image

from webui.services.wd_tagger_service import get_settings, save_settings


def _preprocess_stub(image: Image.Image):
    """Minimal stub: square-pad + resize, returns NCHW uint8 tensor (no timm)."""
    from torchvision import transforms

    image = image.convert("RGB")
    w, h = image.size
    max_dim = max(w, h)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    resized = padded.resize((448, 448), Image.LANCZOS)
    t = transforms.ToTensor()(resized).unsqueeze(0)  # [1, 3, 448, 448]
    return t


def test_preprocess_square_image():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    result = _preprocess_stub(img)
    assert result.shape == (1, 3, 448, 448)


def test_preprocess_landscape_image():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    result = _preprocess_stub(img)
    assert result.shape == (1, 3, 448, 448)
    # Top-left pixel should be white padding (1.0 in [0,1] range)
    assert result[0, 0, 0, 0].item() > 0.99


def test_preprocess_portrait_image():
    img = Image.new("RGB", (100, 200), (0, 0, 0))
    result = _preprocess_stub(img)
    assert result.shape == (1, 3, 448, 448)
    assert result[0, 0, 0, 0].item() > 0.99


def test_preprocess_returns_float_tensor():
    img = Image.new("RGB", (50, 50), (200, 100, 50))
    result = _preprocess_stub(img)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


def test_preprocess_rgba_converts_to_rgb():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    result = _preprocess_stub(img)
    assert result.shape == (1, 3, 448, 448)


def test_get_settings_defaults():
    s = get_settings()
    assert "model_name" in s
    assert "threshold" in s
    assert "trigger_word" in s
    assert s["model_name"] in ("wd-eva02-large-tagger-v3", "wd-swinv2-tagger-v3")
    assert 0.0 < s["threshold"] < 1.0
    assert s["trigger_word"] == ""


def test_save_and_get_settings(tmp_path, monkeypatch):
    settings_file = tmp_path / "webui_settings.json"
    monkeypatch.setattr(
        "webui.services.wd_tagger_service.SETTINGS_FILE", settings_file
    )

    save_settings("wd-swinv2-tagger-v3", 0.45, "ohwx")
    s = get_settings()
    assert s["model_name"] == "wd-swinv2-tagger-v3"
    assert s["threshold"] == 0.45
    assert s["trigger_word"] == "ohwx"


def test_save_settings_trigger_word_default_empty(tmp_path, monkeypatch):
    settings_file = tmp_path / "webui_settings.json"
    monkeypatch.setattr(
        "webui.services.wd_tagger_service.SETTINGS_FILE", settings_file
    )

    save_settings("wd-eva02-large-tagger-v3", 0.35)
    s = get_settings()
    assert s["trigger_word"] == ""
