"""Unit tests for WD tagger service — preprocessing logic (no ONNX model needed)."""

import numpy as np
from PIL import Image

from webui.services.wd_tagger_service import _preprocess, get_settings, save_settings, DEFAULT_MODEL, DEFAULT_THRESHOLD


def test_preprocess_square_image():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    result = _preprocess(img)
    assert result.shape == (1, 448, 448, 3)
    assert result.dtype == np.uint8


def test_preprocess_landscape_image():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    result = _preprocess(img)
    assert result.shape == (1, 448, 448, 3)
    # White padding should be present at top/bottom
    assert result[0, 0, 0, 0] == 255  # top-left is white padding


def test_preprocess_portrait_image():
    img = Image.new("RGB", (100, 200), (0, 0, 0))
    result = _preprocess(img)
    assert result.shape == (1, 448, 448, 3)
    # White padding should be present at left/right
    assert result[0, 0, 0, 0] == 255


def test_preprocess_preserves_dtype():
    img = Image.new("RGB", (50, 50), (200, 100, 50))
    result = _preprocess(img)
    assert result.dtype == np.uint8


def test_preprocess_rgba_converts_to_rgb():
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    result = _preprocess(img)
    assert result.shape == (1, 448, 448, 3)
    assert result.dtype == np.uint8


def test_get_settings_defaults():
    """get_settings should return defaults when no settings file exists."""
    # This test reads the actual settings file — just check structure
    s = get_settings()
    assert "model_name" in s
    assert "threshold" in s
    assert s["model_name"] in ("wd-eva02-large-tagger-v3", "wd-swinv2-tagger-v3")
    assert 0.0 < s["threshold"] < 1.0


def test_save_and_get_settings(tmp_path, monkeypatch):
    """Round-trip save/load settings."""
    settings_file = tmp_path / "webui_settings.json"
    monkeypatch.setattr(
        "webui.services.wd_tagger_service.SETTINGS_FILE", settings_file
    )

    save_settings("wd-swinv2-tagger-v3", 0.45)
    s = get_settings()
    assert s["model_name"] == "wd-swinv2-tagger-v3"
    assert s["threshold"] == 0.45
