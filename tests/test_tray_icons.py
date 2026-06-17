"""Smoke tests for the procedural tray icons.

Verifies every state renders a non-empty RGBA image of the expected size and
that the brand mark + status dot are present (non-transparent pixels exist).
Pure image construction — no pystray / GUI needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from scripts.tray.icons import (  # noqa: E402
    DOT_DOWN,
    DOT_ERROR,
    DOT_IDLE,
    DOT_RUNNING,
    _SIZE,
    icon_for,
    write_ico,
)

STATES = ["idle", "running", "error", "down"]


@pytest.mark.parametrize("state", STATES)
def test_icon_renders_rgba_at_expected_size(state: str) -> None:
    img = icon_for(state)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"
    assert img.size == (_SIZE, _SIZE)


@pytest.mark.parametrize("state", STATES)
def test_icon_has_visible_pixels(state: str) -> None:
    """The mark + dot must produce non-transparent content (not a blank canvas)."""
    img = icon_for(state)
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    assert bbox is not None, f"{state} icon is fully transparent"
    # The icon should occupy a reasonable fraction of the canvas, not a dot.
    assert (bbox[2] - bbox[0]) > _SIZE * 0.4


def test_idle_has_green_dot() -> None:
    """The idle state's bottom-right dot region should contain the green status color."""
    img = icon_for("idle")
    # Sample the dot location (bottom-right ~0.78).
    px = img.getpixel((int(_SIZE * 0.78), int(_SIZE * 0.78)))
    # Allow for the white ring vs. dot center; check the channel balance is green-leaning.
    r, g, b, _a = px
    assert g > r and g > b, f"expected green-ish dot at idle, got {px}"


def test_down_is_dimmed_and_grey() -> None:
    """The 'down' state should be desaturated/dim relative to idle."""
    down = icon_for("down")
    idle = icon_for("idle")
    # Sum of all channel values: the dimmed version should be noticeably lower.
    down_sum = sum(sum(px) for px in down.convert("RGB").get_flattened_data())
    idle_sum = sum(sum(px) for px in idle.convert("RGB").get_flattened_data())
    assert down_sum < idle_sum, "down icon should be dimmer than idle"


def test_running_frame_flicker_changes_pixels() -> None:
    """The two flicker frames should differ (the flame core shifts)."""
    f0 = icon_for("running", frame=0)
    f1 = icon_for("running", frame=1)
    px0 = list(f0.get_flattened_data())
    px1 = list(f1.get_flattened_data())
    assert px0 != px1, "flicker frames are identical"


def test_status_dot_colors_distinct() -> None:
    """Sanity: the dot color constants are the expected distinct hues."""
    assert DOT_IDLE != DOT_RUNNING != DOT_ERROR != DOT_DOWN


def test_write_ico_produces_file(tmp_path: Path) -> None:
    out = tmp_path / "forge.ico"
    write_ico(str(out))
    assert out.exists() and out.stat().st_size > 0
    # PIL can read it back as an ICO.
    reopened = Image.open(out)
    assert reopened.format == "ICO"
