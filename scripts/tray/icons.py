"""Procedural tray icons for the MonadForge daemon tray.

Renders the forge brand mark (hexagon + flame + anvil, matching
``webui/frontend/public/logo.svg``'s palette) with Pillow, then overlays a
status indicator dot so a single glance tells the daemon state:

    idle     → green dot
    running  → amber dot (+ animated flame flicker, toggled by ``frame``)
    error    → red dot
    down     → grey, dimmed mark (daemon not reachable)

No SVG parser is pulled in: everything is Pillow primitives. The base mark is
drawn once and cached; per-state variants clone it and composite the dot.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw

# Brand palette (mirrors logo.svg).
HEX_FILL = (26, 18, 16)  # #1A1210 — dark hex background
FLAME_OUTER = (245, 166, 35)  # #F5A623
FLAME_INNER = (255, 215, 0)  # #FFD700
FLAME_DEEP = (179, 71, 0)  # #B34700
ANVIL_TOP = (184, 115, 51)  # #B87333
ANVIL_BOTTOM = (107, 66, 38)  # #6B4226
ANVIL_EDGE = (212, 148, 58)  # #D4943A
HALO = (245, 166, 35, 31)  # #F5A623 @ 12% — flame glow

# Status dot colors.
DOT_IDLE = (76, 200, 120)  # green
DOT_RUNNING = (245, 166, 35)  # amber
DOT_ERROR = (235, 80, 80)  # red
DOT_DOWN = (130, 130, 130)  # grey

_SIZE = 64


def _hexagon_points(size: int) -> list[tuple[float, float]]:
    """Pointy-top hexagon filling the canvas, matching the SVG's 50,2 … 7,25 … shape."""
    s = size
    return [
        (s * 0.50, s * 0.02),
        (s * 0.93, s * 0.25),
        (s * 0.93, s * 0.75),
        (s * 0.50, s * 0.98),
        (s * 0.07, s * 0.75),
        (s * 0.07, s * 0.25),
    ]


def _draw_flame(draw: ImageDraw.ImageDraw, s: int, *, flicker: bool = False) -> None:
    """Draw the forge flame. ``flicker`` shifts the core for a running animation frame."""
    # Halo behind the flame.
    halo_box = (
        int(s * 0.28),
        int(s * 0.36),
        int(s * 0.72),
        int(s * 0.76),
    )
    draw.ellipse(halo_box, fill=HALO)
    # Outer flame (teardrop).
    outer = [
        (s * 0.50, s * 0.28),
        (s * 0.32, s * 0.60),
        (s * 0.40, s * 0.74),
        (s * 0.60, s * 0.74),
        (s * 0.68, s * 0.60),
    ]
    draw.polygon(outer, fill=FLAME_OUTER)
    # Inner core.
    dy = s * 0.02 if flicker else 0.0
    inner = [
        (s * 0.50, s * 0.38 + dy),
        (s * 0.40, s * 0.58 + dy),
        (s * 0.44, s * 0.67 + dy),
        (s * 0.56, s * 0.67 + dy),
        (s * 0.60, s * 0.58 + dy),
    ]
    draw.polygon(inner, fill=FLAME_INNER)
    # Tip highlight.
    draw.ellipse(
        (s * 0.465, s * 0.395 + dy, s * 0.535, s * 0.465 + dy),
        fill=(255, 251, 230),
    )


def _draw_anvil(draw: ImageDraw.ImageDraw, s: int) -> None:
    """Draw the copper anvil base under the flame."""
    # Tapered base block.
    base = [
        (s * 0.26, s * 0.72),
        (s * 0.74, s * 0.72),
        (s * 0.78, s * 0.80),
        (s * 0.73, s * 0.85),
        (s * 0.27, s * 0.85),
        (s * 0.22, s * 0.80),
    ]
    draw.polygon(base, fill=ANVIL_TOP, outline=ANVIL_EDGE)


def _draw_base_mark(size: int = _SIZE, *, flicker: bool = False) -> Image.Image:
    """The brand mark without any status dot — the cacheable base layer.

    ``flicker`` shifts the flame core up slightly for the running animation.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon(_hexagon_points(size), fill=HEX_FILL)
    _draw_flame(draw, size, flicker=flicker)
    _draw_anvil(draw, size)
    return img


_BASE_CACHE: Optional[Image.Image] = None


def _base(size: int = _SIZE) -> Image.Image:
    global _BASE_CACHE
    if _BASE_CACHE is None or _BASE_CACHE.size != (size, size):
        _BASE_CACHE = _draw_base_mark(size)
    return _BASE_CACHE.copy()


def _status_dot(img: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Composite a small status dot in the bottom-right of the mark."""
    s = img.size[0]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    r = max(4, s // 9)
    cx, cy = int(s * 0.78), int(s * 0.78)
    # White ring for contrast against any background.
    draw.ellipse(
        (cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1), fill=(255, 255, 255, 230)
    )
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (255,))
    return out


def _dim(img: Image.Image) -> Image.Image:
    """Greyscale-dim a copy (used for the 'daemon down' state)."""
    g = img.convert("L").convert("RGBA")
    # Re-apply at reduced opacity to read as 'inactive'.
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(g.point(lambda v: int(v * 0.55)))
    return out


def icon_for(state: str, *, frame: int = 0) -> Image.Image:
    """Return the tray ``PIL.Image`` for a daemon state.

    ``state`` ∈ {"idle","running","error","down"}. ``frame`` (0/1) toggles the
    flame flicker for the running animation; the tray app alternates it each
    refresh so a running job reads as 'live'.
    """
    if state == "down":
        return _status_dot(_dim(_base()), DOT_DOWN)

    # Running: redraw the base with a flicker frame for a subtle 'alive' pulse.
    if state == "running":
        base = _draw_base_mark(flicker=bool(frame))
        return _status_dot(base, DOT_RUNNING)
    if state == "error":
        return _status_dot(_base(), DOT_ERROR)
    # idle (default)
    return _status_dot(_base(), DOT_IDLE)


def write_ico(path: str, size: int = 64) -> None:
    """Write the idle brand mark to ``path`` as a Windows .ico (for the launcher)."""
    icon_for("idle").resize((size, size), Image.LANCZOS).save(path, format="ICO")
