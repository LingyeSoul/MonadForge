"""Which faces keep a glyph's strokes apart at a font px (cjk_ko, 10-12).

A heavy face at a small px fuses a dense syllable's strokes (BlackHanSans 뤽
at 33 px: ㄹ's gaps ~1 px, closed by pose and resampling). The gaps are
measured once per (face, glyph) at ``REF`` px — every background run between
two ink runs along each row and column, as a share of the font px — and
scale with px. A face is kept for a glyph at ``px`` when the median gap is
≥ ``median`` px and the 10th percentile ≥ ``p10`` px; under ``from_px`` every
face is (the 16 px tiers draw small glyphs on purpose).
"""

from __future__ import annotations

from functools import lru_cache

REF = 256


@lru_cache(maxsize=None)
def gaps(font_path: str, glyph: str) -> tuple:
    """``(p10, median)`` of the glyph's stroke gaps, as shares of the font px;
    ``(1, 1)`` with no gap (a glyph with no counter cannot fuse)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, REF, index=0)
    bb = font.getbbox(glyph)
    im = Image.new("L", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), 0)
    ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), glyph, font=font, fill=255)
    a = np.asarray(im) > 127
    runs = []
    for m in (a, a.T):
        for r in m:
            ink = np.flatnonzero(r)
            if len(ink) < 2:
                continue
            d = np.diff(r[ink[0] : ink[-1] + 1].astype(np.int8))
            runs.append(np.flatnonzero(d == 1) - np.flatnonzero(d == -1))
    g = np.concatenate(runs) / REF if runs else np.zeros(0)
    if not len(g):
        return (1.0, 1.0)
    return (float(np.percentile(g, 10)), float(np.median(g)))


def keeps(font_path: str, text: str, px: float, rule: dict) -> bool:
    """Every glyph of ``text`` keeps its strokes apart in ``font_path`` at
    ``px`` under ``rule`` (``{median, p10, from_px}``)."""
    if px < rule["from_px"]:
        return True
    for g in text:
        if g.isspace():
            continue
        p10, med = gaps(font_path, g)
        if med * px < rule["median"] or p10 * px < rule["p10"]:
            return False
    return True


def pick(fonts: list, text: str, px: float, rule: dict) -> list:
    """The faces of ``fonts`` that keep ``text`` legible at ``px``; none, the
    face with the widest median gap (a glyph no face draws apart still gets
    drawn, at its most open)."""
    ok = [f for f in fonts if keeps(f, text, px, rule)]
    if ok:
        return ok
    return [max(fonts, key=lambda f: min(gaps(f, g)[1] for g in text if not g.isspace()))]
