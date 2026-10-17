"""Counterfactual syllables (cjk_ko, user 10-12): a trained syllable with one
jamo left out, the others where they sit in it.

For a trained s = (cho, jung, jong):

- **less the initial** — 언 → ㅓ ㄴ as in 언; row Δ(s) − C = b + V + F
- **less the vowel** — 언 → ㅇ ㄴ as in 언; row Δ(s) − V = b + C[cho, cls] + F
  (cls is the left-out vowel's: the initial keeps its place)
- **less the final** — Δ(s) − F is the CV syllable's own row (어), so the CV
  syllable itself is the item: a row of the run unless it is held out (H)

A form has no code point, so each one rides a host: a syllable outside KS X
1001, which no caption and no read spells. The host's ext row carries the
form's composition during training (``Jamo`` reads the code off ``Run.cf``);
``Jamo.merge_all`` writes every syllable's own composition over it at the end.

Drawn from the font's positional jamo glyphs: HarfBuzz shapes the syllable's
conjoining jamo with the precomposed syllables hidden from its cmap (else the
Hangul shaper composes them into one glyph), so ljmo / vjmo / tjmo pick each
jamo's form for this syllable; the kept glyphs are rasterised by FreeType at
the shaped pen positions. Noto Serif CJK is the one face in the pool with the
positional glyphs (``FACES``).
"""

from __future__ import annotations

import os

from .jamo import ALL, JUNG_CLS, KSX1001, N_CHO, N_JUNG, compose, decompose

DROPS = ("cho", "jung")
# the faces with Hangul positional jamo (ljmo / vjmo / tjmo) in the pool
FACES = "NotoSerifCJK"


def plan(syllables, held=()) -> tuple[list, dict]:
    """``(extra, forms)`` for the trained ``syllables``: ``extra`` the CV
    syllables a final's removal gives that are neither trained nor ``held``;
    ``forms`` {host: {"code": (cho, cls, jung, jong), "drop": "cho" | "jung",
    "src": [syllables]}} — one per distinct composition, hosts in code-point
    order outside KS X 1001 and outside ``syllables``."""
    syllables = list(dict.fromkeys(syllables))
    have, held = set(syllables), set(held)
    by: dict = {}
    extra = []
    for s in syllables:
        cho, jung, jong = decompose(s)
        cls = JUNG_CLS[jung]
        by.setdefault(("cho", (N_CHO, 0, jung, jong)), []).append(s)
        by.setdefault(("jung", (cho, cls, N_JUNG, jong)), []).append(s)
        if jong:
            cv = compose(cho, jung, 0)
            if cv not in have and cv not in held and cv not in extra:
                extra.append(cv)
    ksx = set(KSX1001)
    hosts = (h for h in ALL if h not in ksx and h not in have)
    forms = {
        next(hosts): {"code": code, "drop": drop, "src": src}
        for (drop, code), src in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    }
    return extra, forms


_FACES: dict = {}


def _face(path: str):
    """(HarfBuzz font without the precomposed syllables, FreeType face) for
    ``path`` (index 0), one per process."""
    key = (os.getpid(), path)
    if key not in _FACES:
        import freetype
        import uharfbuzz as hb

        face = hb.Face(hb.Blob.from_file_path(path), 0)
        parent = hb.Font(face)
        font = hb.Font(parent)
        funcs = hb.FontFuncs()
        funcs.set_nominal_glyph_func(
            lambda _f, cp, _d: (
                0 if 0xAC00 <= cp <= 0xD7A3 else (parent.get_nominal_glyph(cp) or 0)
            )
        )
        font.funcs = funcs
        _FACES[key] = (font, face.upem, freetype.Face(path, 0))
    return _FACES[key]


def glyphs(path: str, s: str) -> list:
    """``s``'s positional jamo glyphs in ``path``: [(gid, x, y)] in font units,
    initial, vowel, final order."""
    import uharfbuzz as hb

    font, _upem, _ft = _face(path)
    cho, jung, jong = decompose(s)
    text = chr(0x1100 + cho) + chr(0x1161 + jung) + (chr(0x11A7 + jong) if jong else "")
    b = hb.Buffer()
    b.add_str(text)
    b.guess_segment_properties()
    hb.shape(font, b, {})
    out, pen = [], 0
    for i, p in zip(b.glyph_infos, b.glyph_positions):
        out.append((i.codepoint, pen + p.x_offset, p.y_offset))
        pen += p.x_advance
    assert len(out) == len(text), (s, path, out)
    return out


def mask(path: str, s: str, drop: str, px: int):
    """``s`` at font ``px`` with its ``drop`` jamo left out, as an ``L`` image
    cropped to the ink (``None``: nothing left to draw)."""
    import numpy as np
    from PIL import Image

    import freetype

    _font, upem, ft = _face(path)
    ft.set_pixel_sizes(0, int(px))
    gs = glyphs(path, s)
    keep = [g for k, g in enumerate(gs) if k != DROPS.index(drop)]
    W = H = 3 * int(px)
    can = np.zeros((H, W), np.uint8)
    ox, oy = int(px), 2 * int(px)
    for gid, x, y in keep:
        ft.load_glyph(gid, freetype.FT_LOAD_RENDER)
        bm = ft.glyph.bitmap
        if not bm.rows or not bm.width:
            continue
        a = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, : bm.width]
        x0 = ox + round(x * px / upem) + ft.glyph.bitmap_left
        y0 = oy - round(y * px / upem) - ft.glyph.bitmap_top
        sl = can[y0 : y0 + bm.rows, x0 : x0 + bm.width]
        assert sl.shape == a.shape, (s, drop, px, sl.shape, a.shape)
        np.maximum(sl, a, out=sl)
    im = Image.fromarray(can, "L")
    box = im.getbbox()
    return im.crop(box) if box else None
