"""Render one contact sheet per finding so a human can judge it at a glance.

The audit's verdict rests on evidence a reviewer cannot see from ``report.json``
alone: *which* two boxes SAM drew, and what the tagger read off each one. A sheet
puts all of it on a single page — the full image with the boxes overlaid, the
crops the tagger actually saw (colour-matched to their box), the identity it read
from each, and the caption edit being proposed — so accepting or rejecting a row
is one look rather than a cross-reference.

Deliberately one file per finding rather than one grid for the run: a page per
image can be flipped through, deleted as it is cleared, and named after the
image it judges.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .multiview_audit import (
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
    MultiviewFinding,
)

SHEET_WIDTH = 1400
_MARGIN = 20
_PANEL = 620  # square box the full image is fitted into
_CROP_TILE = 340
# Four header lines at ~26px plus the accent bar and top margin. Fixed rather
# than measured because the panels below are laid out against it.
_HEADER = 156

_BG = (24, 24, 28)
_FG = (232, 232, 236)
_DIM = (150, 150, 158)
_PANEL_BG = (36, 36, 42)

# Per-verdict accent, and a per-instance palette so a crop tile's border matches
# the box it came from — the whole point of the sheet is that pairing.
_VERDICT_COLOR = {
    MULTIPLE_VIEWS: (80, 200, 120),
    EXTRA_CHARACTER: (240, 160, 60),
}
_UNSURE_COLOR = (140, 150, 170)
_BOX_COLORS = [
    (255, 90, 90),
    (90, 170, 255),
    (255, 210, 80),
    (180, 120, 255),
    (90, 230, 200),
    (255, 140, 200),
    (160, 255, 120),
    (255, 175, 100),
]

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """A real TTF if the system or matplotlib has one, else PIL's bitmap default.

    Falls back rather than raising: a sheet with an ugly font is still a usable
    sheet, and this runs on whatever box the dataset lives on.
    """
    names = list(_FONT_CANDIDATES)
    try:
        import matplotlib

        mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        names.insert(
            0, str(mpl / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
        )
    except Exception:
        pass
    if bold:
        names.insert(0, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _fit(image: Image.Image, box: int) -> Image.Image:
    scale = min(box / image.width, box / image.height, 1.0)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    body: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> int:
    """Draw one line, return the y a following line should start at."""
    draw.text(xy, body, font=font, fill=fill)
    return xy[1] + int(font.size * 1.45) if hasattr(font, "size") else xy[1] + 16


def _identity_line(finding_crop) -> str:
    """The identity the verdict used — or why it was thrown away.

    Both discard paths are spelled out rather than shown as a blank, because the
    reviewer's next question after "why wasn't this a view pair" is always which
    of the two fired: a low-score box (mask usually shredded, so whatever the
    tagger read is noise) or a crop with no legible identity in it.
    """
    read = ", ".join(v for _, v in sorted(finding_crop.raw_groups.items()))
    if not finding_crop.reliable:
        return f"low-score box — identity ignored (read: {read or 'nothing'})"
    if finding_crop.groups:
        return ", ".join(v for _, v in sorted(finding_crop.groups.items()))
    if finding_crop.raw_groups:
        rejected = ", ".join(
            f"{v} {finding_crop.group_scores.get(g, 0.0):.2f}"
            for g, v in sorted(finding_crop.raw_groups.items())
        )
        return f"(no confident identity — rejected: {rejected})"
    return "(no identity)"


def render_contact_sheet(
    image: Image.Image,
    finding: MultiviewFinding,
    crops: Sequence[Image.Image] = (),
) -> Image.Image:
    """One page: verdict, boxed original, the tagger's crops, and the caption edit."""
    accent = _VERDICT_COLOR.get(finding.verdict, _UNSURE_COLOR)
    title_font = _font(26, bold=True)
    head_font = _font(17)
    body_font = _font(15)
    small_font = _font(13)

    preview = _fit(image.convert("RGB"), _PANEL)
    scale = preview.width / image.width
    overlay = preview.copy()
    marks = ImageDraw.Draw(overlay)
    for index, crop in enumerate(finding.crops):
        color = _BOX_COLORS[index % len(_BOX_COLORS)]
        x1, y1, x2, y2 = (v * scale for v in crop.box)
        marks.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{index + 1}"
        marks.rectangle([x1, y1, x1 + 26, y1 + 26], fill=color)
        marks.text((x1 + 8, y1 + 4), label, font=head_font, fill=(20, 20, 24))

    # Crops sit in a column beside the original; the sheet grows to fit whichever
    # of the two is taller.
    crop_column_x = _MARGIN + preview.width + 24
    crop_width = SHEET_WIDTH - crop_column_x - _MARGIN
    tile = min(_CROP_TILE, crop_width - 200)
    crop_block_height = max(1, len(finding.crops)) * (tile + 18)

    caption_lines = textwrap.wrap(finding.caption, width=170) or ["(empty)"]
    proposed_lines = (
        textwrap.wrap(finding.proposed, width=170) if finding.proposed else []
    )

    header_height = _HEADER
    body_height = max(preview.height, crop_block_height)
    footer_height = 44 + 22 * (len(caption_lines) + len(proposed_lines) + 2)
    height = header_height + body_height + footer_height + _MARGIN

    sheet = Image.new("RGB", (SHEET_WIDTH, height), _BG)
    draw = ImageDraw.Draw(sheet)

    draw.rectangle([0, 0, SHEET_WIDTH, 6], fill=accent)
    y = _MARGIN + 4
    y = _text(draw, (_MARGIN, y), finding.image, title_font, _FG)
    verdict = f"{finding.verdict.upper()}  ·  {finding.confidence}"
    if finding.suggested_tag:
        verdict += f"  ·  + '{finding.suggested_tag}'"
    else:
        verdict += "  ·  no edit proposed"
    y = _text(draw, (_MARGIN, y), verdict, _font(20, bold=True), accent)
    facts = (
        f"boxes {finding.instances}   caption {finding.girls if finding.girls is not None else '-'}girl"
        f"   P(multiple views) {finding.tagger_multiple_views:.3f}"
        f"   people-head {finding.people_count}"
        f"   identity {finding.identity_agreement if finding.identity_agreement is not None else '-'}"
        f" over {finding.comparable_groups} group(s)   via {finding.source}"
    )
    y = _text(draw, (_MARGIN, y), facts, head_font, _DIM)
    _text(
        draw,
        (_MARGIN, y),
        "witnesses: " + (", ".join(finding.witnesses) or "none"),
        head_font,
        _DIM,
    )

    top = header_height
    sheet.paste(overlay, (_MARGIN, top))

    cy = top
    for index, crop_info in enumerate(finding.crops):
        color = _BOX_COLORS[index % len(_BOX_COLORS)]
        if index < len(crops):
            thumb = _fit(crops[index].convert("RGB"), tile)
            frame = Image.new("RGB", (tile, tile), _PANEL_BG)
            frame.paste(thumb, ((tile - thumb.width) // 2, (tile - thumb.height) // 2))
            sheet.paste(frame, (crop_column_x, cy))
            draw.rectangle(
                [crop_column_x, cy, crop_column_x + tile, cy + tile],
                outline=color,
                width=3,
            )
        draw.rectangle([crop_column_x, cy, crop_column_x + 26, cy + 26], fill=color)
        draw.text(
            (crop_column_x + 8, cy + 4),
            f"{index + 1}",
            font=head_font,
            fill=(20, 20, 24),
        )
        tx = crop_column_x + tile + 14
        ty = cy + 6
        ty = _text(
            draw,
            (tx, ty),
            f"{crop_info.position or 'crop'}  ({crop_info.score:.2f})",
            body_font,
            _FG,
        )
        for line in textwrap.wrap(_identity_line(crop_info), width=34):
            ty = _text(draw, (tx, ty), line, small_font, _DIM)
        if crop_info.name:
            _text(draw, (tx, ty), f"name: {crop_info.name}", small_font, _DIM)
        if crop_info.source != "subject":
            _text(draw, (tx, ty + 18), f"[{crop_info.source} box]", small_font, _DIM)
        cy += tile + 18

    fy = top + body_height + 16
    draw.line([(_MARGIN, fy), (SHEET_WIDTH - _MARGIN, fy)], fill=_PANEL_BG, width=2)
    fy += 12
    fy = _text(draw, (_MARGIN, fy), "caption", body_font, _DIM)
    for line in caption_lines:
        fy = _text(draw, (_MARGIN, fy), line, small_font, _FG)
    if proposed_lines:
        fy += 6
        fy = _text(draw, (_MARGIN, fy), "proposed", body_font, accent)
        for line in proposed_lines:
            fy = _text(draw, (_MARGIN, fy), line, small_font, _FG)
    return sheet


def sheet_path(sheets_dir: Path, finding: MultiviewFinding) -> Path:
    """Mirror the dataset layout, prefixed so a directory listing sorts by verdict.

    Reviewing goes verdict-first — clear all the confident view sheets, then argue
    about the rest — and a filename prefix is the only sort key a plain image
    viewer offers.
    """
    rel = Path(finding.image)
    stem = f"{finding.verdict.replace(' ', '-')}_{finding.confidence}_{rel.stem}.png"
    return sheets_dir / rel.parent / stem
