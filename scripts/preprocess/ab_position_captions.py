"""A/B two `caption-position` configurations side by side, as contact sheets.

Proposes each image **twice** off one detection + tagging pass — the models are
the expensive part and both sides see identical crops, so the only thing that
can differ is the clause rule under test. Default test is the `framing` bind
(`--no_framing` is the A side); `--a_flags` / `--b_flags` take any
`position_captions.py` flag, so the same tool serves the next rule too.

One page per image where the two sides disagree, plus an `index.html` that
stacks them in the order most-changed-first. Images where both sides produce the
same caption are counted and skipped — a sheet nobody needs to look at is worse
than no sheet, because it hides the ones that matter.

Always proposes from the hand-written master under ``--src``, never the derived
caption: after one ``--apply`` the derived side already carries clauses and
``is_candidate`` would reject the whole corpus as ``already-has-clauses``.

Read-only: writes only under `--out`, never a caption.

    make daemon-run ARGS="scripts/preprocess/ab_position_captions.py \\
        --path_pattern 'ama_mitsuki/*'"
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from library.captioning.position_clauses import parse_caption  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.preprocess._dataset import walk_images  # noqa: E402

# Only the drawing helpers are borrowed — the palette below is this sheet's own,
# since an A/B page reads by A-vs-B contrast rather than by verdict.
from library.preprocess.multiview_sheet import _fit, _font, _text  # noqa: E402
from library.preprocess.position_captions import (  # noqa: E402
    PositionCaptionOptions,
    is_candidate,
    load_clause_vocabulary,
    propose_for_image,
)

SHEET_WIDTH = 1400
_MARGIN = 20
_PANEL = 560
_BG = (24, 24, 28)
_FG = (232, 232, 236)
_DIM = (150, 150, 158)
_RULE = (52, 52, 60)
_A = (150, 150, 158)  # the incumbent reads as neutral…
_B = (80, 200, 120)  # …the challenger as the thing to look at
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="image_dataset")
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument("--path_pattern", dest="path_pattern", default="*")
    p.add_argument("--out", default="post_image_dataset/captions/position_ab")
    # GOTCHA: pass these as `--a_flags=--foo`, not `--a_flags --foo` — argparse
    # reads a `-`-leading value as the next option and dies on "expected one
    # argument". An empty side (= shipped defaults) needs `--a_flags=` likewise.
    p.add_argument(
        "--a_flags",
        default="--no_framing",
        help="position_captions.py flags for the A side (the incumbent). Must "
        "be written --a_flags=--foo (see GOTCHA in the source); empty = shipped",
    )
    p.add_argument(
        "--b_flags",
        default="",
        help="…and for the B side (the change), same --b_flags=--foo form",
    )
    p.add_argument("--a_label", default=None)
    p.add_argument("--b_label", default=None)
    p.add_argument("--limit", type=int, default=0, help="0 = no cap")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_options(flags: str) -> tuple[PositionCaptionOptions, argparse.Namespace]:
    """Reuse the real CLI so an A/B can never drift from what ships."""
    from scripts.preprocess.position_captions import build_options_from_args
    from scripts.preprocess.position_captions import parse_args as pc_parse_args

    argv = sys.argv
    sys.argv = [argv[0], *flags.split()]
    try:
        args = pc_parse_args()
    finally:
        sys.argv = argv
    return build_options_from_args(args), args


def clause_map(caption: str | None) -> dict[str, tuple[str, ...]]:
    if not caption:
        return {}
    return {c.position: c.tags for c in parse_caption(caption).clauses}


def render(
    image: Image.Image,
    rel: str,
    a_prop,
    b_prop,
    labels: tuple[str, str],
    original: str,
) -> Image.Image:
    """One page: boxed original on the left, A-vs-B clause columns on the right."""
    title_font = _font(24, bold=True)
    head_font = _font(17, bold=True)
    body_font = _font(15)
    small_font = _font(13)

    shown = b_prop if b_prop.instances else a_prop
    preview = _fit(image.convert("RGB"), _PANEL)
    scale = preview.width / image.width
    marks = ImageDraw.Draw(preview)
    for index, inst in enumerate(shown.instances):
        color = _BOX_COLORS[index % len(_BOX_COLORS)]
        marks.rectangle([v * scale for v in inst.box], outline=color, width=4)
        x1, y1 = inst.box[0] * scale, inst.box[1] * scale
        marks.rectangle([x1, y1, x1 + 26, y1 + 26], fill=color)
        marks.text((x1 + 8, y1 + 4), f"{index + 1}", font=head_font, fill=(20, 20, 24))

    a_clauses, b_clauses = clause_map(a_prop.proposed), clause_map(b_prop.proposed)
    positions = list(dict.fromkeys([*a_clauses, *b_clauses]))
    col_x = _MARGIN + preview.width + 24
    col_w = SHEET_WIDTH - col_x - _MARGIN

    original_lines = textwrap.wrap(original, width=168) or ["(empty)"]
    height = max(
        _PANEL + 190,
        150 + 108 * max(1, len(positions)) + 24 * len(original_lines),
    )
    sheet = Image.new("RGB", (SHEET_WIDTH, height), _BG)
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, SHEET_WIDTH, 6], fill=_B)

    y = _text(draw, (_MARGIN, _MARGIN + 4), rel, title_font, _FG)
    _text(
        draw,
        (_MARGIN, y),
        f"A {labels[0]}: {a_prop.status}    B {labels[1]}: {b_prop.status}",
        body_font,
        _DIM,
    )

    top = 108
    sheet.paste(preview, (_MARGIN, top))

    cy = top
    for index, position in enumerate(positions):
        color = _BOX_COLORS[index % len(_BOX_COLORS)]
        draw.rectangle([col_x, cy, col_x + 6, cy + 92], fill=color)
        ty = _text(draw, (col_x + 16, cy), position, head_font, _FG)
        a_tags, b_tags = a_clauses.get(position, ()), b_clauses.get(position, ())
        gained = [t for t in b_tags if t not in a_tags]
        for label, tags, color_ in ((labels[0], a_tags, _A), (labels[1], b_tags, _B)):
            line = ", ".join(tags) or "—"
            wrapped = textwrap.wrap(f"{label}  {line}", width=int(col_w / 7.2)) or [
                label
            ]
            for part in wrapped[:2]:
                ty = _text(draw, (col_x + 16, ty), part, small_font, color_)
        if gained:
            ty = _text(draw, (col_x + 16, ty), f"+ {', '.join(gained)}", small_font, _B)
        cy += 108

    fy = max(top + preview.height, cy) + 14
    draw.line([(_MARGIN, fy), (SHEET_WIDTH - _MARGIN, fy)], fill=_RULE, width=2)
    fy = _text(draw, (_MARGIN, fy + 10), "caption (master)", body_font, _DIM)
    for line in original_lines:
        fy = _text(draw, (_MARGIN, fy), line, small_font, _FG)
    return sheet


def write_index(out_dir: Path, rows: list[dict], labels: tuple[str, str]) -> Path:
    """A scrollable page beats a directory of PNGs for a side-by-side read."""
    parts = [
        "<meta charset='utf-8'><title>position captions A/B</title>",
        "<style>body{background:#18181c;color:#e8e8ec;font:15px/1.5 system-ui,sans-serif;"
        "margin:0;padding:24px}h1{font-size:20px}img{width:100%;max-width:1400px;"
        "display:block;margin:8px 0 28px;border-radius:6px}"
        ".d{color:#8a8a92;font-size:13px}</style>",
        f"<h1>position captions A/B — A: <code>{html.escape(labels[0])}</code> "
        f"vs B: <code>{html.escape(labels[1])}</code></h1>",
        f"<p class=d>{len(rows)} image(s) where the two sides differ, "
        "most-changed first.</p>",
    ]
    for row in rows:
        parts.append(
            f"<div><b>{html.escape(row['image'])}</b> "
            f"<span class=d>+{row['gained']} tag(s)</span></div>"
            f"<img src='{html.escape(row['sheet'])}' loading='lazy'>"
        )
    path = out_dir / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    src, dst = resolve_under_home(args.src), resolve_under_home(args.dst)
    out_dir = resolve_under_home(args.out)
    (out_dir / "sheets").mkdir(parents=True, exist_ok=True)
    labels = (
        args.a_label or args.a_flags or "shipped",
        args.b_label or args.b_flags or "shipped",
    )

    a_options, detect_args = build_options(args.a_flags)
    b_options, _ = build_options(args.b_flags)

    from scripts.preprocess.position_captions import build_detect_fn

    detect_args.device = args.device
    detect_fn, part_detect_fn, _model, _proc = build_detect_fn(detect_args)

    from library.captioning.anima_tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )

    ckpt = ensure_tagger_checkpoint(
        resolve_under_home(detect_args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    tagger = AnimaTagger(ckpt, device=args.device)
    vocabulary = load_clause_vocabulary(ckpt)

    images = walk_images(dst, recursive=True, pattern=args.path_pattern)
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} image(s) under {args.path_pattern}")

    rows: list[dict] = []
    stats = {"seen": 0, "candidates": 0, "identical": 0, "differ": 0, "no_caption": 0}
    for index, image_path in enumerate(images, 1):
        stats["seen"] += 1
        rel = str(image_path.relative_to(dst))
        # The master, not the derived caption: after one `--apply` the derived
        # side already carries clauses and would be skipped by every rule here.
        caption_path = (src / rel).with_suffix(".txt")
        if not caption_path.exists():
            stats["no_caption"] += 1
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        ok, _reason = is_candidate(caption)
        if not ok:
            continue
        stats["candidates"] += 1

        image = Image.open(image_path).convert("RGB")
        shared = dict(
            detect_fn=detect_fn,
            tag_fn=tagger.predict,
            vocabulary=vocabulary,
            part_detect_fn=part_detect_fn,
        )
        a_prop = propose_for_image(image, caption, options=a_options, **shared)
        b_prop = propose_for_image(image, caption, options=b_options, **shared)
        if a_prop.proposed == b_prop.proposed:
            stats["identical"] += 1
            continue
        stats["differ"] += 1

        a_clauses, b_clauses = clause_map(a_prop.proposed), clause_map(b_prop.proposed)
        gained = sum(
            len([t for t in tags if t not in a_clauses.get(pos, ())])
            for pos, tags in b_clauses.items()
        )
        sheet_rel = (
            f"sheets/{Path(rel).with_suffix('.png').as_posix().replace('/', '__')}"
        )
        render(image, rel, a_prop, b_prop, labels, caption).save(out_dir / sheet_rel)
        rows.append(
            {
                "image": rel,
                "sheet": sheet_rel,
                "gained": gained,
                "a": a_prop.proposed,
                "b": b_prop.proposed,
                "status": [a_prop.status, b_prop.status],
            }
        )
        print(f"  [{index}/{len(images)}] {rel}  +{gained}")

    rows.sort(key=lambda r: -r["gained"])
    (out_dir / "report.json").write_text(
        json.dumps(
            {"labels": list(labels), "summary": stats, "rows": rows},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    index_path = write_index(out_dir, rows, labels)
    print(json.dumps(stats, indent=2))
    print(f"\nindex:  {index_path}\nreport: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
