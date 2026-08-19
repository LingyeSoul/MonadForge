"""Contact sheet for an applied `caption-position` run: image / SAM segment / caption before-after.

`ab_position_captions.py` answers "does rule A differ from rule B". This answers
the other question — "what did the run that already landed actually do to my
captions, and did the detector see what I think it saw". One row per image:
the detection overlay, the mask-blanked crops the tagger was handed, and the
master caption next to the derived one now on disk, with every clause tag
marked as moved / novel / duplicated.

The **after** side is read from disk (`--dst`), not re-proposed, so the sheet
shows what actually trains. Detection is re-run only to recover the crops and
boxes (they are review artifacts and nothing persists them), proposing from the
master under `--src` the way the A/B sheet does — after one `--apply` the
derived caption carries clauses and `is_candidate` would reject the corpus as
`already-has-clauses`. When that fresh proposal disagrees with what is on disk
the row is flagged `drift` — the run used different flags, or the caption was
edited since.

Read-only: writes only under `--out`, never a caption.

    make daemon-run ARGS="scripts/preprocess/review_position_captions.py \\
        --path_pattern 'ama_mitsuki/*|ie_(raarami)/*'"
"""

from __future__ import annotations

import argparse
import html
import json
import sys
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
from library.preprocess.multiview_sheet import _fit, _font  # noqa: E402
from library.preprocess.position_captions import (  # noqa: E402
    is_candidate,
    load_clause_vocabulary,
    propose_for_image,
)

PREVIEW = 900
CROP = 320
# Shared with the A/B sheet so a box and its crop card read as the same subject.
BOX_COLORS = [
    (255, 90, 90),
    (90, 170, 255),
    (255, 210, 80),
    (180, 120, 255),
    (90, 230, 200),
    (255, 140, 200),
    (160, 255, 120),
    (255, 175, 100),
]


class PositionPalette:
    """One color per position phrase, allocated per row in first-seen order.

    Every surface of a row — overlay box, crop card, clause header — looks its
    color up here by the *position name*, so "top left" is the same color
    everywhere. Index-based coloring (the old scheme) desynced: the overlay
    numbered boxes in detection order while the clauses sat in caption order.
    A drifted after-caption whose position no longer exists ("bottom" from an
    older naming run) simply allocates the next color — visibly matching
    nothing, which is the honest rendering.
    """

    def __init__(self) -> None:
        self._map: dict[str, tuple[int, int, int]] = {}

    def __call__(self, position: str) -> tuple[int, int, int]:
        if position not in self._map:
            self._map[position] = BOX_COLORS[len(self._map) % len(BOX_COLORS)]
        return self._map[position]

    def known(self, position: str) -> tuple[int, int, int] | None:
        """The color already allocated to ``position``, without allocating."""
        return self._map.get(position)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="image_dataset", help="Caption master (before)")
    p.add_argument(
        "--dst",
        default="post_image_dataset/resized",
        help="Resized tree — images, and the derived caption (after) as applied",
    )
    p.add_argument("--path_pattern", dest="path_pattern", default="*")
    p.add_argument("--out", default="post_image_dataset/captions/position_review")
    # GOTCHA: pass as `--flags=--foo`, not `--flags --foo` — argparse reads a
    # `-`-leading value as the next option and dies on "expected one argument".
    p.add_argument(
        "--flags",
        default="",
        help="position_captions.py flags to re-derive the crops with (empty = "
        "shipped defaults, i.e. what the preprocess stage runs). Must be "
        "written --flags=--foo",
    )
    p.add_argument("--limit", type=int, default=0, help="0 = no cap")
    p.add_argument(
        "--skips",
        action="store_true",
        help="Also render pre-detection skips (single-subject etc.) as rows. "
        "Off by default — nothing was proposed and there is nothing to see",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_options(flags: str):
    """Reuse the real CLI so the review can never drift from what ships."""
    from scripts.preprocess.position_captions import build_options_from_args
    from scripts.preprocess.position_captions import parse_args as pc_parse_args

    argv = sys.argv
    sys.argv = [argv[0], *flags.split()]
    try:
        args = pc_parse_args()
    finally:
        sys.argv = argv
    return build_options_from_args(args), args


def flat_key_set(caption: str) -> set[str]:
    return {t.strip().lower() for t in parse_caption(caption).flat_tags if t.strip()}


def classify(tag: str, before_bag: set[str], after_bag: set[str]) -> str:
    """How a clause tag got there — read off the two captions, not the proposal.

    Deriving this from the on-disk text rather than ``proposal.moved`` keeps the
    verdict true even when the re-derivation drifts: ``moved`` is the v2 rewrite
    (the bag gave the tag up), ``novel`` is a crop invention the master never
    carried, ``duplicated`` is the v1 shape where it stayed flat as well.
    """
    key = tag.strip().lower()
    if key not in before_bag:
        return "novel"
    return "duplicated" if key in after_bag else "moved"


def render_overlay(
    image: Image.Image,
    boxes: list[list[int]],
    colors: list[tuple[int, int, int]],
) -> Image.Image:
    """The full image with each detection numbered in its crop card's color."""
    preview = _fit(image.convert("RGB"), PREVIEW)
    scale = preview.width / image.width
    draw = ImageDraw.Draw(preview)
    font = _font(22, bold=True)
    for index, (box, color) in enumerate(zip(boxes, colors)):
        draw.rectangle([v * scale for v in box], outline=color, width=4)
        x1, y1 = box[0] * scale, box[1] * scale
        draw.rectangle([x1, y1, x1 + 30, y1 + 30], fill=color)
        draw.text((x1 + 9, y1 + 4), f"{index + 1}", font=font, fill=(20, 20, 24))
    return preview


def caption_html(
    caption: str,
    before_bag: set[str],
    after_bag: set[str],
    palette: PositionPalette,
    fallback_colors: list[tuple[int, int, int]] | None = None,
) -> str:
    """Render one caption, marking what each clause tag did to the flat bag.

    Clause colors are the box colors: a position the palette knows (the fresh
    instances seeded it) matches by name; otherwise the i-th clause takes the
    i-th box's color from ``fallback_colors`` — a drifted after-caption still
    lists its clauses in the reading order of the run that wrote it, so order
    is the next-best identity when the names have since changed.
    """
    if not caption:
        return "<span class=dim>(empty)</span>"
    parsed = parse_caption(caption)
    parts = [f"<span class=bag>{html.escape(', '.join(parsed.flat_tags))}</span>"]
    for index, clause in enumerate(parsed.clauses):
        color = palette.known(clause.position)
        if color is None and fallback_colors and index < len(fallback_colors):
            color = fallback_colors[index]
        if color is None:
            color = palette(clause.position)
        tags = " ".join(
            f"<span class='t {classify(t, before_bag, after_bag)}'>{html.escape(t)}</span>"
            for t in clause.tags
        )
        swatch = "rgb({},{},{})".format(*color)
        tint = "rgba({},{},{},.09)".format(*color)
        parts.append(
            f"<div class=clause style='border-color:{swatch};background:{tint}'>"
            f"<b style='color:{swatch}'>{html.escape(clause.position)}</b> {tags}</div>"
        )
    return "".join(parts)


def write_index(out_dir: Path, rows: list[dict], summary: dict, pattern: str) -> Path:
    parts = [
        "<meta charset='utf-8'><title>position captions — review</title>",
        """<style>
body{background:#18181c;color:#e8e8ec;font:15px/1.55 system-ui,sans-serif;margin:0;padding:24px}
h1{font-size:21px;margin:0 0 4px}
h2{font-size:16px;margin:0 0 10px;font-weight:600}
section{border-top:1px solid #33333c;padding:22px 0}
.dim{color:#8a8a92;font-size:13px}
.sum{color:#8a8a92;font-size:13px;margin-bottom:8px}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-left:8px;
 background:#2c2c34;color:#b8b8c0}
.pill.proposed{background:#1d3a2a;color:#7ad79c}
.pill.drift{background:#4a2a16;color:#ffb570}
.pill.skip{background:#3a2020;color:#ff9a9a}
.shots{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;margin-bottom:14px}
.shots img{border-radius:5px;display:block}
.full img{max-width:min(560px,48vw);height:auto}
.crops{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0}
figcaption{font-size:12px;color:#b8b8c0;margin-top:4px}
.crops img{max-height:260px;width:auto;border:2px solid #33333c}
.caps{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.caps h3{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#8a8a92;margin:0 0 6px}
.box{background:#1f1f25;border-radius:6px;padding:12px;font-size:13.5px}
.bag{color:#c8c8d0}
.clause{border-left:3px solid;padding:3px 8px;margin-top:8px;border-radius:0 4px 4px 0}
.t{display:inline-block;margin-right:6px}
.t.moved{color:#7ad79c}
.t.novel{color:#ffb570;text-decoration:underline dotted}
.t.duplicated{color:#8a8a92}
.legend span{margin-right:14px}
@media(max-width:1100px){.caps{grid-template-columns:1fr}}
</style>""",
        "<h1>position captions — review</h1>",
        f"<p class=sum><code>{html.escape(pattern)}</code> · "
        + " · ".join(f"{k} {v}" for k, v in summary.items())
        + "</p>",
        "<p class='sum legend'>clause tag: "
        "<span class='t moved'>moved out of the bag (v2)</span>"
        "<span class='t novel'>novel — crop invention</span>"
        "<span class='t duplicated'>still flat too (v1 shape)</span></p>",
    ]

    def crop_card(i: int, c: dict) -> str:
        img_style = cap_style = ""
        if c.get("color"):
            swatch = "rgb({},{},{})".format(*c["color"])
            img_style = f" style='border-color:{swatch}'"
            cap_style = f" style='color:{swatch}'"
        return (
            f"<figure><img src='{html.escape(c['path'])}' loading=lazy{img_style}>"
            f"<figcaption{cap_style}>{i + 1} · {html.escape(c['position'])}"
            "</figcaption></figure>"
        )

    for row in rows:
        crops = "".join(crop_card(i, c) for i, c in enumerate(row["crops"]))
        overlay = (
            f"<figure class=full><img src='{html.escape(row['overlay'])}' loading=lazy>"
            f"<figcaption>{row['detected']} detection(s)</figcaption></figure>"
            if row.get("overlay")
            else ""
        )
        pill = f"<span class='pill {row['pill']}'>{html.escape(row['status'])}</span>"
        drift = (
            "<span class='pill drift'>drift vs re-derivation</span>"
            if row.get("drift")
            else ""
        )
        parts.append(
            f"<section><h2>{html.escape(row['image'])}{pill}{drift}</h2>"
            f"<div class=shots>{overlay}<div class=crops>{crops}</div></div>"
            "<div class=caps>"
            f"<div><h3>before — master</h3><div class=box>{row['before']}</div></div>"
            f"<div><h3>after — resized, on disk</h3><div class=box>{row['after']}</div></div>"
            "</div></section>"
        )
    path = out_dir / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    src, dst = resolve_under_home(args.src), resolve_under_home(args.dst)
    out_dir = resolve_under_home(args.out)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    options, detect_args = build_options(args.flags)
    detect_args.device = args.device

    from scripts.preprocess.position_captions import build_detect_fn

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
    summary = {
        "seen": 0,
        "with clauses": 0,
        "proposed": 0,
        "drift": 0,
        "skipped": 0,
        "no caption": 0,
    }
    for index, image_path in enumerate(images, 1):
        summary["seen"] += 1
        rel = image_path.relative_to(dst)
        stem = rel.with_suffix("").as_posix().replace("/", "__")
        before_path = (src / rel).with_suffix(".txt")
        after_path = (dst / rel).with_suffix(".txt")
        if not before_path.exists():
            summary["no caption"] += 1
            continue
        before = before_path.read_text(encoding="utf-8").strip()
        after = (
            after_path.read_text(encoding="utf-8").strip()
            if after_path.exists()
            else ""
        )
        before_bag, after_bag = flat_key_set(before), flat_key_set(after)
        has_clauses = bool(parse_caption(after).clauses)
        if has_clauses:
            summary["with clauses"] += 1

        ok, reason = is_candidate(before)
        if not ok and not args.skips:
            summary["skipped"] += 1
            continue

        image = Image.open(image_path).convert("RGB")
        crops: list[dict] = []
        proposal = None
        if ok:

            def sink(i: int, position: str, crop: Image.Image, _stem=stem) -> str:
                name = f"crops/{_stem}_{i}_{position.replace(' ', '-')}.png"
                _fit(crop, CROP).save(out_dir / name)
                crops.append({"path": name, "position": position})
                return name

            proposal = propose_for_image(
                image,
                before,
                detect_fn=detect_fn,
                tag_fn=tagger.predict,
                vocabulary=vocabulary,
                options=options,
                part_detect_fn=part_detect_fn,
                crop_sink=sink,
            )
            status = proposal.status
        else:
            status = f"skip:{reason}"

        # One color per position name, shared by every surface of the row —
        # overlay box, crop card, clause header. Instances (reading order,
        # matching the crop sink) carry the positions; a gate-rejected image
        # has none, so its raw detections take synthetic keys that clause
        # coloring can never collide with.
        palette = PositionPalette()
        if proposal is not None and proposal.instances:
            boxes = [inst.box for inst in proposal.instances]
            box_colors = [palette(inst.position) for inst in proposal.instances]
            for inst, crop in zip(proposal.instances, crops):
                crop["color"] = palette(inst.position)
        else:
            boxes = [d["box"] for d in proposal.detections] if proposal else []
            box_colors = [palette(f"#{i}") for i in range(len(boxes))]

        overlay_rel = ""
        if boxes:
            overlay_rel = f"overlays/{stem}.png"
            render_overlay(image, boxes, box_colors).save(out_dir / overlay_rel)

        drift = bool(
            proposal and proposal.proposed and after and proposal.proposed != after
        )
        if status == "proposed":
            summary["proposed"] += 1
        else:
            summary["skipped"] += 1
        if drift:
            summary["drift"] += 1

        rows.append(
            {
                "image": rel.as_posix(),
                "status": status,
                "pill": "proposed" if status == "proposed" else "skip",
                "detected": len(boxes),
                "overlay": overlay_rel,
                "crops": crops,
                "drift": drift,
                "has_clauses": has_clauses,
                # After first: its clause positions match the fresh instances,
                # so they reuse the box colors — by name when possible, by
                # clause order when the on-disk names predate a naming change.
                "after": caption_html(
                    after, before_bag, after_bag, palette, box_colors
                ),
                "before": caption_html(
                    before, before_bag, after_bag, palette, box_colors
                ),
                "before_text": before,
                "after_text": after,
                "proposed_text": proposal.proposed if proposal else None,
            }
        )
        print(f"  [{index}/{len(images)}] {rel}  {status}{'  DRIFT' if drift else ''}")

    # Clause-carrying rows first: they are the ones with something to adjudicate.
    rows.sort(key=lambda r: (not r["has_clauses"], not r["drift"], r["image"]))
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "pattern": args.path_pattern,
                "flags": args.flags,
                "summary": summary,
                "rows": [
                    {k: v for k, v in r.items() if k not in {"before", "after"}}
                    for r in rows
                ],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    index_path = write_index(out_dir, rows, summary, args.path_pattern)
    print(json.dumps(summary, indent=2))
    print(f"\nindex:  {index_path}\nreport: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
