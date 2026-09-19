"""A/B two SAM3 subject prompts side by side — text vs text, or text vs a
learned soft prompt — as contact sheets plus a numeric report.

Same shape as `scripts/preprocess/ab_position_captions.py`: one image encode,
one grounding pass per side, a sheet only where the sides disagree, and an
`index.html` ordered most-changed-first. Sides are ``--a`` / ``--b``; a side
ending in ``.safetensors`` is loaded as a soft prompt, anything else is text.

Image set comes from a `build_targets.py` manifest via ``--which``:
``disagree`` (girl vs anime-girl survivor count differs @0.5), ``zero_girl``
(no survivor under girl @0.5), ``train``, or ``all``; ``--path_pattern``
narrows further. The report counts, per side: zero-survivor images, survivors,
degenerate survivors (fill < 0.15), whole-canvas junk (area ≥ 0.95 & fill <
0.10), and — for images whose caption count is unambiguous — how often the
survivor count matches the caption.

    make daemon-run ARGS="bench/sam3_soft_prompt/ab_sam3_prompt.py \\
        --b bench/sam3_soft_prompt/results/<run>/soft_prompt.safetensors --which disagree"
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from bench.sam3_soft_prompt.build_targets import COUNT_TARGETS, survivors  # noqa: E402
from bench.sam3_soft_prompt.common import (  # noqa: E402
    encode_image,
    encode_text,
    ground,
    install_prompt,
    iou_xyxy,
    load_sam3,
    load_soft_prompt,
    proposals,
)
from library.env import resolve_under_home  # noqa: E402
from library.preprocess.multiview_sheet import _font  # noqa: E402

PANEL_W = 640
COLORS = [
    (255, 90, 90),
    (90, 170, 255),
    (90, 220, 120),
    (255, 200, 60),
    (200, 120, 255),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        default="post_image_dataset/captions/sam3_soft_prompt/targets/manifest.json",
    )
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument(
        "--which", default="disagree", choices=["disagree", "zero_girl", "train", "all"]
    )
    p.add_argument(
        "--path_pattern", default=None, help="fnmatch on the manifest's rel path"
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--a", default="girl")
    p.add_argument("--b", required=True)
    p.add_argument(
        "--out",
        default=None,
        help="default post_image_dataset/captions/sam3_soft_prompt/ab_<which>",
    )
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--retry_floor", type=float, default=0.35)
    p.add_argument("--iou_threshold", type=float, default=0.65)
    p.add_argument("--min_area_frac", type=float, default=0.005)
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def side_prompt(model, spec: str, device: str) -> dict:
    if spec.endswith(".safetensors"):
        return load_soft_prompt(resolve_under_home(spec), device)
    return encode_text(model, spec, device)


def draw_side(image: Image.Image, rows: list[dict], title: str) -> Image.Image:
    scale = PANEL_W / image.width
    panel = image.resize((PANEL_W, int(image.height * scale)), Image.LANCZOS).convert(
        "RGBA"
    )
    w, h = panel.size
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, r in enumerate(rows):
        color = COLORS[i % len(COLORS)]
        m = (
            F.interpolate(r["mask"][None, None].float(), size=(h, w), mode="bilinear")[
                0, 0
            ]
            > 0.5
        )
        tint = Image.new("RGBA", panel.size, (*color, 80))
        overlay.paste(
            tint, (0, 0), Image.fromarray((m.cpu().numpy() * 255).astype(np.uint8))
        )
        x0, y0, x1, y1 = (
            r["box"][0] * w,
            r["box"][1] * h,
            r["box"][2] * w,
            r["box"][3] * h,
        )
        draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=3)
        draw.text(
            (x0 + 4, y0 + 4),
            f"{r['score']:.2f} fill {r['fill']:.2f}",
            fill=(*color, 255),
            font=_font(16),
        )
    panel = Image.alpha_composite(panel, overlay)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, w, 26), fill=(0, 0, 0, 180))
    draw.text(
        (6, 4),
        f"{title}  ·  {len(rows)} survivor(s)",
        fill=(255, 255, 255, 255),
        font=_font(16),
    )
    return panel.convert("RGB")


def change_score(a: list[dict], b: list[dict]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0 + abs(len(a) - len(b))
    best = [max(iou_xyxy(ra["box"], rb["box"]) for rb in b) for ra in a]
    return abs(len(a) - len(b)) + (1 - sum(best) / len(best))


def main() -> None:
    args = parse_args()
    dst = resolve_under_home(args.dst)
    manifest = json.loads(resolve_under_home(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if args.which != "all":
        rows = [r for r in rows if r.get(args.which)]
    if args.path_pattern:
        from fnmatch import fnmatch

        rows = [r for r in rows if fnmatch(r["image"], args.path_pattern)]
    if args.limit:
        rows = rows[: args.limit]
    out = resolve_under_home(
        args.out or f"post_image_dataset/captions/sam3_soft_prompt/ab_{args.which}"
    )
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} images ({args.which}) → {out}", flush=True)

    model, processor = load_sam3(resolve_under_home(args.checkpoint), args.device)
    sides = {
        "a": side_prompt(model, args.a, args.device),
        "b": side_prompt(model, args.b, args.device),
    }
    labels = {
        "a": args.a,
        "b": Path(args.b).parent.name + "/" + Path(args.b).name
        if args.b.endswith(".safetensors")
        else args.b,
    }

    stats = {
        k: {
            "zero": 0,
            "survivors": 0,
            "degenerate": 0,
            "whole_canvas_junk": 0,
            "count_match": 0,
            "count_eval": 0,
        }
        for k in sides
    }
    report_rows: list[dict] = []
    for n, r in enumerate(rows, 1):
        if n % 50 == 0:
            print(f"  [{n}/{len(rows)}]", flush=True)
        with Image.open(dst / r["image"]) as h:
            image = h.convert("RGB")
        bo = encode_image(model, processor, image)
        surv: dict[str, list[dict]] = {}
        for k, prompt in sides.items():
            install_prompt(bo, prompt)
            props = proposals(ground(model, processor, bo), args.retry_floor)
            s = survivors(props, args.floor, args.iou_threshold, args.min_area_frac)
            surv[k] = s
            st = stats[k]
            st["zero"] += int(not s)
            st["survivors"] += len(s)
            st["degenerate"] += sum(x["fill"] < 0.15 for x in s)
            st["whole_canvas_junk"] += sum(
                x["area_frac"] >= 0.95 and x["fill"] < 0.10 for x in s
            )
            target_n = COUNT_TARGETS.get(tuple(r["count"]))
            if target_n is not None and not r["multiview"]:
                st["count_eval"] += 1
                st["count_match"] += int(len(s) == target_n)
        delta = change_score(surv["a"], surv["b"])
        row = {
            "image": r["image"],
            "count": r["count"],
            "multiview": r["multiview"],
            "a": [{k: v for k, v in x.items() if k != "mask"} for x in surv["a"]],
            "b": [{k: v for k, v in x.items() if k != "mask"} for x in surv["b"]],
            "change": round(delta, 3),
        }
        if delta > 0.05:
            pa = draw_side(image, surv["a"], f"A: {labels['a']}")
            pb = draw_side(image, surv["b"], f"B: {labels['b']}")
            sheet = Image.new(
                "RGB",
                (pa.width + pb.width + 10, max(pa.height, pb.height)),
                (30, 30, 30),
            )
            sheet.paste(pa, (0, 0))
            sheet.paste(pb, (pa.width + 10, 0))
            name = r["image"].replace("/", "__")
            sheet.save(out / "sheets" / name)
            row["sheet"] = f"sheets/{name}"
        report_rows.append(row)

    report_rows.sort(key=lambda x: -x["change"])
    summary = {
        "images": len(rows),
        "which": args.which,
        "labels": labels,
        "differ": sum("sheet" in x for x in report_rows),
        "sides": stats,
    }
    (out / "report.json").write_text(
        json.dumps({"summary": summary, "rows": report_rows}, indent=1),
        encoding="utf-8",
    )
    parts = [
        "<html><head><meta charset='utf-8'><style>body{background:#111;color:#ddd;font-family:sans-serif}"
        "img{max-width:100%}.row{margin:12px 0;padding:8px;background:#1c1c1c}code{color:#9cf}</style></head><body>",
        f"<h2>SAM3 prompt A/B — {html.escape(args.which)}</h2>",
        f"<p>A = <code>{html.escape(labels['a'])}</code> &nbsp; B = <code>{html.escape(labels['b'])}</code></p>",
        f"<pre>{html.escape(json.dumps(stats, indent=1))}</pre>",
    ]
    for x in report_rows:
        if "sheet" not in x:
            continue
        parts.append(
            f"<div class='row'><div><b>{html.escape(x['image'])}</b> · count {html.escape(str(x['count']))}"
            f"{' · multiview' if x['multiview'] else ''} · change {x['change']}</div>"
            f"<img src='{x['sheet']}'></div>"
        )
    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out / 'index.html'}")


if __name__ == "__main__":
    main()
