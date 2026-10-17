#!/usr/bin/env python3
"""Side-by-side comparison sheet from an eval_cmmd.py run dir.

Samples N prompts and lays out one ROW per prompt, one COLUMN per model
(paired seeds → same noise, so differences are model-driven). Reads the
per-model renders saved under ``<run_dir>/renders/<model>/<stem>.png``.

    uv run python bench/reft/compare_sheet.py \
        bench/reft/results/<ts>-<label>/ [--num 6] [--seed 0] [--height 320]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw

HEADER_H = 28
GAP = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=320)
    args = ap.parse_args()

    renders = args.run_dir / "renders"
    models = sorted(
        (d.name for d in renders.iterdir() if d.is_dir()),
        # base first, then the rest alphabetically (stable column order).
        key=lambda n: (n != "base", n),
    )
    if not models:
        raise SystemExit(f"no render dirs under {renders}")
    stems = sorted(p.stem for p in (renders / models[0]).glob("*.png"))
    common = [
        s for s in stems if all((renders / m / f"{s}.png").exists() for m in models)
    ]
    picks = Random(args.seed).sample(common, min(args.num, len(common)))

    rows = []
    for stem in picks:
        cells = []
        for m in models:
            img = Image.open(renders / m / f"{stem}.png")
            w = round(img.width * args.height / img.height)
            cells.append(img.resize((w, args.height)))
        rows.append((stem, cells))

    col_w = max(c.width for _, cells in rows for c in cells)
    sheet_w = len(models) * (col_w + GAP) + GAP
    sheet_h = HEADER_H + len(rows) * (args.height + GAP) + GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for ci, m in enumerate(models):
        draw.text((GAP + ci * (col_w + GAP) + 4, 8), m, fill="black")
    for ri, (stem, cells) in enumerate(rows):
        y = HEADER_H + GAP + ri * (args.height + GAP)
        for ci, cell in enumerate(cells):
            x = GAP + ci * (col_w + GAP) + (col_w - cell.width) // 2
            sheet.paste(cell, (x, y))

    out = args.run_dir / f"compare_{args.seed}.png"
    sheet.save(out)
    print(f"{len(rows)} prompts x {len(models)} models → {out}")


if __name__ == "__main__":
    main()
