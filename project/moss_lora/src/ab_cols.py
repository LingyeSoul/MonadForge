#!/usr/bin/env python3
"""One file per A/B cell: the arms side by side (stacked when the image is wide).

Cells are matched across grid dirs by the name before ``_x1.0`` (``lora_ab.py``
names carry a per-LoRA digest after it).

    ab_cols.py <out_dir> <glob> <label>=<grid dir> ...
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def cell(name: str) -> str:
    return name.split("_x1.0")[0] if "_x1.0" in name else Path(name).stem


def main() -> None:
    out, pattern = Path(sys.argv[1]), sys.argv[2]
    arms = [a.split("=", 1) for a in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 32)
    files = [{cell(p.name): p for p in Path(d).glob(pattern)} for _, d in arms]
    for key in sorted(files[0]):
        ims = []
        for (label, _), by_cell in zip(arms, files):
            im = Image.open(by_cell[key]).convert("RGB")
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, draw.textlength(label, font=font) + 16, 44], fill=0)
            draw.text((8, 4), label, fill=(255, 255, 0), font=font)
            ims.append(im)
        w, h = ims[0].size
        vertical = w > 1.4 * h
        size = (w, h * len(ims)) if vertical else (w * len(ims), h)
        sheet = Image.new("RGB", size, "white")
        for i, im in enumerate(ims):
            sheet.paste(im, (0, i * h) if vertical else (i * w, 0))
        path = out / f"{key}.jpg"
        sheet.save(path, quality=93)
        print(path, flush=True)


if __name__ == "__main__":
    main()
