"""Phase 0a — pseudo-labels + splits for the SAM3 soft-prompt bench.

No human girl masks exist in this repo (`post_image_dataset/masks/` are
speech-bubble masks), so training targets come from SAM3 itself, filtered
hard enough that they are the *uncontroversial* subset:

- caption count is exactly ``1girl`` (or ``2girls``), no ``multiple views``;
- under the shipped ``girl`` prompt at the primary 0.5 floor, NMS leaves
  exactly that many survivors, every one with box fill ≥ ``--min_fill`` and
  area fraction ≤ ``--max_area`` (the audit's "clean figure" band — see
  `docs/experimental/multiview_audit.md` §5.4).

Those survivors (normalized xyxy box + 288² mask) are the targets. Everything
the prompt is supposed to *fix* is kept out of training and forms the eval
sets: ``zero_girl`` (no survivor under ``girl`` @0.5) and ``disagree``
(``girl`` and ``anime girl`` survivor counts differ @0.5 — the 466-image
population §5.5 flagged). One image encode, two prompt passes per image.

    make daemon-run ARGS="bench/sam3_soft_prompt/build_targets.py"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from bench.sam3_soft_prompt.common import (  # noqa: E402
    encode_image,
    encode_text,
    ground,
    install_prompt,
    load_sam3,
    nms,
    proposals,
)
from library.env import resolve_under_home  # noqa: E402
from library.preprocess._dataset import walk_images  # noqa: E402

COUNT_TARGETS = {("1girl",): 1, ("2girls", "multiple girls"): 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument("--src", default="image_dataset", help="caption master")
    p.add_argument("--path_pattern", default="*")
    p.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    p.add_argument(
        "--out", default="post_image_dataset/captions/sam3_soft_prompt/targets"
    )
    p.add_argument("--prompt", default="girl")
    p.add_argument("--alt_prompt", default="anime girl")
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--retry_floor", type=float, default=0.35)
    p.add_argument("--iou_threshold", type=float, default=0.65)
    p.add_argument("--min_area_frac", type=float, default=0.005)
    p.add_argument("--min_fill", type=float, default=0.2)
    p.add_argument("--max_area", type=float, default=1.01)
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def caption_meta(index: dict, src: Path, rel: str) -> tuple[tuple[str, ...], bool]:
    key = str(Path(rel).with_suffix(""))
    meta = index["image_meta"].get(key)
    if meta is None:
        return (), False
    count = tuple(sorted(meta.get("count", [])))
    try:
        text = (src / meta["path"]).read_text(encoding="utf-8")
    except OSError:
        text = ""
    return count, "multiple views" in text


def survivors(
    rows: list[dict], floor: float, iou: float, min_area: float
) -> list[dict]:
    return nms(
        [r for r in rows if r["score"] >= floor and r["area_frac"] >= min_area], iou
    )


def main() -> None:
    args = parse_args()
    dst = resolve_under_home(args.dst)
    src = resolve_under_home(args.src)
    out = resolve_under_home(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(
        resolve_under_home(args.caption_index).read_text(encoding="utf-8")
    )
    images = walk_images(dst, recursive=True, pattern=args.path_pattern)
    print(f"{len(images)} images", flush=True)

    model, processor = load_sam3(resolve_under_home(args.checkpoint), args.device)
    prompts = {
        name: encode_text(model, name, args.device)
        for name in (args.prompt, args.alt_prompt)
    }

    def load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as handle:
            return handle.convert("RGB")

    pool = ThreadPoolExecutor(max_workers=2)
    pending = deque((p, pool.submit(load_rgb, p)) for p in images[:4])
    upcoming = iter(images[4:])
    rows: list[dict] = []
    n = 0
    while pending:
        path, future = pending.popleft()
        image = future.result()
        ahead = next(upcoming, None)
        if ahead is not None:
            pending.append((ahead, pool.submit(load_rgb, ahead)))
        n += 1
        if n % 100 == 0:
            print(f"  [{n}/{len(images)}]", flush=True)
        rel = str(path.relative_to(dst))
        count, multiview = caption_meta(index, src, rel)
        backbone_out = encode_image(model, processor, image)
        per_prompt: dict[str, dict] = {}
        keep_rows: list[dict] = []
        for name, prompt in prompts.items():
            install_prompt(backbone_out, prompt)
            props = proposals(ground(model, processor, backbone_out), args.retry_floor)
            main_ = survivors(props, args.floor, args.iou_threshold, args.min_area_frac)
            retry = survivors(
                props, args.retry_floor, args.iou_threshold, args.min_area_frac
            )
            per_prompt[name] = {
                "proposals": len(props),
                "survivors": len(main_),
                "survivors_retry": len(retry),
                "min_fill": min((r["fill"] for r in main_), default=None),
                "max_area": max((r["area_frac"] for r in main_), default=None),
            }
            if name == args.prompt:
                keep_rows = main_
        target_n = COUNT_TARGETS.get(count)
        clean = (
            target_n is not None
            and not multiview
            and len(keep_rows) == target_n
            and all(r["fill"] >= args.min_fill for r in keep_rows)
            and all(r["area_frac"] <= args.max_area for r in keep_rows)
        )
        row = {
            "image": rel,
            "count": list(count),
            "multiview": multiview,
            "prompts": per_prompt,
            "train": clean,
            "zero_girl": per_prompt[args.prompt]["survivors"] == 0,
            "disagree": per_prompt[args.prompt]["survivors"]
            != per_prompt[args.alt_prompt]["survivors"],
        }
        if clean:
            tpath = out / (rel.replace("/", "__") + ".npz")
            np.savez_compressed(
                tpath,
                boxes=np.asarray([r["box"] for r in keep_rows], dtype=np.float32),
                masks=np.stack([r["mask"].cpu().numpy() for r in keep_rows]).astype(
                    np.bool_
                ),
                scores=np.asarray([r["score"] for r in keep_rows], dtype=np.float32),
            )
            row["target"] = str(tpath.relative_to(out))
        rows.append(row)
    pool.shutdown()

    summary = {
        "images": len(rows),
        "train": sum(r["train"] for r in rows),
        "train_by_count": {
            str(k): sum(r["train"] and tuple(r["count"]) == k for r in rows)
            for k in COUNT_TARGETS
        },
        "zero_girl": sum(r["zero_girl"] for r in rows),
        "disagree": sum(r["disagree"] for r in rows),
        "zero_alt": sum(r["prompts"][args.alt_prompt]["survivors"] == 0 for r in rows),
        "args": vars(args),
    }
    (out / "manifest.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
