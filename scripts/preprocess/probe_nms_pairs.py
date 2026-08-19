"""How often does NMS keep the worse mask? Sizes the mask-quality tie-break.

`dedupe_detections` ranks duplicates on score alone. On `ama_mitsuki/5847152` that
kept a proposal with 0.077 box-fill over one with 0.560 because it scored 0.035
higher (see `docs/experimental/multiview_audit.md`). One data point is not a
margin, so this walks a corpus, replays the same greedy NMS, and records every
suppression: the two scores, the two mask fills, and their IoU.

Read-only. Prints the distribution needed to answer three things: how often
suppression fires at all, how often the survivor is the *worse* mask, and what
score margin those inversions sit at.

Two requirements are served by one grounding pass per image: the NMS replay
runs at every floor in ``--floors`` (the score gate is a pure re-filter on top
of the shared proposal set, so extra floors are free), and every proposal's box
fill + area fraction is recorded. Both were consumed by the corpus measurement
that fixed the shipped ratio at 2.0 and refuted the degenerate-proposal guard
(`docs/experimental/multiview_audit.md` §5.4).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

PREFETCH_DEPTH = 4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.preprocess._dataset import walk_images  # noqa: E402
from library.preprocess.position_captions import (  # noqa: E402
    box_containment,
    box_iou,
    drop_small_boxes,
)
from scripts.preprocess.position_captions import build_detect_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument("--path_pattern", dest="path_pattern", default="*")
    p.add_argument("--out", default="post_image_dataset/captions/nms_pairs.json")
    p.add_argument("--prompt", default="girl")
    # Every floor gets its own NMS replay off one shared detection pass; the
    # lowest one is what SAM3 is actually asked for. 0.5 = primary pipeline
    # floor, 0.35 = the floor the audit's retry reaches.
    p.add_argument(
        "--floors",
        type=lambda s: sorted((float(x) for x in s.split(",")), reverse=True),
        default=[0.5, 0.35],
    )
    p.add_argument("--retry_score_threshold", type=float, default=0.35)
    p.add_argument("--part_score_threshold", type=float, default=0.5)
    p.add_argument("--iou_threshold", type=float, default=0.65)
    p.add_argument("--min_area_frac", type=float, default=0.005)
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def mask_fill(det, size: tuple[int, int]) -> float | None:
    """Fraction of the detection's own box that its mask actually claims."""
    if det.mask is None:
        return None
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    width, height = size
    x1, y1, x2, y2 = det.box
    box = (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, int(x2)),
        min(height, int(y2)),
    )
    window = mask[box[1] : box[3], box[0] : box[2]] > 0.5
    return float(window.mean()) if window.size else 0.0


def replay_nms(dets: list, iou_threshold: float) -> list[tuple]:
    """Greedy score-ranked NMS mirroring `dedupe_detections`; returns
    (kept, dropped) collision pairs instead of discarding the loser."""
    pairs: list[tuple] = []
    keep: list = []
    for det in sorted(dets, key=lambda d: -d.score):
        hit = next((k for k in keep if box_iou(det.box, k.box) >= iou_threshold), None)
        if hit is None:
            keep.append(det)
        else:
            pairs.append((hit, det))
    return pairs


def main() -> None:
    args = parse_args()
    # SAM3 sees the lowest floor once; each floor's population is a re-filter.
    args.score_threshold = min(args.floors)
    detect_fn, _part, model, processor = build_detect_fn(args)
    dst = resolve_under_home(args.dst)
    images = walk_images(dst, recursive=True, pattern=args.path_pattern)
    print(f"{len(images)} images, floors {args.floors}", flush=True)

    def load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as handle:
            return handle.convert("RGB")

    rows: list[dict] = []
    proposals: list[dict] = []
    # PNG decode is the serial CPU gap between GPU passes — prefetch a few
    # images ahead so the detector never waits on the decoder.
    pool = ThreadPoolExecutor(max_workers=2)
    pending = deque(
        (path, pool.submit(load_rgb, path)) for path in images[:PREFETCH_DEPTH]
    )
    upcoming = iter(images[PREFETCH_DEPTH:])
    index = 0
    while pending:
        path, future = pending.popleft()
        image = future.result()
        ahead = next(upcoming, None)
        if ahead is not None:
            pending.append((ahead, pool.submit(load_rgb, ahead)))
        index += 1
        if index % 100 == 0:
            print(f"  [{index}/{len(images)}]", flush=True)
        dets = drop_small_boxes(
            detect_fn(image, args.score_threshold), image.size, args.min_area_frac
        )
        rel = str(path.relative_to(dst))
        width, height = image.size
        fills = {id(det): mask_fill(det, image.size) for det in dets}
        for det in dets:
            x1, y1, x2, y2 = det.box
            proposals.append(
                {
                    "image": rel,
                    "score": round(float(det.score), 4),
                    "fill": fills[id(det)],
                    "area_frac": round(
                        max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / (width * height),
                        4,
                    ),
                }
            )
        for floor in args.floors:
            floored = [d for d in dets if d.score >= floor]
            for hit, det in replay_nms(floored, args.iou_threshold):
                rows.append(
                    {
                        "image": rel,
                        "floor": floor,
                        "kept_score": round(float(hit.score), 4),
                        "dropped_score": round(float(det.score), 4),
                        "margin": round(float(hit.score - det.score), 4),
                        "kept_fill": fills[id(hit)],
                        "dropped_fill": fills[id(det)],
                        "iou": round(box_iou(det.box, hit.box), 4),
                        "containment": round(box_containment(det.box, hit.box), 4),
                    }
                )

    pool.shutdown()
    del processor, model

    summary: dict = {"images": len(images), "proposals": len(proposals)}
    for floor in args.floors:
        frows = [r for r in rows if r["floor"] == floor]
        inversions = [
            r
            for r in frows
            if r["kept_fill"] is not None
            and r["dropped_fill"] is not None
            and r["dropped_fill"] > r["kept_fill"]
        ]
        bad = [r for r in inversions if r["kept_fill"] < 0.15]
        summary[f"floor_{floor}"] = {
            "suppressions": len(frows),
            "images_with_suppression": len({r["image"] for r in frows}),
            # NMS kept the mask that claims less of its own box than the one it dropped.
            "fill_inversions": len(inversions),
            # ...and the subset where the survivor is degenerate, not merely smaller.
            "degenerate_survivor": len(bad),
            "degenerate_images": sorted({r["image"] for r in bad}),
            "inversion_margins": sorted(r["margin"] for r in inversions),
            "degenerate_margins": sorted(r["margin"] for r in bad),
        }
    # Shape-B gate: is there a clean fill gap among near-whole-canvas proposals?
    whole = [p for p in proposals if p["area_frac"] >= 0.9 and p["fill"] is not None]
    summary["whole_canvas_fills"] = sorted(p["fill"] for p in whole)

    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"summary": summary, "pairs": rows, "proposals": proposals}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
