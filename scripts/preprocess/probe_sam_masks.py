"""Dump SAM3's raw per-instance masks, colour-coded, over one or many images.

Answers "is the mask actually broken, or are we mishandling it?" — the two look
identical downstream once ``crop_instance`` has blanked a crop white. So this
bypasses the audit entirely: run a prompt, print each mask's shape / dtype /
value range / fill against the image it is supposed to align with, and render
every instance in its own colour over the original.

If the overlay shows a clean silhouette, the mask is fine and our indexing is
wrong. If it shows the same speckle the crop did, the mask really is that bad.

Sweeps several prompts over several images in one pass. The image encoding is
computed once per image and re-grounded per prompt (``set_text_prompt`` reuses
the cached ``backbone_out``), so extra prompts cost a grounding pass each, not a
re-encode — which is what makes a body-part prompt cheap to compare against
``girl``. Renders land at ``<out>/<stem>/prompt_<label>.png`` and the numbers in
a ``probe.json`` carrying every proposal's score / box / area fraction / box
fill, the same schema the earlier prompt sweeps under
``post_image_dataset/captions/mask_probe/`` use.
"""

from __future__ import annotations

import argparse
import json
import re
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

from library.env import resolve_under_home  # noqa: E402
from library.preprocess._dataset import walk_images  # noqa: E402

COLORS = [
    (255, 60, 60),
    (60, 140, 255),
    (255, 210, 60),
    (170, 100, 255),
    (60, 230, 190),
    (255, 120, 200),
    (150, 255, 90),
    (255, 165, 80),
]


def parse_prompts(spec: str) -> list[tuple[str, str]]:
    """``"girl,rear=buttocks"`` -> ``[("girl", "girl"), ("rear", "buttocks")]``.

    The label names the output file, so a multi-word prompt needs one; without
    an explicit ``label=`` the prompt is slugified into one.
    """
    out: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, prompt = item.partition("=")
        if not prompt:
            label, prompt = re.sub(r"\W+", "_", label).strip("_"), label
        out.append((label, prompt))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "images",
        nargs="*",
        help="Image paths (repo-relative ok). Omit and pass --path_pattern to "
        "sweep the resized tree instead",
    )
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument(
        "--path_pattern",
        dest="path_pattern",
        default=None,
        help="Glob over --dst, used when no image paths are given",
    )
    p.add_argument(
        "--prompts",
        default="girl",
        help="Comma-separated prompts, each optionally ``label=prompt``. The "
        'body-part fallback ships as "buttocks,hips,thighs"',
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Processor confidence floor — low on purpose, we want to see every "
        "proposal including the ones the audit filters out",
    )
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="post_image_dataset/captions/mask_probe")
    p.add_argument(
        "--summary",
        default=None,
        help="Where the JSON lands (default <out>/probe.json)",
    )
    p.add_argument(
        "--dump_masks",
        action="store_true",
        help="Also write every binary mask as its own PNG (first prompt only)",
    )
    return p.parse_args()


def resolve_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        return [resolve_under_home(i) for i in args.images]
    if not args.path_pattern:
        raise SystemExit("pass image paths or --path_pattern")
    dst = resolve_under_home(args.dst)
    return sorted(walk_images(dst, recursive=True, pattern=args.path_pattern))


def out_names(paths: list[Path]) -> list[str]:
    """One output dir per image, keyed on the stem.

    Stems repeat across artist dirs (`project_booru_id_space_collision`: a bare
    stem is only unique within one folder), so the whole selection falls back to
    `<parent>__<stem>` the moment any two collide — all-or-nothing, so the
    naming doesn't depend on which images happened to be selected.
    """
    stems = [p.stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{p.parent.name}__{p.stem}" for p in paths]


def probe_one(
    processor,
    state,
    image: Image.Image,
    prompt: str,
    *,
    verbose: bool,
    dump_dir: Path | None = None,
) -> tuple[list[dict], np.ndarray, list]:
    """Ground one prompt against an already-encoded image.

    Returns the per-proposal records, the colour overlay, and the raw boxes (the
    caller draws them, so the overlay stays a pure mask render).
    """
    out = processor.set_text_prompt(prompt=prompt, state=state)
    boxes, scores, masks = out["boxes"], out["scores"], out.get("masks")
    if verbose:
        print(f"    returned keys: {sorted(out.keys())}")
    print(f"    instances: {len(boxes)}   masks present: {masks is not None}")

    overlay = np.asarray(image).astype(np.float32)
    records: list[dict] = []
    canvas = image.width * image.height

    for i, (box, score) in enumerate(zip(boxes, scores)):
        import torch

        coords = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
        rec = {
            "score": round(float(score), 4),
            "box": [round(v, 1) for v in coords],
            "area_frac": round(
                max(0.0, coords[2] - coords[0])
                * max(0.0, coords[3] - coords[1])
                / canvas,
                4,
            ),
            "box_fill": None,
            "aligned": None,
        }
        if masks is None or i >= len(masks):
            print(f"    #{i} score={float(score):.4f} box={rec['box']}  mask=MISSING")
            records.append(rec)
            continue

        m = masks[i]
        raw = m.float().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        flat = raw[0] if raw.ndim == 3 else raw
        aligned = flat.shape == (image.height, image.width)
        binary = flat > 0.5
        bx = [
            max(0, int(coords[0])),
            max(0, int(coords[1])),
            min(image.width, int(coords[2])),
            min(image.height, int(coords[3])),
        ]
        in_box = binary[bx[1] : bx[3], bx[0] : bx[2]]
        rec["box_fill"] = round(float(in_box.mean()) if in_box.size else 0.0, 4)
        rec["aligned"] = bool(aligned)
        records.append(rec)

        if verbose:
            print(f"    mask shape={raw.shape} dtype={raw.dtype}")
            print(
                f"    values min={raw.min():.4f} max={raw.max():.4f} "
                f"mean={raw.mean():.4f}"
            )
            print(
                f"    2D shape={flat.shape} vs image (H,W)="
                f"({image.height},{image.width})  "
                f"{'ALIGNED' if aligned else '*** MISMATCH ***'}"
            )
        print(
            f"    #{i} score={float(score):.4f} box={rec['box']} "
            f"area={rec['area_frac']:.3f} fill={rec['box_fill']:.3f}"
            f"{'' if aligned else '  *** MASK MISALIGNED ***'}"
        )
        if dump_dir is not None:
            Image.fromarray((binary * 255).astype(np.uint8)).save(
                dump_dir / f"mask_{i}_raw.png"
            )
        if aligned:
            overlay[binary] = (
                overlay[binary] * 0.45
                + np.array(COLORS[i % len(COLORS)], np.float32) * 0.55
            )

    return records, overlay, boxes


def render(overlay: np.ndarray, boxes, path: Path) -> None:
    import torch

    composite = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(composite)
    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        coords = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
        draw.rectangle(coords, outline=color, width=6)
        draw.rectangle(
            [coords[0], coords[1], coords[0] + 44, coords[1] + 44], fill=color
        )
        draw.text((coords[0] + 14, coords[1] + 12), str(i), fill=(0, 0, 0))
    composite.save(path)


def main() -> None:
    args = parse_args()
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    prompts = parse_prompts(args.prompts)
    if not prompts:
        raise SystemExit("--prompts is empty")
    paths = resolve_images(args)
    out_root = resolve_under_home(args.out)
    # A lone image with a lone prompt is the original single-shot diagnostic —
    # keep it chatty (dtype / value range / alignment), stay terse for a sweep.
    verbose = len(paths) == 1 and len(prompts) == 1
    print(
        f"{len(paths)} image(s) x {len(prompts)} prompt(s): {[p for _, p in prompts]}"
    )

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(resolve_under_home(args.checkpoint)),
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    summary: list[dict] = []
    for path, name in zip(paths, out_names(paths)):
        image = Image.open(path).convert("RGB")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {path}  size={image.size} (W,H)")

        entry: dict = {"image": str(path), "size": list(image.size), "prompts": {}}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for idx, (label, prompt) in enumerate(prompts):
                print(f"  -- {label}: {prompt!r}")
                records, overlay, boxes = probe_one(
                    processor,
                    state,
                    image,
                    prompt,
                    verbose=verbose,
                    dump_dir=out_dir if (args.dump_masks and idx == 0) else None,
                )
                entry["prompts"][label] = {
                    "prompt": prompt,
                    "n": len(records),
                    "proposals": records,
                }
                render(overlay, boxes, out_dir / f"prompt_{label}.png")
                if idx == 0:
                    # `overlay.png` is the first prompt's render under its
                    # historical name, so older references still resolve.
                    render(overlay, boxes, out_dir / "overlay.png")
        summary.append(entry)
        print(f"  wrote: {out_dir}")

    summary_path = (
        resolve_under_home(args.summary) if args.summary else out_root / "probe.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"prompts": dict(prompts), "images": summary}, indent=1)
    )
    print(f"\nwrote: {summary_path}")


if __name__ == "__main__":
    main()
