#!/usr/bin/env python3
"""Generate text/speech-bubble masks for training images using SAM3."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

import cv2
import yaml
from PIL import Image
from tqdm import tqdm

from library.preprocess import walk_images


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    Image.fromarray(alpha_mask, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, required=True, help="YAML config with prompts and params"
    )
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument(
        "--mask-dir", type=str, required=True, help="Output mask directory"
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing masks"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Local SAM3 checkpoint path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="I/O workers for loading/saving (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --image-dir. Mask output mirrors the source "
            "subdir structure under --mask-dir."
        ),
    )
    parser.add_argument(
        "--path-pattern",
        type=str,
        default=None,
        help=(
            "fnmatch glob (| to OR-combine) on each image's path relative to "
            "--image-dir, restricting which images get masked. Same semantics "
            "as the training path_pattern. Overrides the YAML's path_pattern "
            "when given; falls back to it otherwise."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prompts = config["prompts"]
    threshold = config.get("threshold", 0.5)
    dilate = config.get("dilate", 5)
    path_pattern = args.path_pattern or config.get("path_pattern")
    dilate_kernel = np.ones((dilate, dilate), dtype=np.uint8) if dilate > 0 else None

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image_dir = Path(args.image_dir)
    masks_dir = Path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    build_kwargs = {"device": args.device, "eval_mode": True}
    if args.checkpoint:
        build_kwargs["checkpoint_path"] = args.checkpoint
        build_kwargs["load_from_HF"] = False

    print("Loading SAM3 model...")
    model = build_sam3_image_model(**build_kwargs)
    processor = Sam3Processor(model)

    # Per-subdir uniqueness check (the same stem may legitimately appear in
    # multiple subfolders — the nested output layout disambiguates by folder —
    # but two files with the same stem in the *same* folder would overwrite
    # each other's mask). walk_images raises on that collision.
    image_files = walk_images(
        image_dir, recursive=args.recursive, pattern=path_pattern
    )

    # Filter to work items upfront
    work_items = []
    for image_path in image_files:
        try:
            rel = image_path.parent.relative_to(image_dir)
        except ValueError:
            rel = Path("")
        rel_str = str(rel)
        target_dir = masks_dir / rel if rel_str not in ("", ".") else masks_dir
        mask_path = target_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists() and not args.force:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        work_items.append((image_path, mask_path))

    total = len(work_items)
    if total == 0:
        print("No images to process.")
        return

    batch_size = args.batch_size
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    # Prefetch images ahead of GPU to keep it saturated
    prefetch = min(args.workers, total)
    load_futures = [pool.submit(load_image, work_items[j][0]) for j in range(prefetch)]
    save_futures = []

    pbar = tqdm(total=total, desc="Generating masks", ascii=True)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = []
        for i in range(batch_start, batch_end):
            image = load_futures[i].result()
            if i + prefetch < total:
                load_futures.append(
                    pool.submit(load_image, work_items[i + prefetch][0])
                )
            batch.append((work_items[i], image))

        with autocast:
            # Phase 1: encode all images in the batch
            states = []
            for (image_path, mask_path), image in batch:
                states.append(
                    (image_path, mask_path, image, processor.set_image(image))
                )

            # Phase 2: run prompts on each encoded image
            for image_path, mask_path, image, inference_state in states:
                w, h = image.size
                combined_mask = np.zeros((h, w), dtype=np.uint8)

                for prompt in prompts:
                    output = processor.set_text_prompt(
                        state=inference_state, prompt=prompt
                    )
                    for mask, score in zip(output["masks"], output["scores"]):
                        if score < threshold:
                            continue
                        mask_np = (
                            mask.cpu().numpy()
                            if torch.is_tensor(mask)
                            else np.asarray(mask)
                        )
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0]
                        combined_mask = np.maximum(
                            combined_mask, (mask_np > 0.5).astype(np.uint8)
                        )

                pbar.update(1)

                if not combined_mask.any():
                    pbar.set_postfix_str(f"{image_path.name}: skipped")
                    continue

                if dilate_kernel is not None:
                    combined_mask = cv2.dilate(
                        combined_mask, dilate_kernel, iterations=1
                    )

                # Invert: detected=1 → alpha=0 (ignore), no detection → alpha=255 (train)
                alpha_mask = ((1 - combined_mask) * 255).astype(np.uint8)

                save_futures.append(pool.submit(save_mask, mask_path, alpha_mask))

                masked_pct = 100 * np.count_nonzero(combined_mask) / (w * h)
                pbar.set_postfix_str(f"{image_path.name}: {masked_pct:.1f}%")

    pbar.close()

    # Wait for all saves to finish
    for f in save_futures:
        f.result()
    pool.shutdown()

    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    main()
