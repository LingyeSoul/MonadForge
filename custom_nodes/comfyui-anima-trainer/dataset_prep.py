"""Dataset preparation for the Anima LoRA trainer node.

Two modes:

- **Single-image** (a ComfyUI IMAGE tensor was connected): write every frame
  of the IMAGE batch as a PNG into a fresh temp dir, drop one `.txt` sidecar
  per image holding the supplied prompt.
- **Directory**: use the user-provided path as-is. Validate that at least
  one `.txt` caption sidecar exists (training without captions silently
  produces a useless LoRA).

Either mode optionally takes masks for masked-loss training — a ComfyUI MASK
tensor (single-image, written as ``{stem}_mask.png``) or a directory of
``{stem}_mask.png`` files (directory mode). When present, the dataset config
gains a subset ``mask_dir`` and the caller flips ``masked_loss`` on.

Both modes return a ``(src_dir, image_dir, cache_dir, dataset_config_path,
n_images, mask_dir)`` tuple of absolute paths (``mask_dir`` is ``""`` when no
masks were supplied):

- ``src_dir`` holds the original images + caption sidecars — the read-only
  input to ``preprocess-config`` (never resized in place). For single-image
  mode it's a fresh temp dir; for directory mode it's the user's dir as-is.
- ``image_dir`` is where bucket-resized images land; it's the dataset config's
  subset ``image_dir`` (training reads from here, NOT from ``src_dir``).
- ``cache_dir`` holds the VAE/TE caches the trainer's cache-completeness guard
  requires.

The dataset_config.toml mirrors `configs/base.toml`'s `[general]` /
`[[datasets]]` blueprint with a single subset wiring ``image_dir`` + ``cache_dir``.
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl")


def _tensor_to_pil(image_tensor) -> list[Image.Image]:
    """Convert a ComfyUI IMAGE tensor (`[B, H, W, C]`, float32 in [0,1]) to PILs."""
    # ComfyUI keeps IMAGE on CPU, but be defensive.
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected IMAGE of shape [B,H,W,C]; got {arr.shape}")
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return [Image.fromarray(frame) for frame in arr]


def _mask_tensor_to_pil(mask_tensor) -> list[Image.Image]:
    """Convert a ComfyUI MASK tensor (`[B, H, W]` or `[H, W]`, float in [0,1]) to L PILs.

    White (1.0) = keep / full loss weight, matching the repo's `{stem}_mask.png`
    convention. Returns one PIL per batch frame.
    """
    arr = mask_tensor.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected MASK of shape [B,H,W] or [H,W]; got {arr.shape}")
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return [Image.fromarray(frame, mode="L") for frame in arr]


def _toml_str(s: str) -> str:
    """Escape a path for a single-quoted (literal) TOML string."""
    return s.replace(chr(39), chr(39) + chr(39))


def _write_dataset_config(
    image_dir: str, cache_dir: str, output_path: str, mask_dir: str = ""
) -> None:
    cfg = (
        "[general]\n"
        "caption_extension = '.txt'\n"
        "keep_tokens = 3\n"
        "\n"
        "[[datasets]]\n"
        "batch_size = 1\n"
        "\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = '{_toml_str(image_dir)}'\n"
        f"  cache_dir = '{_toml_str(cache_dir)}'\n"
        "  num_repeats = 1\n"
    )
    # mask_dir points the masked-loss path at a dir of `{stem}_mask.png` files.
    # Masks aren't resized/cached by preprocess — the loader reads them raw and
    # NEAREST-resizes to the bucket resolution, so they only need matching stems.
    if mask_dir:
        cfg += f"  mask_dir = '{_toml_str(mask_dir)}'\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cfg)


def _count_sidecar_pairs(directory: str) -> int:
    n = 0
    for entry in os.listdir(directory):
        stem, ext = os.path.splitext(entry)
        if ext.lower() not in _IMAGE_EXTS:
            continue
        if os.path.exists(os.path.join(directory, stem + ".txt")):
            n += 1
    return n


def _count_masks(directory: str) -> int:
    """Count `*_mask.png` files anywhere under ``directory`` (nested or flat)."""
    n = 0
    for _root, _dirs, files in os.walk(directory):
        n += sum(1 for f in files if f.lower().endswith("_mask.png"))
    return n


def _count_images(directory: str) -> int:
    return sum(
        1
        for entry in os.listdir(directory)
        if os.path.splitext(entry)[1].lower() in _IMAGE_EXTS
    )


def prepare_dataset_dir(
    image,
    prompt: str,
    dataset_dir: str,
    tmp_root: str,
    mask=None,
    mask_dir: str = "",
) -> Tuple[str, str, str, str, int, str]:
    """Return ``(src_dir, image_dir, cache_dir, dataset_config_path, n_images, mask_dir)``.

    ``tmp_root`` is where temp dirs are created (usually
    ``anima_lora/output/tmp_trainer``); created on demand. Every job gets a
    fresh ``work_dir`` under it holding ``resized/`` (the dataset config's
    ``image_dir``, where bucket-resized images land) and ``cache/`` (the
    VAE/TE caches). ``src_dir`` — the originals + captions — is either a
    ``src/`` subdir (single-image mode) or the user's dir as-is (directory
    mode); ``preprocess-config`` reads it but never writes to it.

    Masks (optional) enable masked loss. In single-image mode, ``mask`` is a
    ComfyUI MASK tensor written as ``{stem}_mask.png`` into a fresh ``masks/``
    subdir of ``work_dir``. In directory mode, ``mask_dir`` is a user dir of
    ``{stem}_mask.png`` files. The returned ``mask_dir`` (last element) is the
    resolved absolute mask directory, or ``""`` when no masks were supplied —
    the caller flips ``masked_loss`` on iff it's non-empty.
    """
    os.makedirs(tmp_root, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = tempfile.mkdtemp(prefix=f"{ts}_", dir=tmp_root)
    image_dir = os.path.join(work_dir, "resized")
    cache_dir = os.path.join(work_dir, "cache")
    os.makedirs(image_dir)
    os.makedirs(cache_dir)
    dataset_cfg_path = os.path.join(work_dir, "dataset_config.toml")

    if image is not None:
        pils = _tensor_to_pil(image)
        if not pils:
            raise ValueError("IMAGE tensor has no frames.")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError(
                "Single-image mode requires a non-empty `prompt` widget."
            )

        # Optional masks: one MASK frame per image, or a single frame broadcast
        # to every image. Written flat as `{stem}_mask.png` — the resized
        # image_dir is flat in single-image mode, so the flat mask lookup hits.
        mask_pils: list = []
        if mask is not None:
            mask_pils = _mask_tensor_to_pil(mask)
            if len(mask_pils) not in (1, len(pils)):
                raise ValueError(
                    f"MASK batch ({len(mask_pils)}) must be 1 or match the "
                    f"IMAGE batch ({len(pils)})."
                )

        src_dir = os.path.join(work_dir, "src")
        os.makedirs(src_dir)
        resolved_mask_dir = ""
        if mask_pils:
            resolved_mask_dir = os.path.join(work_dir, "masks")
            os.makedirs(resolved_mask_dir)
        for i, pil in enumerate(pils):
            stem = f"img_{i:04d}"
            pil.save(os.path.join(src_dir, f"{stem}.png"), optimize=False)
            with open(
                os.path.join(src_dir, f"{stem}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(prompt)
            if mask_pils:
                mpil = mask_pils[i] if len(mask_pils) == len(pils) else mask_pils[0]
                mpil.save(os.path.join(resolved_mask_dir, f"{stem}_mask.png"))

        _write_dataset_config(
            image_dir, cache_dir, dataset_cfg_path, mask_dir=resolved_mask_dir
        )
        return (
            src_dir,
            image_dir,
            cache_dir,
            dataset_cfg_path,
            len(pils),
            resolved_mask_dir,
        )

    if not dataset_dir:
        raise ValueError(
            "Neither an IMAGE input nor a `dataset_dir` was provided."
        )
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    n_images = _count_images(dataset_dir)
    if n_images == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    if _count_sidecar_pairs(dataset_dir) == 0:
        raise ValueError(
            f"No .txt caption sidecars found in {dataset_dir}. Each image "
            "needs a same-stem .txt next to it."
        )

    resolved_mask_dir = ""
    if mask_dir:
        resolved_mask_dir = os.path.abspath(os.path.expanduser(mask_dir))
        if not os.path.isdir(resolved_mask_dir):
            raise FileNotFoundError(f"mask_dir does not exist: {resolved_mask_dir}")
        if _count_masks(resolved_mask_dir) == 0:
            raise ValueError(
                f"No `*_mask.png` files found in {resolved_mask_dir}. Each image "
                "needs a same-stem `{stem}_mask.png` (white = keep)."
            )

    _write_dataset_config(
        image_dir, cache_dir, dataset_cfg_path, mask_dir=resolved_mask_dir
    )
    return (
        dataset_dir,
        image_dir,
        cache_dir,
        dataset_cfg_path,
        n_images,
        resolved_mask_dir,
    )


def count_captioned_images(directory: str) -> int:
    """Public helper: number of images with .txt sidecars under ``directory``."""
    return _count_sidecar_pairs(directory)
