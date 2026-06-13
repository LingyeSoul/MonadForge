"""Cache VAE latents for a dataset directory.

Orchestration extracted from ``preprocess/cache_latents.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The script keeps only argparse +
VAE load; the walk → group-by-resolution → batched-encode → idempotent-save
loop lives here.

Idempotence note: a single ``{stem}_{WxH}_anima.npz`` can hold *multiple*
resolutions (one ``latents_{H}x{W}`` key each), so the skip is per-resolution
*inside* the encode loop rather than a whole-file existence check.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from library.io.cache import LATENT_CACHE_SUFFIX, resolve_cache_path
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.preprocess._dataset import PreprocessStats, group_by_shape, walk_images
from library.preprocess._progress import ProgressFn


def get_latents_npz_path(
    image_path: Path,
    image_size: tuple[int, int],
    cache_dir: Path | None = None,
    image_dir: Path | None = None,
) -> Path:
    """Match ``AnimaLatentsCachingStrategy`` naming: ``{stem}_{WxH}_anima.npz``.

    With ``cache_dir`` the cache is redirected there (nested under the source
    subpath when ``image_dir`` is given); otherwise it lives next to the image.
    """
    suffix = f"_{image_size[0]:04d}x{image_size[1]:04d}{LATENT_CACHE_SUFFIX}"
    if cache_dir is None:
        return image_path.with_name(image_path.stem + suffix)
    return Path(
        resolve_cache_path(
            str(image_path),
            suffix,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir) if image_dir is not None else None,
        )
    )


def _decode_batch(
    batch_paths: list[Path],
    w: int,
    h: int,
    cache_dir: Path | None,
    data_dir: Path,
) -> tuple[
    list[Path],
    list[tuple[Path, str]],
    list[tuple[Path, tuple[int, int]]],
    "torch.Tensor | None",
]:
    """CPU stage: per-resolution skip-probe + decode + transform a batch.

    Returns ``(skipped, failed, kept, img_batch)`` — already-cached paths,
    ``(path, reason)`` for images that wouldn't decode (truncated / corrupt),
    the ``(path, (w, h))`` survivors, and their stacked CPU tensor (``None`` if
    nothing survived). A bad file is isolated to its own entry rather than
    raising, so one corrupt staged PNG can't abort the whole run. Runs on a
    worker thread (pure PIL/numpy/torch-CPU), overlapping the previous GPU encode."""
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    kept: list[tuple[Path, tuple[int, int]]] = []
    tensors: list[torch.Tensor] = []
    for p in batch_paths:
        npz_path = get_latents_npz_path(p, (w, h), cache_dir=cache_dir, image_dir=data_dir)
        if npz_path.exists():
            latents_size = (h // 8, w // 8)
            key = f"latents_{latents_size[0]}x{latents_size[1]}"
            try:
                if key in np.load(npz_path):
                    skipped.append(p)
                    continue
            except Exception:
                pass
        try:
            img_np = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            failed.append((p, f"{type(e).__name__}: {e}"))
            continue
        tensors.append(IMAGE_TRANSFORMS(img_np))
        kept.append((p, (w, h)))
    img_batch = torch.stack(tensors, dim=0) if tensors else None
    return skipped, failed, kept, img_batch


def _save_batch(items: list[tuple[Path, np.ndarray, tuple[int, int]]]) -> None:
    """IO stage: write each ``(npz_path, latent, size)``. Preserves any
    other-resolution keys already in the file (read-modify-write). Each npz_path
    is written exactly once per ``cache_latents`` call, so threaded saves of a
    batch don't race. Runs on a worker thread, overlapping the next GPU encode."""
    for npz_path, lat_np, size in items:
        key_reso_suffix = f"_{lat_np.shape[-2]}x{lat_np.shape[-1]}"
        kwargs: dict = {}
        if npz_path.exists():
            npz = np.load(npz_path)
            for key in npz.files:
                kwargs[key] = npz[key]
        kwargs[f"latents{key_reso_suffix}"] = lat_np
        kwargs[f"original_size{key_reso_suffix}"] = np.array(list(size))
        kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array([0, 0, size[0], size[1]])
        np.savez(npz_path, **kwargs)


def cache_latents(
    data_dir: Path,
    vae,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    batch_size: int = 4,
    progress: ProgressFn | None = None,
    io_workers: int | None = None,
) -> PreprocessStats:
    """Encode every image under ``data_dir`` through ``vae`` → latent NPZs.

    ``vae`` is supplied loaded + on-device (``device``/``dtype`` are read off
    it). Returns counts; pass ``progress`` for a per-image bar.

    The VAE forward stays serial on the calling thread (single GPU stream); the
    per-batch disk decode + image transform and the npz read-modify-write are
    CPU/IO, so they're farmed to thread pools that overlap the GPU — the GPU no
    longer idles between batches. ``io_workers`` sizes those pools (default
    ``min(8, cpu_count)``). Output is byte-identical to the serial path."""
    image_files = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    reso_groups = group_by_shape(image_files)
    stats = PreprocessStats(seen=len(image_files))

    if progress is not None:
        progress(0, total=len(image_files))

    # Flatten to (w, h, batch_paths) jobs so decode/encode/save pipeline uniformly.
    batches: list[tuple[int, int, list[Path]]] = []
    for (w, h), paths in reso_groups.items():
        for s in range(0, len(paths), batch_size):
            batches.append((w, h, paths[s : s + batch_size]))

    workers = io_workers or min(8, (os.cpu_count() or 4))
    depth = max(2, workers // 2)  # decoded batches kept in flight (bounds host RAM)
    max_saves = max(2, workers)  # in-flight npz writes before backpressure
    it = iter(batches)
    decode_q: deque = deque()
    save_q: deque = deque()
    failed_all: list[tuple[Path, str]] = []

    def _submit_decode(decode_ex) -> bool:
        b = next(it, None)
        if b is None:
            return False
        w, h, bp = b
        decode_q.append(decode_ex.submit(_decode_batch, bp, w, h, cache_dir, data_dir))
        return True

    with (
        ThreadPoolExecutor(max_workers=workers) as decode_ex,
        ThreadPoolExecutor(max_workers=workers) as save_ex,
    ):
        for _ in range(depth):  # prime the decode window
            if not _submit_decode(decode_ex):
                break
        while decode_q:
            skipped, failed, kept, img_batch = decode_q.popleft().result()
            _submit_decode(decode_ex)  # keep a decode in flight

            for p in skipped:
                stats.skipped += 1
                if progress is not None:
                    progress(1, detail=f"skip {p.name}")
            for p, reason in failed:  # isolate corrupt files; don't abort the run
                failed_all.append((p, reason))
                if progress is not None:
                    progress(1, detail=f"FAILED {p.name}")
            if img_batch is None:
                continue

            img_batch = img_batch.to(device=vae.device, dtype=vae.dtype)
            with torch.no_grad():
                latents = vae.encode_pixels_to_latents(img_batch).cpu()

            items: list[tuple[Path, np.ndarray, tuple[int, int]]] = []
            for i, (p, size) in enumerate(kept):
                npz_path = get_latents_npz_path(
                    p, size, cache_dir=cache_dir, image_dir=data_dir
                )
                items.append((npz_path, latents[i].float().numpy(), size))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{p.name} → {size[0]}x{size[1]}")
            save_q.append(save_ex.submit(_save_batch, items))

            while len(save_q) >= max_saves:  # backpressure on the save side
                save_q.popleft().result()
        for f in save_q:  # surface any write error + drain remaining writes
            f.result()

    if failed_all:
        print(
            f"\n⚠ {len(failed_all)} image(s) could not be decoded and were skipped "
            f"(no latent cached — re-stage these, e.g. delete + re-run mangafy):"
        )
        for p, reason in failed_all:
            print(f"  {p}  ({reason})")

    return stats
