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


def cache_latents(
    data_dir: Path,
    vae,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    batch_size: int = 4,
    progress: ProgressFn | None = None,
) -> PreprocessStats:
    """Encode every image under ``data_dir`` through ``vae`` → latent NPZs.

    ``vae`` is supplied loaded + on-device (``device``/``dtype`` are read off
    it). Returns counts; pass ``progress`` for a per-image bar.
    """
    image_files = walk_images(data_dir, recursive=recursive)
    reso_groups = group_by_shape(image_files)
    stats = PreprocessStats(seen=len(image_files))

    if progress is not None:
        progress(0, total=len(image_files))

    for (w, h), paths in reso_groups.items():
        for batch_start in range(0, len(paths), batch_size):
            batch_paths = paths[batch_start : batch_start + batch_size]
            tensors = []

            for p in batch_paths:
                npz_path = get_latents_npz_path(
                    p, (w, h), cache_dir=cache_dir, image_dir=data_dir
                )
                if npz_path.exists():
                    latents_size = (h // 8, w // 8)
                    key = f"latents_{latents_size[0]}x{latents_size[1]}"
                    try:
                        npz = np.load(npz_path)
                        if key in npz:
                            stats.skipped += 1
                            if progress is not None:
                                progress(1, detail=f"skip {p.name}")
                            continue
                    except Exception:
                        pass

                img = Image.open(p).convert("RGB")
                img_np = np.array(img)
                img_tensor = IMAGE_TRANSFORMS(img_np)
                tensors.append((p, img_tensor, (w, h)))

            if not tensors:
                continue

            img_batch = torch.stack([t[1] for t in tensors], dim=0)
            img_batch = img_batch.to(device=vae.device, dtype=vae.dtype)

            with torch.no_grad():
                latents = vae.encode_pixels_to_latents(img_batch).cpu()

            for i, (p, _, size) in enumerate(tensors):
                lat = latents[i]  # (16, H/8, W/8)
                latents_size = lat.shape[-2:]  # H/8, W/8
                key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"

                npz_path = get_latents_npz_path(
                    p, size, cache_dir=cache_dir, image_dir=data_dir
                )
                kwargs = {}
                if npz_path.exists():
                    npz = np.load(npz_path)
                    for key in npz.files:
                        kwargs[key] = npz[key]

                kwargs[f"latents{key_reso_suffix}"] = lat.float().numpy()
                kwargs[f"original_size{key_reso_suffix}"] = np.array(list(size))
                kwargs[f"crop_ltrb{key_reso_suffix}"] = np.array(
                    [0, 0, size[0], size[1]]
                )

                np.savez(npz_path, **kwargs)

                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{p.name} → {size[0]}x{size[1]}")

    return stats
