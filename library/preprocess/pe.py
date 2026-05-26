"""Cache PE-Core (or other registered vision-encoder) patch-token features.

Orchestration extracted from ``preprocess/cache_pe_encoder.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The script keeps only argparse +
encoder load; the walk → group → batched-encode → idempotent-save loop, and the
centroid pooling pass, live here so the daemon / tests / embedding code can
drive them without a CLI attached.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from library.io.cache import resolve_cache_path
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.preprocess._dataset import (
    PreprocessStats,
    group_by_shape,
    partition_cached,
    walk_images,
)
from library.preprocess._progress import ProgressFn
from library.vision.encoder import VisionEncoderBundle, encode_pe_from_imageminus1to1

logger = logging.getLogger(__name__)


def cache_path_for(
    image_path: Path,
    encoder: str,
    cache_dir: Path | None = None,
    image_dir: Path | None = None,
) -> Path:
    """Sidecar path for ``image_path``: ``{stem}_anima_{encoder}.safetensors``.

    With ``cache_dir`` the sidecar is redirected there (nested under the
    source subpath when ``image_dir`` is given); otherwise it lives next to
    the image (legacy layout).
    """
    suffix = f"_anima_{encoder}.safetensors"
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


class _PEImageGroup(Dataset):
    """Reads images from one ``(W, H)`` resolution group.

    Each ``__getitem__`` returns ``(str_path, str_out_path, [3, H, W] tensor in
    [-1, 1])`` so the main thread can write safetensors in batch order without
    holding the PIL.Image object across the worker boundary. Paths are passed
    as strings (lighter to pickle than ``Path``; ``save_file`` takes a string
    anyway).
    """

    def __init__(self, paths: list[Path], out_paths: list[Path]):
        self._paths = [str(p) for p in paths]
        self._out_paths = [str(p) for p in out_paths]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        p = self._paths[idx]
        with Image.open(p) as img:
            tensor = IMAGE_TRANSFORMS(np.array(img.convert("RGB")))
        return p, self._out_paths[idx], tensor


def _collate(batch):
    """Stack tensors into ``[B, 3, H, W]``; group already guarantees same shape."""
    paths, out_paths, tensors = zip(*batch)
    return list(paths), list(out_paths), torch.stack(tensors, dim=0)


def cache_pe_features(
    data_dir: Path,
    bundle: VisionEncoderBundle,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    save_dtype: torch.dtype = torch.bfloat16,
    progress: ProgressFn | None = None,
) -> PreprocessStats:
    """Encode every image under ``data_dir`` through ``bundle`` → sidecars.

    Groups images by ``(W, H)`` (same encoder bucket → one batched forward),
    pre-skips already-cached entries, and writes ``image_features`` per image.
    The encoder is supplied loaded (``load_pe_encoder``) so model setup stays in
    the caller. Returns counts; pass ``progress`` for a per-image bar.
    """
    image_files = walk_images(data_dir, recursive=recursive)
    stats = PreprocessStats(seen=len(image_files))

    # Pre-skip cached files so workers never decode them.
    pending, skipped = partition_cached(
        image_files,
        lambda p: cache_path_for(
            p, bundle.name, cache_dir=cache_dir, image_dir=data_dir
        ),
    )
    stats.skipped = skipped

    reso_groups = group_by_shape(pending)

    metadata = {
        "encoder": bundle.name,
        "d_enc": str(bundle.d_enc),
        "patch": str(bundle.bucket_spec.patch),
    }
    pin_memory = bundle.device.type == "cuda"

    if progress is not None:
        progress(0, total=len(pending))

    from safetensors.torch import save_file

    for paths in reso_groups.values():
        out_paths = [
            cache_path_for(p, bundle.name, cache_dir=cache_dir, image_dir=data_dir)
            for p in paths
        ]
        ds = _PEImageGroup(paths, out_paths)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0 and len(paths) > batch_size),
        )
        for batch_paths, batch_out_paths, img_batch in loader:
            with torch.no_grad():
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, img_batch, same_bucket=True
                )
            for src, dst, feats in zip(batch_paths, batch_out_paths, feats_list):
                save_dict = {
                    "image_features": feats.detach().to(save_dtype).cpu().contiguous()
                }
                save_file(save_dict, dst, metadata=metadata)
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{Path(src).name} → T={feats.shape[0]}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def _pool_pe(feats: torch.Tensor, *, drop_cls: bool = True) -> torch.Tensor:
    """Mean over patch tokens. ``feats`` is ``[T, D]``; returns ``[D]``."""
    if drop_cls and feats.shape[0] > 1:
        feats = feats[1:]
    return feats.mean(dim=0)


def compute_pe_centroid(
    cache_dir: Path, *, encoder: str, limit: int = 0
) -> tuple[int, torch.Tensor]:
    """Stream-pool cached PE features under ``cache_dir`` → ``(n_images, centroid)``.

    Walks ``cache_dir`` recursively (nested caches mirror the source subfolder
    structure). ``centroid`` is the dataset mean of mean-over-patch-token pooled
    features, ``[D]`` fp32. Raises ``FileNotFoundError`` when no matching caches
    exist and ``ValueError`` when none carry an ``image_features`` key.
    """
    from safetensors.torch import load_file

    suffix = f"_anima_{encoder}.safetensors"
    files = sorted(p for p in cache_dir.rglob(f"*{suffix}") if p.is_file())
    files = [p for p in files if not p.name.startswith("anima_pe_centroid")]
    if not files:
        raise FileNotFoundError(f"No '{suffix}' caches under {cache_dir}")
    if limit > 0:
        files = files[:limit]

    centroid: torch.Tensor | None = None
    n = 0
    for p in files:
        sd = load_file(str(p))
        feats = sd.get("image_features")
        if feats is None:
            logger.warning("skip %s: no 'image_features' key", p.name)
            continue
        pool = _pool_pe(feats.to(torch.float32))
        if centroid is None:
            centroid = torch.zeros_like(pool)
        centroid += pool
        n += 1

    if n == 0 or centroid is None:
        raise ValueError(f"No usable PE features found under {cache_dir}")
    return n, centroid / n


def write_pe_centroid(
    cache_dir: Path, out_path: Path, *, encoder: str, limit: int = 0
) -> tuple[int, torch.Tensor]:
    """Compute the centroid (see :func:`compute_pe_centroid`) and save it.

    Returns ``(n_images, centroid)`` so the caller can report a summary.
    """
    from safetensors.torch import save_file

    n, centroid = compute_pe_centroid(cache_dir, encoder=encoder, limit=limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"centroid": centroid.contiguous()},
        str(out_path),
        metadata={
            "encoder": encoder,
            "n_images": str(n),
            "d_enc": str(centroid.shape[0]),
            "pool": "mean_over_patch_tokens",
        },
    )
    return n, centroid
