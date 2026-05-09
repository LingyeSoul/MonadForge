"""Cache builders for the two training paths.

* ``cmd_build_features`` — encode each manifest image through the frozen
  PE-Core encoder, mean-pool patch tokens, write per-stem ``[d_enc]``
  safetensors. Consumed by the frozen-encoder fast path.
* ``cmd_build_resized`` — LANCZOS-resize each manifest image to its PE
  bucket and write per-stem ``uint8 [C, H, W]`` safetensors. Consumed by
  the end-to-end PE-LoRA path (where the encoder is unfrozen and
  pre-pooled features can't track it).

Both modes are idempotent — re-runs only fill in missing entries.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def cmd_build_features(args: argparse.Namespace) -> None:
    from library.captioning.anima_tagger_data import (
        FeatureCacheBuilder,
        TaggerManifest,
    )

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"missing {manifest_path} — run --mode build_vocab first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    cache_dir = out_dir / ".cache" / f"pooled-{args.encoder}"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(
        "build_features: %d manifest entries → %s (device=%s, encoder=%s)",
        len(manifest.stems),
        cache_dir,
        device,
        args.encoder,
    )
    builder = FeatureCacheBuilder(
        manifest=manifest,
        cache_dir=cache_dir,
        device=device,
        encoder_name=args.encoder,
        num_workers=args.feature_cache_workers,
    )
    n_new = builder.build()
    n_total = len(manifest.stems) - len(builder.missing_stems())
    print(f"  cache dir:        {cache_dir}")
    print(f"  newly encoded:    {n_new}")
    print(f"  cached / total:   {n_total} / {len(manifest.stems)}")


def cmd_build_resized(args: argparse.Namespace) -> None:
    from library.captioning.anima_tagger_data import (
        ImageCacheBuilder,
        TaggerManifest,
    )
    from library.vision.encoders import get_encoder_info

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    manifest = TaggerManifest.from_path(manifest_path)
    cache_dir = out_dir / ".cache" / f"resized-{args.encoder}"
    spec = get_encoder_info(args.encoder).bucket_spec
    logger.info(
        "build_resized: %d manifest entries → %s (encoder=%s, patch=%d)",
        len(manifest.stems),
        cache_dir,
        args.encoder,
        spec.patch,
    )
    builder = ImageCacheBuilder(
        manifest=manifest,
        cache_dir=cache_dir,
        spec=spec,
        num_workers=args.feature_cache_workers,
    )
    n_new = builder.build()
    n_total = len(manifest.stems) - len(builder.missing_stems())
    print(f"  cache dir:        {cache_dir}")
    print(f"  newly resized:    {n_new}")
    print(f"  cached / total:   {n_total} / {len(manifest.stems)}")
