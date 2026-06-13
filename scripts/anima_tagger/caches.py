"""Cache builders for the training path.

``cmd_build_features`` encodes each manifest image through both the PE-Core
(``--encoder``) and PE-Spatial (``--aux_encoder``) frozen encoders. Each side
dispatches on its pool kind:

* ``map`` → :class:`TokenCacheBuilder` writes per-stem ``[T, d_enc] bf16`` to
  ``.cache/tokens-<encoder>/`` (consumed by the MAP-pool head).
* ``mean`` → :class:`FeatureCacheBuilder` writes per-stem ``[d_enc] fp32``
  mean-pooled vectors to ``.cache/pooled-<encoder>/`` (consumed by a mean-pool
  trunk side, e.g. production PE-Core).

Idempotent — re-runs only fill in missing entries.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def cache_dir_for(out_dir: Path, pool_kind: str, encoder: str) -> Path:
    """Resolve the per-encoder cache subdir for the given pool kind.

    Shared with the trainer / calibrator so they all agree on the file
    layout — change here, propagate everywhere.
    """
    sub = "tokens" if pool_kind == "map" else "pooled"
    return out_dir / ".cache" / f"{sub}-{encoder}"


def _build_one_encoder(
    args: argparse.Namespace,
    manifest,
    encoder_name: str,
    pool_kind: str,
    device: torch.device,
) -> None:
    """Build a single token / feature cache for one encoder at the given pool_kind."""
    from library.captioning.anima_tagger_data import (
        FeatureCacheBuilder,
        TokenCacheBuilder,
    )

    out_dir = Path(args.out_dir)
    cache_dir = cache_dir_for(out_dir, pool_kind, encoder_name)
    logger.info(
        "build_features: pool_kind=%s  %d manifest entries → %s (device=%s, encoder=%s)",
        pool_kind,
        len(manifest.stems),
        cache_dir,
        device,
        encoder_name,
    )
    builder_kwargs = dict(
        manifest=manifest,
        cache_dir=cache_dir,
        device=device,
        encoder_name=encoder_name,
        num_workers=args.feature_cache_workers,
        batch_size=getattr(args, "feature_cache_batch_size", 8),
    )
    if pool_kind == "map":
        builder = TokenCacheBuilder(**builder_kwargs)
    else:
        builder = FeatureCacheBuilder(**builder_kwargs)
    n_new = builder.build()
    n_total = len(manifest.stems) - len(builder.missing_stems())
    print(f"  [{encoder_name}/{pool_kind}] cache dir:        {cache_dir}")
    print(f"  [{encoder_name}/{pool_kind}] newly encoded:    {n_new}")
    print(
        f"  [{encoder_name}/{pool_kind}] cached / total:   {n_total} / {len(manifest.stems)}"
    )


def cmd_build_features(args: argparse.Namespace) -> None:
    """Build both encoders' token / feature caches.

    Builds the PE-Core cache (``--encoder`` / ``--pool_kind``) and the
    PE-Spatial cache (``--aux_encoder`` / ``--pool_kind_aux``). Each
    ``(encoder, pool_kind)`` pair gets its own
    ``.cache/{tokens,pooled}-<name>/`` subdir so they can be built / refreshed
    independently.
    """
    from library.captioning.anima_tagger_data import TaggerManifest

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    manifest = TaggerManifest.from_path(manifest_path)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    _build_one_encoder(args, manifest, args.encoder, args.pool_kind, device)
    aux_encoder = getattr(args, "aux_encoder", None)
    if not aux_encoder:
        raise SystemExit(
            "build_features requires --aux_encoder (dual encoder is mandatory). "
            "Pass e.g. --aux_encoder pe_spatial."
        )
    _build_one_encoder(args, manifest, aux_encoder, args.pool_kind_aux, device)
