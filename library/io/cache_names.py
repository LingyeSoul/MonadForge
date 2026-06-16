"""Torch-free cache-filename conventions for preprocessed sidecars.

Single source of truth for the suffixes the preprocess pipeline writes next to
(or under ``cache_dir`` for) each source image, plus the by-name classifier /
counter built on them (:func:`classify_cache_file`, :func:`count_preprocess_caches`).
Deliberately stdlib-only at import time — no torch / numpy / safetensors — so
lightweight consumers share the exact naming and counting rules instead of
hand-copying drift-prone string literals.

The PySide6 GUI is the motivating consumer: ``gui/CLAUDE.md`` forbids pulling a
torch-importing module into the GUI (it adds seconds to startup), so the GUI used
to keep its own private copy of these suffixes — which drifted when ``pe_spatial``
became the default REPA encoder and the GUI still only knew ``pe``. Centralizing
the rules here lets ``library.io.cache`` / ``library.preprocess`` (torch-coupled)
and the GUI (torch-free) agree on one definition.
"""

from __future__ import annotations

import os
from pathlib import Path

# VAE latents: ``{stem}_{WxH}_anima.npz`` (resolution infix added by the caller).
LATENT_CACHE_SUFFIX = "_anima.npz"
# Text-encoder cross-attention embeddings: ``{stem}_anima_te.safetensors``.
TE_CACHE_SUFFIX = "_anima_te.safetensors"
# Pooled text embedding sidecar: ``{stem}_anima_pooled.safetensors``.
POOLED_CACHE_SUFFIX = "_anima_pooled.safetensors"

# Default REPA / PE vision encoder. Its sidecars are named
# ``{stem}_anima_pe_spatial.safetensors``; the PE-Core encoder (``pe``) writes
# ``{stem}_anima_pe.safetensors``. The active encoder is configurable via the
# ``repa_encoder`` training knob, so the PE suffix is encoder-parameterized.
DEFAULT_PE_ENCODER = "pe_spatial"


def pe_cache_suffix(encoder: str | None = None) -> str:
    """Sidecar suffix for a PE/vision encoder: ``_anima_{encoder}.safetensors``.

    ``encoder=None`` (or blank) resolves to :data:`DEFAULT_PE_ENCODER`
    (``pe_spatial``), matching the default ``repa_encoder``.
    """
    name = (encoder or "").strip() or DEFAULT_PE_ENCODER
    return f"_anima_{name}.safetensors"


def classify_cache_file(name: str, pe_encoder: str | None = None) -> str | None:
    """Bucket a cache filename into ``"latents"`` / ``"te"`` / ``"pe"`` (or None).

    The single place that maps a sidecar name → cache kind, so no consumer has to
    re-encode the suffix rules. ``pe_encoder`` picks which PE variant counts as
    ``"pe"`` (defaults to the REPA default ``pe_spatial``). Note ``TE`` is tested
    before ``pe`` so the ``pe`` encoder can never shadow a ``_anima_te`` sidecar.
    """
    if name.endswith(TE_CACHE_SUFFIX):
        return "te"
    if name.endswith(pe_cache_suffix(pe_encoder)):
        return "pe"
    if name.endswith(LATENT_CACHE_SUFFIX):
        return "latents"
    return None


def count_preprocess_caches(
    cache_dir: str | os.PathLike,
    path_pattern: str | None = None,
    pe_encoder: str | None = None,
) -> dict[str, int]:
    """Count latent / TE / PE cache sidecars under ``cache_dir`` by filename.

    Returns ``{"latents", "te", "pe"}`` counts (zeros, without raising, if the
    directory is missing). Walks recursively so nested caches mirroring a
    subfoldered source tree are counted; ``path_pattern`` (a glob relative to
    ``cache_dir``) optionally narrows the walk the same way training's
    ``path_pattern`` filter does. ``pe_encoder`` selects the PE variant counted
    (defaults to ``pe_spatial`` — the default ``repa_encoder``).

    Kept here, next to the suffix definitions, so every consumer (the GUI
    included) shares one by-name classifier instead of re-deriving the rules.
    """
    out = {"latents": 0, "te": 0, "pe": 0}
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return out
    paths = [p for p in cache_dir.rglob("*") if p.is_file()]
    if path_pattern and path_pattern != "*":
        # Lazy import: keeps this module a stdlib-only leaf at import time (the
        # GUI imports it just for suffixes); path_filter is itself torch-free.
        from library.datasets.path_filter import filter_paths_by_glob

        keep = filter_paths_by_glob(
            [str(p) for p in paths], str(cache_dir), path_pattern
        )
        paths = [p for p, k in zip(paths, keep) if k]
    for p in paths:
        kind = classify_cache_file(p.name, pe_encoder)
        if kind:
            out[kind] += 1
    return out
