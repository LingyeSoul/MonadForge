"""Dataset-caching orchestration shared by the ``preprocess/`` entry points.

This package holds the reusable "drive the primitives over a *dataset*" logic
that the ``preprocess/cache_*.py`` scripts otherwise inlined in their ``main()``
bodies (see ``docs/proposal/tooling_architecture.md`` §A): the walk/group/skip
loop (``_dataset``) plus one module per cache kind whose function takes the
already-loaded model + explicit paths and returns a :class:`PreprocessStats`.
Entry points keep argparse + model load + an optional ``tqdm`` progress bar.

Imports are **lazy** (PEP 562): the public names below resolve to their
submodules only when first accessed. This keeps ``import
library.preprocess.resize_preview`` (a torch-free leaf used by the GUI) from
dragging torch/cv2 into the process via this ``__init__`` — see
``gui/CLAUDE.md`` (torch must not appear in GUI startup importtime).
"""

from importlib import import_module

# public name -> (submodule, attribute-in-submodule)
_LAZY = {
    "PreprocessStats": ("_dataset", "PreprocessStats"),
    "group_by_shape": ("_dataset", "group_by_shape"),
    "partition_cached": ("_dataset", "partition_cached"),
    "walk_images": ("_dataset", "walk_images"),
    "ProgressFn": ("_progress", "ProgressFn"),
    "tqdm_progress": ("_progress", "tqdm_progress"),
    "process_image": ("images", "process_image"),
    "resize_to_buckets": ("images", "resize_to_buckets"),
    "cache_latents": ("latents", "cache_latents"),
    "count_pending_latents": ("latents", "count_pending_latents"),
    "get_latents_npz_path": ("latents", "get_latents_npz_path"),
    "cache_pe_features": ("pe", "cache_pe_features"),
    "compute_pe_centroid": ("pe", "compute_pe_centroid"),
    "count_pending_pe": ("pe", "count_pending_pe"),
    "write_pe_centroid": ("pe", "write_pe_centroid"),
    "pe_cache_path_for": ("pe", "cache_path_for"),
    "find_stale_caches": ("reconcile", "find_stale_caches"),
    "delete_stale": ("reconcile", "delete_stale"),
    "reconcile_caches": ("reconcile", "reconcile_caches"),
    "StaleCaches": ("reconcile", "StaleCaches"),
    "cache_text_embeddings": ("text", "cache_text_embeddings"),
    "count_pending_text": ("text", "count_pending_text"),
    # Torch-free leaf (caption_variants) so the caption-correction step / GUI can
    # import these without dragging torch in via the text submodule.
    "generate_caption_variants": ("caption_variants", "generate_caption_variants"),
    "build_erasure_token_pool": ("caption_variants", "build_erasure_token_pool"),
    "variants_sidecar_path": ("caption_variants", "variants_sidecar_path"),
    "write_variants_sidecar": ("caption_variants", "write_variants_sidecar"),
    "read_variants_sidecar": ("caption_variants", "read_variants_sidecar"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    try:
        submodule, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f"{__name__}.{submodule}"), attr)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(__all__)
