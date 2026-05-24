"""Dataset-caching orchestration shared by the ``preprocess/`` entry points.

This package holds the reusable "drive the primitives over a *dataset*" logic
that the ``preprocess/cache_*.py`` scripts otherwise inlined in their ``main()``
bodies (see ``docs/proposal/tooling_architecture.md`` §A): the walk/group/skip
loop (``_dataset``) plus one module per cache kind whose function takes the
already-loaded model + explicit paths and returns a :class:`PreprocessStats`.
Entry points keep argparse + model load + an optional ``tqdm`` progress bar.
"""

from library.preprocess._dataset import (
    PreprocessStats,
    group_by_shape,
    partition_cached,
    walk_images,
)
from library.preprocess._progress import ProgressFn, tqdm_progress
from library.preprocess.images import process_image, resize_to_buckets
from library.preprocess.latents import cache_latents, get_latents_npz_path
from library.preprocess.pe import (
    cache_path_for as pe_cache_path_for,
    cache_pe_features,
    compute_pe_centroid,
    write_pe_centroid,
)
from library.preprocess.text import (
    cache_pooled_text,
    cache_text_embeddings,
    generate_caption_variants,
)

__all__ = [
    # dataset walk / group / skip
    "PreprocessStats",
    "group_by_shape",
    "partition_cached",
    "walk_images",
    # progress
    "ProgressFn",
    "tqdm_progress",
    # cache functions
    "cache_latents",
    "get_latents_npz_path",
    "cache_text_embeddings",
    "cache_pooled_text",
    "generate_caption_variants",
    "cache_pe_features",
    "compute_pe_centroid",
    "write_pe_centroid",
    "pe_cache_path_for",
    "resize_to_buckets",
    "process_image",
]
