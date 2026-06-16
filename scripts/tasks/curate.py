"""Dataset curation tasks — organize / select images, distinct from preprocess.

``preprocess-*`` makes data training-ready (resize → latent/text/PE caches);
``curate-*`` is about *curation* — grouping, dedup, coverage — over the native
source tree. Command bodies for the ``make curate-*`` targets live here.
"""

from __future__ import annotations

from ._common import PY, _path, run


def cmd_curate_group(extra):
    """Group dataset images by PE-Spatial visual similarity.

    Writes ``post_image_dataset/groups/groups.json`` (per-artist
    connected-components over the same near-twin grid gate the miner uses — two
    images group when ``match_frac >= --match-frac-min`` at per-cell floor
    ``--cell-match-min``). The GUI Dataset tab reads the manifest to filter the
    image list by group. Tune via ``ARGS="--match-frac-min 0.4 --cell-match-min
    0.9"`` (higher = tighter) / ``ARGS="--min-size 2"``. Reuses the shared PE
    feature cache, so re-runs and threshold sweeps are cheap.
    """
    run(
        [
            PY,
            "scripts/curate/build_groups.py",
            "--source-dir",
            _path("source_image_dir", "image_dataset"),
            *extra,
        ]
    )
