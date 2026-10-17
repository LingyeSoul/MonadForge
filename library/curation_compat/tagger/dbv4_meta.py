"""Torch-free facts about the dbv4 tagger backbone.

Our half of the tagger (vocab / rules / thresholds / sidecar) ships from
``sorryhyun/anima-tagger`` into ``models/captioners/``; the backbone ships from
a gated, GPL-3.0 upstream repo into the HuggingFace hub cache under the user's
own token. ``tagger.py`` re-exports every name below.
"""

from __future__ import annotations

from pathlib import Path

from library.curation_compat._json import read_json
from library.curation_compat.contract import (
    DBV4_BACKBONE_FILES,
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    TAGGER_OPTIONAL_FILES,
    TAGGER_REQUIRED_FILES,
)

__all__ = [
    "DBV4_BACKBONE_FILES",
    "DBV4_OPTIONAL_FILES",
    "DBV4_REQUIRED_FILES",
    "TAGGER_OPTIONAL_FILES",
    "TAGGER_REQUIRED_FILES",
]

DEFAULT_DBV4_REPO = "animetimm/caformer_b36.dbv4-full"
DEFAULT_DBV4_ARCH = "caformer_b36"
DEFAULT_DBV4_IMG_SIZE = 384

# Our half of the tagger: auto-fetched when ckpt_dir is missing required files.
# The live checkpoint sits at the repo root — vocab / rules / groups /
# thresholds / sidecar only (the `dbv4/` and legacy `v5/` subfolders were
# removed 2026-09-13). The file sets are contract (the trainer probes them
# too), so they live in ``anime_tools.contract`` and are re-exported here.
TAGGER_HF_REPO = "sorryhyun/anima-tagger"
TAGGER_HF_SUBFOLDER = ""
DEFAULT_TAGGER_DIR = "models/captioners/anima-tagger-dbv4"


def gated_hint(repo: str) -> str:
    """Recovery text for a failed backbone fetch (token / terms)."""
    return f"hf auth login, then accept the terms at https://huggingface.co/{repo}"


def backbone_repo_for(ckpt_dir: str | Path) -> str:
    """Backbone repo the checkpoint at ``ckpt_dir`` was built against.

    Reads ``config.json["dbv4"]["repo"]``; falls back to the default when the
    checkpoint is absent or doesn't name one.
    """
    try:
        cfg = read_json(Path(ckpt_dir) / "config.json")
    except (OSError, ValueError):
        return DEFAULT_DBV4_REPO
    repo = (cfg.get("dbv4") or {}).get("repo")
    return repo if isinstance(repo, str) and repo else DEFAULT_DBV4_REPO


def backbone_cached(repo: str) -> bool:
    """True when every backbone file is already in the local hub cache (offline)."""
    from library.curation_compat._hf import hf_file_cached

    return all(hf_file_cached(repo, f) for f in DBV4_BACKBONE_FILES)
