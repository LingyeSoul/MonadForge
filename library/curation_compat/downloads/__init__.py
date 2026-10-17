"""Catalog of every model weight the stages need: repo, destination and an
offline installed-probe. Torch-free, and the public surface of the package — the
loaders import their repo / filename / default path from here so a Download
button can't write where a loader won't look.

| Half | What is in it |
|---|---|
| :mod:`._locations` | where each weight lives: repo, filename, destination path |
| :mod:`._assets` | what a row is (:class:`Asset`, :class:`Pack`) and the fetch engine |
| :mod:`._catalog` | the rows themselves, and :func:`by_id` / :func:`by_pack` / :func:`expand` |
| :mod:`._cli` | ``python -m anime_tools.downloads`` |

Nothing here is auto-run; loaders still fetch on first use.
"""

from __future__ import annotations

from library.curation_compat.downloads._assets import PACK_BY_ID, PACKS, Asset, Pack
from library.curation_compat.downloads._catalog import by_id, by_pack, catalog, expand
from library.curation_compat.downloads._cli import build_parser, main
from library.curation_compat.downloads._locations import (
    ANIMETEXT_DIR,
    ANIMETEXT_FILES,
    ANIMETEXT_REPO,
    ANIMETEXT_SUBFOLDER,
    ANIMETEXT_WEIGHTS,
    DANBOORU_TAGS_GH_REPO,
    DANBOORU_TAGS_URL,
    DANBOORU_WIKI_FILE,
    DANBOORU_WIKI_REPO,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SUBJECT_PROMPT_EMBED,
    PE_SPATIAL_FILENAME,
    PE_SPATIAL_REPO,
    SAM3_DIR,
    SAM3_FILENAME,
    SAM3_REPO,
    SFX_READER_ADAPTER_FILES,
    SFX_READER_DIR,
    SFX_READER_FILES,
    SFX_READER_LM_FILE,
    SFX_READER_REPO,
    SFX_READER_REVISION,
    SFX_READER_SUBFOLDER,
    SOFT_PROMPT_DIR,
    SOFT_PROMPT_FILENAME,
    SOFT_PROMPT_GH_REPO,
    SOFT_PROMPT_URL,
    VL16_BASE_DIR,
    VL16_BASE_FILES,
    VL16_BASE_REPO,
    default_animetext_dir,
    default_pe_spatial_path,
    default_sfx_reader_dir,
    default_vl16_base_dir,
    http_timeout,
)

__all__ = [
    "ANIMETEXT_DIR",
    "ANIMETEXT_FILES",
    "ANIMETEXT_REPO",
    "ANIMETEXT_SUBFOLDER",
    "ANIMETEXT_WEIGHTS",
    "DANBOORU_TAGS_GH_REPO",
    "DANBOORU_TAGS_URL",
    "DANBOORU_WIKI_FILE",
    "DANBOORU_WIKI_REPO",
    "DEFAULT_SAM3_CHECKPOINT",
    "DEFAULT_SUBJECT_PROMPT_EMBED",
    "PACKS",
    "PACK_BY_ID",
    "PE_SPATIAL_FILENAME",
    "PE_SPATIAL_REPO",
    "SAM3_DIR",
    "SAM3_FILENAME",
    "SAM3_REPO",
    "SFX_READER_ADAPTER_FILES",
    "SFX_READER_DIR",
    "SFX_READER_FILES",
    "SFX_READER_LM_FILE",
    "SFX_READER_REPO",
    "SFX_READER_REVISION",
    "SFX_READER_SUBFOLDER",
    "SOFT_PROMPT_DIR",
    "SOFT_PROMPT_FILENAME",
    "SOFT_PROMPT_GH_REPO",
    "SOFT_PROMPT_URL",
    "VL16_BASE_DIR",
    "VL16_BASE_FILES",
    "VL16_BASE_REPO",
    "Asset",
    "Pack",
    "build_parser",
    "by_id",
    "by_pack",
    "catalog",
    "default_animetext_dir",
    "default_pe_spatial_path",
    "default_sfx_reader_dir",
    "default_vl16_base_dir",
    "expand",
    "http_timeout",
    "main",
]
