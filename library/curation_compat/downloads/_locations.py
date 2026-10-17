"""Where every model weight lives — the one spelling of each repo, filename and
destination path. Torch-free, and the leaf of this package: the loaders import
their repo / filename / default path from here (through
:mod:`anime_tools.downloads`, which re-exports it) so a Download button cannot
write where a loader will not look.

Destinations are of two kinds: a path under the curation home (``models/…``),
where a ``--checkpoint`` / ``--tagger_dir`` default points, so the file has to
land exactly there; or the HuggingFace hub cache, for assets whose loader
fetches by repo id — probed with :func:`anime_tools._hf.hf_file_cached`, never
by guessing a cache path.
"""

from __future__ import annotations

import os
from pathlib import Path

from library.curation_compat._env import models_dir, resolve_path

# SAM 3 — gated on the Hub. The CLIs pass ``--checkpoint`` with
# ``load_from_HF=False``, so the file must land on this path, not merely in the
# hub cache sam3's own downloader would use.
SAM3_REPO = "facebook/sam3"
SAM3_FILENAME = "sam3.pt"
SAM3_DIR = "models/sam3"
DEFAULT_SAM3_CHECKPOINT = f"{SAM3_DIR}/{SAM3_FILENAME}"

# PE-Spatial-B16-512 — weights for the grouping embedder's frozen tower
# (:mod:`anime_tools.vision.pe` builds the architecture).
PE_SPATIAL_REPO = "facebook/PE-Spatial-B16-512"
PE_SPATIAL_FILENAME = "PE-Spatial-B16-512.pt"

# SAM3 subject soft prompt — textual inversion of ``anime girl``, the default
# ``--prompt_embed`` of the position stage and the multiview audit. Not on the
# Hub, so it comes straight from GitHub.
SOFT_PROMPT_GH_REPO = "sorryhyun/anima_lora"
SOFT_PROMPT_DIR = "networks/calibration"
SOFT_PROMPT_FILENAME = "sam3_girl_prompt.safetensors"
DEFAULT_SUBJECT_PROMPT_EMBED = f"{SOFT_PROMPT_DIR}/{SOFT_PROMPT_FILENAME}"
SOFT_PROMPT_URL = (
    f"https://raw.githubusercontent.com/{SOFT_PROMPT_GH_REPO}/main/{SOFT_PROMPT_DIR}"
)

# PaddleOCR-VL-1.6 — the 0.9 B VLM base the SFX reader is fine-tuned from
# (Apache-2.0; ERNIE LM + NaViT tower). A flat checkpoint dir, so the remote
# modeling files ride along with the weights. `anime_tools.ocr.sfx` reads it.
VL16_BASE_REPO = "PaddlePaddle/PaddleOCR-VL-1.6"
VL16_BASE_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "configuration_paddleocr_vl.py",
    "image_processing_paddleocr_vl.py",
    "modeling_paddleocr_vl.py",
    "processing_paddleocr_vl.py",
)
VL16_BASE_DIR = "models/paddleocr_vl_1.6"

# The manga SFX reader, **v4** under the repo's ``v4/`` subfolder: a peft LoRA
# on the base's vision tower + projector plus its fully fine-tuned ERNIE
# decoder layers, trained on Manga109-s / COO and KO / ZH pseudo-labelled SFX
# (attribution and citations on the model card). Weights only — no Manga109-s
# image ships. The repo root still holds v3 (LM LoRA + ``tower.safetensors``).
SFX_READER_REPO = "sorryhyun/paddleocr-vl-1.6-manga-lora"
SFX_READER_SUBFOLDER = "v4"
SFX_READER_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
SFX_READER_LM_FILE = "lm.safetensors"
SFX_READER_FILES = (*SFX_READER_ADAPTER_FILES, SFX_READER_LM_FILE)
SFX_READER_DIR = "models/paddleocr_vl_1.6_manga_lora"
SFX_READER_REVISION = "926eafb13514baf26c8b3fea9eb4ed8c365013bf"
"""Hub commit of the reader the package ships — **v4** (``vl16_p2b_kozh``:
tower LoRA + LM fine-tune, 2 epochs, KO / ZH restored to the stock level;
the 1-epoch v4 was ``9fe3d41e``; v3 ``26292839``, v2 ``4caffe65``, v1
``3b5fe022``, all at the repo root). The
adapter file names are shared with v3, so the catalog row pins this and stamps
it under the dest — an install that predates the pin re-fetches instead of
reading a v3 adapter under a v4 name."""

# The AnimeText text-block detector: a YOLO12-l trained on deepghs/AnimeText
# (735k anime / manga pages, one class), the OCR stage's one detector.
# **Runtime download only, never bundled**: the model card is
# GPL-3.0 and the dataset CC-BY-NC-SA-4.0 — neither may ship inside this MIT
# package, the trainer, or a node. ``threshold.json`` carries the card's F1
# threshold (0.426) for reference; the detector's own default is lower.
ANIMETEXT_REPO = "deepghs/AnimeText_yolo"
ANIMETEXT_SUBFOLDER = "yolo12l_animetext"
ANIMETEXT_WEIGHTS = "model.pt"
ANIMETEXT_FILES = (ANIMETEXT_WEIGHTS, "threshold.json")
ANIMETEXT_DIR = "models/animetext"

# Danbooru tag KB — the ~114k-row classified tag table the correction pass types
# tags against and the GUI's tag panel reads. A CSV in a GitHub repo, so it
# rides ``_fetch_http``. Its descriptions are Korean; the English sibling is
# built, not hosted.
DANBOORU_TAGS_GH_REPO = "Localsmile/danbooru_KR_wiki_tag_search"
DANBOORU_TAGS_URL = f"https://raw.githubusercontent.com/{DANBOORU_TAGS_GH_REPO}/main"

# The Danbooru wiki mirrored as one parquet on the Hub — an input to the
# ``danbooru_tags_en`` row's join, so the 45 MB file stays in the hub cache.
DANBOORU_WIKI_REPO = "isek-ai/danbooru-wiki-2024"
DANBOORU_WIKI_FILE = "data/train-00000-of-00001.parquet"


def default_pe_spatial_path() -> Path:
    """``<models_dir>/pe/PE-Spatial-B16-512.pt``."""
    return models_dir() / "pe" / PE_SPATIAL_FILENAME


def default_vl16_base_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6`` — the SFX reader's base checkpoint."""
    return resolve_path(VL16_BASE_DIR)


def default_sfx_reader_dir() -> Path:
    """``<home>/models/paddleocr_vl_1.6_manga_lora`` — adapter + fine-tuned
    tower, what :class:`anime_tools.ocr.sfx.SfxReader` loads."""
    return resolve_path(SFX_READER_DIR)


def default_animetext_dir() -> Path:
    """``<home>/models/animetext`` — the AnimeText detector's weights, what
    :class:`anime_tools.ocr.animetext.AnimeTextDetector` loads."""
    return resolve_path(ANIMETEXT_DIR)


def http_timeout() -> float:
    """Socket timeout for the plain-HTTPS rows; shares ``ANIMA_HF_TIMEOUT``."""
    return float(os.environ.get("ANIMA_HF_TIMEOUT", "30"))
