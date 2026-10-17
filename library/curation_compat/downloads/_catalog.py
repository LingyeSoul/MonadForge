"""The catalog: one :class:`~anime_tools.downloads._assets.Asset` row per model
the stages need, and the lookups a caller addresses them by. Torch-free.

:func:`catalog` is rebuilt per call rather than held as a constant, because both
things a row resolves against move at runtime: the curation home
(``ANIME_TOOLS_HOME``) and the dbv4 backbone repo, which follows whichever
checkpoint is installed. Nothing here is auto-run; loaders still fetch on first
use, and this only moves the wait somewhere the user chose.

Row ids and pack ids are interchangeable wherever one is accepted —
:func:`expand` is the one place that turns either into row ids.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from library.curation_compat._env import models_dir, resolve_path
from library.curation_compat.captions.correction import TAG_CSV_EN_NAME, TAG_CSV_NAME
from library.curation_compat.downloads._assets import PACKS, Asset
from library.curation_compat.downloads._locations import (
    ANIMETEXT_FILES,
    ANIMETEXT_REPO,
    ANIMETEXT_SUBFOLDER,
    DANBOORU_TAGS_GH_REPO,
    DANBOORU_TAGS_URL,
    DANBOORU_WIKI_FILE,
    DANBOORU_WIKI_REPO,
    PE_SPATIAL_FILENAME,
    PE_SPATIAL_REPO,
    SAM3_DIR,
    SAM3_FILENAME,
    SAM3_REPO,
    SFX_READER_FILES,
    SFX_READER_REPO,
    SFX_READER_REVISION,
    SFX_READER_SUBFOLDER,
    SOFT_PROMPT_DIR,
    SOFT_PROMPT_FILENAME,
    SOFT_PROMPT_GH_REPO,
    SOFT_PROMPT_URL,
    VL16_BASE_FILES,
    VL16_BASE_REPO,
    default_animetext_dir,
    default_pe_spatial_path,
    default_sfx_reader_dir,
    default_vl16_base_dir,
)
from library.curation_compat.tagger.dbv4_meta import (
    DBV4_BACKBONE_FILES,
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    DEFAULT_TAGGER_DIR,
    TAGGER_HF_REPO,
    TAGGER_HF_SUBFOLDER,
    backbone_repo_for,
)

__all__ = ["by_id", "by_pack", "catalog", "expand"]


def _build_english_tag_csv(dest: Path, log: Callable[[str], None]) -> None:
    """``danbooru_tags_en``'s post-fetch step: join the cached wiki parquet
    against the base CSV and write the English CSV."""
    from library.curation_compat.tagger.cli.build_english_tag_csv import build

    build(dest / TAG_CSV_NAME, dest / TAG_CSV_EN_NAME, revision=None, log=log)


def catalog() -> tuple[Asset, ...]:
    """The full catalog, rebuilt per call: the home moves with
    ``ANIME_TOOLS_HOME`` and the backbone repo follows the installed
    checkpoint."""
    tagger_dir = resolve_path(DEFAULT_TAGGER_DIR)
    backbone = backbone_repo_for(tagger_dir)
    return (
        Asset(
            id="tagger",
            pack="tagger",
            title="Anima Tagger checkpoint",
            repo=TAGGER_HF_REPO,
            subfolder=TAGGER_HF_SUBFOLDER,
            files=DBV4_REQUIRED_FILES,
            optional=DBV4_OPTIONAL_FILES,
            dest=tagger_dir,
            used_by="Autotag captions · Position captions · Multiview audit",
            stages=("autotag", "position", "audit"),
            notes="Vocab, rules, groups, thresholds, sidecar. Small — the "
            "weights are the backbone below.",
        ),
        Asset(
            id="tagger_backbone",
            pack="tagger",
            title="dbv4 tagger backbone",
            repo=backbone,
            files=DBV4_BACKBONE_FILES,
            used_by="the Anima Tagger checkpoint above",
            stages=("autotag", "position", "audit"),
            gated=f"https://huggingface.co/{backbone}",
            notes="GPL-3.0 and never vendored; the terms auto-approve.",
        ),
        Asset(
            id="sam3",
            pack="masking",
            title="SAM 3",
            repo=SAM3_REPO,
            files=(SAM3_FILENAME,),
            dest=resolve_path(SAM3_DIR),
            used_by="Position captions · Multiview audit · SAM3 subject masks",
            stages=("position", "audit", "masks_sam"),
            gated=f"https://huggingface.co/{SAM3_REPO}",
            notes="Lands on the --checkpoint default of every SAM3 stage.",
        ),
        Asset(
            id="pe_spatial",
            pack="grouping",
            title="PE-Spatial-B16-512",
            repo=PE_SPATIAL_REPO,
            files=(PE_SPATIAL_FILENAME,),
            dest=default_pe_spatial_path().parent,
            used_by="Build groups (near-twin / same-concept grouping)",
            stages=("groups",),
        ),
        Asset(
            id="soft_prompt",
            pack="masking",
            title="SAM3 subject soft prompt",
            repo=SOFT_PROMPT_GH_REPO,
            url=SOFT_PROMPT_URL,
            files=(SOFT_PROMPT_FILENAME,),
            dest=resolve_path(SOFT_PROMPT_DIR),
            used_by="Position captions · Multiview audit (subject detection)",
            stages=("position", "audit"),
            notes="The default --prompt_embed (161 KB); without it both "
            "stages fall back to the prompt `girl` and find fewer subjects.",
        ),
        Asset(
            id="danbooru_tags",
            pack="tags",
            title="Danbooru tag KB",
            repo=DANBOORU_TAGS_GH_REPO,
            url=DANBOORU_TAGS_URL,
            files=(TAG_CSV_NAME,),
            dest=models_dir(),
            used_by="Correct + mirror captions · the caption panel's tag descriptions",
            stages=("correct",),
            notes="~114k tags with category, post count and a wiki blurb. "
            "Blurbs are Korean; the row below rewrites them in English.",
        ),
        Asset(
            id="danbooru_tags_en",
            pack="tags",
            title="Danbooru tag descriptions (English)",
            repo=DANBOORU_WIKI_REPO,
            repo_type="dataset",
            files=(DANBOORU_WIKI_FILE,),
            derived=(TAG_CSV_EN_NAME,),
            build=_build_english_tag_csv,
            dest=models_dir(),
            used_by="the caption panel's tag descriptions",
            stages=(),
            notes="Optional; needs the row above. Rewrites its blurbs in "
            "English, which the caption panel prefers. The 45 MB mirror it "
            "joins stays in the hub cache.",
        ),
        Asset(
            id="vl16_base",
            pack="ocr",
            title="PaddleOCR-VL-1.6 base",
            repo=VL16_BASE_REPO,
            files=VL16_BASE_FILES,
            dest=default_vl16_base_dir(),
            used_by="OCR text (the manga VL reader's base; anime_tools.ocr.sfx)",
            stages=("ocr",),
            notes="1.9 GB, Apache-2.0. The VLM the SFX reader below is a "
            "fine-tune of; the reader loads this and merges the row below "
            "onto it.",
        ),
        Asset(
            id="sfx_reader",
            pack="ocr",
            title="Manga SFX reader v4 (VL-1.6 tower LoRA + LM)",
            repo=SFX_READER_REPO,
            subfolder=SFX_READER_SUBFOLDER,
            files=SFX_READER_FILES,
            revision=SFX_READER_REVISION,
            dest=default_sfx_reader_dir(),
            used_by="OCR text (the manga VL reader; anime_tools.ocr.sfx)",
            stages=("ocr",),
            notes="0.65 GB (tower LoRA 143 MB + fine-tuned LM layers 510 MB). Reads every "
            "box the detector finds — balloon speech, hearts, small kana and the "
            "hand-lettered onomatopoeia a print recognizer garbles; a crop reader, "
            "not a detector. Needs the base above; both fetched on first use. "
            "Trained on Manga109-s (COO) plus KO / ZH SFX; reads JA, KO and ZH.",
        ),
        Asset(
            id="animetext_det",
            pack="ocr",
            title="AnimeText text-block detector (YOLO12-l)",
            repo=ANIMETEXT_REPO,
            subfolder=ANIMETEXT_SUBFOLDER,
            files=ANIMETEXT_FILES,
            dest=default_animetext_dir(),
            used_by="OCR text (the text-block detector; anime_tools.ocr.animetext)",
            stages=("ocr",),
            notes="54 MB. One detector for balloon lines and the SFX drawn "
            "onto the artwork; every box it finds is read by the manga VL reader. "
            "Fetched on first use, never bundled: weights GPL-3.0, training data "
            "CC-BY-NC-SA-4.0 — a shipped build defaulting to it is a licence call.",
        ),
    )


def by_id(rows: tuple[Asset, ...] | None = None) -> dict[str, Asset]:
    """Rows keyed by id. ``rows`` defaults to the catalog."""
    return {a.id: a for a in (catalog() if rows is None else rows)}


def by_pack(rows: tuple[Asset, ...] | None = None) -> dict[str, tuple[Asset, ...]]:
    """Rows bucketed by pack, in :data:`PACKS` order, catalog order inside each;
    a pack with no rows is left out. ``rows`` defaults to the catalog."""
    rows = catalog() if rows is None else rows
    out: dict[str, tuple[Asset, ...]] = {}
    for pack in PACKS:
        picked = tuple(a for a in rows if a.pack == pack.id)
        if picked:
            out[pack.id] = picked
    return out


def expand(
    names: list[str] | tuple[str, ...], rows: tuple[Asset, ...] | None = None
) -> list[str]:
    """Row ids and/or pack ids → row ids, catalog order, deduped.

    Raises ``KeyError`` naming the first token that is neither, so a typo fails
    loudly rather than downloading nothing.

    Every lookup here takes ``rows``, because building the catalog resolves the
    curation home and probes the installed checkpoint for its backbone repo: a
    caller that needs two views of it builds it once and hands it down.
    """
    rows = catalog() if rows is None else rows
    assets = by_id(rows)
    packed = by_pack(rows)
    picked: list[str] = []
    for name in names:
        if name in packed:
            ids = [a.id for a in packed[name]]
        elif name in assets:
            ids = [name]
        else:
            raise KeyError(
                f"unknown model or pack id {name!r} — rows: {', '.join(assets)}; "
                f"packs: {', '.join(packed)}"
            )
        picked.extend(i for i in ids if i not in picked)
    order = list(assets)
    return sorted(picked, key=order.index)
