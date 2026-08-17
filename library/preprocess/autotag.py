"""Batch auto-tagging: image → Anima Tagger → caption master.

The dataset-wide counterpart to the Dataset tab's single-image autotag button
(``scripts.anima_tagger.autotag``, which is stdout-only and tags one file).
This walks the resized tree and writes ``.txt`` sidecars back into the caption
**master** under ``image_dataset/`` — the same write target as
:mod:`library.preprocess.position_captions`, and for the same reason: the master
is what ``preprocess-captions`` mirrors and the TE step then encodes.

Three modes, because the right policy depends on whether the corpus is
hand-captioned:

``missing``
    Tag only images with no caption sidecar. Never touches an existing caption
    — the safe default, and the only mode that cannot lose hand-written text.
``merge``
    Tag everything, but only *append* tags the caption does not already carry.
    Position clauses are preserved verbatim and their tags count as present, so
    a merge after ``caption-position`` cannot re-flatten a bound tag back into
    the bag.
``overwrite``
    Replace the caption with the tagger's output. Destructive — an ``--apply``
    run in this mode discards hand-written captions.

**Dry-run is the default** (the caller passes ``apply``), mirroring the
position-clause pass. Caption edits do *not* invalidate the TE caches, so an
applied run must be followed by ``make preprocess-te``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from PIL import Image

from library.captioning.position_clauses import compose_caption, parse_caption
from library.captioning.taxonomy import RATING_LITERALS

MODES = ("missing", "merge", "overwrite")


@dataclass
class AutotagOptions:
    """Policy knobs for one batch autotag pass."""

    mode: str = "missing"
    # Extra probability floor on top of the tagger's per-tag F1 thresholds.
    # Forwarded to ``predict_caption``; 0.0 leaves its own decisions untouched.
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence!r}"
            )


@dataclass
class AutotagProposal:
    """One image's before/after, whether or not it was written."""

    image: str = ""
    caption_path: str = ""
    existing: str = ""
    proposed: str = ""
    added: tuple[str, ...] = ()
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class AutotagStats:
    seen: int = 0
    candidates: int = 0
    proposed: int = 0
    written: int = 0
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def _has_rating(tags: tuple[str, ...]) -> bool:
    return any(t.strip().lower() in RATING_LITERALS for t in tags)


def merge_tags(existing: str, tagged: str) -> tuple[str, tuple[str, ...]]:
    """Append the tagger's novel tags to ``existing``, preserving its clauses.

    Returns ``(caption, added)``. A tag already bound inside a position clause
    counts as present, so merging after ``caption-position`` never duplicates a
    clause tag back into the flat bag. The tagger always emits a rating first;
    it is dropped when the caption already carries one, since two rating tokens
    are a contradiction rather than an addition.
    """
    parsed = parse_caption(existing)
    seen = {t.strip().lower() for t in parsed.flat_tags}
    for clause in parsed.clauses:
        seen.update(t.strip().lower() for t in clause.tags)

    keep_rating = not _has_rating(parsed.flat_tags)
    added: list[str] = []
    for index, tag in enumerate(t.strip() for t in tagged.split(",")):
        if not tag:
            continue
        low = tag.lower()
        if low in seen:
            continue
        # ``predict_caption`` always puts the rating in slot 0.
        if index == 0 and low in RATING_LITERALS and not keep_rating:
            continue
        seen.add(low)
        added.append(tag)

    if not added:
        return existing, ()
    return compose_caption((*parsed.flat_tags, *added), parsed.clauses), tuple(added)


def run_autotag_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    tag_fn: Callable[[Image.Image], str],
    options: AutotagOptions | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[AutotagProposal], AutotagStats]:
    """Walk the resized tree, tag, and (with ``apply``) write the caption master.

    Tagging runs on the *resized* image because that is the pixel data training
    actually sees; the caption lands next to the original under ``source_dir``.
    ``tag_fn`` takes a PIL image and returns a caption string — normally
    ``AnimaTagger.predict_caption`` bound to ``min_confidence``.
    """
    from library.preprocess._dataset import walk_images

    options = options or AutotagOptions()
    stats = AutotagStats()
    rows: list[AutotagProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)

    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        caption_path = source_dir / rel
        if progress is not None:
            progress(index, len(images), str(rel))

        existing = ""
        if caption_path.exists():
            existing = caption_path.read_text(encoding="utf-8").strip()
        if existing and options.mode == "missing":
            stats.skip("has-caption")
            continue
        stats.candidates += 1

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        tagged = str(tag_fn(image)).strip()

        proposal = AutotagProposal(
            image=str(image_path.relative_to(resized_dir)),
            caption_path=str(rel),
            existing=existing,
        )
        if not tagged:
            proposal.status = "skip:no-tags"
            stats.skip("no-tags")
            rows.append(proposal)
            continue

        if options.mode == "merge" and existing:
            proposal.proposed, proposal.added = merge_tags(existing, tagged)
        else:
            proposal.proposed = tagged
            proposal.added = tuple(t.strip() for t in tagged.split(",") if t.strip())

        if proposal.proposed == existing:
            proposal.status = "skip:unchanged"
            stats.skip("unchanged")
            rows.append(proposal)
            continue

        rows.append(proposal)
        stats.proposed += 1
        if apply:
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption_path.write_text(proposal.proposed, encoding="utf-8")
            stats.written += 1

    return rows, stats


def build_tag_fn(
    ckpt_dir: str | Path | None = None,
    device: str = "cuda",
    min_confidence: float = 0.0,
) -> tuple[Callable[[Image.Image], str], Mapping[str, object]]:
    """Load the Anima Tagger and return ``(tag_fn, info)``.

    Torch-importing, so it lives behind a function rather than at module import
    — ``run_autotag_captions`` itself is torch-free and unit-testable with a
    stub ``tag_fn``. The checkpoint is auto-fetched when absent, matching the
    Dataset tab's one-click autotag.
    """
    from library.captioning.anima_tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )
    from library.env import resolve_under_home

    resolved = resolve_under_home(str(ckpt_dir or DEFAULT_TAGGER_DIR))
    ensure_tagger_checkpoint(resolved)
    tagger = AnimaTagger(ckpt_dir=resolved, device=device)

    def tag_fn(image: Image.Image) -> str:
        return tagger.predict_caption(image, min_confidence=float(min_confidence))

    return tag_fn, {"tagger_dir": str(resolved), "device": device}
