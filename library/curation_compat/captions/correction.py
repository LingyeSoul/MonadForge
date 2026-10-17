"""Caption ordering helpers for Anima-style Danbooru captions.

Pure-stdlib and GUI-safe: the tag database is loaded lazily from a local CSV
only when correction is requested.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from library.curation_compat._env import models_dir
from library.curation_compat.captions.position_clauses import (
    compose_caption as compose_position_caption,
)
from library.curation_compat.captions.position_clauses import (
    parse_caption as parse_position_caption,
)
from library.curation_compat.captions.tag_drop_groups import should_drop_tag
from library.curation_compat.captions.taxonomy import (
    canonical_rating,
    is_artist_tag,
    is_count_tag,
    is_rating_tag,
    normalize_tag,
)

_CATEGORY_RE = re.compile(r"^\s*\[([^\]]+)\]")

# :mod:`anime_tools.downloads` imports these for its catalog row, so the file a
# Download button fetches is the file this module looks for.
TAG_CSV_NAME = "danbooru_tags_classified.csv"
"""The base table: English tag names / categories, **Korean** descriptions."""
TAG_CSV_EN_NAME = "danbooru_tags_classified.en.csv"
"""Its English sibling, built by
``python -m anime_tools.tagger.cli.build_english_tag_csv``."""

# Fallback for this source tree; the GUI passes ROOT explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DANBOORU_NUMERIC_CATEGORIES = {
    "0": "general",
    "1": "artist",
    "3": "copyright",
    "4": "character",
    "5": "meta",
}

# Injected aesthetic/quality-conditioning tags. Not real booru tags (absent from
# the KB), so without this set they fall through to the general tail. Stored
# space-form because ``normalize_tag`` turns ``_``→`` `` before classification;
# ``score_N`` gets its underscore back on emit (``_quality_emit_form``).
# Emitted at the very front of the caption.
_SCORE_QUALITY_TAGS = frozenset(f"score {n}" for n in range(1, 10))
_QUALITY_TAGS = (
    frozenset(
        {
            "masterpiece",
            "best quality",
            "good quality",
            "normal quality",
            "low quality",
            "worst quality",
        }
    )
    | _SCORE_QUALITY_TAGS
)


@dataclass(frozen=True)
class CaptionCorrectionOptions:
    """User-visible caption correction switches."""

    insert_no_artist: bool = True
    validate_artist_tags: bool = False
    trigger_word: str = ""
    trigger_at_front: bool = False
    # Tag groups to strip from every caption: slugs from
    # ``tag_drop_groups.DROP_GROUP_PATHS`` or literal KB taxonomy-path prefixes.
    # Applied to the flat bag AND inside position clauses (a clause emptied by
    # the drop is removed). The trigger word and ``@no-artist`` never drop.
    drop_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaptionCorrectionResult:
    text: str
    changed: bool
    inserted_no_artist: bool
    unknown_tags: tuple[str, ...]
    dropped_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrectedTags:
    """What :func:`correct_tags` returns: the bag, still a tuple of tags."""

    tags: tuple[str, ...]
    inserted_no_artist: bool
    unknown_tags: tuple[str, ...]
    dropped_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TagInfo:
    name: str
    kind: str
    category_path: str
    description: str = ""
    post_count: int = 0


class TagKnowledgeBase:
    """Minimal Danbooru tag classifier backed by ``danbooru_tags_classified.csv``."""

    def __init__(self, tags: dict[str, TagInfo], source: Path):
        self.tags = tags
        self.source = source
        self._ranked: list[TagInfo] | None = None

    def classify(self, tag: str) -> str | None:
        info = self.tags.get(tag_key(tag))
        return info.kind if info else None

    def has_artist(self, tag: str) -> bool:
        bare = tag.removeprefix("@")
        return self.classify(bare) == "artist"

    def describe(self, tag: str) -> TagInfo | None:
        return self.tags.get(tag_key(tag))

    def ranked_infos(self) -> list[TagInfo]:
        """Every known tag ordered by Danbooru ``post_count`` (popular first)."""
        if self._ranked is None:
            self._ranked = sorted(
                self.tags.values(), key=lambda info: info.post_count, reverse=True
            )
        return self._ranked


def default_tag_csv_candidates(
    root: Path | None = None, lang: str | None = None
) -> list[Path]:
    """Likely local CSV locations, in priority order.

    The base CSV carries Korean descriptions. For a non-Korean ``lang`` the
    lookup prefers a same-language sibling ``danbooru_tags_classified.<lang>.csv``
    then the English one; the base file is the final fallback.
    """

    stems = [TAG_CSV_NAME]
    if lang and lang != "ko":
        if lang != "en":
            stems.insert(0, TAG_CSV_EN_NAME)
        stems.insert(0, f"danbooru_tags_classified.{lang}.csv")

    paths: list[Path] = []
    for stem in stems:
        if root is not None:
            paths.append(root / "models" / stem)
        # models_dir() honours ANIME_TOOLS_MODELS, where the `danbooru_tags`
        # download row puts the file.
        paths.append(models_dir() / stem)
        paths.append(_REPO_ROOT / "models" / stem)
    env = os.environ.get("ANIMA_DANBOORU_TAGS_CSV")
    if env:
        paths.append(Path(env))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def find_tag_csv(root: Path | None = None, lang: str | None = None) -> Path | None:
    for path in default_tag_csv_candidates(root, lang):
        if path.exists():
            return path
    return None


def load_tag_knowledge_base(path: str | Path) -> TagKnowledgeBase:
    # Hot path: ~114k rows, hence ``csv.reader`` rather than ``DictReader`` and
    # one ``normalize_tag`` per row.
    csv_path = Path(path)
    tags: dict[str, TagInfo] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        cols = {name: i for i, name in enumerate(header)}
        i_name = cols.get("name", 0)
        i_category = cols.get("category", 1)
        i_post = cols.get("post_count", 2)
        i_desc = cols.get("description", 3)
        ncols = len(header)
        for row in reader:
            if len(row) < ncols:
                continue
            raw_name = row[i_name].strip()
            if not raw_name:
                continue
            description = row[i_desc]
            match = _CATEGORY_RE.match(description)
            if match:
                category_path = normalize_tag(match.group(1).replace(">", " > "))
                body = description[match.end() :].strip()
            else:
                category_path = ""
                body = description.strip()
            name = normalize_tag(raw_name)
            tags[_key_from_normalized(name)] = TagInfo(
                name=name,
                kind=_kind_from_category(category_path, row[i_category]),
                category_path=category_path,
                description=body,
                post_count=_parse_post_count(row[i_post]),
            )
    return TagKnowledgeBase(tags=tags, source=csv_path)


def tag_key(tag: str) -> str:
    return _key_from_normalized(normalize_tag(tag))


def _key_from_normalized(name: str) -> str:
    """Lookup key from an already ``normalize_tag``-ed name (no re-normalize)."""
    name = name.removeprefix("@")
    return name.replace(" ", "_")


def correct_caption(
    text: str,
    kb: TagKnowledgeBase,
    *,
    options: CaptionCorrectionOptions | None = None,
) -> CaptionCorrectionResult:
    options = options or CaptionCorrectionOptions()
    # One parse, and it is the grammar's: the bag reaches :func:`correct_tags`
    # as the tuple ``parse_caption`` cut, never as text this module splits
    # again. A tag may legally contain a comma inside quotes (``sign reading
    # "stop, now"``), and a bare ``split(",")`` cuts it in two.
    #
    # Newlines first: a hand-written master may separate its tags with them and
    # the grammar does not — nothing this package writes has one inside a
    # caption, text clauses included, so the substitution can only ever turn a
    # legacy separator into the canonical one.
    parsed = parse_position_caption(text.replace("\n", ","))
    flat = correct_tags(parsed.flat_tags, kb, options=options)
    if parsed.has_clauses:
        # Bucket-reordering would dissolve a clause tag's binding back into the
        # bag, so only the flat bag is corrected and the clauses re-appended.
        clauses, clause_dropped = _drop_clause_tags(
            parsed.clauses, _group_dropper(kb, options.drop_groups)
        )
        corrected = compose_position_caption(list(flat.tags), clauses)
        dropped = (*flat.dropped_tags, *clause_dropped)
    else:
        corrected = ", ".join(flat.tags)
        dropped = flat.dropped_tags
    return CaptionCorrectionResult(
        text=corrected,
        changed=corrected != text.strip(),
        inserted_no_artist=flat.inserted_no_artist,
        unknown_tags=flat.unknown_tags,
        dropped_tags=dropped,
    )


@dataclass(frozen=True)
class GroupDropResult:
    text: str
    dropped_tags: tuple[str, ...] = ()


def drop_caption_groups(
    text: str,
    kb: TagKnowledgeBase,
    selectors: Iterable[str],
    *,
    keep: Iterable[str] = (),
) -> GroupDropResult:
    """Strip every tag under ``selectors`` from ``text``, and change nothing else.

    :func:`correct_caption`'s drop without the bucket reorder, trigger word and
    ``@no-artist`` it wraps around it: what is left keeps the order it was
    written in, so the one change a reader sees is the tags that went. A caption
    with nothing to drop comes back as it arrived — a re-compose would still
    respace its commas, and that is not a drop. ``keep`` names tags no selector
    may take, compared by :func:`tag_key`.
    """
    drops = _group_dropper(kb, selectors, keep)
    if drops is None:
        return GroupDropResult(text)
    parsed = parse_position_caption(text.replace("\n", ","))
    flat: list[str] = []
    dropped: list[str] = []
    for tag in parsed.flat_tags:
        (dropped if drops(tag) else flat).append(tag)
    clauses, clause_dropped = _drop_clause_tags(parsed.clauses, drops)
    if not dropped and not clause_dropped:
        return GroupDropResult(text)
    return GroupDropResult(
        compose_position_caption(flat, clauses), (*dropped, *clause_dropped)
    )


def correct_tags(
    raw_tags: Iterable[str],
    kb: TagKnowledgeBase,
    *,
    options: CaptionCorrectionOptions | None = None,
) -> CorrectedTags:
    """Correct a flat tag bag, tag by tag, with no caption text in between.

    This is the entry point for a caller that already holds the bag — the
    clause branch of :func:`correct_caption`, above. It never serialises the
    bag, so a tag containing a comma survives.
    """
    options = options or CaptionCorrectionOptions()
    tags = [normalize_tag(tag) for tag in raw_tags if tag.strip()]
    if not tags:
        return CorrectedTags(
            tags=(), inserted_no_artist=False, unknown_tags=(), dropped_tags=()
        )

    buckets: dict[str, list[str]] = {
        "front": [],
        "quality": [],
        "meta": [],
        "year": [],
        "safety": [],
        "count": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "general": [],
    }
    unknown: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    trigger = options.trigger_word.strip()
    trigger_key = tag_key(trigger) if trigger else ""

    for tag in tags:
        key = tag_key(tag)
        if trigger and key == trigger_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        if options.drop_groups and should_drop_tag(tag, kb, options.drop_groups):
            dropped.append(tag)
            continue
        kind = _classify_tag(tag, kb, options)
        if kind is None:
            unknown.append(tag)
            kind = "general"
        buckets[kind].append(tag)

    if trigger:
        if options.trigger_at_front:
            buckets["front"].append(trigger)
        else:
            buckets["artist"].insert(0, trigger)

    inserted_no_artist = False
    if options.insert_no_artist and not buckets["artist"]:
        buckets["artist"].append("@no-artist")
        inserted_no_artist = True

    ordered = [
        *buckets["front"],
        *buckets["quality"],
        *buckets["meta"],
        *buckets["year"],
        # Rewrite legacy booru ratings to Anima's band; dict.fromkeys drops the
        # duplicate a caption carrying both spellings collapses into.
        *dict.fromkeys(canonical_rating(t) or t for t in buckets["safety"]),
        *buckets["count"],
        *buckets["character"],
        *buckets["copyright"],
        *buckets["artist"],
        *buckets["general"],
    ]
    return CorrectedTags(
        tags=tuple(_quality_emit_form(tag) for tag in ordered),
        inserted_no_artist=inserted_no_artist,
        unknown_tags=tuple(unknown),
        dropped_tags=tuple(dropped),
    )


def _group_dropper(
    kb: TagKnowledgeBase, selectors: Iterable[str], keep: Iterable[str] = ()
) -> Callable[[str], bool] | None:
    """``tag -> drop it?`` over drop-group ``selectors``, or ``None`` when there
    is nothing to drop — the one rule the bag and the clauses are cut by."""
    sels = tuple(selectors)
    if not sels:
        return None
    kept = {tag_key(t) for t in keep if t.strip()}
    return lambda tag: (
        tag_key(tag) not in kept and should_drop_tag(normalize_tag(tag), kb, sels)
    )


def _drop_clause_tags(clauses, drops: Callable[[str], bool] | None):
    """Apply a group drop inside position clauses.

    Only the dropped tags are removed; a clause left with no tags is removed
    whole, since a bare ``On the left`` binds nothing. A text clause's tags
    are quoted lines, not taxonomy tags — it passes through untouched.
    """
    if drops is None:
        return list(clauses), ()
    from dataclasses import replace

    kept = []
    dropped: list[str] = []
    for clause in clauses:
        if clause.is_text:
            kept.append(clause)
            continue
        tags = []
        for tag in clause.tags:
            if drops(tag):
                dropped.append(tag)
            else:
                tags.append(tag)
        if tags:
            kept.append(replace(clause, tags=tuple(tags)))
    return kept, tuple(dropped)


def tag_kind(tag: str, kb: TagKnowledgeBase) -> str:
    """The :func:`correct_tags` bucket ``tag`` sorts into under default options
    (``quality`` / ``meta`` / ``year`` / ``safety`` / ``count`` / ``character``
    / ``copyright`` / ``artist`` / ``general``); unknown to the KB is
    ``general``."""
    kind = _classify_tag(normalize_tag(tag), kb, CaptionCorrectionOptions())
    return kind or "general"


def _classify_tag(
    tag: str, kb: TagKnowledgeBase, options: CaptionCorrectionOptions
) -> str | None:
    if tag == "@no-artist":
        return "artist"
    if is_artist_tag(tag):
        if not options.validate_artist_tags or kb.has_artist(tag):
            return "artist"
        return "general"
    # Front region, in emit order: quality → meta → year → safety.
    if tag in _QUALITY_TAGS:
        return "quality"
    if is_rating_tag(tag):
        return "safety"
    if _is_year_tag(tag):
        return "year"
    if is_count_tag(tag):
        return "count"
    if _is_commentary_tag(tag):
        return "general"  # demoted to the tail, not the early meta slot
    # Remaining Danbooru metadata-category tags (highres, translated, …) keep
    # their own "meta" slot ahead of year/safety.
    return kb.classify(tag)


def _parse_post_count(value: object) -> int:
    try:
        return int(str(value or "").strip() or 0)
    except ValueError:
        return 0


def _kind_from_category(category_path: str, numeric_category: str) -> str:
    numeric_kind = _DANBOORU_NUMERIC_CATEGORIES.get(numeric_category.strip())
    if numeric_kind is not None:
        return numeric_kind
    # Korean fallback path, for rows with no numeric Danbooru category.
    if category_path:
        root = category_path.split(">")[0].strip()
        if root == "작가":
            return "artist"
        if root == "캐릭터":
            return "character"
        if root == "작품":
            return "copyright"
        if root == "메타":
            return "meta"
        if category_path.startswith("인물 > 인원수"):
            return "count"
    return "general"


def _is_year_tag(tag: str) -> bool:
    return len(tag) == 4 and tag.isdigit() and 1900 <= int(tag) <= 2100


def _quality_emit_form(tag: str) -> str:
    """``score_N`` quality tags are restored to their underscore spelling (Pony
    convention); every other tag keeps the space form."""
    return tag.replace(" ", "_") if tag in _SCORE_QUALITY_TAGS else tag


def _is_commentary_tag(tag: str) -> bool:
    """Danbooru commentary metadata tags (``commentary request``, ``<language>
    commentary``, …) all carry the substring."""
    return "commentary" in tag
