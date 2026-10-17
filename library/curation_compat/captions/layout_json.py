"""Anima caption → MOSS-Image layout-JSON prompt.

MOSS-Image reads a layout prompt as plain text: a compact JSON object whose
``summary.caption`` describes the scene and whose ``items`` carry one element
each (``label`` / ``caption`` / ``text``). This maps an Anima caption onto that
shape without inventing anything — every tag lands in exactly one field::

    sensitive, 1girl, hatsune miku, vocaloid, @sincos, long hair.
    On the left, necktie. Japanese text reads as "満足でしょうか"

    {"summary":{"meta":["sensitive"],"characters":["hatsune miku"],
     "series":["vocaloid"],"artists":["@sincos"],"caption":"1girl, long hair"},
     "items":[{"label":"subject","position":"left","caption":"necktie"},
              {"label":"text","caption":"Japanese text","text":"満足でしょうか"}]}

Position clauses keep their relative position word (no boxes: the position
stage does not persist them); a clause's character names move to the item's
``characters``. Each quoted line of a text clause is its own ``text`` item, in
reading order. Artists carry the Anima ``@`` (added to a bare KB artist name);
``@no-artist`` is an Anima sentinel and is left out. Danbooru meta tags that
describe the pixels (``cropped``, ``scan``, ``jpeg artifacts``, … —
``_VISUAL_META``) go to ``summary.caption``; the rest of the meta category stays
in ``summary.meta``. Empty fields are omitted, except ``summary.caption``.

The serialization is compact (``","`` / ``":"``, no ASCII escaping) like the
MOSS gallery prompts, and deterministic for a given caption and KB — training
and inference must produce the same string. Pure stdlib.
"""

from __future__ import annotations

import json

from library.curation_compat.captions.correction import TagKnowledgeBase, tag_kind
from library.curation_compat.captions.position_clauses import (
    TEXT_PREFIXES,
    PositionClause,
    as_parsed,
)
from library.curation_compat.captions.taxonomy import (
    canonical_rating,
    normalize_tag,
    strip_artist_prefix,
)

# correct_tags buckets that read as image metadata, in Anima emit order.
_META_KINDS = ("quality", "meta", "year", "safety")

# Danbooru meta-category tags that describe what is in the image, not the post.
_VISUAL_META = frozenset(
    {
        "cropped",
        "scan",
        "magazine scan",
        "tall image",
        "wide image",
        "jpeg artifacts",
        "lowres",
        "upscaled",
        "rotated",
        "color halftone",
        "colorized",
        "decensored",
        "hard-translated",
        "nude filter",
        "photo (medium)",
        "game cg",
    }
)

_TEXT_CAPTIONS = {
    TEXT_PREFIXES[0]: "Japanese text",
    TEXT_PREFIXES[1]: "Japanese sound effect",
}


def _summary(tags, kb: TagKnowledgeBase) -> dict:
    meta: dict[str, list[str]] = {k: [] for k in _META_KINDS}
    characters: list[str] = []
    series: list[str] = []
    artists: list[str] = []
    counts: list[str] = []
    general: list[str] = []
    for tag in tags:
        kind = tag_kind(tag, kb)
        if kind == "meta" and normalize_tag(tag) in _VISUAL_META:
            general.append(tag)
        elif kind in meta:
            meta[kind].append(canonical_rating(tag) or tag)
        elif kind == "character":
            characters.append(tag)
        elif kind == "copyright":
            series.append(tag)
        elif kind == "artist":
            if tag != "@no-artist":
                artists.append("@" + strip_artist_prefix(tag))
        elif kind == "count":
            counts.append(tag)
        else:
            general.append(tag)
    out: dict = {}
    if metas := [t for k in _META_KINDS for t in meta[k]]:
        out["meta"] = list(dict.fromkeys(metas))
    if characters:
        out["characters"] = characters
    if series:
        out["series"] = series
    if artists:
        out["artists"] = artists
    # Counts lead the description, as they lead an Anima bag.
    out["caption"] = ", ".join((*counts, *general))
    return out


def _subject_item(clause: PositionClause, kb: TagKnowledgeBase) -> dict:
    characters = [t for t in clause.tags if tag_kind(t, kb) == "character"]
    rest = [t for t in clause.tags if tag_kind(t, kb) != "character"]
    item: dict = {"label": "subject", "position": clause.position}
    if characters:
        item["characters"] = characters
    if rest:
        item["caption"] = ", ".join(rest)
    return item


def _unquote(line: str) -> str:
    # ``text_clause`` wraps each line in ASCII quotes; a 「…」 inside is content.
    if len(line) >= 2 and line[0] == '"' == line[-1]:
        return line[1:-1]
    return line


def _text_items(clause: PositionClause) -> list[dict]:
    caption = _TEXT_CAPTIONS.get(clause.prefix, "text")
    return [
        {"label": "text", "caption": caption, "text": _unquote(line)}
        for line in clause.tags
    ]


def caption_to_layout(caption: str, kb: TagKnowledgeBase) -> dict:
    """The layout object for one Anima caption (see the module docstring)."""
    parsed = as_parsed(caption.replace("\n", ","))
    items: list[dict] = []
    for clause in parsed.position_clauses:
        items.append(_subject_item(clause, kb))
    for clause in parsed.text_clauses:
        items.extend(_text_items(clause))
    layout: dict = {"summary": _summary(parsed.flat_tags, kb)}
    if items:
        layout["items"] = items
    return layout


def layout_json(caption: str, kb: TagKnowledgeBase) -> str:
    """:func:`caption_to_layout`, serialized the way MOSS prompts are."""
    return json.dumps(
        caption_to_layout(caption, kb), ensure_ascii=False, separators=(",", ":")
    )
