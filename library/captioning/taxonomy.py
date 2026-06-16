"""Low-level danbooru tag-*shape* primitives — the single source of truth.

These recognize the *form* of a tag (artist ``@``-prefix, count tag, raw rating
literal) without any vocab or model. They are shared by every consumer that
types tags so the two categorization paths can't silently drift:

* the Anima Tagger vocab build — ``scripts/anima_tagger/vocab.py::categorize``
  (image→tag model's view of the corpus), and
* the dataset caption index — ``scripts/preprocess/build_caption_index.py``
  (method-agnostic typed-tag index for identity pairing / analytics).

Pure stdlib by design: importing this must NOT pull in torch. The richer,
*content*-aware heuristics (vocab-membership classification, danbooru
``name (series)`` paren recovery, positional bare-name recovery) stay with the
caption-index builder — they exist to compensate for the tagger's frozen vocab
and have no model-side counterpart.
"""

from __future__ import annotations

import re

# People-count tags — shared definition that classify_people and the vocab categorizer both key off. The caption-index builder additionally counts "no girls"/"no boys", but those sit after @artist so they never reach the pre-artist span; keeping them out here avoids mistyping them in the model vocab.
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

_LEADING_INT_RE = re.compile(r"^(\d+)")


def is_count_tag(tag: str) -> bool:
    """True for people-count tags (``1girl``, ``2girls``, ``multiple_boys``…)."""
    return bool(_COUNT_RE.match(tag))


def is_artist_tag(tag: str) -> bool:
    """True for Anima artist tags: a leading ``@`` immediately followed by a
    non-whitespace character (``@sincos``, ``@sumiyao (amam)``).

    The non-whitespace guard excludes booru emoticons like ``@ @`` (``@_@``
    after ``_``→`` `` normalization), which are general tags, not artists.
    """
    return len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()


def strip_artist_prefix(tag: str) -> str:
    """Drop a leading ``@`` so the bare name can be looked up in a tag cache."""
    return tag[1:] if tag.startswith("@") else tag


# Raw-caption 4-class danbooru rating vocabulary (for stripping the leading rating band). Intentionally a superset of the tagger's 3-class MODEL-OUTPUT ratings (library.captioning.anima_tagger.RATINGS), which collapse questionable → sensitive.
CAPTION_RATINGS = frozenset({"general", "sensitive", "questionable", "explicit"})
