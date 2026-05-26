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

# ── Count tags ─────────────────────────────────────────────────────────────
# Matches ``1girl``, ``2girls``, ``1boy``, ``3others``, ``6+girls`` (booru "≥6
# of"), and ``multiple_girls`` / ``multiple girls`` (underscore or space). This
# is the tagger's long-standing definition; ``classify_people`` and the vocab
# categorizer both key off it. (The caption-index builder additionally treats
# ``no girls`` / ``no boys`` as part of the leading count band, but those are
# general tags that sit *after* ``@artist`` in practice, so they never reach the
# pre-artist character span — keeping them out of ``is_count_tag`` avoids
# mistyping them in the model vocab.)
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

# Pull the leading integer off a count tag like "3girls" / "6+girls" → 3 / 6.
_LEADING_INT_RE = re.compile(r"^(\d+)")


def is_count_tag(tag: str) -> bool:
    """True for people-count tags (``1girl``, ``2girls``, ``multiple_boys``…)."""
    return bool(_COUNT_RE.match(tag))


# ── Artist tags ────────────────────────────────────────────────────────────


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


# ── Rating literals ──────────────────────────────────────────────────────────
# Raw captions carry the *4-class* danbooru rating vocabulary. Use this to strip
# the leading rating band when parsing a caption. Note this is intentionally a
# superset of the tagger's MODEL-OUTPUT ratings
# (``library.captioning.anima_tagger.RATINGS``), which is the 3-class set after
# the ``questionable → sensitive`` collapse the head is trained on.
CAPTION_RATINGS = frozenset({"general", "sensitive", "questionable", "explicit"})
