"""Low-level danbooru tag-*shape* primitives.

These recognize the *form* of a tag (artist ``@``-prefix, count tag, raw rating
literal) without any vocab or model, and are shared by the tagger vocab build
(``tagger/cli/vocab.py::categorize``) and the caption index
(``captions/index.py``). Pure stdlib: importing this must NOT pull in torch.
Content-aware heuristics stay with the caption-index builder.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable

# People-count tags. "no girls"/"no boys" are out of the set: they sit after
# @artist so they never reach the pre-artist span, and including them would
# mistype them in the model vocab.
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

_LEADING_INT_RE = re.compile(r"^(\d+)")

_SPACE_RE = re.compile(r"\s+")

# The exact / open-ended halves of :data:`_COUNT_RE`, split apart so
# :func:`count_of` can pull the number and the noun out of the match. ``6+girls``
# reads as "unknown", not as 6.
_EXACT_COUNT_RE = re.compile(r"^(\d+)(girl|boy|other)s?$")
_OPEN_COUNT_RE = re.compile(r"^\d+\+(girl|boy|other)s?$")


def normalize_tag(tag: str) -> str:
    """A tag's canonical space form: trimmed, ``_``→`` ``, lowercased, collapsed.

    ``speech_bubble`` and ``Speech  Bubble`` both key ``speech bubble``. It is
    a *key* function that happens also to be the corrector's output form, so
    callers that only compare must not write the result back.
    """
    return _SPACE_RE.sub(" ", tag.strip().replace("_", " ").lower())


def is_count_tag(tag: str) -> bool:
    """True for people-count tags (``1girl``, ``2girls``, ``multiple_boys``…)."""
    return bool(_COUNT_RE.match(tag))


def classify_people(tags: Iterable[str]) -> int:
    """Derive the 8-class :data:`PEOPLE_COUNT_LABELS` index for a parsed-tag list.

    Buckets 0-7 are named by the ``return`` comments below; ``others`` counts
    ride into ``multi``, the head being girls/boys-shaped. Booru fires
    ``multiple_girls`` / ``multiple_boys`` at ≥2, not ≥3, so an explicit numeric
    count wins and ``multiple_*`` only floors at 2 when no numeric tag for that
    gender was seen. Tag order doesn't matter.
    """
    girls = boys = 0
    saw_multi_g = saw_multi_b = False
    saw_other = False
    for t in tags:
        if not is_count_tag(t):
            continue
        if t.startswith("multiple"):
            if "girl" in t:
                saw_multi_g = True
            elif "boy" in t:
                saw_multi_b = True
            elif "other" in t:
                saw_other = True
            continue
        m = _LEADING_INT_RE.match(t)
        if m is None:  # e.g. malformed; defensive
            continue
        n = int(m.group(1))
        if "girl" in t:
            girls = max(girls, n)
        elif "boy" in t:
            boys = max(boys, n)
        # "others" counts go to the "multi" indicator (no 7-bucket fit).
        elif "other" in t:
            saw_other = True
    # ``multiple_*`` only kicks in when the numeric tag is missing, and means
    # ≥2, not ≥3.
    if saw_multi_g and girls == 0:
        girls = 2
    if saw_multi_b and boys == 0:
        boys = 2
    if saw_other or girls >= 3 or boys >= 3 or (boys >= 2 and girls >= 2):
        return 7  # multi: 3+girls / 3+boys / 2g+2b+ / lonely multiple_* / Nothers
    if girls == 0 and boys == 0:
        return 0  # no_people (only when no count tag fired)
    if girls == 1 and boys == 0:
        return 1  # 1girl
    if girls == 1 and boys == 1:
        return 2  # 1girl_1boy
    if girls == 2 and boys == 0:
        return 3  # 2girls
    if girls == 2 and boys == 1:
        return 4  # 2girls_1boy
    if girls == 1 and boys == 2:
        return 5  # 2boys_1girl
    if girls == 0 and boys == 1:
        return 6  # 1boy
    return 7  # fallback (e.g. 0g/2b without "others")


def exact_count(tag: str, noun: str) -> int | None:
    """The number in an exact ``N<noun>s`` tag (``3girls`` → 3), else ``None``.

    Exact means countable: the open-ended ``6+girls`` and the ``multiple_*``
    implication answer ``None``.
    """
    m = _EXACT_COUNT_RE.match(normalize_tag(tag))
    return int(m.group(1)) if m is not None and m.group(2) == noun else None


def count_of(tags: Iterable[str], noun: str) -> int | None:
    """How many ``girl``/``boy``/``other`` subjects ``tags`` claims, if it says.

    * an integer — the largest exact ``N<noun>s`` count present, ``0`` when the
      bag carries no count for this noun at all
    * ``None`` — "more than one, count unknown": ``multiple girls`` or an
      open-ended ``6+girls`` with no exact count to override it.

    An exact count always wins over an open-ended or ``multiple_*`` tag. Tags
    are read through :func:`normalize_tag`, so ``multiple_girls`` /
    ``multiple girls`` agree.
    """
    plural = f"multiple {noun}s"
    exact: list[int] = []
    unknown = False
    for raw in tags:
        tag = normalize_tag(raw)
        if (n := exact_count(tag, noun)) is not None:
            exact.append(n)
        elif tag == plural or ((m := _OPEN_COUNT_RE.match(tag)) and m.group(1) == noun):
            unknown = True
    if exact:
        return max(exact)
    return None if unknown else 0


# ``solo`` rides alongside ``1girl``/``1boy`` when there is exactly one figure,
# so the single-subject predicate is "one of these fired, and no *other* count
# tag did". These names also match :data:`_COUNT_RE`, so they must be excluded
# from the multi test rather than merely not counted.
SINGLE_COUNT_NAMES = frozenset({"solo", "1girl", "1boy", "1other"})


def is_solo_names(names: Iterable[str]) -> bool:
    """True when a tag-name set describes a single subject — what
    ``softmax_when_solo`` groups gate on."""
    names = set(names)
    has_multi = any(n not in SINGLE_COUNT_NAMES and is_count_tag(n) for n in names)
    return bool(names & SINGLE_COUNT_NAMES) and not has_multi


def solo_multi_indices(vocab_tags: Iterable[dict]) -> tuple[set[int], set[int]]:
    """``(single_count_indices, multi_count_indices)`` over ``vocab.json`` rows.

    Same precedence as :func:`is_solo_names`: a single-count name is never also
    counted as multi.
    """
    single: set[int] = set()
    multi: set[int] = set()
    for t in vocab_tags:
        name = t["name"]
        idx = int(t["index"])
        if name in SINGLE_COUNT_NAMES:
            single.add(idx)
        elif is_count_tag(name):
            multi.add(idx)
    return single, multi


def is_artist_tag(tag: str) -> bool:
    """True for Anima artist tags: a leading ``@`` immediately followed by a
    non-whitespace character (``@sincos``, ``@sumiyao (amam)``).

    The non-whitespace guard excludes booru emoticons like ``@ @`` (``@_@``
    after ``_``→`` `` normalization), which are general tags.
    """
    return len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()


def strip_artist_prefix(tag: str) -> str:
    """Drop a leading ``@`` so the bare name can be looked up in a tag cache."""
    return tag.removeprefix("@")


def trailing_qualifier(tag: str) -> str | None:
    """The balanced trailing parenthetical of ``tag``, or ``None``.

    ``ayaka (genshin impact)`` → ``genshin impact``; nested handles stay whole:
    ``kuronuma mayu (kat (bu-kunn))`` → ``kat (bu-kunn)``.
    """
    t = tag.rstrip()
    if not t.endswith(")"):
        return None
    depth = 0
    for i in range(len(t) - 1, -1, -1):
        ch = t[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return t[i + 1 : -1].strip() or None
    return None


def artist_oc_handle(tag: str, artist_handles: Collection[str]) -> str | None:
    """The artist handle an original-character tag is qualified by, or ``None``.

    Booru spells an artist's own recurring OC as ``name (handle)`` —
    ``shiro (mignon)``, ``akiyama fumika (pepper0)`` — the same trailing
    parenthetical a franchise qualifier uses (``ayaka (genshin impact)``).
    What tells them apart is the qualifier being an *artist*: the bare handle
    of an ``@artist`` tag. ``artist_handles`` is that set, without the ``@``.
    The tag's own category is not consulted — booru mistypes some OCs as
    general, and the name is an OC either way.
    """
    handle = trailing_qualifier(tag)
    return handle if handle is not None and handle in artist_handles else None


def artist_handles_of(tags: Iterable[str]) -> frozenset[str]:
    """Bare handles of every artist tag in ``tags`` (``@mignon`` → ``mignon``)."""
    return frozenset(strip_artist_prefix(t) for t in tags if is_artist_tag(t))


# Anima's 4-class rating vocabulary — the leading safety band of a caption, and
# the set the tagger's rating head predicts (``tagger.RATINGS`` fixes the order).
CAPTION_RATINGS = frozenset({"safe", "sensitive", "nsfw", "explicit"})

# Danbooru's own rating literals mapped onto the Anima band, so raw booru
# captions read as ratings instead of falling through to the ``general``
# *category*.
LEGACY_RATING_ALIASES = {"general": "safe", "questionable": "nsfw"}

# Every literal that reads as a rating: canonical band + accepted legacy spellings.
RATING_LITERALS = CAPTION_RATINGS | frozenset(LEGACY_RATING_ALIASES)


def is_rating_tag(tag: str) -> bool:
    """True for any accepted rating literal (canonical Anima or legacy booru)."""
    return tag in RATING_LITERALS


def canonical_rating(tag: str) -> str | None:
    """The canonical Anima rating for ``tag``, or None when it isn't a rating."""
    if tag in CAPTION_RATINGS:
        return tag
    return LEGACY_RATING_ALIASES.get(tag)
