"""Positional caption clauses — parse / compose / position vocabulary.

The dataset's hand-written convention for binding attributes to subjects is a
run of trailing clauses appended to the flat tag bag::

    safe, 3girls, akita neru, ..., white socks. On the left, akita neru,
    yellow eyes. On the middle, hatsune miku, twintails. On the right,
    kasane teto.

Structurally that is ``<flat tags>`` followed by one ``. On the <position>,
<tags>`` segment per subject, terminated by a period. The **period** is the
clause delimiter; commas separate tags *within* a segment. That distinction is
the whole reason this module exists: a naive ``caption.split(",")`` glues the
clause header onto the previous tag (``"white socks. On the left"``) and every
downstream consumer that keys off ``tag.startswith("On the ")`` then silently
fails to see any section at all — shredding the clauses under shuffle.

Pure stdlib by design (no torch, no model): the caption-variant generator, the
order-correction pass, and the auto-caption pipeline all parse clauses through
here so the three can't drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# Clause headers the convention uses. ``In the`` is accepted on read (it appears
# in a handful of hand-written captions for scene regions) but never emitted.
CLAUSE_PREFIXES = ("On the ", "In the ")

# ``. `` immediately before a clause header is the segment delimiter. Matched
# case-insensitively on read so a hand-written ``on the left`` still parses;
# emission always uses the canonical capitalized form.
_CLAUSE_SPLIT_RE = re.compile(r"\.\s*(?=(?:On|In)\s+the\s)", re.IGNORECASE)
_CLAUSE_HEADER_RE = re.compile(r"^(On|In)\s+the\s+(.+)$", re.IGNORECASE)

# Horizontal position words for N side-by-side subjects.
_ORDINALS = (
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
)
# Row words, indexed by (n_rows, row_index). Beyond 3 rows there is no natural
# vocabulary, so the caller falls back to pure left→right ordering.
_ROW_WORDS = {
    1: ("",),
    2: ("top", "bottom"),
    3: ("top", "middle", "bottom"),
}
MAX_ROWS = max(_ROW_WORDS)


@dataclass(frozen=True)
class PositionClause:
    """One ``On the <position>, <tags>`` segment.

    ``position`` is the bare position phrase (``"left"``, ``"top right"``);
    ``prefix`` is the header verb kept verbatim from the source so a round-trip
    of a hand-written ``In the background`` clause stays byte-stable.
    """

    position: str
    tags: tuple[str, ...]
    prefix: str = "On the "

    @property
    def header(self) -> str:
        return f"{self.prefix}{self.position}"

    def render(self) -> str:
        body = ", ".join(self.tags)
        return f"{self.header}, {body}" if body else self.header


@dataclass(frozen=True)
class ParsedCaption:
    """A caption split into its flat tag bag and its trailing position clauses."""

    flat_tags: tuple[str, ...]
    clauses: tuple[PositionClause, ...]

    @property
    def has_clauses(self) -> bool:
        return bool(self.clauses)

    def render(self) -> str:
        return compose_caption(self.flat_tags, self.clauses)


def _strip_trailing_period(tag: str) -> str:
    """Drop the caption-terminating ``.`` from the final tag of a segment.

    Guarded on the remainder still carrying an alphanumeric so punctuation-only
    booru tags (``:d``, ``>_<``, ``...``) survive intact.
    """
    if tag.endswith(".") and any(c.isalnum() for c in tag[:-1]):
        return tag[:-1].rstrip()
    return tag


def _split_tags(segment: str) -> list[str]:
    return [t for t in (raw.strip() for raw in segment.split(",")) if t]


def is_clause_header(tag: str) -> bool:
    """True for a bare clause header token (``"On the left"``).

    Used by consumers that already hold a comma-split token list — e.g. a
    caption written in the *comma* form (``…, white socks, On the left, …``),
    which :func:`parse_caption` also accepts.
    """
    return tag.startswith(CLAUSE_PREFIXES)


def has_clauses(caption: str) -> bool:
    """Cheap "does this caption already carry positional clauses?" check.

    The candidate prefilter uses it to leave hand-written captions alone.
    """
    return bool(_CLAUSE_SPLIT_RE.search(caption)) or any(
        is_clause_header(t) for t in _split_tags(caption)
    )


def parse_caption(caption: str) -> ParsedCaption:
    """Split ``caption`` into its flat tag bag and its position clauses.

    Accepts both written forms: the canonical period-delimited one and the
    comma form where the header happens to be its own comma token. A caption
    with no clauses round-trips to ``flat_tags`` alone, so callers can parse
    unconditionally.
    """
    tokens: list[tuple[str, bool]] = []  # (text, is_header)
    for i, segment in enumerate(_CLAUSE_SPLIT_RE.split(caption)):
        parts = _split_tags(segment)
        if not parts:
            continue
        # Every segment past the first starts at a clause header by construction
        # of the split; within a segment only the comma form can introduce one.
        for j, part in enumerate(parts):
            header = is_clause_header(part) and (j == 0 or i == 0)
            tokens.append((_strip_trailing_period(part), header))

    flat: list[str] = []
    clauses: list[list] = []
    for text, header in tokens:
        if header:
            m = _CLAUSE_HEADER_RE.match(text)
            prefix = f"{m.group(1).capitalize()} the " if m else "On the "
            position = m.group(2).strip() if m else text
            clauses.append([prefix, position, []])
        elif clauses:
            clauses[-1][2].append(text)
        else:
            flat.append(text)

    return ParsedCaption(
        flat_tags=tuple(flat),
        clauses=tuple(
            PositionClause(position=pos, tags=tuple(tags), prefix=prefix)
            for prefix, pos, tags in clauses
        ),
    )


def flatten_caption(caption: str) -> str:
    """Merge every clause's tags back into the flat bag, dropping the clauses.

    The inverse of the v2 rewrite, and the reason that rewrite is safe to apply
    in place: v2 *moves* a bound tag out of the bag rather than deleting it, so
    every tag is still in the caption — just inside a clause. Flattening
    recovers a clause-free caption (bag order first, then each clause left→right,
    duplicates dropped), which is both the undo for an ``--apply`` run and the
    control arm for a clause A/B.

    Order is *not* guaranteed byte-identical to the pre-rewrite caption — a moved
    tag comes back at the end rather than its original slot — but the tag *set*
    is exactly restored, and the order-correction pass re-buckets it anyway.
    """
    parsed = parse_caption(caption)
    seen: set[str] = set()
    flat: list[str] = []
    for tag in (*parsed.flat_tags, *(t for c in parsed.clauses for t in c.tags)):
        key = tag.strip().lower()
        if key and key not in seen:
            seen.add(key)
            flat.append(tag)
    return compose_caption(flat)


def compose_caption(
    flat_tags: Iterable[str], clauses: Iterable[PositionClause] = ()
) -> str:
    """Render a flat tag bag + clauses back into the hand-written convention.

    Inverse of :func:`parse_caption` (modulo whitespace normalization around
    commas). With no clauses this is a plain ``", "`` join, so it is safe to
    route every caption through it.
    """
    flat = ", ".join(t for t in flat_tags if t)
    parts = [c.render() for c in clauses if c.tags or c.position]
    if not parts:
        return flat
    body = ". ".join(parts) + "."
    return f"{flat}. {body}" if flat else body


# ---------------------------------------------------------------------------
# Position vocabulary
# ---------------------------------------------------------------------------


def horizontal_names(n: int) -> list[str]:
    """Left→right position words for ``n`` subjects in one row.

    ``left/right`` for a pair, ``left/middle/right`` for a trio, and
    ``leftmost / second from left / … / rightmost`` beyond that — the vocabulary
    the hand-written captions use.
    """
    if n <= 0:
        return []
    if n == 1:
        return ["center"]
    if n == 2:
        return ["left", "right"]
    if n == 3:
        return ["left", "middle", "right"]
    names = ["leftmost"]
    for i in range(1, n - 1):
        ordinal = _ORDINALS[i - 1] if i - 1 < len(_ORDINALS) else f"{i + 1}th"
        names.append(f"{ordinal} from left")
    names.append("rightmost")
    return names


def cluster_rows(
    centers_y: Sequence[float], height: float, tol: float = 0.25
) -> list[int]:
    """Assign each subject a row index by clustering its box center-y.

    Grid sheets (``multiple views`` 2×2 contact sheets) interleave under pure
    x-ordering, so rows are resolved first. Single-linkage on center-y with a
    gap threshold of ``tol × height``: subjects whose centers are closer than
    the threshold share a row. Returns a row index per input, top row = 0.
    """
    if not centers_y:
        return []
    gap = tol * float(height)
    order = sorted(range(len(centers_y)), key=lambda i: centers_y[i])
    rows = [0] * len(centers_y)
    row = 0
    rows[order[0]] = 0
    for prev, cur in zip(order, order[1:]):
        if centers_y[cur] - centers_y[prev] > gap:
            row += 1
        rows[cur] = row
    return rows


def assign_positions(
    boxes: Sequence[Sequence[float]],
    size: tuple[int, int],
    *,
    row_tol: float = 0.25,
) -> list[str]:
    """Position phrase per box, ordered to match ``boxes``.

    Rows are clustered first (``top``/``bottom``), then each row is named
    left→right. With one row — or more rows than the row vocabulary covers
    (:data:`MAX_ROWS`) — this degrades to the plain horizontal names, which is
    also the ordering the hand-written captions use for side-by-side subjects.
    """
    if not boxes:
        return []
    width, height = size
    centers_x = [(float(b[0]) + float(b[2])) / 2 for b in boxes]
    centers_y = [(float(b[1]) + float(b[3])) / 2 for b in boxes]
    rows = cluster_rows(centers_y, height, tol=row_tol)
    n_rows = max(rows) + 1

    if n_rows == 1 or n_rows > MAX_ROWS:
        names = horizontal_names(len(boxes))
        order = sorted(range(len(boxes)), key=lambda i: centers_x[i])
        out = [""] * len(boxes)
        for slot, idx in enumerate(order):
            out[idx] = names[slot]
        return out

    row_words = _ROW_WORDS[n_rows]
    out = [""] * len(boxes)
    for r in range(n_rows):
        members = sorted(
            (i for i in range(len(boxes)) if rows[i] == r), key=lambda i: centers_x[i]
        )
        # A row holding a single subject reads better as bare "top" than
        # "top center" — the row word alone is already unambiguous.
        names = horizontal_names(len(members))
        for slot, idx in enumerate(members):
            out[idx] = (
                row_words[r] if len(members) == 1 else f"{row_words[r]} {names[slot]}"
            )
    return out


def ordered_indices(
    boxes: Sequence[Sequence[float]], size: tuple[int, int], *, row_tol: float = 0.25
) -> list[int]:
    """Reading order (row-major, left→right within a row) for ``boxes``."""
    if not boxes:
        return []
    _, height = size
    centers_x = [(float(b[0]) + float(b[2])) / 2 for b in boxes]
    centers_y = [(float(b[1]) + float(b[3])) / 2 for b in boxes]
    rows = cluster_rows(centers_y, height, tol=row_tol)
    return sorted(range(len(boxes)), key=lambda i: (rows[i], centers_x[i]))
