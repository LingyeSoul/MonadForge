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

# Clause headers the convention uses, in canonical emission form. ``In the`` is
# accepted on read (it appears in a handful of hand-written captions for scene
# regions) but never emitted.
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

    @property
    def tag_keys(self) -> frozenset[str]:
        """The flat bag as a lookup set, lowercased and whitespace-stripped.

        Every "does the caption already say this?" test keys off this form —
        the rewrite compares crop tags against it, the prefilter matches count
        and layout tags in it. Shared so the normalization can't drift between
        the consumer that writes a tag and the one that looks it up.
        """
        return frozenset(t.strip().lower() for t in self.flat_tags)

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

    Deliberately case-SENSITIVE, unlike ``_CLAUSE_SPLIT_RE``. With no period to
    delimit it, a bare lowercase ``on the beach`` in the middle of a tag bag is
    a scene tag, not a clause header — only the capitalized canonical form is
    unambiguous enough to promote. The period-delimited form has the delimiter
    to go on, so :func:`parse_caption` accepts either case *there*.
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
        # of the split, so trust the position rather than re-testing the text —
        # the split regex is case-insensitive, and a hand-written
        # ``safe. on the left, red hair.`` used to fail the capitalized
        # ``is_clause_header`` here and yield ZERO clauses while
        # ``has_clauses`` said yes: the position pass then skipped the caption
        # as already annotated and the correction/variant passes shuffled the
        # lowercase header around as an ordinary flat tag. Within a segment only
        # the comma form can introduce a header, and that one stays strict.
        for j, part in enumerate(parts):
            header = (i > 0 and j == 0) or (
                is_clause_header(part) and (j == 0 or i == 0)
            )
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


def flat_tag_set(caption: str) -> frozenset[str]:
    """``caption``'s flat bag as a lookup set — clause tags excluded.

    Shorthand for ``parse_caption(caption).tag_keys``; see that property for why
    the normalization is shared.
    """
    return parse_caption(caption).tag_keys


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


# A lone box in a group is qualified with the end it hugs ("top left") only
# when it sits within _EDGE_HUG of that end of the subjects' combined extent
# AND leaves at least _EDGE_CLEAR of the other end empty — a full-height
# subject beside stacked panels stays bare "right", and two same-height girls
# stay bare "left"/"right" however their boxes wobble.
#
# _EDGE_CLEAR is calibrated on two curated judgment calls (2026-08-19), and
# the margin is thin: `9760121`'s top-left panel (bottom gap 0.488) reads
# "top left", while `6183990`'s near-identical layout (0.452) reads bare
# "left" — the intuition being "qualify only when the panel stops clearly
# above the halfway line". Both sit within box-jitter of the threshold; if a
# review sweep shows flapping, resolve it toward the user's calls above.
_EDGE_HUG = 0.15
_EDGE_CLEAR = 0.47


def _cluster_intervals(
    intervals: Sequence[tuple[float, float]], min_overlap: float
) -> list[int]:
    """Single-linkage grouping of 1-D intervals by fractional overlap.

    Two intervals share a group when they overlap by at least ``min_overlap``
    of the narrower one, and groups chain through a shared member (a
    full-height box bridges every panel it overlaps — which is exactly the
    signal that they are NOT stacked rows). Center-distance clustering, which
    this replaced, split a tall box from a short neighbour it overlapped by
    80%+, misnaming a magazine layout ``top``/``bottom``. Returns a group
    index per input, group 0 = lowest coordinate.
    """
    if not intervals:
        return []
    order = sorted(range(len(intervals)), key=lambda i: intervals[i])
    groups = [0] * len(intervals)
    group = 0
    lo, hi = intervals[order[0]]
    for cur in order[1:]:
        c0, c1 = float(intervals[cur][0]), float(intervals[cur][1])
        overlap = min(hi, c1) - max(lo, c0)
        narrow = min(hi - lo, c1 - c0)
        if narrow <= 0 or overlap < min_overlap * narrow:
            group += 1
            lo, hi = c0, c1
        else:
            lo, hi = min(lo, c0), max(hi, c1)
        groups[cur] = group
    return groups


# Two subjects share a lane (column) when their center-x gap is under
# _LANE_GAP of the NARROWER box's width. Deliberately not interval overlap:
# a wide panel whose content bleeds under the neighbouring subject (a leg
# drawn across the sheet) overlaps that subject's x-extent completely, but its
# center still sits squarely in its own lane — overlap-chaining glued such a
# layout into one column and degraded it to left/middle/right.
_LANE_GAP = 0.5


def _cluster_lanes(intervals: Sequence[tuple[float, float]]) -> list[int]:
    """Lane grouping of 1-D intervals by center gap vs the narrower width.

    Adjacent-in-center-order comparison, chained: a new lane starts when the
    gap to the previous interval's center exceeds ``_LANE_GAP`` of the
    narrower of the two. Returns a group index per input, lane 0 = leftmost.
    """
    if not intervals:
        return []
    order = sorted(
        range(len(intervals)),
        key=lambda i: (intervals[i][0] + intervals[i][1]) / 2,
    )
    groups = [0] * len(intervals)
    group = 0
    for prev, cur in zip(order, order[1:]):
        p0, p1 = intervals[prev]
        c0, c1 = intervals[cur]
        gap = (c0 + c1) / 2 - (p0 + p1) / 2
        narrow = min(p1 - p0, c1 - c0)
        if gap > _LANE_GAP * narrow:
            group += 1
        groups[cur] = group
    return groups


def _layout(boxes: Sequence[Sequence[float]], tol: float) -> tuple[str, list[int]]:
    """Group boxes along the axis that actually separates them.

    Rows first (grid sheets read row-major, matching the hand-written
    convention), by y-interval overlap — rows only exist where nothing spans
    them. When every box lands in one y-group — the magazine layout: a
    full-height subject bridging a column of stacked panels — the x-axis is
    grouped into center-gap lanes instead (see :func:`_cluster_lanes` for why
    not overlap). Returns ``("row"|"column", group_index_per_box)``.
    """
    y_groups = _cluster_intervals([(float(b[1]), float(b[3])) for b in boxes], tol)
    if max(y_groups) + 1 >= 2:
        return "row", y_groups
    x_groups = _cluster_lanes([(float(b[0]), float(b[2])) for b in boxes])
    return "column", x_groups


def _end_word(
    lo: float, hi: float, frame: tuple[float, float], words: tuple[str, str]
) -> str:
    """``top``/``bottom`` (or ``left``/``right``) for a box hugging one end of
    the subjects' combined extent, ``""`` for one that spans it."""
    f0, f1 = frame
    span = f1 - f0
    if span <= 0:
        return ""
    near0 = (lo - f0) / span
    near1 = (f1 - hi) / span
    if near0 <= _EDGE_HUG and near1 >= _EDGE_CLEAR:
        return words[0]
    if near1 <= _EDGE_HUG and near0 >= _EDGE_CLEAR:
        return words[1]
    return ""


def assign_positions(
    boxes: Sequence[Sequence[float]],
    size: tuple[int, int],
    *,
    row_tol: float = 0.25,
) -> list[str]:
    """Position phrase per box, ordered to match ``boxes``.

    Boxes are grouped by *interval overlap* (``row_tol`` = the minimum
    fractional overlap of the narrower extent), rows first: row groups are
    named ``top``/``bottom`` with left→right names inside a row, and a row's
    lone subject takes the bare row word — plus the side it hugs when it
    leaves the other side clear (a diagonal pair reads ``top left`` /
    ``bottom right``). When everything shares one row — the magazine layout: a
    full-height subject beside a column of stacked panels, which center-y
    clustering used to split into fake ``top``/``bottom`` rows — columns are
    named left→right instead, a stacked column takes ``top left``/``bottom
    left``, and the full-height subject stays bare ``right`` (an end-hugging
    lone panel is qualified: ``top left``). Degrades to the plain horizontal
    names when nothing separates, or when a grouping outgrows the row
    vocabulary (:data:`MAX_ROWS`).

    ``size`` is unused (the frame is the subjects' own combined extent) but
    kept for signature stability.
    """
    if not boxes:
        return []
    n = len(boxes)
    centers_x = [(float(b[0]) + float(b[2])) / 2 for b in boxes]
    centers_y = [(float(b[1]) + float(b[3])) / 2 for b in boxes]
    frame_x = (min(float(b[0]) for b in boxes), max(float(b[2]) for b in boxes))
    frame_y = (min(float(b[1]) for b in boxes), max(float(b[3]) for b in boxes))
    axis, groups = _layout(boxes, tol=row_tol)
    n_groups = max(groups) + 1

    def fallback() -> list[str]:
        names = horizontal_names(n)
        order = sorted(range(n), key=lambda i: centers_x[i])
        out = [""] * n
        for slot, idx in enumerate(order):
            out[idx] = names[slot]
        return out

    if n_groups == 1:
        return fallback()

    out = [""] * n
    if axis == "row":
        if n_groups > MAX_ROWS:
            return fallback()
        row_words = _ROW_WORDS[n_groups]
        for r in range(n_groups):
            members = sorted(
                (i for i in range(n) if groups[i] == r), key=lambda i: centers_x[i]
            )
            if len(members) == 1:
                # Bare "top" reads better than "top center"; the side is added
                # only when it genuinely places the subject ("top left").
                idx = members[0]
                side = _end_word(
                    float(boxes[idx][0]),
                    float(boxes[idx][2]),
                    frame_x,
                    ("left", "right"),
                )
                out[idx] = f"{row_words[r]} {side}".strip()
            else:
                names = horizontal_names(len(members))
                for slot, idx in enumerate(members):
                    out[idx] = f"{row_words[r]} {names[slot]}"
        return out

    by_column = [
        sorted((i for i in range(n) if groups[i] == c), key=lambda i: centers_y[i])
        for c in range(n_groups)
    ]
    if any(len(members) > MAX_ROWS for members in by_column):
        return fallback()
    col_names = horizontal_names(n_groups)
    for c, members in enumerate(by_column):
        if len(members) == 1:
            idx = members[0]
            vert = _end_word(
                float(boxes[idx][1]),
                float(boxes[idx][3]),
                frame_y,
                ("top", "bottom"),
            )
            out[idx] = f"{vert} {col_names[c]}".strip()
        else:
            row_words = _ROW_WORDS[len(members)]
            for slot, idx in enumerate(members):
                out[idx] = f"{row_words[slot]} {col_names[c]}"
    return out


def ordered_indices(
    boxes: Sequence[Sequence[float]], size: tuple[int, int], *, row_tol: float = 0.25
) -> list[int]:
    """Reading order for ``boxes`` — row-major (left→right within a row), or
    column-major (top→bottom within a column) on a magazine layout, matching
    the grouping :func:`assign_positions` names by."""
    if not boxes:
        return []
    centers_x = [(float(b[0]) + float(b[2])) / 2 for b in boxes]
    centers_y = [(float(b[1]) + float(b[3])) / 2 for b in boxes]
    axis, groups = _layout(boxes, tol=row_tol)
    # One undivided group means neither axis separates (nested/overlapping
    # boxes): assign_positions names that fallback left→right, so it must also
    # READ left→right — sorting it by center-y would open with "On the right".
    if axis == "row" or max(groups) == 0:
        minor = centers_x
    else:
        minor = centers_y
    return sorted(range(len(boxes)), key=lambda i: (groups[i], minor[i]))
