"""Positional caption clauses — parse / compose / position vocabulary.

The caption grammar lives here. A caption is a flat tag bag followed by trailing
clauses that bind attributes to subjects::

    safe, 3girls, akita neru, ..., white socks. On the left, akita neru,
    yellow eyes. On the middle, hatsune miku, twintails. On the right,
    kasane teto.

GOTCHA: the **period** is the clause delimiter; commas separate tags *within* a
segment. A naive ``caption.split(",")`` glues the header onto the previous tag
(``"white socks. On the left"``), shredding the clause. Pure stdlib.

Quoted lines (2026-09-05): a tag may be a quoted text line — ``「大丈夫、本当に」``,
``『…』`` or ASCII ``"Are you okay? I'm fine, really."`` — and a comma or
``. On the`` *inside an open quote pair* is content, not a separator. The
quote pairs are :data:`QUOTE_PAIRS`; an opener with no closer is plain text.

Text clauses (2026-09-05): a caption may end with the page's OCR'd lines as a
sentence of its own kind, ``Japanese text reads as "…", "…"`` (or ``Japanese
SFX reads as …``) — :data:`TEXT_PREFIXES`. It parses as a
:class:`PositionClause` with an empty ``position`` and the quoted lines as its
tags, in reading order; :func:`compose_caption` renders text clauses last,
after every position clause, and the variants pass keeps them whole.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from library.curation_compat.captions.taxonomy import is_artist_tag, normalize_tag

# Clause headers, in canonical emission form. ``In the`` is accepted on read
# (hand-written captions use it for scene regions) but never emitted.
CLAUSE_PREFIXES = ("On the ", "In the ")

# Text-clause headers: the sentence that carries a page's OCR'd lines as quoted
# tags (``Japanese text reads as "…", "…"``). Exact, case-sensitive — the
# trainer emits the canonical form and nothing hand-written collides with it.
# The ``order`` phrase (``Japanese text in following order: …``) is a flat tag.
TEXT_PREFIXES = ("Japanese text reads as ", "Japanese SFX reads as ")

# ``. `` immediately before a clause header is the segment delimiter. Position
# headers are matched case-insensitively on read so a hand-written ``on the
# left`` still parses; text headers exactly.
_POSITION_SPLIT_RE = re.compile(r"\.\s*(?=(?:On|In)\s+the\s)", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(
    r"\.\s*(?=(?i:(?:On|In)\s+the\s)|"
    + "|".join(re.escape(p) for p in TEXT_PREFIXES)
    + ")"
)

# Quote pairs whose content is opaque to the grammar (no tag / clause split
# inside). Same three pairs the trainer's ext-vocab span rule recognises.
QUOTE_PAIRS: tuple[tuple[str, str], ...] = (("「", "」"), ("『", "』"), ('"', '"'))
_QUOTE_RE = re.compile(
    "|".join(
        f"{re.escape(o)}[^{re.escape(o)}{re.escape(c)}]*?{re.escape(c)}"
        for o, c in QUOTE_PAIRS
    )
)


def quoted_spans(caption: str) -> tuple[tuple[int, int], ...]:
    """``[start, end)`` of every closed quote pair in ``caption``, delimiters
    included. Left to right, non-nesting; a stray opener matches nothing."""
    return tuple((m.start(), m.end()) for m in _QUOTE_RE.finditer(caption))


def _outside_quotes(caption: str):
    """Predicate: is offset ``i`` outside every closed quote pair?"""
    spans = quoted_spans(caption)
    if not spans:
        return lambda i: True
    return lambda i: not any(a <= i < b for a, b in spans)


def _find_outside(caption: str, ch: str, at: int, hi: int, ok) -> int:
    """``caption.find(ch, at, hi)`` skipping hits inside a quote pair."""
    while True:
        i = caption.find(ch, at, hi)
        if i < 0 or ok(i):
            return i
        at = i + 1


def _split_outside(caption: str, sep: str) -> list[str]:
    """``caption.split(sep)`` that leaves quoted content whole."""
    ok = _outside_quotes(caption)
    out: list[str] = []
    at = 0
    while True:
        i = _find_outside(caption, sep, at, len(caption), ok)
        if i < 0:
            out.append(caption[at:])
            return out
        out.append(caption[at:i])
        at = i + 1


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
# Row words, indexed by (n_rows, row_index). Beyond 3 rows the caller falls
# back to pure left→right ordering.
_ROW_WORDS = {
    1: ("",),
    2: ("top", "bottom"),
    3: ("top", "middle", "bottom"),
}
MAX_ROWS = max(_ROW_WORDS)


@dataclass(frozen=True)
class PositionClause:
    """One ``On the <position>, <tags>`` segment; ``prefix`` is the header verb
    kept verbatim, so ``In the background`` round-trips byte-stable.

    A **text clause** (``prefix`` in :data:`TEXT_PREFIXES`) has no position:
    its tags are the quoted lines and it renders ``Japanese text reads as
    "…", "…"`` — build one with :func:`text_clause`.
    """

    position: str
    tags: tuple[str, ...]
    prefix: str = "On the "

    @property
    def is_text(self) -> bool:
        return self.prefix in TEXT_PREFIXES

    @property
    def header(self) -> str:
        if self.is_text:
            return self.prefix.rstrip()
        return f"{self.prefix}{self.position}"

    def render(self) -> str:
        body = ", ".join(self.tags)
        if self.is_text:
            return f"{self.prefix}{body}" if body else self.header
        return f"{self.header}, {body}" if body else self.header


def text_clause(lines: Iterable[str], *, sfx: bool = False) -> PositionClause:
    """A text clause carrying ``lines`` in order, each as a ``"…"`` tag.

    An inner ASCII double quote would close the pair early, so it is
    rewritten to ``”``; commas and periods inside the pair are content.
    """
    prefix = TEXT_PREFIXES[1] if sfx else TEXT_PREFIXES[0]
    tags = tuple(f'"{ln.replace(chr(34), "”")}"' for ln in lines if ln)
    return PositionClause(position="", tags=tags, prefix=prefix)


@dataclass(frozen=True)
class ParsedCaption:
    """A caption split into its flat tag bag and its trailing position clauses."""

    flat_tags: tuple[str, ...]
    clauses: tuple[PositionClause, ...]

    @property
    def has_clauses(self) -> bool:
        """Anything beyond the flat bag — position *or* text clauses."""
        return bool(self.clauses)

    @property
    def position_clauses(self) -> tuple[PositionClause, ...]:
        return tuple(c for c in self.clauses if not c.is_text)

    @property
    def text_clauses(self) -> tuple[PositionClause, ...]:
        return tuple(c for c in self.clauses if c.is_text)

    @property
    def tag_keys(self) -> frozenset[str]:
        """The flat bag as a lookup set, in :func:`normalize_tag` form.

        Every "does the caption already say this?" test keys off this form: the
        tagger emits ``speech bubble`` where a hand-written caption may hold
        ``speech_bubble``.
        """
        return frozenset(normalize_tag(t) for t in self.flat_tags)

    def render(self) -> str:
        return compose_caption(self.flat_tags, self.clauses)


@dataclass(frozen=True)
class TagSpan:
    """One tag's half-open ``[start, end)`` slice of the raw caption string.

    Whitespace, the comma and the caption-terminating period fall outside the
    span, so it covers exactly the characters the parse reads as the tag.

    ``kind`` is ``"header"`` for a clause header (``On the left``), ``"artist"``
    for an Anima artist handle and ``"tag"`` otherwise; ``clause`` is ``-1`` in
    the flat bag and otherwise the index of the owning :class:`PositionClause`,
    a header carrying the index of the clause it opens.
    """

    start: int
    end: int
    text: str
    kind: str
    clause: int


def _strip_trailing_period(tag: str) -> str:
    """Drop the caption-terminating ``.`` from the final tag of a segment.

    Guarded on an alphanumeric remainder so punctuation-only booru tags
    (``:d``, ``>_<``, ``...``) survive intact; a closed quote pair counts as
    a remainder too, so a quoted ``"♡"`` at the caption's end sheds it.
    """
    if tag.endswith(".") and (
        any(c.isalnum() for c in tag[:-1]) or tag[:-1].endswith(_QUOTE_CLOSERS)
    ):
        return tag[:-1].rstrip()
    return tag


_QUOTE_CLOSERS = tuple(c for _, c in QUOTE_PAIRS)


def _text_prefix(tag: str) -> str | None:
    """The :data:`TEXT_PREFIXES` entry ``tag`` opens with — as a whole
    comma-token (``Japanese text reads as "…"``) or as the bare header —
    else ``None``."""
    for prefix in TEXT_PREFIXES:
        if tag.startswith(prefix) or tag == prefix.rstrip():
            return prefix
    return None


def _split_tags(segment: str) -> list[str]:
    return [t for t in (raw.strip() for raw in _split_outside(segment, ",")) if t]


def is_position_header(tag: str) -> bool:
    """True for a bare position-clause header token (``"On the left"``)."""
    return tag.startswith(CLAUSE_PREFIXES)


def is_clause_header(tag: str) -> bool:
    """True for a clause header token — a bare position header (``"On the
    left"``) or a text header with its first line (``Japanese text reads as
    "…"``).

    GOTCHA: case-SENSITIVE, unlike ``_CLAUSE_SPLIT_RE`` — with no period to
    delimit it, a lowercase ``on the beach`` mid-bag is a scene tag, not a
    header. The period-delimited form has the delimiter to go on, so
    :func:`parse_caption` accepts either case there.
    """
    return is_position_header(tag) or _text_prefix(tag) is not None


def has_clauses(caption: str) -> bool:
    """Cheap "does this caption already carry *positional* clauses?" check.

    A text clause alone does not count — it binds no subject, and every
    caller asks this to decide whether a position rewrite already happened.
    See :func:`has_text_clauses`.
    """
    outside = _outside_quotes(caption)
    return any(outside(m.start()) for m in _POSITION_SPLIT_RE.finditer(caption)) or any(
        is_position_header(t) for t in _split_tags(caption)
    )


def has_text_clauses(caption: str) -> bool:
    """Does ``caption`` carry a :data:`TEXT_PREFIXES` sentence?"""
    return any(c.is_text for c in parse_caption(caption).clauses)


def tag_spans(caption: str) -> tuple[TagSpan, ...]:
    """Every tag in ``caption``, in reading order, with where it sits.

    :func:`parse_caption` is written on top of this, so a span can never
    disagree with the parse about where a tag ends or which clause owns it.
    """
    spans: list[TagSpan] = []
    clause = -1

    outside = _outside_quotes(caption)

    def comma_parts(lo: int, hi: int) -> list[tuple[int, int]]:
        """``caption[lo:hi].split(",")`` as offsets — the pieces, not the text.
        A comma inside a quote pair is content and never splits."""
        out: list[tuple[int, int]] = []
        at = lo
        while True:
            comma = _find_outside(caption, ",", at, hi, outside)
            out.append((at, hi if comma < 0 else comma))
            if comma < 0:
                return out
            at = comma + 1

    # ``_CLAUSE_SPLIT_RE.split`` drops the delimiter it matched; walking the
    # matches instead keeps each segment's offset. A ``. On the`` inside a
    # quoted line is content too.
    bounds: list[tuple[int, int]] = []
    pos = 0
    for m in _CLAUSE_SPLIT_RE.finditer(caption):
        if not outside(m.start()):
            continue
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, len(caption)))

    for i, (seg_start, seg_end) in enumerate(bounds):
        j = 0  # index among the segment's *non-empty* tags, as ``_split_tags``
        for lo, hi in comma_parts(seg_start, seg_end):
            start, end = lo, hi
            while start < end and caption[start].isspace():
                start += 1
            while end > start and caption[end - 1].isspace():
                end -= 1
            if end == start:
                continue
            # ``_strip_trailing_period`` only ever trims from the right, so the
            # start holds and the shortened text re-fixes the end.
            text = _strip_trailing_period(caption[start:end])
            end = start + len(text)
            # GOTCHA: trust segment position, not ``is_clause_header``, for
            # whether a segment starts a clause — that check is case-sensitive,
            # and a hand-written ``safe. on the left, red hair.`` must not drop
            # its clause into the flat bag.
            header = (i > 0 and j == 0) or (
                is_clause_header(text) and (j == 0 or i == 0)
            )
            j += 1
            if header:
                clause += 1
                text_prefix = _text_prefix(text)
                if text_prefix is not None:
                    # The header is the prefix alone; the first quoted line
                    # riding the same comma-token is that clause's first tag.
                    head = text_prefix.rstrip()
                    spans.append(
                        TagSpan(start, start + len(head), head, "header", clause)
                    )
                    rest_raw = text[len(text_prefix) :]
                    rest = rest_raw.strip()
                    if rest:
                        rest_start = (
                            start
                            + len(text_prefix)
                            + (len(rest_raw) - len(rest_raw.lstrip()))
                        )
                        spans.append(
                            TagSpan(
                                rest_start, rest_start + len(rest), rest, "tag", clause
                            )
                        )
                    continue
            spans.append(
                TagSpan(
                    start=start,
                    end=end,
                    text=text,
                    kind="header"
                    if header
                    else ("artist" if is_artist_tag(text) else "tag"),
                    clause=clause,
                )
            )
    return tuple(spans)


def parse_caption(caption: str) -> ParsedCaption:
    """Split ``caption`` into its flat tag bag and its position clauses.

    Accepts the canonical period-delimited form and the comma form where the
    header is its own comma token. A caption with no clauses round-trips to
    ``flat_tags`` alone, so callers can parse unconditionally.
    """
    flat: list[str] = []
    clauses: list[list] = []
    for span in tag_spans(caption):
        if span.kind == "header":
            text_prefix = _text_prefix(span.text)
            if text_prefix is not None:
                clauses.append([text_prefix, "", []])
                continue
            m = _CLAUSE_HEADER_RE.match(span.text)
            prefix = f"{m.group(1).capitalize()} the " if m else "On the "
            position = m.group(2).strip() if m else span.text
            clauses.append([prefix, position, []])
        elif clauses:
            clauses[-1][2].append(span.text)
        else:
            flat.append(span.text)

    return ParsedCaption(
        flat_tags=tuple(flat),
        clauses=tuple(
            PositionClause(position=pos, tags=tuple(tags), prefix=prefix)
            for prefix, pos, tags in clauses
        ),
    )


def as_parsed(caption: str | ParsedCaption) -> ParsedCaption:
    """``caption``, parsed — or itself, when it is already a parse.

    The entry point for a chain of text-only questions about one caption. Each
    of them alone would call :func:`parse_caption`, so asking three of them
    parses the same string three times; taking the parse instead makes the
    chain one.
    """
    return caption if isinstance(caption, ParsedCaption) else parse_caption(caption)


def flat_tag_set(caption: str | ParsedCaption) -> frozenset[str]:
    """``caption``'s flat bag as a lookup set — clause tags excluded."""
    return as_parsed(caption).tag_keys


def flatten_caption(caption: str) -> str:
    """Merge every position clause's tags back into the flat bag, dropping
    the clauses. A text clause is not a binding and stays as it is.

    Bag order first, then each clause left-to-right, duplicates dropped.

    NOTE: order is not byte-identical to the pre-rewrite caption (a moved tag
    returns at the end, not its original slot), but the tag *set* is restored.
    """
    parsed = parse_caption(caption)
    seen: set[str] = set()
    flat: list[str] = []
    for tag in (
        *parsed.flat_tags,
        *(t for c in parsed.position_clauses for t in c.tags),
    ):
        # Same key as `tag_keys`: a tag moved out in space form must not come
        # back beside its own underscore spelling.
        key = normalize_tag(tag)
        if key and key not in seen:
            seen.add(key)
            flat.append(tag)
    return compose_caption(flat, parsed.text_clauses)


def compose_caption(
    flat_tags: Iterable[str], clauses: Iterable[PositionClause] = ()
) -> str:
    """Render a flat tag bag + clauses back into the hand-written convention.

    Inverse of :func:`parse_caption` modulo whitespace around commas; with no
    clauses it is a plain ``", "`` join. Text clauses render last, after
    every position clause, whatever order they were given in.
    """
    flat = ", ".join(t for t in flat_tags if t)
    clauses = list(clauses)
    ordered = [c for c in clauses if not c.is_text] + [c for c in clauses if c.is_text]
    parts = [c.render() for c in ordered if c.tags or c.position]
    if not parts:
        return flat
    body = ". ".join(parts) + "."
    return f"{flat}. {body}" if flat else body


# ---------------------------------------------------------------------------
# Position vocabulary
# ---------------------------------------------------------------------------


def horizontal_names(n: int) -> list[str]:
    """Left→right position words for ``n`` subjects in one row: ``left/right``,
    ``left/middle/right``, then ``leftmost / second from left / … /
    rightmost``."""
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
# AND leaves at least _EDGE_CLEAR of the other end empty.
#
# _EDGE_CLEAR is calibrated on two curated judgment calls, and the margin is
# thin: bottom gap 0.488 reads "top left", 0.452 reads bare "left".
_EDGE_HUG = 0.15
_EDGE_CLEAR = 0.47


def _cluster_intervals(
    intervals: Sequence[tuple[float, float]], min_overlap: float
) -> list[int]:
    """Single-linkage grouping of 1-D intervals by fractional overlap.

    Two intervals share a group when they overlap by at least ``min_overlap``
    of the narrower one, and groups chain through a shared member — a
    full-height box bridging every panel it overlaps is the signal that they are
    NOT stacked rows. Returns a group index per input, group 0 = lowest.
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


# Two subjects share a lane (column) when their center-x gap is under _LANE_GAP
# of the NARROWER box's width. Not interval overlap: a wide panel whose content
# bleeds under its neighbour overlaps that subject's x-extent completely while
# its center still sits in its own lane.
_LANE_GAP = 0.5


def _cluster_lanes(intervals: Sequence[tuple[float, float]]) -> list[int]:
    """Lane grouping of 1-D intervals by center gap vs the narrower width.

    A new lane starts when the gap to the previous interval's center exceeds
    ``_LANE_GAP`` of the narrower of the two. Lane 0 = leftmost.
    """
    if not intervals:
        return []
    order = sorted(
        range(len(intervals)),
        key=lambda i: (intervals[i][0] + intervals[i][1]) / 2,
    )
    groups = [0] * len(intervals)
    group = 0
    for prev, cur in itertools.pairwise(order):
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

    Rows first (grid sheets read row-major), by y-interval overlap. When every
    box lands in one y-group — the magazine layout: a full-height subject
    bridging a column of stacked panels — the x-axis is grouped into center-gap
    lanes instead. Returns ``("row"|"column", group_index_per_box)``.
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
    fractional overlap of the narrower extent), rows first: row groups are named
    ``top``/``bottom`` with left→right names inside a row, and a row's lone
    subject takes the bare row word plus the side it hugs when it leaves the
    other side clear (a diagonal pair reads ``top left`` / ``bottom right``).
    When everything shares one row — the magazine layout — columns are named
    left→right instead, a stacked column takes ``top left``/``bottom left``, and
    the full-height subject stays bare ``right``. Degrades to the plain
    horizontal names when nothing separates, or when a grouping outgrows the row
    vocabulary (:data:`MAX_ROWS`).

    ``size`` is unused — the frame is the subjects' own combined extent.
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
                # Bare "top", with a side added only when it genuinely places
                # the subject ("top left").
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
    """Reading order for ``boxes`` — row-major, or column-major on a magazine
    layout, matching the grouping :func:`assign_positions` names by."""
    if not boxes:
        return []
    centers_x = [(float(b[0]) + float(b[2])) / 2 for b in boxes]
    centers_y = [(float(b[1]) + float(b[3])) / 2 for b in boxes]
    axis, groups = _layout(boxes, tol=row_tol)
    # One undivided group means neither axis separates (nested/overlapping
    # boxes): assign_positions names that fallback left→right, so it must also
    # READ left→right.
    if axis == "row" or max(groups) == 0:
        minor = centers_x
    else:
        minor = centers_y
    return sorted(range(len(boxes)), key=lambda i: (groups[i], minor[i]))
