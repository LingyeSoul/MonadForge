"""What a caption claims about its own layout — counts, panels, candidacy.

The text-only prefilter for the position-clause pipeline and the multiview
audit: how many bindable subjects the caption claims, whether it describes one
character drawn several times (``multiple views`` / a comic page), and whether
it is worth running a detector over at all. Pure stdlib — no model, no pixels.

Every function here takes a caption *or* an already-parsed one
(:func:`~anime_tools.captions.position_clauses.as_parsed`), because a caller
asks several of them about the same caption in a row: ``is_candidate`` asks
three and ``propose_for_image`` five more, and each one parsing the string for
itself is the same parse over and over.
"""

from __future__ import annotations

import re

from library.curation_compat.captions.position_clauses import ParsedCaption, as_parsed, flat_tag_set
from library.curation_compat.captions.taxonomy import count_of

# ``2koma`` / ``4koma`` name the panel count. Anchored, so the open-ended
# ``multiple 4koma`` does not match and stays unbounded.
KOMA_COUNT_RE = re.compile(r"^(\d+)koma$")
MULTI_VIEW_TAGS = frozenset({"multiple views", "multiple_views"})

# Panel layouts: a comic page draws the same character once per panel, so like
# `multiple views` its girls-count counts *characters*, not bindable subjects —
# `1girl, 2koma` is routinely two.
#
# `page number` is NOT in the set: it marks a scanned art-book page, not a
# layout, and every image it catches is a single illustration.
PANEL_LAYOUT_TAGS = frozenset(
    {
        "comic",
        "silent comic",
        "silent_comic",
        "sequential",
        "2koma",
        "3koma",
        "4koma",
        "multiple 4koma",
        "multiple_4koma",
    }
)

# Every layout tag that decouples the girls-count from the bindable-subject
# count.
LAYOUT_TAGS = MULTI_VIEW_TAGS | PANEL_LAYOUT_TAGS


def caption_subject_count(caption: str | ParsedCaption) -> int | None:
    """How many bindable subjects the caption itself claims, if it says.

    ``Ngirls`` gives a number; ``None`` means "more than one, count unknown",
    and the count-consistency check then trusts detection instead of skipping.

    A layout tag (:data:`LAYOUT_TAGS`) forces ``None`` even alongside a
    girls-count, because that count tags *characters* while each view/panel is
    its own bindable subject (``1girl, multiple views`` is routinely four).
    ``multiple girls`` / open-ended ``N+girls`` are ``None`` too.
    """
    tags = flat_tag_set(caption)
    if tags & LAYOUT_TAGS:
        return None
    return count_of(tags, "girl")


def caption_panel_ceiling(caption: str | ParsedCaption) -> int | None:
    """Most bindable subjects an ``Nkoma`` page can hold, or ``None`` if unbounded.

    A layout tag waives the count check entirely; this restores a backstop for a
    comic page. ``Nkoma`` names the panel count, so the ceiling is
    ``panels x (girls + boys)`` — generous by construction. Plain ``comic`` /
    ``multiple views`` carry no panel count and stay unbounded, as does any
    caption with an unknown term.
    """
    tags = flat_tag_set(caption)
    panels = [int(m.group(1)) for t in tags if (m := KOMA_COUNT_RE.match(t))]
    if not panels:
        return None
    girls = count_of(tags, "girl")
    boys = count_of(tags, "boy")
    if girls is None or boys is None:
        return None
    # A page with no counted character at all still draws somebody per panel.
    per_panel = max(girls + boys, 1)
    return max(panels) * per_panel


def caption_boy_count(caption: str | ParsedCaption) -> int | None:
    """How many *male* subjects the caption claims — the count check's slack.

    The SAM3 ``girl`` prompt does not reliably exclude males, so the count gate
    accepts the range ``girls .. girls + boys``. ``None`` = "some boys, count
    unknown", which drops the upper bound entirely.
    """
    return count_of(flat_tag_set(caption), "boy")


def is_repeated_subject_layout(caption: str | ParsedCaption) -> bool:
    """Is this one character drawn several times, rather than several characters?

    Any :data:`LAYOUT_TAGS` member says yes: whatever belongs to *her*
    discriminates nothing between panels, so ``ClauseVocabulary.select`` drops
    the whole class (``view_invariant``). The bag keeps every suppressed tag,
    so only the per-panel binding is lost.
    """
    return bool(flat_tag_set(caption) & LAYOUT_TAGS)


def is_candidate(caption: str | ParsedCaption) -> tuple[bool, str]:
    """Should this caption go through detection? Returns ``(ok, reason)``.

    "Already has clauses" is *position* clauses (a text clause binds no
    subject), which is what ``parsed.position_clauses`` is.
    """
    parsed = as_parsed(caption)
    if parsed.position_clauses:
        return False, "already-has-clauses"
    tags = parsed.tag_keys
    if tags & MULTI_VIEW_TAGS:
        return True, "multiple-views"
    if tags & PANEL_LAYOUT_TAGS:
        return True, "panel-layout"
    expected = caption_subject_count(parsed)
    if expected is None or expected > 1:
        return True, "multi-girl"
    return False, "single-subject"
