#!/usr/bin/env python3
"""Color-only caption filter for the colorization EasyControl task.

The colorization condition (mangafied B&W lineart + screentone) already encodes
*everything spatial* — composition, poses, objects, layout. The one variable it
cannot encode is **hue/chroma**. So the text channel for colorization should
carry *only* color information: every surviving token is then a color fact the
model can't get from structure, which gives a strong text→color coupling and
makes color prompts actually steer at inference.

:func:`filter_to_colors` reduces a full Anima caption to its color tags only:

  * hair / eye / skin color tags ("blue hair", "aqua eyes", "dark skin", plus
    the multi-color escapes: "two-tone hair", "heterochromia", …)
  * any ``<color> <noun>`` tag — clothing, objects, background
    ("yellow shirt", "red ribbon", "white background", "blue sky")
  * standalone palette descriptors ("monochrome", "colorful", "pastel colors")

Everything else is dropped. Images whose caption has no color tag collapse to an
empty caption (→ unconditional / auto-color sample), which is fine — that's the
empty-prompt colorization mode.

Tag order is preserved (the slot order is irrelevant once non-color tags are
gone). Pure stdlib so it stays importable from the preprocess path and unit
tests without pulling torch.

:func:`filter_to_colors_and_copyright` is the variant used by the colorize prep
when copyright tags should ride along with the color tags ("genshin impact,
pink hair, blue eyes"). The manga cond can't encode *which series* a page is
from, so copyright is genuinely-ambiguous text the model can bind to — and
unlike a hue it shouldn't be tag-dropped, so it's emitted as a protected prefix
(see ``protect_fn`` in :func:`library.preprocess.generate_caption_variants`).
Copyright tags are identified against the corpus copyright vocab in the caption
index (``post_image_dataset/captions/caption_index.json`` ``groups.copyright``).

:func:`filter_to_colors_and_protected` generalizes that prefix to any set of
genuinely-ambiguous tags: copyright (series identity) and **comic/panel-format**
tags (``comic``, ``4koma``, …). The mangafied cond carries lineart + screentone
but the multi-panel *page format* is text the model should bind so a "comic"
prompt produces panelled output — kept first and dropout-protected, like
copyright. Comic tags match a fixed booru format vocab (:data:`COMIC_TAGS`), no
caption-index lookup needed.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Callable

# Repo root: easycontrol_adapters/colorization/color_caption.py → ../../..
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTION_INDEX = (
    _REPO_ROOT / "post_image_dataset" / "captions" / "caption_index.json"
)

# Single color words that head a "<color> <noun>" tag or end a "<x> hair/eyes".
# Booru's stable color vocab + a few common extended hues. Kept deliberately
# broad on the permissive side: a false-keep ("blue rose") is still color info,
# a false-drop loses a hue the model needs.
COLOR_WORDS: frozenset[str] = frozenset(
    {
        "aqua",
        "black",
        "blonde",
        "blue",
        "brown",
        "green",
        "grey",
        "gray",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "white",
        "yellow",
        "tan",
        "teal",
        "cyan",
        "magenta",
        "maroon",
        "navy",
        "olive",
        "violet",
        "beige",
        "gold",
        "golden",
        "cream",
        "crimson",
        "scarlet",
        "turquoise",
        "lavender",
        "peach",
        "azure",
        "ivory",
        "bronze",
    }
)

# Leading modifiers that precede a color word ("light blue hair", "dark green
# dress", "pale ...").
COLOR_MODIFIERS: frozenset[str] = frozenset(
    {"light", "dark", "pale", "deep", "bright", "pastel"}
)

# Multi-word / non-"<color> X" tags that are still pure color information and
# should be kept whole.
COLOR_PHRASES: frozenset[str] = frozenset(
    {
        "monochrome",
        "greyscale",
        "grayscale",
        "colorful",
        "muted color",
        "pastel colors",
        "limited palette",
        "spot color",
        "rainbow order",
        # hair/eye multi-color escapes (booru groups.yaml escapes)
        "two-tone hair",
        "multicolored hair",
        "gradient hair",
        "streaked hair",
        "split-color hair",
        "rainbow hair",
        "colored inner hair",
        "multicolored eyes",
        "heterochromia",
        "two-tone eyes",
        # skin tone
        "dark skin",
        "pale skin",
        "tan",
        "sun tan",
        "tanlines",
        "tan lines",
        "dark-skinned female",
        "dark-skinned male",
    }
)


# Comic / panel-format tags. The mangafied cond synthesizes lineart + screentone
# but doesn't reliably hand the model the "this is a multi-panel comic page"
# signal, so — like copyright — it's genuinely-ambiguous text worth binding: kept
# as a protected, dropout-immune leading token so the adapter learns the `comic`
# tag and a "comic" prompt steers toward panelled output. A fixed booru
# format-tag vocab (no caption-index lookup, unlike copyright); the koma variants
# are the same comic-page family.
COMIC_TAGS: frozenset[str] = frozenset(
    {
        "comic",
        "manga",
        "doujinshi",
        "2koma",
        "3koma",
        "4koma",
        "5koma",
        "multiple 4koma",
    }
)


def is_comic_tag(tag: str) -> bool:
    """True if ``tag`` is a comic / panel-format tag worth keeping + binding."""
    return tag.strip().lower() in COMIC_TAGS


def is_color_tag(tag: str) -> bool:
    """True if ``tag`` carries color/hue information worth keeping."""
    t = tag.strip().lower()
    if not t:
        return False
    if t in COLOR_PHRASES:
        return True
    words = t.split()
    # "<...> hair/eyes/skin" with a color word anywhere → colored body feature.
    if words[-1] in ("hair", "eyes", "skin") and any(w in COLOR_WORDS for w in words):
        return True
    # Leading color word: "<color> <noun>" (clothing, object, background, …).
    if words[0] in COLOR_WORDS:
        return True
    # Modifier + color word: "light blue dress", "dark green skirt".
    if words[0] in COLOR_MODIFIERS and len(words) >= 2 and words[1] in COLOR_WORDS:
        return True
    return False


def filter_to_colors(caption: str) -> str:
    """Reduce a comma-separated Anima caption to its color tags only.

    Returns a comma-joined string of the kept tags (original order), or ``""``
    when the caption has no color tags.
    """
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    kept = [t for t in tags if is_color_tag(t)]
    return ", ".join(kept)


# Copyright-group entries that name "no series" rather than a real franchise.
# ``original`` is Danbooru's copyright tag for non-derivative works — it's the
# single most common copyright tag in the corpus, but it carries no series
# identity the manga cond can't already encode. Keeping it would inject a
# constant, dropout-protected leading token on ~a quarter of the captions and
# dilute the color→text coupling the filter exists to sharpen. So it's stripped
# from the copyright vocab (keep + protect paths both read this set).
_NON_SERIES_COPYRIGHT: frozenset[str] = frozenset({"original", "original character"})


@functools.lru_cache(maxsize=4)
def load_copyright_tags(
    index_path: str | Path = DEFAULT_CAPTION_INDEX,
) -> frozenset[str]:
    """Lowercased set of every copyright tag name from the caption index.

    Reads ``groups.copyright`` (a ``{copyright_name: [image_ids]}`` map) from the
    method-agnostic typed-tag index. Returns an empty set if the index is missing
    so callers degrade to "no copyright kept" rather than crashing. Cached because
    the index is a couple-thousand-image JSON and this runs once per caption.

    Non-series meta-copyright tags (``_NON_SERIES_COPYRIGHT``, e.g. ``original``)
    are excluded — they name "no franchise", so they're not worth riding along as
    a protected caption prefix.
    """
    p = Path(index_path)
    if not p.exists():
        return frozenset()
    data = json.loads(p.read_text(encoding="utf-8"))
    groups = data.get("groups", {})
    copyright_group = groups.get("copyright", {})
    return frozenset(
        name.strip().lower()
        for name in copyright_group
        if name.strip() and name.strip().lower() not in _NON_SERIES_COPYRIGHT
    )


def is_copyright_tag(tag: str, copyright_tags: frozenset[str]) -> bool:
    """True if ``tag`` is a known copyright/series name (case-insensitive)."""
    return tag.strip().lower() in copyright_tags


def filter_to_colors_and_protected(
    caption: str,
    copyright_tags: frozenset[str] | None = None,
    *,
    keep_copyright: bool = True,
    keep_comic: bool = False,
) -> str:
    """Reduce a caption to its protected leading tags followed by its color tags.

    The protected prefix is, in order: copyright/series tags (``keep_copyright``)
    then comic/panel-format tags (``keep_comic``). They lead the color tags so
    they form a contiguous run, mirroring the ``@artist``-prefix convention the
    variant generator already protects, and a prompt reads naturally — "genshin
    impact, comic, pink hair, blue eyes". Each tag is emitted once: a tag that is
    both copyright/comic and color-shaped stays in its protected slot. Pass
    ``copyright_tags`` to avoid re-reading the caption index per call; defaults to
    :func:`load_copyright_tags` (only consulted when ``keep_copyright``).
    """
    if copyright_tags is None:
        copyright_tags = load_copyright_tags() if keep_copyright else frozenset()
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    protected: list[str] = []
    seen: set[str] = set()

    def _take(pred: Callable[[str], bool]) -> None:
        for t in tags:
            if t.lower() not in seen and pred(t):
                protected.append(t)
                seen.add(t.lower())

    if keep_copyright:
        _take(lambda t: is_copyright_tag(t, copyright_tags))
    if keep_comic:
        _take(is_comic_tag)
    colors = [t for t in tags if t.lower() not in seen and is_color_tag(t)]
    return ", ".join(protected + colors)


def filter_to_colors_and_copyright(
    caption: str, copyright_tags: frozenset[str] | None = None
) -> str:
    """Reduce a caption to its copyright tags (first) followed by its color tags.

    Thin back-compat wrapper over :func:`filter_to_colors_and_protected` with only
    the copyright prefix enabled.
    """
    return filter_to_colors_and_protected(
        caption, copyright_tags, keep_copyright=True, keep_comic=False
    )
