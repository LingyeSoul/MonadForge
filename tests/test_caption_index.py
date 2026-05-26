"""Tests for build_caption_index._classify, focused on the danbooru
`name (series)` character-recovery heuristic that rescues characters the tagger
vocab predates (e.g. `endministrator (arknights)`)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_caption_index",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "preprocess"
    / "build_caption_index.py",
)
bci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bci)


# Minimal vocab sets: deliberately omits the newer characters so recovery must
# carry them. "genshin impact" is a known copyright; "miside" is not (only
# reachable via the same-caption bare tag).
VSETS = {
    "character": {"hatsune miku"},
    "copyright": {"genshin impact", "vocaloid", "hololive", "pokemon", "original"},
    "count": {"1girl", "1boy"},
}


def test_recovers_paren_character_via_known_copyright():
    typed = bci._classify("1girl, mualani (genshin impact), genshin impact", VSETS)
    assert "mualani (genshin impact)" in typed["character"]
    assert "genshin impact" in typed["copyright"]
    assert "1girl" in typed["count"]


def test_recovers_via_same_caption_bare_tag_when_copyright_unknown():
    # "miside" is not in the vocab copyright set, but appears as a bare tag.
    typed = bci._classify("cool mita (miside), miside, 1girl", VSETS)
    assert "cool mita (miside)" in typed["character"]
    assert "miside" in typed["copyright"]


def test_generic_qualifier_not_recovered():
    # `X (cosplay)` must not make X a character (cosplay is a generic qualifier),
    # even though "cosplay" appears as a bare tag.
    typed = bci._classify("frieren (cosplay), cosplay, 1girl", VSETS)
    assert typed["character"] == []


def test_unknown_series_not_recovered():
    # Parenthetical whose series is neither a known copyright nor a bare tag.
    typed = bci._classify("someone (obscure game), 1girl", VSETS)
    assert typed["character"] == []


def test_exact_vocab_character_still_classified():
    typed = bci._classify("hatsune miku, vocaloid, 1girl", VSETS)
    assert typed["character"] == ["hatsune miku"]
    assert "vocaloid" in typed["copyright"]


def test_recover_paren_can_be_disabled():
    typed = bci._classify(
        "mualani (genshin impact), genshin impact", VSETS, recover_paren=False
    )
    assert typed["character"] == []
    assert "genshin impact" in typed["copyright"]


def test_artist_prefix_unaffected():
    typed = bci._classify("@some artist, mualani (genshin impact), genshin impact", VSETS)
    assert typed["artist"] == ["@some artist"]
    assert "mualani (genshin impact)" in typed["character"]


# ── positional bare-name character recovery ─────────────────────────────────


def test_positional_recovers_bare_name_character():
    # `nakiri ayame` is a bare name (no parens, not in vocab) sitting in the
    # pre-@artist band → recovered as character; "hololive" stays copyright.
    typed = bci._classify(
        "sensitive, 1girl, nakiri ayame, hololive, @drawfag, black hair", VSETS
    )
    assert typed["character"] == ["nakiri ayame"]
    assert "1girl" in typed["count"]


def test_positional_excludes_franchise_subtitle():
    # `pokemon scarlet and violet` shares the word "pokemon" with the known
    # copyright → it's a franchise sub-title (copyright), NOT a character. Only
    # the bare-name character `hilda` is recovered.
    typed = bci._classify(
        "explicit, 1girl, hilda, pokemon, pokemon scarlet and violet, @x",
        VSETS,
    )
    assert typed["character"] == ["hilda"]
    assert "pokemon scarlet and violet" not in typed["character"]


def test_positional_skips_general_tags_after_artist():
    # Descriptive tags live after @artist and must never be read as characters.
    typed = bci._classify(
        "1girl, yukihana lamy, hololive, @y, blue eyes, smile, looking at viewer",
        VSETS,
    )
    assert typed["character"] == ["yukihana lamy"]
    assert "blue eyes" not in typed["character"]


def test_positional_excludes_count_like_tags():
    # A count tag the vocab missed ("2others") must not become a character.
    typed = bci._classify("sensitive, 1girl, 2others, asaka karin, hololive, @z", VSETS)
    assert "2others" not in typed["character"]
    assert "asaka karin" in typed["character"]


def test_positional_needs_artist_anchor():
    # No @artist → no reliable band boundary → no positional recovery.
    typed = bci._classify("1girl, mystery name, hololive", VSETS)
    assert "mystery name" not in typed["character"]


def test_positional_can_be_disabled():
    typed = bci._classify(
        "1girl, nakiri ayame, hololive, @drawfag", VSETS, recover_positional=False
    )
    assert typed["character"] == []


# ── `original` sole-copyright clears characters (OC images) ─────────────────


def test_original_only_clears_character():
    # `original` is the sole copyright → an OC image, no named character, even
    # though a character tag is present (vocab or otherwise).
    typed = bci._classify("1girl, hatsune miku, original, @x", VSETS)
    assert typed["character"] == []
    assert "original" in typed["copyright"]


def test_original_crossover_keeps_character():
    # `original` co-occurring with a real franchise (pokemon) is a crossover —
    # the franchise character survives.
    typed = bci._classify("1girl, dawn (pokemon), pokemon, original, @x", VSETS)
    assert "dawn (pokemon)" in typed["character"]
