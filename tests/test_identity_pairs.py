"""Tests for the IP-Adapter distinct-pair (identity) reference sampler.

Covers the tiered character → copyright → artist back-off, the cross-artist
constraint, self-fallback, candidate-pool restriction, and the shuffled
(unrelated) negative used by the validation gate.
"""

from __future__ import annotations

import json
import random

import pytest

from library.datasets.identity_pairs import IdentityPairSampler


def _write_index(tmp_path, image_meta):
    """Build a caption_index.json (image_meta + inverted groups) from a
    {stem: {path, character, copyright, artist}} dict."""
    groups = {"character": {}, "copyright": {}, "artist": {}}
    for stem, meta in image_meta.items():
        for axis in groups:
            for tag in meta.get(axis, []):
                groups[axis].setdefault(tag, []).append(stem)
    for axis in groups:
        groups[axis] = {t: sorted(s) for t, s in sorted(groups[axis].items())}
    index = {
        "meta": {"n_images": len(image_meta)},
        "image_meta": image_meta,
        "groups": groups,
    }
    p = tmp_path / "caption_index.json"
    p.write_text(json.dumps(index), encoding="utf-8")
    return str(p)


@pytest.fixture
def index_path(tmp_path):
    meta = {
        # frieren: 3 imgs across 2 artists
        "a1": {
            "path": "art_x/a1.txt",
            "character": ["frieren"],
            "copyright": ["sousou no frieren"],
            "artist": ["@x"],
        },
        "a2": {
            "path": "art_x/a2.txt",
            "character": ["frieren"],
            "copyright": ["sousou no frieren"],
            "artist": ["@x"],
        },
        "a3": {
            "path": "art_y/a3.txt",
            "character": ["frieren"],
            "copyright": ["sousou no frieren"],
            "artist": ["@y"],
        },
        # fern: same franchise, different character, single image (no char positive)
        "b1": {
            "path": "art_y/b1.txt",
            "character": ["fern"],
            "copyright": ["sousou no frieren"],
            "artist": ["@y"],
        },
        # unrelated identity, shares artist @y only
        "c1": {
            "path": "art_y/c1.txt",
            "character": ["miku"],
            "copyright": ["vocaloid"],
            "artist": ["@y"],
        },
    }
    return _write_index(tmp_path, meta)


def test_same_character_preferred(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    # a1 has same-character partners a2, a3 → must resolve at character tier.
    refs = {s.resolve("a1", rng) for _ in range(20)}
    chosen = {stem for stem, _ in refs}
    assert chosen <= {"a2", "a3"}
    assert all(level == "character" for _, level in refs)


def test_franchise_fallback(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(0)
    # b1 (fern) is the only fern → no character positive → falls back to the
    # franchise tier (sousou no frieren): a1/a2/a3.
    stem, level = s.resolve("b1", rng)
    assert level == "copyright"
    assert stem in {"a1", "a2", "a3"}


def test_min_level_blocks_fallback(index_path):
    s = IdentityPairSampler(index_path, min_level="character")
    rng = random.Random(0)
    # b1 has no character positive and franchise/artist tiers are disallowed
    # by min_level="character" → self.
    assert s.resolve("b1", rng) == ("b1", "self")


def test_cross_artist_excludes_same_artist(index_path):
    s = IdentityPairSampler(index_path, min_level="artist", cross_artist=True)
    rng = random.Random(0)
    # a1 is @x; cross-artist character match must be the @y image a3, never a2 (@x).
    refs = {s.resolve("a1", rng)[0] for _ in range(20)}
    assert refs == {"a3"}


def test_restrict_stems(index_path):
    # Exclude a3 (validation split) → only a2 reachable for a1 at char tier.
    s = IdentityPairSampler(
        index_path, min_level="artist", restrict_stems={"a1", "a2", "b1", "c1"}
    )
    rng = random.Random(0)
    refs = {s.resolve("a1", rng)[0] for _ in range(20)}
    assert refs == {"a2"}


def test_shuffled_avoids_identity(index_path):
    s = IdentityPairSampler(index_path, min_level="artist")
    rng = random.Random(1)
    # a1's identity = {frieren, sousou no frieren}; shuffled should pick an
    # unrelated stem (c1 shares neither; b1 shares the franchise).
    picks = {s.shuffled("a1", rng)[0] for _ in range(40)}
    assert "c1" in picks
    assert not (picks & {"a1"})  # never the target


def test_unknown_stem_self(index_path):
    s = IdentityPairSampler(index_path)
    assert s.resolve("nonexistent", random.Random(0)) == ("nonexistent", "self")


def test_rel_dir(index_path):
    s = IdentityPairSampler(index_path)
    assert s.rel_dir("a3") == "art_y"


def test_invalid_min_level(index_path):
    with pytest.raises(ValueError):
        IdentityPairSampler(index_path, min_level="franchise")
