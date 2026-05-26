"""Tests for the shared tag-shape primitives in
``library.captioning.taxonomy`` and the contract that the Anima Tagger vocab
build and the caption-index builder type tag *shape* identically (no drift)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from library.captioning import taxonomy as tx

# The caption-index builder, loaded as a standalone module (it's a script).
_SPEC = importlib.util.spec_from_file_location(
    "build_caption_index",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "preprocess"
    / "build_caption_index.py",
)
bci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bci)


def test_taxonomy_is_torch_free():
    # Importing the primitives must not drag torch (the caption-index script
    # relies on staying lightweight / method-agnostic). Check in a fresh
    # subprocess so a torch import elsewhere in the suite can't mask a
    # regression here.
    import subprocess
    import sys

    code = "import library.captioning.taxonomy, sys; assert 'torch' not in sys.modules"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        "importing library.captioning.taxonomy pulled in torch:\n" + r.stderr
    )


def test_is_artist_tag():
    assert tx.is_artist_tag("@sincos")
    assert tx.is_artist_tag("@sumiyao (amam)")
    # Booru emoticon `@_@` → `@ @` after normalization is NOT an artist.
    assert not tx.is_artist_tag("@ @")
    assert not tx.is_artist_tag("@")
    assert not tx.is_artist_tag("blue eyes")


def test_strip_artist_prefix():
    assert tx.strip_artist_prefix("@sincos") == "sincos"
    assert tx.strip_artist_prefix("hatsune miku") == "hatsune miku"


def test_is_count_tag():
    for t in [
        "1girl",
        "2girls",
        "1boy",
        "3others",
        "6+girls",
        "multiple girls",
        "multiple_boys",
    ]:
        assert tx.is_count_tag(t), t
    for t in ["no girls", "blue eyes", "1girl_1boy", "original"]:
        assert not tx.is_count_tag(t), t


def test_caption_ratings_superset_of_model_ratings():
    # CAPTION_RATINGS (raw, 4-class) is a superset of the tagger's model-output
    # RATINGS (3-class, after questionable→sensitive collapse).
    from library.captioning.anima_tagger import RATINGS

    assert set(RATINGS) <= tx.CAPTION_RATINGS
    assert "questionable" in tx.CAPTION_RATINGS
    assert "questionable" not in RATINGS


def test_index_and_tagger_agree_on_tag_shape():
    """The two categorization paths must classify artist/count identically —
    both now key off the single shared primitives."""
    from scripts.anima_tagger import vocab as v

    for tag in [
        "@sincos",
        "@ @",
        "1girl",
        "2others",
        "multiple girls",
        "no girls",
        "blue eyes",
    ]:
        tagger_cat = v.categorize(tag, cache={}, overrides=None)
        if bci.is_artist_tag(tag):
            assert tagger_cat == "artist", tag
        elif bci.is_count_tag(tag):
            assert tagger_cat == "count", tag
        else:
            # Neither artist nor count shape → tagger must not call it one.
            assert tagger_cat not in ("artist", "count"), tag


def test_constants_reexports_for_back_compat():
    # train_common imports _COUNT_RE from scripts.anima_tagger.constants.
    from scripts.anima_tagger import constants as c

    assert c.is_count_tag is tx.is_count_tag
    assert c._COUNT_RE is tx._COUNT_RE
