"""Batch auto-tagging — mode policy and the merge invariants.

The stage writes the caption master, so the invariants worth pinning are all
about what it is *not* allowed to destroy:

1. **``missing`` never touches an existing caption.** It is the default and the
   only mode that cannot lose hand-written text.
2. **``merge`` never re-flattens a bound tag.** A tag already inside a position
   clause counts as present, so merging after ``caption-position`` cannot copy
   it back into the flat bag and re-introduce the exact ambiguity clauses exist
   to remove. Clauses themselves round-trip verbatim.
3. **Only one rating survives a merge.** ``predict_caption`` always emits a
   rating in slot 0; appending it to a caption that already has one produces a
   contradiction, not an addition.
4. **Dry run writes nothing.** ``apply`` defaults off everywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.preprocess.autotag import (  # noqa: E402
    AutotagOptions,
    merge_tags,
    run_autotag_captions,
)


def _dataset(tmp_path: Path, images: dict[str, str | None]) -> tuple[Path, Path]:
    """Build ``(resized_dir, source_dir)``; ``None`` caption → no sidecar."""
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    resized.mkdir()
    source.mkdir()
    for stem, caption in images.items():
        Image.new("RGB", (8, 8), (10, 20, 30)).save(resized / f"{stem}.png")
        if caption is not None:
            (source / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return resized, source


def _run(resized, source, tagged, **kwargs):
    return run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_fn=lambda _img: tagged,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# merge_tags
# ---------------------------------------------------------------------------


def test_merge_appends_only_novel_tags():
    merged, added = merge_tags(
        "safe, 1girl, blue hair", "safe, 1girl, blue hair, smile, outdoors"
    )
    assert merged == "safe, 1girl, blue hair, smile, outdoors"
    assert added == ("smile", "outdoors")


def test_merge_preserves_clauses_and_treats_their_tags_as_present():
    """A clause-bound tag must not be copied back into the flat bag."""
    existing = "safe, 2girls. On the left, akita neru. On the right, kasane teto."
    merged, added = merge_tags(existing, "safe, 2girls, akita neru, smile")

    assert added == ("smile",)
    assert merged == (
        "safe, 2girls, smile. On the left, akita neru. On the right, kasane teto."
    )
    # The clause text round-trips byte-for-byte.
    assert merged.endswith("On the left, akita neru. On the right, kasane teto.")


def test_merge_drops_a_second_rating():
    merged, added = merge_tags("safe, 1girl", "explicit, 1girl, nude")
    assert added == ("nude",)
    assert merged == "safe, 1girl, nude"
    assert "explicit" not in merged


def test_merge_keeps_the_rating_when_the_caption_has_none():
    merged, added = merge_tags("1girl, blue hair", "sensitive, 1girl, smile")
    assert added == ("sensitive", "smile")
    assert merged == "1girl, blue hair, sensitive, smile"


def test_merge_is_case_insensitive_and_idempotent():
    first, added = merge_tags("safe, 1girl, Blue Hair", "safe, 1girl, blue hair")
    assert added == ()
    assert first == "safe, 1girl, Blue Hair"
    second, added_again = merge_tags(first, "safe, 1girl, blue hair, smile")
    assert added_again == ("smile",)
    assert merge_tags(second, "safe, smile")[1] == ()


# ---------------------------------------------------------------------------
# run_autotag_captions
# ---------------------------------------------------------------------------


def test_missing_mode_only_fills_gaps(tmp_path):
    resized, source = _dataset(
        tmp_path, {"a": "hand written, 1girl", "b": None, "c": None}
    )

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="missing"),
        apply=True,
    )

    assert stats.seen == 3
    assert stats.candidates == 2
    assert stats.written == 2
    assert stats.skipped["has-caption"] == 1
    # The hand-written caption is byte-identical.
    assert (source / "a.txt").read_text(encoding="utf-8") == "hand written, 1girl"
    assert (source / "b.txt").read_text(encoding="utf-8") == "safe, 1girl, smile"
    assert {r.caption_path for r in rows} == {"b.txt", "c.txt"}


def test_overwrite_mode_replaces_every_caption(tmp_path):
    resized, source = _dataset(tmp_path, {"a": "hand written, 1girl", "b": None})

    _, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="overwrite"),
        apply=True,
    )

    assert stats.written == 2
    assert (source / "a.txt").read_text(encoding="utf-8") == "safe, 1girl, smile"


def test_merge_mode_leaves_a_saturated_caption_untouched(tmp_path):
    """No novel tags → no write, and the row is reported as unchanged."""
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl, smile"})

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="merge"),
        apply=True,
    )

    assert stats.written == 0
    assert stats.skipped["unchanged"] == 1
    assert rows[0].status == "skip:unchanged"


def test_dry_run_writes_nothing(tmp_path):
    resized, source = _dataset(tmp_path, {"a": None})

    rows, stats = _run(
        resized, source, "safe, 1girl", options=AutotagOptions(mode="missing")
    )

    assert stats.proposed == 1
    assert stats.written == 0
    assert not (source / "a.txt").exists()
    assert rows[0].proposed == "safe, 1girl"


def test_missing_mode_creates_nested_caption_dirs(tmp_path):
    """The master mirrors the resized tree; a new artist subdir must be made."""
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    (resized / "artist_x").mkdir(parents=True)
    source.mkdir()
    Image.new("RGB", (8, 8)).save(resized / "artist_x" / "a.png")

    _, stats = _run(resized, source, "safe, 1girl", apply=True)

    assert stats.written == 1
    assert (source / "artist_x" / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"


def test_empty_tagger_output_is_a_skip_not_a_blank_caption(tmp_path):
    resized, source = _dataset(tmp_path, {"a": None})

    rows, stats = _run(resized, source, "  ", apply=True)

    assert stats.written == 0
    assert stats.skipped["no-tags"] == 1
    assert rows[0].status == "skip:no-tags"
    assert not (source / "a.txt").exists()


def test_bad_options_fail_at_construction():
    with pytest.raises(ValueError):
        AutotagOptions(mode="clobber")
    with pytest.raises(ValueError):
        AutotagOptions(min_confidence=1.5)
