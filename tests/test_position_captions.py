"""Position-aware caption clauses — parse/compose, variants, and the v1 pipeline.

Three invariants worth pinning:

1. **Clause parsing round-trips.** The convention delimits clauses with ``.``
   and tags with ``,``. A naive ``split(",")`` glues the header onto the
   preceding tag (``"white socks. On the left"``), which is what used to make
   every ``On the …``-aware consumer silently see no sections at all.
2. **Clauses are atomic under variant generation.** A shuffled variant must
   never move a clause tag into the flat bag or into a *different* clause —
   that reassigns an attribute to the wrong subject, the exact ambiguity the
   feature exists to remove.
3. **The pipeline never writes a clause it can't ground.** Count disagreement,
   too few instances, and hallucinated character names all skip.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from library.captioning.position_clauses import (  # noqa: E402
    assign_positions,
    compose_caption,
    has_clauses,
    horizontal_names,
    ordered_indices,
    parse_caption,
)

GT = (
    "sensitive, 2girls, original, @rurudo, cat girl, navel. "
    "On the left, red eyes, gray hair. "
    "On the right, aqua eyes, pink hair."
)


# ----- parse / compose ----------------------------------------------------


def test_parse_splits_flat_bag_from_clauses():
    parsed = parse_caption(GT)
    assert parsed.flat_tags == (
        "sensitive",
        "2girls",
        "original",
        "@rurudo",
        "cat girl",
        "navel",
    )
    assert [c.position for c in parsed.clauses] == ["left", "right"]
    assert parsed.clauses[0].tags == ("red eyes", "gray hair")
    # The clause-terminating period is not part of the last tag.
    assert parsed.clauses[1].tags == ("aqua eyes", "pink hair")


def test_parse_compose_round_trip():
    assert parse_caption(GT).render() == GT


def test_parse_caption_without_clauses_is_flat():
    parsed = parse_caption("1girl, blue hair, smile")
    assert not parsed.has_clauses
    assert parsed.render() == "1girl, blue hair, smile"


def test_parse_accepts_comma_form_header():
    # Some hand-written captions separate the first clause with a comma rather
    # than a period; both parse, and compose normalizes to the period form.
    parsed = parse_caption("1girl, white socks, On the left, red eyes")
    assert parsed.flat_tags == ("1girl", "white socks")
    assert parsed.clauses[0].tags == ("red eyes",)
    assert parsed.render() == "1girl, white socks. On the left, red eyes."


def test_punctuation_tags_survive_the_period_strip():
    parsed = parse_caption("1girl, :d, >_<. On the left, :3.")
    assert parsed.flat_tags == ("1girl", ":d", ">_<")
    assert parsed.clauses[0].tags == (":3",)


def test_has_clauses_detects_both_forms():
    assert has_clauses(GT)
    assert has_clauses("1girl, On the left, red eyes")
    assert not has_clauses("1girl, blue hair, on the beach")


def test_compose_without_clauses_is_a_plain_join():
    assert compose_caption(["a", "b"]) == "a, b"


# ----- position vocabulary ------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, ["left", "right"]),
        (3, ["left", "middle", "right"]),
        (4, ["leftmost", "second from left", "third from left", "rightmost"]),
    ],
)
def test_horizontal_names(n, expected):
    assert horizontal_names(n) == expected


def test_assign_positions_orders_left_to_right():
    # Deliberately out of order: the names must follow geometry, not input order.
    boxes = [(600, 0, 800, 400), (0, 0, 200, 400), (300, 0, 500, 400)]
    assert assign_positions(boxes, (1000, 400)) == ["right", "left", "middle"]


def test_assign_positions_is_row_aware_for_grid_sheets():
    # A 2x2 contact sheet: pure x-ordering would interleave the rows and call
    # the bottom-left view "left" alongside the top-left one.
    boxes = [
        (0, 0, 400, 400),  # top left
        (600, 0, 1000, 400),  # top right
        (0, 600, 400, 1000),  # bottom left
        (600, 600, 1000, 1000),  # bottom right
    ]
    assert assign_positions(boxes, (1000, 1000)) == [
        "top left",
        "top right",
        "bottom left",
        "bottom right",
    ]
    assert ordered_indices(boxes, (1000, 1000)) == [0, 1, 2, 3]


def test_single_subject_row_gets_the_bare_row_word():
    boxes = [(400, 0, 600, 300), (0, 700, 300, 1000), (700, 700, 1000, 1000)]
    assert assign_positions(boxes, (1000, 1000)) == [
        "top",
        "bottom left",
        "bottom right",
    ]


# ----- caption variants: clauses are atomic -------------------------------


def _variants(*args, **kwargs):
    from library.preprocess import generate_caption_variants

    return generate_caption_variants(*args, **kwargs)


def test_clause_tags_never_leak_into_the_flat_bag():
    random.seed(0)
    for text in _variants(GT, num_variants=12, tag_dropout_rate=0.3):
        parsed = parse_caption(text)
        flat = set(parsed.flat_tags)
        for clause_tag in ("red eyes", "gray hair", "aqua eyes", "pink hair"):
            assert clause_tag not in flat, text


def test_clause_tags_never_cross_between_clauses():
    random.seed(1)
    left, right = {"red eyes", "gray hair"}, {"aqua eyes", "pink hair"}
    for text in _variants(GT, num_variants=12, tag_dropout_rate=0.3):
        for clause in parse_caption(text).clauses:
            owner = left if clause.position == "left" else right
            assert set(clause.tags) <= owner, text


def test_clause_is_dropped_whole_or_kept_whole():
    random.seed(2)
    for text in _variants(GT, num_variants=16, tag_dropout_rate=0.5):
        for clause in parse_caption(text).clauses:
            expected = (
                ("red eyes", "gray hair")
                if clause.position == "left"
                else (
                    "aqua eyes",
                    "pink hair",
                )
            )
            assert sorted(clause.tags) == sorted(expected), text


def test_clause_dropout_rate_one_removes_every_clause():
    random.seed(3)
    out = _variants(GT, num_variants=6, tag_dropout_rate=0.0, clause_dropout_rate=1.0)
    assert out[0] == GT  # v0 is always pristine
    for text in out[1:]:
        assert not parse_caption(text).has_clauses


def test_clause_dropout_rate_zero_keeps_every_clause():
    random.seed(4)
    for text in _variants(
        GT, num_variants=8, tag_dropout_rate=0.9, clause_dropout_rate=0.0
    ):
        assert len(parse_caption(text).clauses) == 2


def test_artist_prefix_still_protected_with_clauses():
    random.seed(5)
    for text in _variants(GT, num_variants=8, tag_dropout_rate=1.0):
        assert "@rurudo" in parse_caption(text).flat_tags


def test_no_clause_caption_is_byte_identical_at_v0():
    raw = "@sincos,blue hair  ,1girl"
    assert _variants(raw, num_variants=1, tag_dropout_rate=0.0)[0] == raw


def test_clause_headers_are_never_identity_randomized():
    random.seed(6)
    pool = ["swing", "sodium", "awards"]
    for text in _variants(
        GT,
        num_variants=8,
        tag_dropout_rate=0.0,
        clause_dropout_rate=0.0,
        tag_randomize_rate=1.0,
        erasure_pool=pool,
    )[1:]:
        assert [c.position for c in parse_caption(text).clauses] == ["left", "right"]


# ----- order correction keeps clauses intact ------------------------------


def test_correct_caption_does_not_dissolve_clauses():
    from library.captioning.correction import TagKnowledgeBase, correct_caption

    kb = TagKnowledgeBase({}, Path("stub.csv"))
    out = correct_caption(GT, kb).text
    parsed = parse_caption(out)
    assert [c.position for c in parsed.clauses] == ["left", "right"]
    assert parsed.clauses[0].tags == ("red eyes", "gray hair")
    # The flat bag is still reordered into buckets (rating first, artist slot).
    assert parsed.flat_tags[0] == "sensitive"
    assert "@rurudo" in parsed.flat_tags


# ----- pipeline -----------------------------------------------------------


@pytest.fixture
def pipeline_bits():
    from PIL import Image

    from library.preprocess.position_captions import (
        ClauseVocabulary,
        Detection,
        PositionCaptionOptions,
    )

    vocabulary = ClauseVocabulary(
        characters=frozenset({"akita neru", "hatsune miku"}),
        excluded=frozenset({"vocaloid"}),
        exclusive_groups=frozenset({"hair_color", "eye_color"}),
        tag_to_group={
            "blonde hair": "hair_color",
            "aqua hair": "hair_color",
            "green hair": "hair_color",
            "red eyes": "eye_color",
            "twintails": "hairstyle",
            "simple background": "background_detail",
            "white background": "background_detail",
        },
    )
    image = Image.new("RGB", (1000, 500), "white")
    return image, vocabulary, Detection, PositionCaptionOptions


def _detector(boxes_by_threshold):
    """Stub ``detect_fn``: threshold → list of (box, score)."""
    from library.preprocess.position_captions import Detection

    def detect(image, score_threshold):
        for thr in sorted(boxes_by_threshold, reverse=True):
            if score_threshold >= thr:
                return [Detection(box=b, score=s) for b, s in boxes_by_threshold[thr]]
        lowest = min(boxes_by_threshold)
        return [Detection(box=b, score=s) for b, s in boxes_by_threshold[lowest]]

    return detect


def _tagger(per_crop):
    """Stub ``tag_fn`` returning ``per_crop`` predictions in call order."""
    calls = {"n": 0}

    def tag(crop):
        out = per_crop[calls["n"] % len(per_crop)]
        calls["n"] += 1
        return out

    return tag


def test_propose_binds_hair_color_to_each_side(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    caption = "safe, 2girls, akita neru, hatsune miku, @channel, blonde hair, aqua hair, simple background"
    proposal = propose_for_image(
        image,
        caption,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {
                        "akita neru": 0.9,
                        "blonde hair": 0.8,
                        "simple background": 0.7,
                    },
                    "groups": {"hair_color": "blonde hair"},
                },
                {
                    "kept": {
                        "hatsune miku": 0.9,
                        "aqua hair": 0.8,
                        "twintails": 0.7,
                        "simple background": 0.7,
                    },
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.ok
    parsed = parse_caption(proposal.proposed)
    # v1 is additive: the flat bag comes through untouched.
    assert parsed.flat_tags == tuple(t.strip() for t in caption.split(","))
    assert parsed.clauses[0].tags[:2] == ("akita neru", "blonde hair")
    assert parsed.clauses[1].tags[:2] == ("hatsune miku", "aqua hair")
    # Kept on BOTH crops → not attributable → stays out of every clause.
    assert not any("simple background" in c.tags for c in parsed.clauses)


def test_count_mismatch_is_skipped_not_guessed(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 3girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.3: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger([{"kept": {}, "groups": {}}]),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:count-mismatch"
    assert proposal.proposed is None


def test_low_threshold_retry_recovers_the_missing_instance(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    detect = _detector(
        {
            0.5: [((0, 0, 300, 500), 0.9), ((350, 0, 650, 500), 0.9)],
            0.3: [
                ((0, 0, 300, 500), 0.9),
                ((350, 0, 650, 500), 0.9),
                ((700, 0, 1000, 500), 0.35),
            ],
        }
    )
    proposal = propose_for_image(
        image,
        "safe, 3girls, blonde hair, aqua hair, green hair",
        detect_fn=detect,
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8}, "groups": {}},
                {"kept": {"aqua hair": 0.8}, "groups": {}},
                {"kept": {"green hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.detected == 3
    assert [c.position for c in parse_caption(proposal.proposed).clauses] == [
        "left",
        "middle",
        "right",
    ]


def test_unlisted_character_name_is_rejected(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair, aqua hair",  # no character named
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {"akita neru": 0.99, "blonde hair": 0.8}, "groups": {}},
                {"kept": {"hatsune miku": 0.99, "aqua hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    clauses = parse_caption(proposal.proposed).clauses
    assert all("akita neru" not in c.tags for c in clauses)
    assert clauses[0].tags == ("blonde hair",)


def test_multiple_views_clauses_carry_only_what_differs(pipeline_bits):
    # A `1girl, multiple views` outfit sheet: same character, same hair, same
    # eyes in every view. Repeating those four times binds nothing — the clause
    # must carry the outfit, which is the only thing that varies.
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    shared = {"hatsune miku": 0.99, "aqua hair": 0.9, "twintails": 0.8}
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, hatsune miku, aqua hair, twintails, maid, swimsuit",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {**shared, "maid": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
                {
                    "kept": {**shared, "swimsuit": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    clauses = parse_caption(proposal.proposed).clauses
    assert clauses[0].tags == ("maid",)
    assert clauses[1].tags == ("swimsuit",)
    # ...and the shared attributes are still asserted, in the untouched flat bag.
    assert "hatsune miku" in parse_caption(proposal.proposed).flat_tags


def test_keep_shared_tags_restores_the_repeated_attributes(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    shared = {"hatsune miku": 0.99, "aqua hair": 0.9}
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, hatsune miku, aqua hair, maid, swimsuit",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {**shared, "maid": 0.8}, "groups": {}},
                {"kept": {**shared, "swimsuit": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(discriminative_only=False),
    )
    assert "hatsune miku" in parse_caption(proposal.proposed).clauses[0].tags


def test_indistinguishable_subjects_are_skipped(pipeline_bits):
    # Nothing varies between the crops → no clause can be grounded → skip
    # rather than emit two identical, information-free clauses.
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    same = {"hatsune miku": 0.99, "aqua hair": 0.9, "twintails": 0.8}
    proposal = propose_for_image(
        image,
        "safe, 2girls, hatsune miku, aqua hair, twintails",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger([{"kept": same, "groups": {}}, {"kept": same, "groups": {}}]),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:no-discriminative-tags"
    assert proposal.proposed is None


def test_differing_hair_colors_still_bind_for_two_characters(pipeline_bits):
    # The discriminative rule must NOT suppress the multi-girl case it exists
    # to serve: two girls with different hair keep their hair in their clause.
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, akita neru, hatsune miku, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {"akita neru": 0.9, "blonde hair": 0.8},
                    "groups": {"hair_color": "blonde hair"},
                },
                {
                    "kept": {"hatsune miku": 0.9, "aqua hair": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    clauses = parse_caption(proposal.proposed).clauses
    assert clauses[0].tags == ("akita neru", "blonde hair")
    assert clauses[1].tags == ("hatsune miku", "aqua hair")


def test_copyright_tags_stay_out_of_clauses(pipeline_bits):
    # A franchise tag fires on every crop and describes the image, not a
    # subject — it must not ride the ranked path into a clause.
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, vocaloid, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8, "vocaloid": 0.95}, "groups": {}},
                {"kept": {"aqua hair": 0.8, "twintails": 0.7}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert all(
        "vocaloid" not in c.tags for c in parse_caption(proposal.proposed).clauses
    )


def test_only_one_member_of_an_exclusive_group_per_clause(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, green hair, aqua hair, blonde hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                # Two hair colors kept on one crop; the group winner wins and the
                # runner-up must not follow it in through the ranked path.
                {
                    "kept": {"green hair": 0.9, "aqua hair": 0.85, "twintails": 0.7},
                    "groups": {"hair_color": "green hair"},
                },
                {"kept": {"blonde hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    left = parse_caption(proposal.proposed).clauses[0].tags
    assert "green hair" in left
    assert "aqua hair" not in left


def test_single_detection_is_skipped(pipeline_bits):
    from library.preprocess.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair",
        detect_fn=_detector({0.3: [((0, 0, 400, 500), 0.9)]}),
        tag_fn=_tagger([{"kept": {}, "groups": {}}]),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:too-few-instances"


# ----- candidate prefilter ------------------------------------------------


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("safe, 3girls, blonde hair", True),
        ("safe, 1girl, multiple views, maid", True),
        ("safe, multiple girls, blonde hair", True),
        ("safe, 1girl, blonde hair", False),
        (GT, False),  # already has clauses
    ],
)
def test_is_candidate(caption, expected):
    from library.preprocess.position_captions import is_candidate

    assert is_candidate(caption)[0] is expected


def test_multiple_views_defers_the_count_to_detection():
    # `1girl, multiple views` is routinely four bindable views — the girls-count
    # must NOT gate it, or every outfit sheet skips on count-mismatch.
    from library.preprocess.position_captions import caption_subject_count

    assert caption_subject_count("safe, 1girl, multiple views, maid") is None
    assert caption_subject_count("safe, 3girls, blonde hair") == 3
    assert caption_subject_count("safe, 1girl, blonde hair") == 1


def test_mask_blanking_removes_the_neighbour(pipeline_bits):
    import numpy as np
    from PIL import Image

    from library.preprocess.position_captions import Detection, crop_instance

    # Left half red (the subject), right half blue (a neighbour intruding into
    # the padded box). Blanking must leave no blue in the crop.
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[:, :100] = (255, 0, 0)
    pixels[:, 100:] = (0, 0, 255)
    image = Image.fromarray(pixels)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[:, :100] = 1

    crop = np.asarray(
        crop_instance(
            image, Detection(box=(0, 0, 200, 100), score=0.9, mask=mask), pad=0.0
        )
    )
    assert not (crop == (0, 0, 255)).all(axis=-1).any()
    assert (crop == (255, 0, 0)).all(axis=-1).any()
