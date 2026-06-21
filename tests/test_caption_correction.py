from __future__ import annotations

from pathlib import Path

from library.captioning.correction import (
    CaptionCorrectionOptions,
    correct_caption,
    default_tag_csv_candidates,
    find_tag_csv,
    load_tag_knowledge_base,
)


def _csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,category,post_count,description",
                '1girl,0,10,"[인물 > 인원수] count"',
                'solo,0,10,"[인물 > 인원수] count"',
                'hatsune_miku,4,10,"[캐릭터 > vocaloid] character"',
                'vocaloid,3,10,"[작품 > series] copyright"',
                'sincos,1,10,"[작가 > illustrator] artist"',
                'best_quality,5,10,"[메타 > 화질] quality"',
                'highres,5,10,"[메타 > 화질] resolution meta"',
                'commentary,5,10,"[메타 > 정보_요청] artist commentary"',
                'long_hair,0,10,"[머리카락 > 머리 길이] general"',
                'copyright_notice,0,10,"[메타 > 정보_요청] misleading description"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_correct_caption_orders_known_sections_and_preserves_general_order(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, @sincos, hatsune miku, 1girl, best quality",
        kb,
        options=CaptionCorrectionOptions(insert_no_artist=True),
    )

    # "best quality" is a quality-conditioning tag — emitted at the very front.
    assert result.text == (
        "best quality, 1girl, hatsune miku, vocaloid, @sincos, long hair"
    )
    assert result.changed
    assert not result.inserted_no_artist


def test_front_region_orders_quality_meta_year_safety_and_demotes_commentary(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, commentary, highres, best quality, score_9, 1girl, sensitive, 2008",
        kb,
        options=CaptionCorrectionOptions(insert_no_artist=False),
    )

    # Front region: quality (best quality, score_9) → meta (highres) →
    # year (2008) → safety (sensitive). Commentary is demoted to the general
    # tail; quality `score_N` tags are recognized despite being absent from KB
    # and keep their underscore spelling on emit.
    assert result.text == (
        "best quality, score_9, highres, 2008, sensitive, 1girl, long hair, commentary"
    )


def test_correct_caption_inserts_no_artist_at_artist_position(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption("long hair, vocaloid, hatsune miku, solo", kb)

    assert result.text == "hatsune miku, vocaloid, @no-artist, long hair, solo"
    assert result.inserted_no_artist


def test_trigger_defaults_to_artist_slot_and_suppresses_no_artist(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, @sincos, vocaloid, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
        ),
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @dataset-trigger, @sincos, long hair"
    )
    assert not result.inserted_no_artist


def test_existing_artist_trigger_is_not_duplicated(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, @dataset-trigger, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
        ),
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @dataset-trigger, long hair"
    )
    assert not result.inserted_no_artist


def test_trigger_preserves_underscores_when_inserted(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset_trigger",
        ),
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @dataset_trigger, long hair"
    )


def test_existing_general_trigger_moves_to_artist_slot(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, dataset trigger, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset_trigger",
        ),
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @dataset_trigger, long hair"
    )
    assert not result.inserted_no_artist


def test_front_trigger_allows_no_artist_when_no_artist_marker(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
            trigger_at_front=True,
        ),
    )

    assert result.text == (
        "@dataset-trigger, 1girl, hatsune miku, vocaloid, @no-artist, long hair"
    )
    assert result.inserted_no_artist


def test_existing_front_trigger_moves_to_front_and_allows_no_artist(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, @dataset-trigger, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
            trigger_at_front=True,
        ),
    )

    assert result.text == (
        "@dataset-trigger, 1girl, hatsune miku, vocaloid, @no-artist, long hair"
    )
    assert result.inserted_no_artist


def test_artist_validation_keeps_unknown_at_tag_in_general_tail(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "@trigger-word, hatsune miku, vocaloid, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            validate_artist_tags=True,
        ),
    )

    assert result.text == ("1girl, hatsune miku, vocaloid, @no-artist, @trigger-word")


def test_artist_validation_does_not_reclassify_at_prefixed_character(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "@hatsune miku, vocaloid, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            validate_artist_tags=True,
        ),
    )

    assert result.text == "1girl, vocaloid, @no-artist, @hatsune miku"


def test_numeric_category_wins_over_description_prefix(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "copyright notice, hatsune miku, vocaloid, 1girl",
        kb,
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @no-artist, copyright notice"
    )


def test_describe_returns_body_category_and_post_count(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))

    info = kb.describe("long hair")
    assert info is not None
    assert info.name == "long hair"
    assert info.kind == "general"
    assert info.category_path == "머리카락 > 머리 길이"
    assert info.description == "general"  # bracketed category prefix stripped
    assert info.post_count == 10

    assert kb.describe("not_a_real_tag") is None


def test_ranked_infos_orders_by_post_count_and_is_cached(tmp_path):
    path = _csv(tmp_path / "tags.csv")
    path.write_text(
        "\n".join(
            [
                "name,category,post_count,description",
                'rare_tag,0,5,"[머리카락 > x] rare"',
                'common_tag,0,9000,"[머리카락 > x] common"',
                'mid_tag,0,500,"[머리카락 > x] mid"',
            ]
        ),
        encoding="utf-8",
    )
    kb = load_tag_knowledge_base(path)

    ranked = kb.ranked_infos()
    assert [info.name for info in ranked] == ["common tag", "mid tag", "rare tag"]
    assert kb.ranked_infos() is ranked  # cached identity


def test_default_tag_csv_prefers_models_dir_over_env(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    model_csv = root / "models" / "danbooru_tags_classified.csv"
    env_csv = tmp_path / "env.csv"
    model_csv.parent.mkdir(parents=True)
    model_csv.write_text("name,category,post_count,description\n", encoding="utf-8")
    env_csv.write_text("name,category,post_count,description\n", encoding="utf-8")
    monkeypatch.setenv("ANIMA_DANBOORU_TAGS_CSV", str(env_csv))

    candidates = default_tag_csv_candidates(root)

    assert candidates[0] == model_csv
    assert root.parent / "danbooru_tags_classified.csv" not in candidates
    assert find_tag_csv(root) == model_csv
