import pytest

from library.datasets.buckets import buckets_for_edges, choose_edge
from library.preprocess.resize_preview import (
    compute_resize_preview,
    normalize_target_res,
    normalize_crop_margins,
    parse_bucket_resos,
    select_resize_bucket,
)


def test_resize_preview_uses_preprocess_bucket_tier():
    preview = compute_resize_preview(1440, 2560, [768, 1024])

    assert preview.target_edge == choose_edge(1440, 2560, [768, 1024])
    assert preview.bucket_size in buckets_for_edges([preview.target_edge])
    assert preview.kept_rect.width == pytest.approx(1440)
    assert preview.kept_rect.left == pytest.approx(0)
    assert preview.kept_rect.top > 0


def test_resize_preview_keeps_full_frame_when_aspect_matches_bucket():
    preview = compute_resize_preview(1008, 1024, 1024)

    assert preview.bucket_size == (1008, 1024)
    assert preview.kept_rect.left == pytest.approx(0)
    assert preview.kept_rect.top == pytest.approx(0)
    assert preview.kept_rect.width == pytest.approx(1008)
    assert preview.kept_rect.height == pytest.approx(1024)


def test_normalize_target_res_accepts_config_shapes():
    assert normalize_target_res("768, 1024") == [768, 1024]
    assert normalize_target_res(1024) == [1024]
    assert normalize_target_res([896, 1024]) == [896, 1024]


def test_resize_preview_applies_crop_anchor():
    center = compute_resize_preview(1600, 1000, 1024, bucket_resos=["1024x1008"])
    right = compute_resize_preview(
        1600,
        1000,
        1024,
        crop_anchor="right",
        bucket_resos=["1024x1008"],
    )

    assert center.kept_rect.left > 0
    assert right.kept_rect.left > center.kept_rect.left
    assert right.crop_anchor == "right"


def test_resize_preview_can_force_bucket_resolution():
    preview = compute_resize_preview(
        1440,
        2560,
        [896, 1024],
        bucket_resos=["576x1344"],
    )

    assert preview.bucket_size == (576, 1344)
    assert preview.target_edge == 896


def test_resize_bucket_filter_keeps_nearest_aspect_selection():
    buckets = [f"{width}x{height}" for width, height in buckets_for_edges([1024])]

    edge, bucket = select_resize_bucket(1440, 2560, 1024, buckets)

    assert edge == 1024
    assert bucket == (768, 1344)


def test_parse_bucket_resos_accepts_gui_and_cli_shapes():
    assert parse_bucket_resos(["1024x1008", "896:1344"]) == [
        (896, 1344),
        (1024, 1008),
    ]
    assert parse_bucket_resos("1024x1008, 896x1344") == [
        (896, 1344),
        (1024, 1008),
    ]


def test_resize_preview_applies_crop_margins_before_bucket_crop():
    preview = compute_resize_preview(
        1000,
        1000,
        1024,
        crop_margins={"top": 10, "right": 0, "bottom": 0, "left": 20},
        bucket_resos=["1008x1024"],
    )

    assert preview.margin_rect.left == pytest.approx(200)
    assert preview.margin_rect.top == pytest.approx(100)
    assert preview.margin_rect.width == pytest.approx(800)
    assert preview.margin_rect.height == pytest.approx(900)
    assert preview.kept_rect.left >= preview.margin_rect.left
    assert preview.kept_rect.top >= preview.margin_rect.top


def test_normalize_crop_margins_clamps_axis_totals():
    margins = normalize_crop_margins({"left": 80, "right": 80})

    assert margins["left"] + margins["right"] == pytest.approx(95)
