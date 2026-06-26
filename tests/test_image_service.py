"""WebUI image_service — ``resolve_image_path`` filename safety.

Regression: the service used a positive character whitelist
``^[a-zA-Z0-9_./-]+$`` on the relative image path. Dataset filenames
legitimately contain characters outside that set (spaces, CJK, ``@``,
parens), and such images silently failed to load — they appeared in the
``list_images`` listing (which never applied the regex) but their
``<img>`` / caption / mask fetches all 404'd. Same root cause + symptom as
the training-preview fix in ``test_preview.py``.

Containment is now enforced by ``resolve() + relative_to(base)``; the
regex only rejects traversal vectors (``..`` segments, control bytes).
Relative paths MAY contain ``/`` because ``list_images`` recurses via
``rglob`` and returns sub-directory-relative paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def image_env(tmp_path, monkeypatch):
    """Point image_service's ROOT at a tmp sandbox and register a dataset dir
    via ``resolve_directory``'s absolute-path fallback."""
    monkeypatch.setattr("webui.services.image_service.ROOT", tmp_path)
    dataset = tmp_path / "ds"
    dataset.mkdir()
    return dataset, tmp_path


def _write_png(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _resolve(dataset: Path, rel_path: str) -> Path | None:
    from webui.services.image_service import resolve_image_path

    return resolve_image_path(str(dataset), rel_path)


def test_image_with_spaces_is_resolved(image_env):
    """``my image.png`` is a legitimate dataset filename."""
    dataset, _ = image_env
    _write_png(dataset / "my image.png")
    assert _resolve(dataset, "my image.png") == (dataset / "my image.png")


def test_image_with_at_sign_is_resolved(image_env):
    """``girl @ tag.png`` — the ``@`` that broke the preview gallery."""
    dataset, _ = image_env
    _write_png(dataset / "girl @ tag.png")
    assert _resolve(dataset, "girl @ tag.png") == (dataset / "girl @ tag.png")


def test_image_with_cjk_is_resolved(image_env):
    """CJK filenames are common in user datasets."""
    dataset, _ = image_env
    _write_png(dataset / "我的图片.png")
    assert _resolve(dataset, "我的图片.png") == (dataset / "我的图片.png")


def test_image_in_subdirectory_is_resolved(image_env):
    """``list_images`` recurses with rglob and returns sub-dir-relative paths,
    so ``sub/img.png`` (with a ``/``) must resolve — the whitelist used to
    reject slashes only because the path happened to be all-ASCII."""
    dataset, _ = image_env
    _write_png(dataset / "subset" / "img.png")
    assert _resolve(dataset, "subset/img.png") == (dataset / "subset" / "img.png")


@pytest.mark.parametrize(
    "bad_rel",
    [
        "../escape.png",  # parent traversal
        "sub/../../escape.png",  # traversal after a legit prefix
        "img\x00.png",  # NUL byte
    ],
)
def test_traversal_vectors_rejected(image_env, bad_rel):
    """Traversal / control-byte paths are still rejected — the blacklist
    keeps the defense-in-depth the whitelist once (over-broadly) provided,
    and ``relative_to`` re-asserts containment."""
    dataset, _ = image_env
    # The escape target exists but lives outside the dataset dir.
    _write_png(dataset.parent / "escape.png")
    assert _resolve(dataset, bad_rel) is None


def test_nonexistent_returns_none(image_env):
    """A missing image resolves to None (not a raise) — callers raise
    FileNotFoundError for a friendlier 404 message."""
    dataset, _ = image_env
    assert _resolve(dataset, "nope.png") is None


# ── Tag index tests ──────────────────────────────────────────────

def _write_txt(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_build_tag_index_basic(image_env):
    dataset, _ = image_env
    _write_png(dataset / "a.png")
    _write_txt(dataset / "a.txt", "1girl, solo, blue eyes")
    _write_png(dataset / "b.png")
    _write_txt(dataset / "b.txt", "1girl, cat ears")

    from webui.services.image_service import build_tag_index

    result = build_tag_index(str(dataset))
    assert result["total_images"] == 2
    assert result["tags"]["1girl"] == 2
    assert result["tags"]["solo"] == 1
    assert result["tags"]["cat ears"] == 1


def test_build_tag_index_empty(image_env):
    dataset, _ = image_env
    from webui.services.image_service import build_tag_index

    result = build_tag_index(str(dataset))
    assert result == {"tags": {}, "total_images": 0}


def test_build_tag_index_no_caption(image_env):
    dataset, _ = image_env
    _write_png(dataset / "a.png")

    from webui.services.image_service import build_tag_index

    result = build_tag_index(str(dataset))
    assert result["total_images"] == 0


# ── Batch caption update tests ───────────────────────────────────

def test_batch_append(image_env):
    dataset, _ = image_env
    _write_png(dataset / "a.png")
    _write_txt(dataset / "a.txt", "1girl, solo")

    from webui.services.image_service import batch_update_captions

    result = batch_update_captions(str(dataset), ["a.png"], "append", tag="smile, wink")
    assert result["updated"] == 1
    assert result["failed"] == 0
    content = (dataset / "a.txt").read_text(encoding="utf-8")
    assert "smile" in content
    assert "wink" in content


def test_batch_remove(image_env):
    dataset, _ = image_env
    _write_png(dataset / "a.png")
    _write_txt(dataset / "a.txt", "1girl, solo, smile")

    from webui.services.image_service import batch_update_captions

    result = batch_update_captions(str(dataset), ["a.png"], "remove", tag="solo")
    assert result["updated"] == 1
    content = (dataset / "a.txt").read_text(encoding="utf-8")
    assert "solo" not in content
    assert "1girl" in content


def test_batch_replace(image_env):
    dataset, _ = image_env
    _write_png(dataset / "a.png")
    _write_txt(dataset / "a.txt", "1girl, blue_eyes, solo")

    from webui.services.image_service import batch_update_captions

    result = batch_update_captions(str(dataset), ["a.png"], "replace", find="blue_eyes", replace="green_eyes")
    assert result["updated"] == 1
    content = (dataset / "a.txt").read_text(encoding="utf-8")
    assert "green_eyes" in content
    assert "blue_eyes" not in content


def test_batch_replace_regex(image_env):
    dataset, _ = image_env
    _write_png(dataset / "a.png")
    _write_txt(dataset / "a.txt", "1girl, by artist1, solo")

    from webui.services.image_service import batch_update_captions

    result = batch_update_captions(str(dataset), ["a.png"], "replace", find=r"by \w+", replace="by newartist", use_regex=True)
    assert result["updated"] == 1
    content = (dataset / "a.txt").read_text(encoding="utf-8")
    assert "by newartist" in content
    assert "by artist1" not in content


def test_batch_missing_image(image_env):
    dataset, _ = image_env
    from webui.services.image_service import batch_update_captions

    result = batch_update_captions(str(dataset), ["nonexistent.png"], "append", tag="test")
    assert result["updated"] == 0
    assert result["failed"] == 1
    assert len(result["errors"]) == 1
