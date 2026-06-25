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
