from __future__ import annotations

import urllib.error
from io import BytesIO

import pytest

from scripts.tasks import downloads


def test_danbooru_tags_download_url_points_to_source_repo():
    assert downloads.DANBOORU_TAGS_URLS == (
        "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
    )


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self._payload


def test_download_danbooru_tags_writes_models_file(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",))
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda _req, timeout=60: _FakeResponse(
            b"name,category,post_count,description\n1girl,0,1,test\n"
        ),
    )

    downloads.cmd_download_danbooru_tags([])

    assert dest.read_text(encoding="utf-8").startswith("name,category")


def test_download_danbooru_tags_skips_existing_without_force(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    called = False

    def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        return _FakeResponse(BytesIO().read())

    monkeypatch.setattr(downloads.urllib.request, "urlopen", _fail_if_called)

    downloads.cmd_download_danbooru_tags([])

    assert not called
    assert dest.read_text(encoding="utf-8") == "existing"


def test_download_danbooru_tags_failure_names_source_repo(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",))
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("nope")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        downloads.cmd_download_danbooru_tags([])

    assert "Localsmile/danbooru_KR_wiki_tag_search" in str(exc.value)
