from __future__ import annotations

from pathlib import Path

import pytest

from library.runtime import ms_download


class _FakeStream:
    """Minimal urlopen() response: context manager + status + headers + read(n)."""

    def __init__(self, payload: bytes, status: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status = status
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        if n <= 0:
            data, self._payload = self._payload, b""
            return data
        data, self._payload = self._payload[:n], self._payload[n:]
        return data

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_resolve_url_quotes_path():
    assert ms_download.resolve_url("org/name", "a b/c.bin") == (
        "https://modelscope.cn/models/org/name/resolve/master/a%20b/c.bin"
    )


def test_source_enabled_env_gated(monkeypatch):
    monkeypatch.delenv("ANIMA_DOWNLOAD_SOURCE", raising=False)
    assert ms_download.source_enabled() is False
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    assert ms_download.source_enabled() is True
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "ModelScope")
    assert ms_download.source_enabled() is True


def test_mirror_repo_mapping():
    assert ms_download.mirror_repo("facebook/PE-Core-L14-336") == (
        "AI-ModelScope/PE-Core-L14-336"
    )
    assert ms_download.mirror_repo("sorryhyun/anima-tagger") is None


def test_repo_files_filters_blobs(monkeypatch):
    payload = b"""
    {"Data": {"Files": [
        {"Type": "tree", "Path": "sub", "Size": 0},
        {"Type": "blob", "Path": "sub/b.bin", "Size": 7},
        {"Type": "blob", "Path": "a.bin", "Size": 3}
    ]}}
    """
    monkeypatch.setattr(ms_download, "_open", lambda url: _FakeStream(payload))
    assert ms_download.repo_files("org/name") == [("sub/b.bin", 7), ("a.bin", 3)]


def test_download_file_resumes_part(tmp_path, monkeypatch):
    dst = tmp_path / "f.bin"
    part = dst.with_name("f.bin.part")
    part.write_bytes(b"abcd")

    captured = {}

    def fake_open(url, headers=None):
        captured["headers"] = headers or {}
        return _FakeStream(
            b"efghij", status=206, headers={"Content-Range": "bytes 4-9/10"}
        )

    monkeypatch.setattr(ms_download, "_open", fake_open)
    out = ms_download.download_file("https://x/f.bin", dst, expected_size=10)

    assert out == dst
    assert dst.read_bytes() == b"abcdefghij"
    assert not part.exists()
    assert captured["headers"]["Range"] == "bytes=4-"


def test_download_file_restarts_when_range_ignored(tmp_path, monkeypatch):
    dst = tmp_path / "f.bin"
    dst.with_name("f.bin.part").write_bytes(b"abcd")
    monkeypatch.setattr(
        ms_download, "_open", lambda url, headers=None: _FakeStream(b"abcdefghij")
    )
    ms_download.download_file("https://x/f.bin", dst, expected_size=10)
    assert dst.read_bytes() == b"abcdefghij"


def test_download_file_restarts_on_mismatched_content_range(tmp_path, monkeypatch):
    """A 206 that resumes at the wrong offset would corrupt the midsection on
    append — it must restart from zero instead."""
    dst = tmp_path / "f.bin"
    part = dst.with_name("f.bin.part")
    part.write_bytes(b"abcd")
    captured = {}

    class _Resp(_FakeStream):
        def __init__(self, payload):
            super().__init__(payload, status=206)
            self.headers = {"Content-Range": "bytes 2-9/10"}

    def fake_open(url, headers=None):
        captured["headers"] = headers or {}
        # Server claims a resume from byte 2 (not the requested 4); after the
        # client restarts from zero it must still write the full body cleanly.
        return _Resp(b"abcdefghij")

    monkeypatch.setattr(ms_download, "_open", fake_open)
    ms_download.download_file("https://x/f.bin", dst, expected_size=10)

    assert dst.read_bytes() == b"abcdefghij"  # full 10 bytes, no shifted append
    assert captured["headers"]["Range"] == "bytes=4-"


def test_range_start_parses_header():
    assert ms_download._range_start("bytes 4-9/10") == 4
    assert ms_download._range_start("bytes 0-0/1") == 0
    assert ms_download._range_start(None) is None
    assert ms_download._range_start("garbage") is None


def test_download_file_incomplete_keeps_part(tmp_path, monkeypatch):
    dst = tmp_path / "f.bin"
    monkeypatch.setattr(
        ms_download, "_open", lambda url, headers=None: _FakeStream(b"abc")
    )
    with pytest.raises(IOError, match="incomplete"):
        ms_download.download_file("https://x/f.bin", dst, expected_size=10)
    assert dst.with_name("f.bin.part").read_bytes() == b"abc"


def test_snapshot_download_include_and_skip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ms_download,
        "repo_files",
        lambda repo, rev="master": [("a.bin", 3), ("sub/b.bin", 5)],
    )
    fetched: list[str] = []

    def fake_download(url, dst, *, expected_size=None, what="asset"):
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x" * (expected_size or 0))
        fetched.append(dst.name)
        return dst

    monkeypatch.setattr(ms_download, "download_file", fake_download)

    out = ms_download.snapshot_download("org/name", tmp_path, include=["sub/*"])
    assert [p.name for p in out] == ["b.bin"]
    assert (tmp_path / "sub" / "b.bin").read_bytes() == b"xxxxx"

    out = ms_download.snapshot_download("org/name", tmp_path)
    assert [p.name for p in out] == ["a.bin"]  # b.bin complete -> skipped

    out = ms_download.snapshot_download("org/name", tmp_path, force=True)
    assert {p.name for p in out} == {"a.bin", "b.bin"}


def test_snapshot_download_missing_filename_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ms_download, "repo_files", lambda repo, rev="master": [("a.bin", 1)]
    )
    with pytest.raises(FileNotFoundError, match="gone.bin"):
        ms_download.snapshot_download("org/name", tmp_path, filenames=["gone.bin"])


def test_fetch_file_matches_by_basename(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ms_download,
        "repo_files",
        lambda repo, rev="master": [("nested/PE-Core-L14-336.pt", 5)],
    )
    fetched: list[Path] = []

    def fake_download(url, dst, *, expected_size=None, what="asset", force=False):
        Path(dst).write_bytes(b"pptpt")
        fetched.append(Path(dst))
        return Path(dst)

    monkeypatch.setattr(ms_download, "download_file", fake_download)
    out = ms_download.fetch_file(
        ms_repo="m", filename="PE-Core-L14-336.pt", local_dir=tmp_path
    )
    assert out == tmp_path / "PE-Core-L14-336.pt"
    assert out.read_bytes() == b"pptpt"

    # Complete copy is returned untouched (no download call possible to assert
    # directly here — the size check short-circuits before download_file).
    again = ms_download.fetch_file(
        ms_repo="m", filename="PE-Core-L14-336.pt", local_dir=tmp_path
    )
    assert again == out
    assert len(fetched) == 1

    # --force re-fetches even though the copy is complete.
    ms_download.fetch_file(
        ms_repo="m", filename="PE-Core-L14-336.pt", local_dir=tmp_path, force=True
    )
    assert len(fetched) == 2

    with pytest.raises(FileNotFoundError, match="no file named"):
        ms_download.fetch_file(ms_repo="m", filename="other.pt", local_dir=tmp_path)


def test_fetch_file_duplicate_basename_prefers_shallowest(monkeypatch, tmp_path):
    """When several repo paths share the basename the pick must be
    deterministic (shallowest path), not listing-order luck."""
    monkeypatch.setattr(
        ms_download,
        "repo_files",
        lambda repo, rev="master": [
            ("deep/nest/weights.bin", 99),
            ("weights.bin", 5),
            ("aaa/weights.bin", 7),
        ],
    )
    seen: list[str] = []

    def fake_download(url, dst, *, expected_size=None, what="asset", force=False):
        seen.append(url)
        return Path(dst)

    monkeypatch.setattr(ms_download, "download_file", fake_download)
    ms_download.fetch_file(ms_repo="m", filename="weights.bin", local_dir=tmp_path)
    assert seen == ["https://modelscope.cn/models/m/resolve/master/weights.bin"]


def test_maybe_ms_download_gated_by_env_and_mirror(monkeypatch, tmp_path):
    monkeypatch.delenv("ANIMA_DOWNLOAD_SOURCE", raising=False)
    monkeypatch.setattr(
        ms_download,
        "_open",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network off-limits")),
    )
    assert (
        ms_download.maybe_ms_download(
            repo_id="facebook/PE-Core-L14-336",
            filename="PE-Core-L14-336.pt",
            local_dir=tmp_path,
        )
        is None
    )

    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    # Selected source + no mirror must raise, NOT silently fall back to HF —
    # the user picked ModelScope because HF is unreachable.
    with pytest.raises(FileNotFoundError, match="no ModelScope mirror"):
        ms_download.maybe_ms_download(
            repo_id="facebook/PE-Spatial-B16-512", filename="x.pt", local_dir=tmp_path
        )

    monkeypatch.setattr(
        ms_download,
        "fetch_file",
        lambda *, ms_repo, filename, local_dir, what="asset", force=False: (
            Path(local_dir) / filename
        ),
    )
    out = ms_download.maybe_ms_download(
        repo_id="facebook/PE-Core-L14-336",
        filename="PE-Core-L14-336.pt",
        local_dir=tmp_path,
    )
    assert out == tmp_path / "PE-Core-L14-336.pt"
