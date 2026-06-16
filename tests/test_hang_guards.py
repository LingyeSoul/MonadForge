"""Regression tests for the indefinite-hang guards added to the pipeline.

These close the "I clicked Train/Preprocess and it spins forever" class:
  1. ``safe_walk`` — a symlink cycle in the (symlinked) dataset tree used to
     make ``os.walk(followlinks=True)`` loop forever;
  2. ``hf_download`` error classification — only *transport* failures (which
     hang) get translated to a fail-fast error; 404-style "not found" must
     propagate so callers' specific handlers still work.

(The upstream daemon stall-watchdog guard was guard #3 here, but this fork
removes the training daemon in favor of the WebUI TaskService, so that suite
was dropped with it.)
"""

from __future__ import annotations

import os
import socket
import time

import pytest


# ----- 1. safe_walk cycle guard -----


def _make_cyclic_tree(root: str) -> None:
    os.makedirs(os.path.join(root, "a"))
    open(os.path.join(root, "top.txt"), "w").close()
    open(os.path.join(root, "a", "img1.png"), "w").close()
    os.symlink(root, os.path.join(root, "a", "back"))  # a/back -> root (cycle)
    os.symlink(os.path.join(root, "a"), os.path.join(root, "a", "self2"))  # diamond


def test_safe_walk_terminates_on_symlink_cycle(tmp_path):
    from library.io.walk import safe_walk

    _make_cyclic_tree(str(tmp_path))
    files = []
    t0 = time.time()
    for _dp, _dn, fn in safe_walk(str(tmp_path), followlinks=True):
        files.extend(fn)
        assert time.time() - t0 < 10, "safe_walk did not terminate on a cycle"
    # Each real file surfaces exactly once despite the cycle + diamond link.
    assert sorted(files) == ["img1.png", "top.txt"]


def test_safe_walk_matches_oswalk_without_links(tmp_path):
    from library.io.walk import safe_walk

    os.makedirs(tmp_path / "sub")
    open(tmp_path / "a.txt", "w").close()
    open(tmp_path / "sub" / "b.txt", "w").close()
    got = sorted(f for _dp, _dn, fn in safe_walk(str(tmp_path)) for f in fn)
    assert got == ["a.txt", "b.txt"]


# ----- 2. hf_download transport-vs-status classification -----


def test_hf_download_classifies_only_transport_errors():
    from library.runtime import hf_download as H

    import requests

    assert H._is_network_error(requests.exceptions.ConnectionError()) is True
    assert H._is_network_error(requests.exceptions.ConnectTimeout()) is True
    assert H._is_network_error(requests.exceptions.ReadTimeout()) is True
    assert H._is_network_error(socket.timeout()) is True
    assert H._is_network_error(TimeoutError()) is True
    # Non-transport must propagate unchanged (e.g. a 404 EntryNotFoundError the
    # tagger catches for best-effort optional files).
    assert H._is_network_error(ValueError()) is False
    assert H._is_network_error(KeyError()) is False


def test_ensure_hf_timeouts_pins_env(monkeypatch):
    from library.runtime.hf_download import ensure_hf_timeouts

    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    ensure_hf_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]
    assert os.environ["HF_HUB_ETAG_TIMEOUT"]
    # Respects a user-set value rather than clobbering it.
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    ensure_hf_timeouts()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "5"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
