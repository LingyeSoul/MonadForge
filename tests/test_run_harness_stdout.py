"""Regression tests for ``scripts/tasks/_common.py::run`` stdout handling.

The training daemon (introduced to host the WebUI) launches
``tasks.py <command>`` with stdout/stderr redirected to a regular file
(``<job_dir>/stdout.log``) via ``spawn_detached``. ``tasks.py`` then
launches ``train.py`` (the actual training subprocess) via ``run``.

Before the fix, ``run`` on Windows-without-console unconditionally
redirected the grandchild's stdout to ``DEVNULL`` — a guard against a
parent pipe stalling on high-volume tqdm output. Under the daemon that
guard is *counterproductive*: our own stdout is a regular file (not a
pipe), so there's no risk of buffer fill, but the redirect throws away
the grandchild's tqdm redraws and the WebUI's live-log view shows a
single frozen log line (the trainer's Python ``logging`` output on
stderr is unaffected — that's why the dashboard wasn't completely
empty).

These tests pin both halves of the fix: the ``_stdout_is_regular_file``
helper, and the ``run`` redirect decision.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest


def _make_module():
    """Import the helper module under its real name.

    The test doesn't run the actual training subprocess — it just
    exercises the ``_stdout_is_regular_file`` detector and inspects
    how ``run`` chooses its kwargs.
    """
    from scripts.tasks import _common

    return _common


def test_stdout_is_regular_file_true_for_disk_file(tmp_path: Path):
    """A regular file on disk reads as a regular file."""
    _common = _make_module()
    log = tmp_path / "stdout.log"
    with log.open("wb", buffering=0) as fh:
        with mock.patch.object(sys, "stdout", fh):
            assert _common._stdout_is_regular_file() is True


def test_stdout_is_regular_file_false_for_pipe():
    """subprocess.PIPE is a pipe, not a regular file."""
    _common = _make_module()
    # subprocess.PIPE is a sentinel; the real fd doesn't exist until
    # Popen runs. We can't ``fileno()`` it, so the helper should
    # gracefully return False (the "not a regular file" branch).
    fake_pipe = mock.MagicMock(spec=subprocess.PIPE)
    with mock.patch.object(sys, "stdout", fake_pipe):
        # fileno() raises ValueError on the PIPE sentinel — the helper
        # should catch it and treat the stream as not-a-regular-file.
        assert _common._stdout_is_regular_file() is False


def test_stdout_is_regular_file_false_for_devnull():
    """DEVNULL is not a regular file — fall back to the pipe-safe branch."""
    _common = _make_module()
    with mock.patch.object(sys, "stdout", subprocess.DEVNULL):
        assert _common._stdout_is_regular_file() is False


def test_stdout_is_regular_file_false_for_stringio():
    """An in-memory buffer (e.g. test capture) is not a regular file."""
    _common = _make_module()
    with mock.patch.object(sys, "stdout", io.StringIO()):
        # StringIO has no fileno() — should fall through to False.
        assert _common._stdout_is_regular_file() is False


def test_run_does_not_redirect_to_devnull_when_stdout_is_a_file(
    monkeypatch, tmp_path: Path
):
    """Regression: under the daemon, stdout is a file — don't DEVNULL.

    The original code unconditionally redirected the grandchild's
    stdout to ``DEVNULL`` whenever the parent had no console. Under
    the daemon the parent is ``tasks.py`` with stdout already pointing
    at ``<job_dir>/stdout.log`` — a regular file, not a pipe. Throwing
    the grandchild's tqdm output to ``DEVNULL`` silently strands the
    WebUI's live-log view: only stderr (Python ``logging``) lands in
    the file, the dashboard shows a single frozen log line.
    """
    _common = _make_module()

    log = tmp_path / "stdout.log"
    log.write_bytes(b"")
    # Text mode + line-buffering mimics the daemon's ``open(..., "ab", 0)``
    # but accepts the ``print(...)`` strings ``run`` writes internally.
    # (The file handle returned is still backed by a regular file, which
    # is what the detector checks.)
    with log.open("a", buffering=1, encoding="utf-8") as stdout_fh:
        # ``subprocess.run`` is mocked, so its return value would be a
        # ``MagicMock`` whose ``returncode`` is truthy and triggers
        # ``sys.exit``. Pretend the grandchild exited 0.
        run_mock = mock.MagicMock(return_value=mock.MagicMock(returncode=0))
        with mock.patch.object(sys, "stdout", stdout_fh), mock.patch.object(
            sys, "stderr", stdout_fh
        ), mock.patch.object(
            _common, "_has_console", return_value=False
        ), mock.patch(
            "subprocess.run", run_mock
        ):
            _common.run(["python", "-c", "pass"])

    _, kwargs = run_mock.call_args
    # The grandchild must inherit our stdout (i.e. the daemon's file),
    # not get redirected to DEVNULL — that's the fix.
    assert kwargs.get("stdout") in (None, stdout_fh), (
        f"run() redirected grandchild stdout to {kwargs.get('stdout')!r} "
        "under the daemon's regular-file stdout; that strands the WebUI "
        "live-log view. Should inherit (None) or the file handle."
    )


def test_run_still_redirects_to_devnull_when_stdout_is_a_pipe(monkeypatch, tmp_path: Path):
    """The pipe-safe DEVNULL redirect is preserved for non-daemon callers.

    The legacy path (WebUI / QProcess direct subprocess) keeps the
    parent's stdout as a pipe. The trainer's tqdm output would fill
    the OS pipe buffer and stall. The redirect to DEVNULL still
    applies in that case — only the daemon's regular-file stdout
    skips it.
    """
    _common = _make_module()

    # Simulate a parent whose stdout is a pipe: ``fstat`` on a pipe
    # returns ``S_IFIFO`` (``stat.S_ISFIFO``), not ``S_ISREG``, so
    # ``_stdout_is_regular_file()`` returns False. We use a real
    # ``os.pipe()`` so ``sys.stdout`` is a real text file handle that
    # accepts ``print()`` writes (the read end is intentionally
    # unused — ``run()`` mocks ``subprocess.run`` so no grandchild
    # inherits anything).
    r_fd, w_fd = os.pipe()
    try:
        pipe_text = os.fdopen(w_fd, "w", encoding="utf-8", buffering=1)
        run_mock = mock.MagicMock(return_value=mock.MagicMock(returncode=0))
        with mock.patch.object(sys, "stdout", pipe_text), mock.patch.object(
            _common, "_has_console", return_value=False
        ), mock.patch("subprocess.run", run_mock):
            _common.run(["python", "-c", "pass"])
    finally:
        os.close(r_fd)

    _, kwargs = run_mock.call_args
    # Pipe stdout must still get the DEVNULL treatment so high-volume
    # tqdm output doesn't deadlock the grandchild.
    assert kwargs.get("stdout") is subprocess.DEVNULL, (
        "run() failed to apply the pipe-safe DEVNULL redirect; "
        "the grandchild's tqdm output would fill the parent's pipe "
        "buffer and stall the trainer."
    )
