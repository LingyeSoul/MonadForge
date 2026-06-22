"""Tests for ``scripts/daemon/manager.py::_classify_exit``.

The function turns a process exit code into a user-facing diagnostic string.
The interesting branch is ``rc is None``: the adopted-orphan path
(``popen is None``) cannot ``poll()`` a process that vanished while the daemon
was down, so it lands here with no exit code at all. A prior version fell
through the signal checks and reported ``"process exited (code=None) — crashed
before finishing. See the last traceback above."`` — which pointed the user at a
non-existent traceback. The fix returns a distinct, actionable "exit code
unavailable" message.
"""

from __future__ import annotations

import pytest

from scripts.daemon.manager import _classify_exit


def test_rc_none_returns_actionable_unavailable_message():
    """``rc is None`` must NOT claim a crash with a traceback pointer."""
    msg = _classify_exit(None)

    assert "unavailable" in msg.lower()
    # The misleading pre-fix phrasing must be gone.
    assert "crashed before finishing" not in msg
    assert "traceback" not in msg.lower()


@pytest.mark.parametrize(
    "rc, expected_substring",
    [
        (-9, "killed (SIGKILL)"),
        (-6, "aborted (SIGABRT)"),
        (-11, "segfault (SIGSEGV)"),
        (-15, "terminated (SIGTERM)"),
        # 128 + signal → shell/launcher relay (e.g. accelerate launch)
        (128 + 9, "killed (SIGKILL)"),
        (128 + 6, "aborted (SIGABRT)"),
    ],
)
def test_signal_deaths_keep_their_actionable_hint(rc, expected_substring):
    """Signal-death classification is unchanged by the rc-is-None fix."""
    msg = _classify_exit(rc)
    assert expected_substring in msg
    assert f"code={rc}" in msg


def test_nonzero_non_signal_exit_keeps_generic_crash_message():
    """A plain nonzero exit (no signal) keeps the "see traceback" pointer."""
    msg = _classify_exit(1)

    assert "crashed before finishing" in msg
    assert "traceback" in msg.lower()
    assert "code=1" in msg


def test_rc_zero_is_still_classified():
    """``_classify_exit`` is only called on nonzero exits in practice, but it
    must remain total — rc=0 falls through to the generic branch (no signal
    match) without raising."""
    msg = _classify_exit(0)

    # No signal matches rc=0, so the generic branch runs.
    assert "code=0" in msg
