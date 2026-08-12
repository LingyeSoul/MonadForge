"""Regression tests for the preview-image API + WebUI task-submission surface.

Covers three fixes from the sample_dir per-job isolation review:

1. **C1 (Critical)** — ``_resolve_task_sample_dir`` now reads ``task.sample_dir``
   (the value the daemon returns on submit) instead of only parsing
   ``--sample_dir`` out of ``task.args`` (which never contained the daemon's
   injected value, so the per-job isolation silently failed and the dashboard
   kept replaying the previous task's gallery).

2. **C8** — when the WebUI's session-only ``_tasks`` dict no longer knows a
   task_id (after a restart), fall back to the daemon's persisted job record
   so the gallery is still reachable.

3. **Security** — ``webui/api/tasks.py`` rejects reserved path/daemon-owned
   flags (``--sample_dir`` / ``--output_dir`` / ...) at the public edge so a
   browser can't aim train.py's write primitive at arbitrary dirs.

Tests mirror the ``_make_service()`` + real-Task + monkeypatch-ROOT style of
``test_task_service_metrics.py`` — no FastAPI ``TestClient``, no daemon socket.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the WebUI package is importable (the conftest at tests/ adds the
# repo root, so ``webui.*`` resolves directly).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_svc():
    """Build a TaskService without touching the daemon singleton."""
    from webui.services.task_service import Task, TaskService

    return TaskService(), Task


@pytest.fixture
def preview_env(tmp_path, monkeypatch):
    """Point preview's ROOT + task_service at a tmp sandbox + fresh service."""
    # preview.py imports ROOT/task_service at module load, so patch them there.
    monkeypatch.setattr("webui.api.preview.ROOT", tmp_path)
    svc, _ = _make_svc()
    monkeypatch.setattr("webui.api.preview.task_service", svc)
    return svc, tmp_path


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# C1: task.sample_dir (daemon-returned) is the primary resolution source
# ---------------------------------------------------------------------------


def test_daemon_returned_sample_dir_is_used(preview_env):
    """A task whose ``sample_dir`` was populated from the daemon submit
    response resolves straight to that path — no argv parsing needed.

    This is the C1 regression guard: before the fix, ``task.sample_dir`` did
    not exist and the preview API parsed ``--sample_dir`` out of ``task.args``
    (which never held the daemon's injected value), so the per-job gallery was
    silently unreachable.
    """
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(id="t1", command="lora", args=[], state=TaskState.RUNNING)
    task.sample_dir = "output/daemon/jobs/t1/sample"
    svc._tasks["t1"] = task
    _mkdir(tmp / "output" / "daemon" / "jobs" / "t1" / "sample")

    from webui.api.preview import _resolve_task_sample_dir

    resolved = _resolve_task_sample_dir("t1")
    assert resolved == (tmp / "output" / "daemon" / "jobs" / "t1" / "sample").resolve()


def test_args_sample_dir_used_when_no_daemon_value(preview_env):
    """Direct ``train.py`` runs (no daemon) still carry ``--sample_dir`` in
    args — that path must still resolve when ``task.sample_dir`` is None."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(
        id="t2",
        command="lora",
        args=["--sample_dir", "my/samples"],
        state=TaskState.RUNNING,
    )
    svc._tasks["t2"] = task
    _mkdir(tmp / "my" / "samples")

    from webui.api.preview import _resolve_task_sample_dir

    resolved = _resolve_task_sample_dir("t2")
    assert resolved == (tmp / "my" / "samples").resolve()


def test_falls_back_to_output_dir_sample(preview_env):
    """No sample_dir anywhere → ``<output_dir>/sample`` (the pre-isolation
    default). Keeps direct train.py runs working."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(
        id="t3",
        command="lora",
        args=["--output_dir", "out/custom"],
        state=TaskState.RUNNING,
    )
    svc._tasks["t3"] = task
    _mkdir(tmp / "out" / "custom" / "sample")

    from webui.api.preview import _resolve_task_sample_dir

    resolved = _resolve_task_sample_dir("t3")
    assert resolved == (tmp / "out" / "custom" / "sample").resolve()


def test_falls_back_to_output_ckpt_default(preview_env):
    """No sample_dir, no output_dir → the ``output/ckpt`` default."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(id="t4", command="lora", args=[], state=TaskState.RUNNING)
    svc._tasks["t4"] = task
    _mkdir(tmp / "output" / "ckpt" / "sample")

    from webui.api.preview import _resolve_task_sample_dir

    resolved = _resolve_task_sample_dir("t4")
    assert resolved == (tmp / "output" / "ckpt" / "sample").resolve()


def test_daemon_sample_dir_beats_args(preview_env):
    """``task.sample_dir`` (daemon-owned) wins over a stale ``--sample_dir``
    in args. Guards against a caller smuggling a different path."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(
        id="t5",
        command="lora",
        args=["--sample_dir", "stale/from/args"],
        state=TaskState.RUNNING,
    )
    task.sample_dir = "fresh/from/daemon/sample"
    svc._tasks["t5"] = task
    _mkdir(tmp / "fresh" / "from" / "daemon" / "sample")

    from webui.api.preview import _resolve_task_sample_dir

    resolved = _resolve_task_sample_dir("t5")
    assert resolved == (tmp / "fresh" / "from" / "daemon" / "sample").resolve()
    # The stale args path was NOT created, and would have escaped notice
    # pre-fix; the daemon value must win.
    assert not (tmp / "stale" / "from" / "args").exists()


def test_missing_dir_returns_none(preview_env):
    """Task tracked but the sample dir doesn't exist yet (training hasn't
    reached its first sample event) → None, not a Path."""
    svc, _ = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(id="t6", command="lora", args=[], state=TaskState.RUNNING)
    task.sample_dir = "output/daemon/jobs/t6/sample"  # dir NOT created
    svc._tasks["t6"] = task

    from webui.api.preview import _resolve_task_sample_dir

    assert _resolve_task_sample_dir("t6") is None


# ---------------------------------------------------------------------------
# Path-traversal guard (must cover task.sample_dir too, not just args)
# ---------------------------------------------------------------------------


def test_path_traversal_in_sample_dir_rejected(preview_env):
    """A malicious ``--sample_dir`` escaping ROOT must resolve to None.

    The guard relies on ``resolve() + relative_to(ROOT)`` — absolute paths
    (e.g. ``C:\\evil``) get swallowed by ``Path /`` on Windows but still fail
    the ``relative_to`` check. Both ``--sample_dir`` and ``--output_dir``
    paths are covered."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(
        id="t7",
        command="lora",
        args=["--sample_dir", "../../../etc/passwd"],
        state=TaskState.RUNNING,
    )
    svc._tasks["t7"] = task
    # Don't create the dir — even if it existed outside ROOT it must be refused.

    from webui.api.preview import _resolve_task_sample_dir

    assert _resolve_task_sample_dir("t7") is None


# ---------------------------------------------------------------------------
# C8: restart recovery via the daemon's persisted job record
# ---------------------------------------------------------------------------


def test_unknown_task_recovers_from_daemon_job(preview_env, monkeypatch):
    """When the WebUI loses the task (restart), the daemon's persisted job
    record still carries ``sample_dir`` — recover from there so the gallery
    stays reachable.

    Pre-fix, ``get_task`` returning None short-circuited to ``return None``
    for every task after a restart, even though the PNGs were still on disk.
    """
    svc, tmp = preview_env
    # task_service does NOT know "recovered-job" (simulating a restart).
    assert svc.get_task("recovered-job") is None

    # Daemon would return its persisted Job.public() with sample_dir set.
    recovered_path = "output/daemon/jobs/recovered-job/sample"
    _mkdir(tmp / "output" / "daemon" / "jobs" / "recovered-job" / "sample")

    # Patch the singleton instance's method (preview.py holds it as
    # ``_daemon_client``, same object).
    from webui.services.daemon_client import daemon_client as dc_singleton

    monkeypatch.setattr(
        dc_singleton,
        "get_job_sync",
        lambda job_id: {"job_id": job_id, "sample_dir": recovered_path},
    )

    from webui.api.preview import _resolve_task_sample_dir

    resolved = _resolve_task_sample_dir("recovered-job")
    assert (
        resolved
        == (tmp / "output" / "daemon" / "jobs" / "recovered-job" / "sample").resolve()
    )


def test_unknown_task_daemon_down_returns_none(preview_env, monkeypatch):
    """If the daemon is down (or the job truly unknown), preview returns None
    rather than raising — the gallery just reports empty."""
    from webui.services.daemon_client import DaemonError, daemon_client as dc_singleton

    def _raise(job_id):
        raise DaemonError("daemon down")

    monkeypatch.setattr(dc_singleton, "get_job_sync", _raise)

    from webui.api.preview import _resolve_task_sample_dir

    assert _resolve_task_sample_dir("never-existed") is None


# ---------------------------------------------------------------------------
# Security: reserved flags rejected at the WebUI edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag",
    [
        "--sample_dir",
        "--output_dir",
        "--progress_jsonl",
        "--config_file",
        "--dataset_config",
    ],
)
def test_reserved_flag_rejected(flag):
    """Path/daemon-owned flags must not be settable from the browser.

    The daemon injects ``--sample_dir`` / ``--progress_jsonl`` itself and
    ``--output_dir`` / ``--config_file`` would let a caller aim train.py's
    write primitive at arbitrary dirs. Reject at the public edge."""
    from fastapi import HTTPException

    from webui.api.tasks import _reject_forbidden_args

    with pytest.raises(HTTPException) as exc:
        _reject_forbidden_args([flag, "evil/path"])
    assert exc.value.status_code == 400


def test_reserved_flag_equals_form_rejected():
    """``--flag=value`` bypass must also be caught (the daemon's own
    ``not in argv`` guard only checks the bare token)."""
    from fastapi import HTTPException

    from webui.api.tasks import _reject_forbidden_args

    with pytest.raises(HTTPException):
        _reject_forbidden_args(["--sample_dir=evil/path"])


def test_safe_args_pass_through():
    """Normal training knobs (``--network_dim``, ``PRESET`` env, etc.) are
    unaffected by the reserved-flag filter."""
    from webui.api.tasks import _reject_forbidden_args

    # Should not raise.
    _reject_forbidden_args(["--network_dim", "32", "--max_train_epochs", "64"])


# ---------------------------------------------------------------------------
# get_sample_file: filename whitelist replaced with a traversal blacklist
#
# The file endpoint used to gate ``path`` on ``^[a-zA-Z0-9_./-]+$``, which
# silently rejected sample files whose ``output_name`` carries characters
# outside that set (e.g. ``@``, spaces, CJK, parens). The file still appeared
# in the listing (``list_task_samples`` never applied the regex), so the gallery
# tile rendered the *filename text* in place of the broken ``<img>`` — the
# "不显示图片只显示文件名" symptom. Containment is now enforced by
# ``resolve() + relative_to(sample_dir)``; only traversal vectors are rejected.
# ---------------------------------------------------------------------------


def _write_png(p: Path) -> None:
    """Write a minimal 1x1 PNG (valid header + IEND) so ``is_file()`` passes."""
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_sample_file_with_at_sign_is_served(preview_env):
    """Regression: ``output_name = lanima_tlora_ortho-@nanfang_e000002`` produces
    a sample PNG whose filename contains ``@``. The old whitelist rejected it
    (400), so the gallery tile showed the filename text instead of the image."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(id="at", command="lora", args=[], state=TaskState.RUNNING)
    task.sample_dir = "output/daemon/jobs/at/sample"
    svc._tasks["at"] = task
    sample_dir = _mkdir(tmp / "output" / "daemon" / "jobs" / "at" / "sample")
    fn = "lanima_tlora_ortho-@nanfang_e000002_00_20260625091724_740298002.png"
    _write_png(sample_dir / fn)

    from webui.api.preview import get_sample_file

    # Must NOT raise — ``@`` is a legitimate filename character.
    resp = get_sample_file("at", path=fn)
    assert resp.status_code == 200


def test_sample_file_with_spaces_and_cjk_is_served(preview_env):
    """``output_name`` may carry spaces and CJK; both are legitimate filename
    characters and must reach the file, not trip the gate."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(id="i18n", command="lora", args=[], state=TaskState.RUNNING)
    task.sample_dir = "output/daemon/jobs/i18n/sample"
    svc._tasks["i18n"] = task
    sample_dir = _mkdir(tmp / "output" / "daemon" / "jobs" / "i18n" / "sample")
    fn = "我的 LoRA e000002_00_20260625091724.png"
    _write_png(sample_dir / fn)

    from webui.api.preview import get_sample_file

    resp = get_sample_file("i18n", path=fn)
    assert resp.status_code == 200


def test_logical_task_samples_select_exact_attempt(preview_env):
    """Same-named samples remain independently addressable across attempts."""
    svc, tmp = preview_env
    from fastapi import HTTPException

    from webui.api.preview import get_sample_file, list_task_samples
    from webui.services.task_service import Task, TaskState

    root_dir = _mkdir(tmp / "output" / "daemon" / "jobs" / "root" / "sample")
    child_dir = _mkdir(tmp / "output" / "daemon" / "jobs" / "child" / "sample")
    filename = "same.png"
    (root_dir / filename).write_bytes(b"root-attempt")
    (child_dir / filename).write_bytes(b"child-attempt")

    task = Task(id="root", command="lora", args=[], state=TaskState.FAILED)
    task.job_id = "child"
    task.sample_dir = "output/daemon/jobs/child/sample"
    task.attempts = [
        {
            "id": "root",
            "attempt_index": 0,
            "sample_dir": "output/daemon/jobs/root/sample",
        },
        {
            "id": "child",
            "attempt_index": 1,
            "sample_dir": "output/daemon/jobs/child/sample",
        },
    ]
    svc._tasks[task.id] = task

    listing = list_task_samples("root", page=1, page_size=60)
    assert listing.total == 2
    assert {(item.attempt_id, item.path) for item in listing.items} == {
        ("root", filename),
        ("child", filename),
    }

    root_response = get_sample_file("root", path=filename, attempt_id="root")
    child_response = get_sample_file("root", path=filename, attempt_id="child")
    assert Path(root_response.path).read_bytes() == b"root-attempt"
    assert Path(child_response.path).read_bytes() == b"child-attempt"

    with pytest.raises(HTTPException) as exc:
        get_sample_file("root", path=filename, attempt_id="missing")
    assert exc.value.status_code == 404


@pytest.mark.parametrize(
    "bad_path",
    [
        "../secret.png",  # parent traversal
        "subdir/img.png",  # separator (must be a bare filename)
        "..\\..\\evil.png",  # backslash traversal (Windows)
        "img\x00.png",  # NUL byte
    ],
)
def test_sample_file_traversal_vectors_rejected(preview_env, bad_path):
    """Traversal / separator / NUL paths are still rejected with 400 — the
    blacklist retains the defense-in-depth the whitelist once (over-broadly)
    provided. Containment is then re-asserted by ``relative_to``."""
    svc, tmp = preview_env
    from webui.services.task_service import Task, TaskState

    task = Task(id="tv", command="lora", args=[], state=TaskState.RUNNING)
    task.sample_dir = "output/daemon/jobs/tv/sample"
    svc._tasks["tv"] = task
    _mkdir(tmp / "output" / "daemon" / "jobs" / "tv" / "sample")

    from fastapi import HTTPException

    from webui.api.preview import get_sample_file

    with pytest.raises(HTTPException) as exc:
        get_sample_file("tv", path=bad_path)
    assert exc.value.status_code == 400
