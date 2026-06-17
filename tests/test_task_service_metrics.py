"""Regression tests for ``webui/services/task_service.py``.

The WebUI used to display ``— / —`` for the training step counter until the
first tqdm bar leaked through the daemon's stdout file (often minutes
into a run, given the daemon throttles tqdm via ``TQDM_MININTERVAL=10``).
It also used to flood the WebSocket on training end: every JSONL ``step``
event produced a separate ``metrics`` message carrying the full growing
``loss_history``, so a long run queued thousands of messages whose final
burst arrived at the UI as a single visible flood right before ``done``.

Fixes (see webui/services/task_service.py):
  * ``run_start`` JSONL event populates ``total_steps`` / ``total_epochs``
    on the parser metrics so the dashboard renders ``step / total`` from
    step 1 — not when the first tqdm bar finally surfaces.
  * Step-event metrics WS emits are **debounced** (~0.3 s) so a burst of
    step events lands as a single message; the final pre-``done`` flush
    guarantees subscribers still see the last logged step.
  * Training tasks (``is_training=True``) suppress the parallel stdout
    ``metrics`` emit so two streams don't race and double-append history.

These tests don't touch the real daemon — they exercise the watcher and
the debounce helpers directly via in-process asyncio.run + a tempdir
JSONL file. They avoid ``pytest-asyncio`` (not in this project's deps)
by driving each test from a single top-level ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Make sure the WebUI package is importable (the conftest at tests/ adds
# the repo root, so ``webui.*`` resolves directly).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_service():
    """Build a TaskService without touching the daemon singleton.

    The production constructor pulls in ``daemon_client`` lazily, but the
    methods we exercise here don't talk to it. We use the real class so
    we test the same code paths the running WebUI hits.
    """
    from webui.services.task_service import TaskService, Task, TaskState

    svc = TaskService()
    return svc, Task, TaskState


async def _drain_queue(queue: asyncio.Queue, *, timeout: float = 0.1) -> list[dict]:
    """Drain everything available in *queue* right now."""
    out: list[dict] = []
    while True:
        try:
            out.append(await asyncio.wait_for(queue.get(), timeout=timeout))
        except asyncio.TimeoutError:
            return out


def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Append *events* to *path* as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _drive(coro):
    """Run an async test body from a sync pytest function.

    The project doesn't ship pytest-asyncio; this thin shim keeps the
    test bodies readable while staying within the configured deps.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 0. WebUI command jobs tail the daemon's per-job progress JSONL
# ---------------------------------------------------------------------------


def test_lora_gui_progress_jsonl_path_uses_daemon_job_file():
    """``lora-gui`` progress must be isolated per daemon job.

    The old fallback tailed ``output/logs/<output_name>.progress.jsonl``. That
    file is shared across runs, so a new dashboard could replay stale metrics
    immediately at task start. The daemon gives every job its own progress file;
    command-style training jobs are launched with ``--progress_jsonl`` pointing
    there, and the WebUI watcher tails the same per-job file.
    """
    svc, _, _ = _make_service()

    path = Path(
        svc._derive_progress_jsonl_path(
            {"job_id": "20260617-163000-abcdef"},
            "lora-gui",
            ["lora-8gb"],
            {"PRESET": "test"},
        )
    )

    assert path.name == "progress.jsonl"
    assert path.parent.name == "20260617-163000-abcdef"
    assert path.parent.parent.name == "jobs"


def test_non_training_command_has_no_progress_jsonl_watcher():
    """Preprocess/mask command jobs should not start the training JSONL watcher."""
    svc, _, _ = _make_service()

    assert (
        svc._derive_progress_jsonl_path(
            {"job_id": "20260617-163000-abcdef"},
            "preprocess",
            [],
            {},
        )
        is None
    )


def test_progress_jsonl_explicit_path_wins_for_command_job(tmp_path: Path):
    """A caller-provided ``--progress_jsonl`` must match train.py semantics."""
    svc, _, _ = _make_service()
    custom = tmp_path / "job.progress.jsonl"

    path = svc._derive_progress_jsonl_path(
        {"job_id": "ignored"},
        "lora-gui",
        ["lora-8gb", "--progress_jsonl", str(custom)],
        {"PRESET": "test"},
    )

    assert path == str(custom)


# ---------------------------------------------------------------------------
# 0b. JSONL step timing derives speed / elapsed / ETA
# ---------------------------------------------------------------------------


def test_jsonl_step_timing_derives_speed_elapsed_eta():
    """Speed should not depend solely on stdout tqdm redraws.

    The structured JSONL stream carries relative timestamps. Deriving timing
    fields from consecutive step events lets the dashboard show speed even when
    stdout tqdm is throttled or delayed.
    """
    svc, Task, TaskState = _make_service()
    task = Task(id="tim", command="lora-gui", args=[], state=TaskState.RUNNING)
    metrics = task.parser.metrics
    metrics.total_steps = 22
    task.progress_started_at = 0.0
    task.progress_last_step = 0
    task.progress_last_ts = 0.0

    svc._update_jsonl_timing_metrics(task, {"global_step": 8, "ts": 24.0})

    assert metrics.elapsed == "00:24"
    assert metrics.speed == "3.00 s/it"
    assert metrics.eta == "00:42"


# ---------------------------------------------------------------------------
# 1. run_start event populates total_steps / total_epochs
# ---------------------------------------------------------------------------


def test_run_start_populates_total_steps(tmp_path: Path):
    """The dashboard needs total_steps before the first step completes.

    Regression: under the daemon, tqdm was throttled to 10 s so the
    ``step / total`` denominator was missing for the first minute of
    training. The JSONL ``run_start`` event always carries it though, so
    we now read it eagerly.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    task = Task(id="t1", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    svc._tasks["t1"] = task
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)

    # Write run_start first, then the watcher; the file already exists
    # so the watcher's "wait for it to appear" loop exits immediately.
    _write_jsonl(jsonl, [{"ev": "run_start", "total_steps": 1000, "total_epochs": 10}])

    async def _run():
        # Schedule a stopper that flips state to SUCCESS after the
        # watcher has had a chance to process the run_start event.
        async def _stop():
            await asyncio.sleep(0.15)
            task.state = TaskState.SUCCESS

        stopper = asyncio.create_task(_stop())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await stopper
        # Flush any pending debounced metrics.
        await svc._flush_metrics_emit(task)

    _drive(_run())

    # The run_start handler should have set total_steps / total_epochs
    # on the parser metrics.
    assert task.parser.metrics.total_steps == 1000
    assert task.parser.metrics.total_epochs == 10

    # And a metrics message should have been pushed carrying the new
    # total — this is what lets a late-joining subscriber render
    # ``step / total`` from step 1 onward.
    msgs = _drive(_drain_queue(sub, timeout=0.1))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    assert metrics_msgs, "run_start should have triggered a metrics emit"
    assert metrics_msgs[0]["data"]["total_steps"] == 1000
    assert metrics_msgs[0]["data"]["total_epochs"] == 10


# ---------------------------------------------------------------------------
# 2. Debounce coalesces a burst of step events into one WS message
# ---------------------------------------------------------------------------


def test_step_events_emit_metrics_immediately(tmp_path: Path):
    """A burst of step events produces one metrics message per event.

    The earlier debounce stacked every emit on the same 0.3 s window and
    kept getting reset by the next event, so the dashboard never saw
    a metrics message until the watcher's ``finally`` flush. We now
    forward each ``step`` event directly — the trainer's
    ``log_every_n_steps`` cadence (default 2) bounds the WS load, and
    the loss / step history is deduped by ``global_step`` so the
    direct emit can't double-append.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    task = Task(id="deb", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["deb"] = task

    # Pre-create the file so the watcher's first poll is a no-op
    # until we append events; then drop 50 step events on disk before
    # the watcher reads. The first poll reads all 50 at once.
    jsonl.write_text("", encoding="utf-8")
    _write_jsonl(
        jsonl,
        [
            {
                "ev": "step",
                "global_step": i,
                "epoch": 0,
                "avr_loss": 1.0 / (i + 1),
                "lr": 1e-4,
            }
            for i in range(1, 51)
        ],
    )

    async def _run():
        async def _stop():
            await asyncio.sleep(0.5)
            task.state = TaskState.SUCCESS

        stopper = asyncio.create_task(_stop())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await stopper

    _drive(_run())

    msgs = _drive(_drain_queue(sub, timeout=0.2))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]

    # Each step event must produce its own metrics message — 50 events
    # in the file means 50 messages to the frontend. The dashboard
    # receives them as the watcher reads them, not all at the end.
    assert len(metrics_msgs) == 50, (
        f"expected 50 per-step metrics messages, got {len(metrics_msgs)}: "
        "regression — step events aren't reaching the dashboard live"
    )
    # The last message carries the freshest step; the history grew
    # monotonically across the batch.
    assert metrics_msgs[-1]["data"]["step"] == 50
    assert len(metrics_msgs[-1]["data"]["step_history"]) == 50
    assert len(metrics_msgs[-1]["data"]["loss_history"]) == 50


# ---------------------------------------------------------------------------
# 2b. JSONL ``step`` events use accelerate's ``group/key`` field naming
# ---------------------------------------------------------------------------


def test_jsonl_step_maps_loss_average_and_lr_unet(tmp_path: Path):
    """The trainer writes scalars under ``loss/average`` and ``lr/unet``.

    Regression: the watcher used to look for the legacy flat names
    (``avr_loss``, ``lr``). Under the real trainer those keys never
    appear in the JSONL — the dashboard rendered ``— / —`` for step
    and loss even though the JSONL had fresh values on every line. The
    watcher now translates the accelerate-style names into the
    dashboard's expected field names.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    jsonl.write_text("", encoding="utf-8")
    task = Task(id="map", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["map"] = task

    # Real trainer JSONL output: accelerate keys, not the legacy names.
    _write_jsonl(
        jsonl,
        [
            {
                "ev": "step",
                "ts": 1.0,
                "loss/current": 0.3,
                "loss/average": 0.28,
                "lr/unet": 9.5e-5,
                "global_step": 4,
                "epoch": 1,
            }
        ],
    )

    async def _run():
        async def _stop():
            await asyncio.sleep(0.6)
            task.state = TaskState.SUCCESS

        stopper = asyncio.create_task(_stop())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await stopper

    _drive(_run())

    msgs = _drive(_drain_queue(sub, timeout=0.2))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    assert metrics_msgs, "JSONL step event should have produced a metrics emit"
    data = metrics_msgs[-1]["data"]
    # Field name mapping: ``loss/average`` → ``avr_loss``,
    # ``lr/unet`` → ``lr``. Without the mapping, these stay at 0 and
    # the dashboard shows ``—`` for both.
    assert data["avr_loss"] == pytest.approx(0.28)
    assert data["lr"] == pytest.approx(9.5e-5)
    assert data["step"] == 4
    assert data["epoch"] == 1
    # And the history was appended (one new point, deduped by step).
    assert data["step_history"] == [4]
    assert data["loss_history"] == [pytest.approx(0.28)]


# ---------------------------------------------------------------------------
# 3. Stdout parser IS the source of speed / total_steps / elapsed / eta
#    (the JSONL doesn't carry these — only ``avr_loss`` / ``lr`` / ``epoch``)
# ---------------------------------------------------------------------------


def test_emit_line_emits_metrics_for_training_task():
    """Training tasks still get metrics from stdout — it's the only source
    of tqdm-derived scalars.

    The JSONL ``step`` event carries ``avr_loss``, ``lr/unet``, ``epoch``,
    and ``global_step``. The ``speed``, ``total_steps``, ``elapsed``, and
    ``eta`` fields come from the stdout tqdm bar — they're never written
    to the JSONL stream. Suppressing the stdout emit for training tasks
    (an earlier over-aggressive fix) silently dropped those fields; the
    dashboard's progress ring and speed/elapsed cards went blank. Both
    paths must update the metrics stream.
    """
    svc, Task, TaskState = _make_service()
    task = Task(id="st", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["st"] = task

    # A tqdm line that the parser would normally match and emit metrics
    # for. Format matches the ``_TQDM_RE`` in training_log_parser.py.
    tqdm_line = (
        "steps:  10%|...| 100/1000 [00:10<01:30,  1.50it/s, avr_loss=0.1234, "
        "lr=1.00e-04, router_H=1.0]"
    )

    async def _run():
        assert task.parser.feed(tqdm_line)  # parser still updates metrics
        await svc._emit_line(task, tqdm_line, replace=False)

    _drive(_run())
    msgs = _drive(_drain_queue(sub, timeout=0.05))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    # The stdout path MUST still emit a metrics message for training
    # tasks — that's where ``total_steps`` / ``speed`` / ``elapsed``
    # come from.
    assert metrics_msgs, (
        "training task stdout tqdm redraw must still emit a metrics "
        "message; the JSONL has no speed/elapsed/total_steps fields"
    )
    data = metrics_msgs[0]["data"]
    assert data["step"] == 100
    assert data["total_steps"] == 1000
    assert data["speed"] == "1.50 it/s"


def test_emit_line_emits_metrics_for_non_training_task():
    """Non-training tasks still get metrics from the stdout path.

    The JSONL watcher only runs for tasks that derive a progress JSONL
    path (training). For everything else, the stdout parser is the only
    source — the suppression must not bleed into those tasks.
    """
    svc, Task, TaskState = _make_service()
    task = Task(id="nt", command="preprocess", args=[], state=TaskState.RUNNING)
    task.is_training = False  # default
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["nt"] = task

    tqdm_line = "steps:  10%|...| 100/1000 [00:10<01:30,  1.50it/s, avr_loss=0.5]"

    async def _run():
        await svc._emit_line(task, tqdm_line, replace=False)

    _drive(_run())
    msgs = _drive(_drain_queue(sub, timeout=0.05))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    assert len(metrics_msgs) == 1
    assert metrics_msgs[0]["data"]["step"] == 100


# ---------------------------------------------------------------------------
# 4. (removed — flush was a debounce helper; direct emit replaced it)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. run_end emits a final-step metric so the dashboard ends on the
#    trainer's authoritative final_step
# ---------------------------------------------------------------------------


def test_run_end_emits_final_step_metric(tmp_path: Path):
    """The ``run_end`` event surfaces ``final_step`` to the dashboard.

    The trainer's ``run_end`` event carries the authoritative
    ``final_step`` — the exact step training reached, which can be a
    few steps past the last logged ``step`` event (the loss / sample
    logs are at ``log_every_n_steps`` cadence, not every step). The
    watcher updates ``metrics.step`` and emits a fresh metrics
    message so the dashboard ends on the trainer's reported value,
    not the last logged one.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    task = Task(id="re", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["re"] = task

    # Write the last ``step`` event first, then ``run_end`` — both
    # before the watcher's first poll. Both messages are forwarded
    # directly to subscribers; the final one carries ``final_step``.
    jsonl.write_text("", encoding="utf-8")
    _write_jsonl(
        jsonl,
        [
            {"ev": "step", "global_step": 999, "epoch": 0, "avr_loss": 0.1, "lr": 1e-4},
            {"ev": "run_end", "status": "ok", "final_step": 1000},
        ],
    )

    async def _run():
        async def _stop():
            await asyncio.sleep(0.5)
            task.state = TaskState.SUCCESS

        stopper = asyncio.create_task(_stop())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await stopper

    _drive(_run())

    msgs = _drive(_drain_queue(sub, timeout=0.2))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    assert metrics_msgs, "expected at least one metrics message"
    # The final metrics message must reflect the run_end final_step
    # (the trainer's authoritative "we stopped at step 1000" signal —
    # not the last ``step`` event, which logged 999).
    assert metrics_msgs[-1]["data"]["step"] == 1000
    # Two messages expected: one for the step event, one for the
    # run_end event. (Previously the debounce collapsed them into one
    # — the dashboard's loss curve still rendered correctly because
    # both shared the same history, but the run_end's final_step
    # update was being lost in the debounce window.)
    assert len(metrics_msgs) == 2, (
        f"expected 2 metrics messages (step + run_end), got "
        f"{len(metrics_msgs)}"
    )


# ---------------------------------------------------------------------------
# 6. Watcher must wait for the daemon to launch the subprocess
# ---------------------------------------------------------------------------


def test_watcher_waits_for_running_state(tmp_path: Path):
    """The JSONL watcher must wait for ``task.state == RUNNING`` first.

    Regression: under the daemon design, ``start_task`` sets
    ``task.state = PENDING`` (the daemon's serial queue may hold the
    job for a while — GPU guard, queue pause, another job ahead).
    The old direct-subprocess design set ``RUNNING`` directly so the
    JSONL watcher's ``while state == RUNNING`` loop was always true on
    the first iteration. The naive port kept that condition; the
    watcher then exited on its first check and the dashboard never
    saw a single step event, even though the trainer wrote hundreds
    of them. The watcher must wait for ``_poll_daemon_job`` to flip
    the state to ``RUNNING`` before tailing the file.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    # Critical: start as PENDING (the real ``start_task`` path).
    task = Task(id="pen", command="lora", args=[], state=TaskState.PENDING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["pen"] = task

    # Pre-create the file with a step event so once the watcher
    # transitions to RUNNING and the file is found, it has something
    # to read.
    jsonl.write_text("", encoding="utf-8")
    _write_jsonl(
        jsonl,
        [
            {
                "ev": "step",
                "global_step": 7,
                "epoch": 0,
                "avr_loss": 0.5,
                "lr": 1e-4,
            }
        ],
    )

    async def _run():
        # Schedule a transition to RUNNING on a small delay — this
        # mimics ``_poll_daemon_job`` flipping the state once the
        # daemon's worker dequeues the job.
        async def _flip_to_running():
            await asyncio.sleep(0.3)
            task.state = TaskState.RUNNING

        # Then schedule termination after the watcher has had time to
        # process at least one batch.
        async def _terminate():
            await asyncio.sleep(1.0)
            task.state = TaskState.SUCCESS

        flipper = asyncio.create_task(_flip_to_running())
        terminator = asyncio.create_task(_terminate())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await flipper
            await terminator

    _drive(_run())

    msgs = _drive(_drain_queue(sub, timeout=0.2))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    # The watcher must have processed the step event — without the
    # wait-for-RUNNING fix, the queue would be empty (the watcher
    # would have exited on its first ``state != RUNNING`` check).
    assert metrics_msgs, (
        "watcher exited before the daemon launched the subprocess; "
        "dashboard would never see any step events"
    )
    assert metrics_msgs[-1]["data"]["step"] == 7
