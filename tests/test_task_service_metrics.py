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


def test_step_events_are_debounced(tmp_path: Path):
    """A burst of step events should land as a single metrics message.

    Regression: without debouncing, every step event sent its own
    ``metrics`` WS message. Over a long run the queue backed up and the
    final pre-``done`` flood dumped the whole history on the UI at once.
    The fix coalesces step events inside a ~0.3 s window.
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
            # > debounce window so the scheduled send fires; long
            # headroom for slow CI runners.
            await asyncio.sleep(0.8)
            task.state = TaskState.SUCCESS

        stopper = asyncio.create_task(_stop())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await stopper

    _drive(_run())

    # Drain everything (the finally: clause's _flush_metrics_emit
    # guarantees the last pending snapshot reaches the queue).
    msgs = _drive(_drain_queue(sub, timeout=0.2))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]

    # We expect ONE metrics message, not 50. The snapshot inside carries
    # the *last* step the trainer wrote.
    assert len(metrics_msgs) == 1, (
        f"expected 1 debounced metrics message, got {len(metrics_msgs)}: "
        "regression — step events aren't being coalesced"
    )
    assert metrics_msgs[0]["data"]["step"] == 50
    # History has all 50 points (we accumulate regardless of debounce —
    # the debounce only controls WS traffic).
    assert len(metrics_msgs[0]["data"]["step_history"]) == 50
    assert len(metrics_msgs[0]["data"]["loss_history"]) == 50


# ---------------------------------------------------------------------------
# 3. Stdout parser does not emit metrics for training tasks
# ---------------------------------------------------------------------------


def test_emit_line_skips_metrics_for_training_task():
    """For training tasks, only the JSONL watcher owns the metrics channel.

    Regression: both the stdout parser and the JSONL watcher used to
    emit metrics for the same step, racing on the same parser metrics
    object. The stdout emit is now suppressed for training tasks so the
    WebSocket traffic isn't doubled (and the loss_history append isn't
    double-counted — the step-number dedup saved the data, but doubled
    the WS load).
    """
    svc, Task, TaskState = _make_service()
    task = Task(id="st", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True  # <-- the gate
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
    assert metrics_msgs == [], (
        "training task should not emit a metrics message from the stdout "
        "path; the JSONL watcher owns that channel"
    )
    # The parser did update the underlying metrics object (so a late
    # REST /metrics request still sees step=100).
    assert task.parser.metrics.step == 100
    assert task.parser.metrics.total_steps == 1000


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
# 4. Final flush guarantees the last step reaches subscribers
# ---------------------------------------------------------------------------


def test_final_flush_emits_pending_metrics():
    """A step event that lands just before ``done`` must reach the UI.

    Without the final flush, a step event scheduled for a debounce send
    300 ms later could lose the race to the terminal ``done`` message —
    the user would see the dashboard stuck on the second-to-last step.
    The watcher's ``finally`` clause calls ``_flush_metrics_emit``, which
    cancels the pending timer and sends the latest snapshot synchronously.
    """
    svc, Task, TaskState = _make_service()
    task = Task(id="fl", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)

    async def _run():
        # Schedule a pending snapshot directly (mimics what the JSONL
        # watcher does when a step event arrives inside the debounce
        # window).
        snapshot = task.parser.metrics.snapshot()
        snapshot["step"] = 999
        await svc._schedule_metrics_emit(task, snapshot)
        # Immediately flush — the debounce window hasn't elapsed yet,
        # so without the flush the message would not be in the queue.
        await svc._flush_metrics_emit(task)
        # The pending handle is cleared — a second flush is a no-op.
        assert task._pending_metrics is None
        assert task._pending_metrics_handle is None
        await svc._flush_metrics_emit(task)
        # Drain here in the same event loop so the queue stays bound
        # to this loop (``asyncio.run`` finalizes the loop on return,
        # so a second ``asyncio.run`` from the test body would see a
        # "bound to a different event loop" error on the queue).
        return await _drain_queue(sub, timeout=0.05)

    msgs = _drive(_run())
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    assert len(metrics_msgs) == 1, (
        f"expected 1 metrics message (the one we just flushed), got "
        f"{len(metrics_msgs)}"
    )
    assert metrics_msgs[0]["data"]["step"] == 999


# ---------------------------------------------------------------------------
# 5. run_end flushes pending metrics so the last step lands before done
# ---------------------------------------------------------------------------


def test_run_end_flushes_pending_metrics(tmp_path: Path):
    """A pending debounced metrics emit is flushed on the run_end event.

    Regression: the trainer writes the final step event, then writes
    ``run_end`` ~microseconds later. The debounced send of the final
    step's snapshot is still 0.3 s away when ``run_end`` lands. The
    handler must flush immediately so the last step reaches the UI
    *before* the WebUI processes the terminal ``done`` message — not
    after, which would briefly leave the dashboard on the prior step.
    The handler also schedules a fresh snapshot carrying the run_end
    ``final_step`` (which can be a few steps past the last logged one)
    so the dashboard ends on the exact value training reported.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    task = Task(id="re", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["re"] = task

    # Write step event first, then run_end — both before the watcher's
    # first poll. The debounced emit for the step is still pending when
    # run_end arrives; the handler must flush it.
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
            # > debounce window so any pending scheduled send fires.
            await asyncio.sleep(0.6)
            task.state = TaskState.SUCCESS

        stopper = asyncio.create_task(_stop())
        try:
            await svc._watch_progress_jsonl(task, str(jsonl))
        finally:
            await stopper
        # Final flush in the watcher's `finally:` clause has already
        # run; no extra cleanup needed here.

    _drive(_run())

    msgs = _drive(_drain_queue(sub, timeout=0.2))
    metrics_msgs = [m for m in msgs if m.get("type") == "metrics"]
    assert metrics_msgs, "expected at least one metrics message"
    # The final metrics message should reflect the run_end final_step
    # (the trainer's authoritative "we stopped at step 1000" signal —
    # not the last ``step`` event, which logged 999).
    assert metrics_msgs[-1]["data"]["step"] == 1000
    # And we should NOT have flooded the queue: at most a handful of
    # messages (one for the last step's pending flush, one for the
    # run_end's fresh snapshot). The old behavior queued one per
    # step event, which is what we explicitly do NOT want.
    assert len(metrics_msgs) <= 2, (
        f"expected at most 2 debounced metrics messages, got "
        f"{len(metrics_msgs)}: regression — run_end isn't flushing / "
        "step events aren't being coalesced"
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
