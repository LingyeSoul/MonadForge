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


# Regression: the WebUI's command-name set once drifted from the daemon's
# (only {lora, lora-gui, easycontrol}), so exp-* training commands that route
# through train.py were not watched even though the daemon redirected their
# --progress_jsonl. The two surfaces must agree via the shared helper.
@pytest.mark.parametrize("command", ["lora", "lora-gui", "easycontrol", "exp-chimera"])
def test_training_command_gets_progress_jsonl_watcher(command):
    """Commands that route through train.py must tail the per-job JSONL."""
    svc, _, _ = _make_service()

    path = svc._derive_progress_jsonl_path(
        {"job_id": "20260617-163000-abcdef"},
        command,
        [],
        {},
    )

    assert path is not None
    # Path is absolute + platform-specific separators, so assert on components
    # (matches test_lora_gui_progress_jsonl_path_uses_daemon_job_file's style).
    p = Path(path)
    assert p.name == "progress.jsonl"
    assert p.parent.name == "20260617-163000-abcdef"
    assert p.parent.parent.name == "jobs"


@pytest.mark.parametrize("command", ["turbo", "exp-spd"])
def test_bespoke_loop_command_has_no_progress_jsonl_watcher(command):
    """turbo/exp-spd bypass train.py and write no progress.jsonl — no watcher."""
    svc, _, _ = _make_service()

    assert (
        svc._derive_progress_jsonl_path(
            {"job_id": "20260617-163000-abcdef"},
            command,
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


def test_legacy_jsonl_prefers_d_star_lr_over_base_lr(tmp_path: Path):
    """Legacy Prodigy logs carry the base lr under ``lr/unet`` and the
    effective lr under ``lr/d*lr/unet``.

    Regression for "WebUI always shows lr=1.0 with Prodigy": those older
    progress files wrote the user-set base multiplier (1.0) to ``lr/unet`` and
    the real, rising ``d * lr`` only to ``lr/d*lr/unet``. The watcher must
    prefer the effective value so a Prodigy run's dashboard tracks the real lr
    instead of a flat 1.0.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    jsonl.write_text("", encoding="utf-8")
    task = Task(id="prod", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["prod"] = task

    # Legacy shape: base lr under lr/unet, effective under lr/d*lr/unet.
    _write_jsonl(
        jsonl,
        [
            {
                "ev": "step",
                "ts": 1.0,
                "loss/average": 0.28,
                "lr/unet": 1.0,           # base multiplier (the bug value)
                "lr/d*lr/unet": 0.01,     # effective d*lr (the real value)
                "global_step": 2,
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
    data = metrics_msgs[-1]["data"]
    # Effective lr wins over the flat base multiplier.
    assert data["lr"] == pytest.approx(0.01), (
        "legacy Prodigy logs must surface d*lr (0.01), not the base 1.0"
    )
    assert data["lr_history"] == [pytest.approx(0.01)]


# ---------------------------------------------------------------------------
# 2c. Preview sampling must not dip the LR / loss curves to zero
# ---------------------------------------------------------------------------


def test_sample_step_does_not_zero_first_lr_history_point(tmp_path: Path):
    """The LR curve's first point must be the real first lr, not ``0.0``.

    Regression: the watcher parsed ``lr`` AFTER appending to
    ``lr_history``. Since ``TrainingMetrics.lr`` defaults to ``0.0``,
    the very first logged ``step`` event appended ``0.0`` and only then
    recorded the event's real lr — so the dashboard's LR curve started
    with a hard zero and the loss curve (index-aligned with it) showed a
    matching kink. This was especially visible on runs with
    ``--sample_at_first`` (a preview lands at step 0 just before the
    first ``step`` event), which is why it surfaced as "curves dip to
    zero during preview sampling".
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    jsonl.write_text("", encoding="utf-8")
    task = Task(id="z", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["z"] = task

    # sample_at_first fires before the first logged step, then a normal
    # cadence. The first step event carries a non-zero lr/unet.
    _write_jsonl(
        jsonl,
        [
            {
                "ev": "sample",
                "ts": 0.5,
                "global_step": 0,
                "epoch": None,
                "path": "output/sample/000.png",
                "prompt": "p",
            },
            {
                "ev": "step",
                "ts": 1.0,
                "loss/current": 0.3,
                "loss/average": 0.28,
                "lr/unet": 9.5e-5,
                "global_step": 2,
                "epoch": 1,
            },
            {
                "ev": "step",
                "ts": 2.0,
                "loss/current": 0.25,
                "loss/average": 0.26,
                "lr/unet": 8.0e-5,
                "global_step": 4,
                "epoch": 1,
            },
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
    data = metrics_msgs[-1]["data"]

    # The first LR history point must be the real lr from step 2, not
    # the default 0.0 — this is the exact "curve dips to zero at the
    # sample step" regression.
    assert data["lr_history"][0] == pytest.approx(9.5e-5), (
        "first lr_history entry should be the first step's real lr, not "
        "the default 0.0 (parse-after-append regression)"
    )
    # Histories are index-aligned and not shifted by one step.
    assert data["step_history"] == [2, 4]
    assert data["loss_history"] == [pytest.approx(0.28), pytest.approx(0.26)]
    assert data["lr_history"] == [pytest.approx(9.5e-5), pytest.approx(8.0e-5)]
    # No spurious zero point was ever appended.
    assert all(v > 0 for v in data["lr_history"])


def test_sample_step_does_not_kink_lr_at_sampling_boundary(tmp_path: Path):
    """An ``--sample_every_n_steps`` boundary must not record a stale lr.

    The trainer emits the ``sample`` event for step S immediately before
    the matching ``step`` event for S. With the old parse-after-append
    order, the step-S lr_history entry held step-(S-2)'s lr (the
    previous value), so the LR curve visibly kinked at every sampling
    boundary. The fix records the current step's lr, matching the
    already-correct ``training_log_parser._parse_tqdm``.
    """
    svc, Task, TaskState = _make_service()
    jsonl = tmp_path / "run.progress.jsonl"
    jsonl.write_text("", encoding="utf-8")
    task = Task(id="k", command="lora", args=[], state=TaskState.RUNNING)
    task.is_training = True
    sub: asyncio.Queue = asyncio.Queue()
    task._subscribers.append(sub)
    svc._tasks["k"] = task

    _write_jsonl(
        jsonl,
        [
            {
                "ev": "step",
                "ts": 1.0,
                "loss/average": 0.10,
                "lr/unet": 1.0e-4,
                "global_step": 2,
                "epoch": 1,
            },
            {
                "ev": "step",
                "ts": 2.0,
                "loss/average": 0.09,
                "lr/unet": 8.0e-5,
                "global_step": 4,
                "epoch": 1,
            },
            # Sampling boundary at step 6: sample event then step event.
            {
                "ev": "sample",
                "ts": 2.5,
                "global_step": 6,
                "epoch": None,
                "path": "output/sample/006.png",
                "prompt": "p",
            },
            {
                "ev": "step",
                "ts": 3.0,
                "loss/average": 0.085,
                "lr/unet": 6.0e-5,
                "global_step": 6,
                "epoch": 1,
            },
            {
                "ev": "step",
                "ts": 4.0,
                "loss/average": 0.08,
                "lr/unet": 4.0e-5,
                "global_step": 8,
                "epoch": 1,
            },
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
    data = metrics_msgs[-1]["data"]

    # Each lr_history entry is its OWN step's lr (no one-step shift);
    # in particular step 6 — the sampling boundary — carries 6.0e-5,
    # not the stale 8.0e-5 from step 4.
    assert data["step_history"] == [2, 4, 6, 8]
    assert data["loss_history"] == [
        pytest.approx(0.10),
        pytest.approx(0.09),
        pytest.approx(0.085),
        pytest.approx(0.08),
    ]
    assert data["lr_history"] == [
        pytest.approx(1.0e-4),
        pytest.approx(8.0e-5),
        pytest.approx(6.0e-5),
        pytest.approx(4.0e-5),
    ]


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
