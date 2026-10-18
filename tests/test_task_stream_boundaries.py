"""Task telemetry remains complete across partial writes and lifecycle races."""

from __future__ import annotations

import asyncio
import json

import pytest

from webui.services.task_service import Task, TaskService, TaskState


def test_jsonl_partial_utf8_record_keeps_exact_byte_offset(tmp_path):
    path = tmp_path / "progress.jsonl"
    first = b'{"ev":"step","global_step":1}\n'
    second = (
        json.dumps({"ev": "sample", "prompt": "测试"}, ensure_ascii=False) + "\n"
    ).encode()
    split = second.index("测".encode()) + 1
    path.write_bytes(first + second[:split])

    lines, offset = TaskService._read_jsonl_bytes(str(path), 0)
    assert lines == [first.decode().strip()]
    assert offset == len(first)
    assert TaskService._read_jsonl_bytes(str(path), offset) == ([], offset)
    with path.open("ab") as stream:
        stream.write(second[split:])
    lines, offset = TaskService._read_jsonl_bytes(str(path), offset)
    assert json.loads(lines[0])["prompt"] == "测试"
    assert offset == path.stat().st_size

    path.write_bytes(b'{"ev":')
    assert TaskService._read_jsonl_bytes(str(path), offset) is None
    assert TaskService._read_jsonl_bytes(str(path), 0) == ([], 0)


def test_history_skips_damaged_events_and_preserves_valid_steps(tmp_path, caplog):
    path = tmp_path / "progress.jsonl"
    path.write_text(
        "\n".join(
            [
                "null",
                "[]",
                '{"ev":"step","global_step":"broken"}',
                '{"ev":"step","global_step":1,"avr_loss":NaN}',
                '{"ev":"step","global_step":2,"avr_loss":0.4,"lr":0.01}',
                '{"ev":"run_end","final_step":3}',
            ]
        )
    )
    task = TaskService()._task_from_daemon(
        {
            "id": "history",
            "kind": "train",
            "method": "lora",
            "state": "done",
            "progress_path": str(path),
        }
    )
    assert task.parser.metrics.step_history == [2]
    assert task.parser.metrics.loss_history == [0.4]
    assert task.parser.metrics.step == 3
    assert "Ignoring malformed progress event" in caplog.text


def test_legacy_stdout_only_history_retains_chart(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text(
        "steps:  10%|...| 100/1000 [00:10<01:30,  1.50it/s, avr_loss=0.1234, lr=1.00e-04]\n"
    )
    task = TaskService()._task_from_daemon(
        {
            "id": "legacy",
            "kind": "train",
            "method": "lora",
            "state": "done",
            "stdout_path": str(path),
        }
    )
    assert task.parser.metrics.step == 100
    assert task.parser.metrics.step_history == [100]
    assert task.parser.metrics.loss_history == [0.1234]


def test_attempt_replacement_starts_new_monitors_immediately(monkeypatch):
    svc = TaskService()
    task = Task(id="root", command="lora", args=[], job_id="root", is_training=True)

    async def wait_forever(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(svc, "_poll_daemon_job", wait_forever)
    monkeypatch.setattr(svc, "_watch_progress_jsonl", wait_forever)

    async def exercise():
        svc._ensure_monitors(task, "first.jsonl")
        old = [svc._pollers[task.id], svc._progress_watchers[task.id]]
        task.job_id = "child"
        svc._ensure_monitors(task, "child.jsonl")
        new = [svc._pollers[task.id], svc._progress_watchers[task.id]]
        assert all(before is not after for before, after in zip(old, new))
        await asyncio.gather(*old, return_exceptions=True)
        assert svc._pollers[task.id] is new[0]
        assert svc._progress_watchers[task.id] is new[1]
        await svc.close()

    asyncio.run(exercise())


def test_direct_subscription_starts_restored_task_monitors(monkeypatch):
    from webui.services import task_service as task_module

    svc = TaskService()
    calls = []
    monkeypatch.setattr(
        task_module.daemon_client,
        "get_job_group_sync",
        lambda _: {
            "id": "restored",
            "kind": "train",
            "method": "lora",
            "state": "running",
            "progress_path": "restored.jsonl",
        },
    )
    monkeypatch.setattr(
        svc, "_ensure_monitors", lambda task, path: calls.append((task.id, path))
    )
    svc.subscribe("restored")
    assert calls == [("restored", "restored.jsonl")]


def test_idle_task_websocket_disconnect_unsubscribes(monkeypatch):
    from webui.api import ws

    queue = asyncio.Queue()
    unsubscribed = []
    monkeypatch.setattr(ws.task_service, "subscribe", lambda _: queue)
    monkeypatch.setattr(
        ws.task_service, "unsubscribe", lambda *args: unsubscribed.append(args)
    )

    class Socket:
        async def accept(self):
            pass

        async def receive(self):
            return {"type": "websocket.disconnect"}

        async def send_json(self, _message):
            raise AssertionError("an idle task has no output to forward")

        async def close(self):
            pass

    asyncio.run(asyncio.wait_for(ws.task_log_ws(Socket(), "queued"), timeout=1))
    assert unsubscribed == [("queued", queue)]


@pytest.mark.parametrize("state", [TaskState.RUNNING, TaskState.STOPPING])
def test_terminal_poll_drains_last_progress_before_done(tmp_path, state):
    svc = TaskService()
    task = Task(
        id="job", command="lora", args=[], job_id="job", state=state, is_training=True
    )
    path = tmp_path / "progress.jsonl"
    path.write_text('{"ev":"step","global_step":3,"avr_loss":0.4}\n')
    queue = asyncio.Queue()
    task._subscribers.append(queue)

    async def exercise():
        svc._progress_watchers[task.id] = asyncio.create_task(
            svc._watch_progress_jsonl(task, str(path))
        )
        # Finalize before the watcher gets its first timeslice: it must still
        # drain the finished file before clients receive the terminal message.
        await svc._finalize_from_daemon(task, {"state": "stopped", "rc": 0})

    asyncio.run(exercise())
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    assert messages[-1]["type"] == "done"
    assert messages[-2]["data"]["step_history"] == [3]


@pytest.mark.parametrize("known_attempt", [False, True])
@pytest.mark.parametrize("terminal", ["done", "error", "stopped"])
def test_list_reconcile_preserves_final_stdout_and_done(
    tmp_path, monkeypatch, known_attempt, terminal
):
    from webui.services import task_service as task_module

    svc = TaskService()
    path = tmp_path / "stdout.log"
    path.write_text("before completion\n")
    task = Task(
        id="job",
        command="probe",
        args=[],
        job_id="job",
        state=TaskState.RUNNING,
        stdout_path=str(path),
    )
    if known_attempt:
        task.attempts = [{"id": "job", "state": "running"}]
        task.loaded_attempt_ids = ["job"]
    svc._tasks[task.id] = task
    queue = asyncio.Queue()
    task._subscribers.append(queue)

    async def get_job(_job_id):
        return {"id": "job", "state": "running", "stdout_path": str(path)}

    monkeypatch.setattr(task_module.daemon_client, "get_job", get_job)

    async def exercise():
        svc._ensure_monitors(task)
        poller = svc._pollers[task.id]
        first = await asyncio.wait_for(queue.get(), 3)
        assert first["type"] == "log"
        assert first["line"] == "before completion"
        with path.open("a") as stream:
            stream.write("final stdout\n")
        record = {
            "id": "job",
            "state": terminal,
            "rc": 0,
            "kind": "command",
            "method": "probe",
            "stdout_path": str(path),
        }
        svc._reconcile_job_list([record])
        await asyncio.wait_for(poller, 3)
        svc._reconcile_job_list([record])
        await svc.close()

    asyncio.run(exercise())
    messages = []
    while not queue.empty():
        messages.append(queue.get_nowait())
    assert any(message.get("line") == "final stdout" for message in messages)
    assert [message["type"] for message in messages].count("done") == 1
    assert messages[-1]["type"] == "done"
    assert task.lines.count("final stdout") == 1


def test_rate_limited_metrics_flush_while_training_is_idle(tmp_path, monkeypatch):
    from webui.services import task_service as task_module

    svc = TaskService()
    task = Task(id="job", command="lora", args=[], state=TaskState.RUNNING)
    path = tmp_path / "progress.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"ev": "step", "global_step": step, "avr_loss": 1 / step})
            for step in range(1, 51)
        )
        + "\n"
    )
    queue = asyncio.Queue()
    task._subscribers.append(queue)
    clock = [10.0]
    monkeypatch.setattr(task_module.time, "monotonic", lambda: clock[0])

    async def tick(_delay):
        clock[0] += 0.5
        if queue.qsize() >= 2:
            assert task.state == TaskState.RUNNING
            latest = list(queue._queue)[-1]["data"]
            assert latest["step_history"] == list(range(1, 51))
            assert latest["loss_history"] == [1 / step for step in range(1, 51)]
            task.state = TaskState.SUCCESS

    monkeypatch.setattr(task_module.asyncio, "sleep", tick)
    asyncio.run(svc._watch_progress_jsonl(task, str(path)))
    assert queue.qsize() == 2


@pytest.mark.parametrize(
    "state,daemon_state",
    [
        ("pending", "queued"),
        ("running", "running"),
        ("paused", "paused"),
        ("active", "queued,running,paused"),
    ],
)
def test_task_list_forwards_state_filter(monkeypatch, state, daemon_state):
    from webui.services import task_service as task_module

    svc = TaskService()
    calls = []

    async def list_groups(**kwargs):
        calls.append(kwargs)
        return {"groups": [], "total": 0}

    monkeypatch.setattr(task_module.daemon_client, "list_job_groups_page", list_groups)
    assert asyncio.run(svc.list_tasks_page(state=state)) == ([], 0)
    assert calls[0]["state"] == daemon_state


def test_paused_task_restores_and_resumes_same_attempt(monkeypatch):
    from webui.services import task_service as task_module

    svc = TaskService()
    job = {"id": "paused-job", "kind": "train", "method": "lora", "state": "paused"}
    task = svc._task_from_daemon(job)
    svc._tasks[task.id] = task
    assert task.state == TaskState.PAUSED
    assert svc._map_daemon_state("paused", stop_requested=True) == TaskState.STOPPING
    calls = []

    async def resume(job_id):
        calls.append(job_id)
        return {"job_id": job_id, "state": "running"}

    async def group(job_id):
        return svc._physical_job_group({**job, "state": "running"})

    monkeypatch.setattr(task_module.daemon_client, "resume", resume)
    monkeypatch.setattr(task_module.daemon_client, "get_job_group", group)
    monkeypatch.setattr(svc, "_ensure_monitors", lambda *_args: None)
    assert asyncio.run(svc.resume_task(task.id)) is task
    assert calls == ["paused-job"]
    assert task.job_id == "paused-job"
    assert task.info()["attempt_count"] == 1
    assert task.state == TaskState.RUNNING


def test_paused_task_can_stop_and_poller_reaches_terminal(monkeypatch):
    from webui.services import task_service as task_module

    svc = TaskService()
    task = Task(
        id="paused-job",
        command="lora",
        args=[],
        job_id="paused-job",
        state=TaskState.PAUSED,
    )
    svc._tasks[task.id] = task

    async def stop(job_id):
        return {"job_id": job_id, "state": "paused"}

    async def get_job(job_id):
        return {"id": job_id, "state": "stopped", "rc": 0}

    monkeypatch.setattr(task_module.daemon_client, "stop", stop)
    monkeypatch.setattr(task_module.daemon_client, "get_job", get_job)

    async def exercise():
        assert await svc.cancel_task(task.id)
        assert task.state == TaskState.STOPPING
        await svc._poll_daemon_job(task)

    asyncio.run(exercise())
    assert task.state == TaskState.CANCELLED


def test_poller_tracks_pause_and_unpause_without_exiting(monkeypatch):
    from webui.services import task_service as task_module

    svc = TaskService()
    task = Task(
        id="paused-job",
        command="lora",
        args=[],
        job_id="paused-job",
        state=TaskState.RUNNING,
    )
    replies = iter(["paused", "running", "done"])
    seen = []

    async def get_job(job_id):
        return {"id": job_id, "state": next(replies), "rc": 0}

    async def sleep(_delay):
        seen.append(task.state)

    monkeypatch.setattr(task_module.daemon_client, "get_job", get_job)
    monkeypatch.setattr(task_module.asyncio, "sleep", sleep)
    asyncio.run(svc._poll_daemon_job(task))
    assert seen == [TaskState.PAUSED, TaskState.RUNNING]
    assert task.state == TaskState.SUCCESS
