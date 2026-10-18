"""Native HTTP callers and queue controls retain authority before spawning."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from anima_daemon import config, manager as manager_module
from anima_daemon.manager import JobManager
from anima_daemon.server import _Handler, serve


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "JOBS_DIR", config.STATE_DIR / "jobs")
    monkeypatch.setattr(manager_module, "_POLL_INTERVAL", 0.01)
    config.ensure_state_dirs()
    result = JobManager()
    monkeypatch.setattr(result, "_gpu_guard", lambda _job: None)
    return result


@pytest.fixture
def http_server(manager):
    server = serve(manager, port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", manager
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://foreign.example", "Content-Type": "text/plain"},
        {"Origin": "null", "Content-Type": "application/json"},
        {"Sec-Fetch-Site": "cross-site", "Content-Type": "application/json"},
        {"Sec-Fetch-Site": "same-site", "Content-Type": "application/json"},
    ],
)
def test_browser_submit_is_rejected_before_queueing(http_server, tmp_path, headers):
    base, manager = http_server
    marker = tmp_path / "must-not-execute"
    body = {
        "kind": "command",
        "argv": ["-c", f"open({str(marker)!r}, 'w').write('executed')"],
    }
    request = urllib.request.Request(
        base + "/jobs", data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 403
    assert manager.list_jobs() == []
    assert not marker.exists()


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/queue/pause"), ("POST", "/shutdown"), ("DELETE", "/jobs/job")],
)
def test_browser_cannot_mutate_queue_or_history(http_server, method, path):
    base, manager = http_server
    request = urllib.request.Request(
        base + path, method=method, headers={"Origin": "https://foreign.example"}
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 403
    assert not manager.is_paused()
    assert not manager._stopping


def test_plain_text_body_without_origin_is_rejected(http_server):
    base, manager = http_server
    request = urllib.request.Request(
        base + "/jobs",
        data=b'{"kind":"command","argv":["-c","pass"]}',
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 415
    assert manager.list_jobs() == []


def test_native_json_client_still_executes_command(http_server, tmp_path):
    base, manager = http_server
    marker = tmp_path / "native-executed"
    request = urllib.request.Request(
        base + "/jobs",
        data=json.dumps(
            {
                "kind": "command",
                "argv": ["-c", f"open({str(marker)!r}, 'w').write('executed')"],
            }
        ).encode(),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 201
        job_id = json.load(response)["job_id"]
    manager._process_one(job_id)
    assert marker.read_text() == "executed"
    assert manager.get(job_id).state == "done"


def test_log_stream_remains_open_while_paused(monkeypatch, tmp_path):
    from anima_daemon import server as server_module

    job = SimpleNamespace(state="paused", stdout_path=str(tmp_path / "stdout.log"))
    messages = []
    handler = _Handler.__new__(_Handler)
    handler.server = SimpleNamespace(manager=SimpleNamespace(get=lambda _id: job))
    handler._open_sse = lambda: None
    handler._sse = lambda message: messages.append(message) or True

    def follow(*_args, **_kwargs):
        yield ""
        assert messages == []
        job.state = "done"
        yield ""

    monkeypatch.setattr(server_module.tail, "follow", follow)
    handler._handle_logs("job")
    assert messages == [{"ev": "eof", "state": "done"}]


@pytest.mark.parametrize("boundary", ["_gpu_guard", "_build_cmd"])
@pytest.mark.parametrize("control", ["cancel", "pause", "shutdown"])
def test_queue_controls_during_preparation_prevent_launch(
    manager, tmp_path, monkeypatch, boundary, control
):
    entered = threading.Event()
    release = threading.Event()
    held = threading.Event()
    original = getattr(manager, boundary)
    original_broadcast = manager._broadcast

    def preparation(job):
        entered.set()
        assert release.wait(5)
        return original(job)

    def broadcast(event):
        if event.get("ev") == "queue_held":
            held.set()
        original_broadcast(event)

    monkeypatch.setattr(manager, boundary, preparation)
    monkeypatch.setattr(manager, "_broadcast", broadcast)
    marker = tmp_path / "child-executed"
    job = manager.submit_command(
        label="launch-control-probe",
        argv=["-c", f"open({str(marker)!r}, 'w').write('executed')"],
    )
    failures = []

    def process():
        try:
            manager._process_one(job.id)
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=process, daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        if control == "cancel":
            manager.stop(job.id)
            assert job.state == "stopped"
        elif control == "pause":
            manager.pause()
        else:
            manager.shutdown(mode="detach")
        release.set()
        if control == "pause":
            assert held.wait(5)
            assert job.state == "queued"
            assert not marker.exists()
            manager.resume()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert failures == []
        if control == "pause":
            assert marker.read_text() == "executed"
            assert job.state == "done"
        else:
            assert not marker.exists()
            assert job.state == ("stopped" if control == "cancel" else "queued")
    finally:
        release.set()
        manager.resume()
        worker.join(timeout=5)
