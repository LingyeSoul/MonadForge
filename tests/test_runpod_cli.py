"""anima_daemon.runpod — request shapes and CLI plumbing against a fake urlopen."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

from anima_daemon.runpod import api, cli


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_http(monkeypatch):
    """Records each request; replies from ``routes[(method, path)]``."""
    calls: list[dict] = []
    routes: dict = {}

    def urlopen(req, timeout=None):
        url = urllib.parse.urlsplit(req.full_url)
        calls.append(
            {
                "method": req.get_method(),
                "host": url.netloc,
                "path": url.path,
                "query": urllib.parse.parse_qs(url.query),
                "body": json.loads(req.data) if req.data else None,
                "auth": req.get_header("Authorization"),
            }
        )
        reply = routes.get((req.get_method(), url.path), {})
        if isinstance(reply, urllib.error.HTTPError):
            raise reply
        return _Resp(json.dumps(reply).encode() if reply is not None else b"")

    monkeypatch.setattr(api.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("RUNPOD_API_KEY", "rk_test")
    return calls, routes


def test_api_key_from_runpodctl_config(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('apikey = "rk_file"\n')
    monkeypatch.setattr(api, "CONFIG_PATH", cfg)
    assert api.resolve_api_key() == "rk_file"
    monkeypatch.setattr(api, "CONFIG_PATH", tmp_path / "missing.toml")
    assert api.resolve_api_key() is None
    with pytest.raises(api.RunPodError):
        api.RunPodClient()


def test_gpu_catalog_hits_v2_with_availability(fake_http):
    calls, routes = fake_http
    routes[("GET", "/v2/catalog/gpus")] = {"gpus": [{"id": "A"}]}
    assert api.RunPodClient().gpu_types(cloud="COMMUNITY", count=2) == [{"id": "A"}]
    c = calls[0]
    assert c["host"] == "api.runpod.io" and c["auth"] == "Bearer rk_test"
    assert c["query"] == {
        "include": ["AVAILABILITY"],
        "product": ["POD"],
        "cloud": ["COMMUNITY"],
        "count": ["2"],
    }


def test_pod_verbs_hit_v1_paths(fake_http):
    calls, routes = fake_http
    routes[("DELETE", "/v1/pods/p1")] = None
    cl = api.RunPodClient()
    cl.start_pod("p1")
    cl.stop_pod("p1")
    cl.terminate_pod("p1")
    assert [(c["method"], c["path"]) for c in calls] == [
        ("POST", "/v1/pods/p1/start"),
        ("POST", "/v1/pods/p1/stop"),
        ("DELETE", "/v1/pods/p1"),
    ]
    assert {c["host"] for c in calls} == {"rest.runpod.io"}


def test_http_error_carries_status_and_message(fake_http):
    _, routes = fake_http
    routes[("GET", "/v1/pods/nope")] = urllib.error.HTTPError(
        "u", 404, "Not Found", {}, io.BytesIO(b'{"error": "pod not found"}')
    )
    with pytest.raises(api.RunPodError) as e:
        api.RunPodClient().get_pod("nope")
    assert e.value.status == 404 and e.value.message == "pod not found"


def test_create_spec_prefixes_name_and_injects_pubkey(tmp_path, monkeypatch):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_ed25519.pub").write_text("ssh-ed25519 AAAA me\n")
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    a = cli.build_parser().parse_args(
        [
            "create",
            "--gpu",
            "G1",
            "--gpu",
            "G2",
            "--image",
            "img",
            "--name",
            "run1",
            "--env",
            "A=b=c",
            "--spot",
        ]
    )
    spec = cli.build_create_spec(a)
    assert spec["name"] == "anima-run1"
    assert spec["gpuTypeIds"] == ["G1", "G2"]
    assert spec["env"] == {"A": "b=c", "PUBLIC_KEY": "ssh-ed25519 AAAA me"}
    assert spec["interruptible"] is True and spec["volumeInGb"] == 0
    assert spec["containerDiskInGb"] == 200
    assert "networkVolumeId" not in spec

    a = cli.build_parser().parse_args(
        [
            "create",
            "--gpu",
            "G1",
            "--image",
            "img",
            "--no-ssh-key",
            "--network-volume",
            "nv1",
        ]
    )
    spec = cli.build_create_spec(a)
    assert "PUBLIC_KEY" not in spec["env"]
    assert spec["networkVolumeId"] == "nv1" and "volumeInGb" not in spec


def test_create_dry_run_needs_no_key(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(api, "CONFIG_PATH", api.Path("/nonexistent/config.toml"))
    rc = cli.main(["create", "--gpu", "G", "--no-ssh-key", "--dry-run"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["imageName"] == api.DEFAULT_IMAGE


def test_wait_ssh_polls_until_mapping(fake_http, monkeypatch):
    _, routes = fake_http
    states = iter(
        [
            {"desiredStatus": "RUNNING", "publicIp": "", "portMappings": {}},
            {
                "desiredStatus": "RUNNING",
                "publicIp": "1.2.3.4",
                "portMappings": {"22": 40022},
            },
        ]
    )
    monkeypatch.setattr(api.RunPodClient, "get_pod", lambda self, pid: next(states))
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    pod = api.RunPodClient().wait_ssh("p1", probe=False)
    assert api.ssh_endpoint(pod) == ("1.2.3.4", 40022)


def test_wait_ssh_fails_fast_on_exited(fake_http, monkeypatch):
    monkeypatch.setattr(
        api.RunPodClient, "get_pod", lambda self, pid: {"desiredStatus": "EXITED"}
    )
    with pytest.raises(api.RunPodError):
        api.RunPodClient().wait_ssh("p1", probe=False)


def test_ssh_argv():
    pod = {"id": "p1", "publicIp": "1.2.3.4", "portMappings": {"22": 40022}}
    argv = cli.ssh_argv(pod, remote=["nvidia-smi"])
    assert argv[:3] == ["ssh", "-p", "40022"]
    assert argv[-2:] == ["root@1.2.3.4", "nvidia-smi"]
    with pytest.raises(SystemExit):
        cli.ssh_argv({"id": "p2", "publicIp": None})
