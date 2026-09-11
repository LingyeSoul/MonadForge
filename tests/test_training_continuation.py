from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
import torch
from safetensors.torch import save_file

from library.io.output_layout import resolve_output_layout
from library.training.continuation import (
    blueprint_from_config,
    complete_continuation_blueprint,
    dataset_blueprint_path,
    scheduler_budget,
    setup_continuation,
    state_fingerprint,
    validate_continuation_signatures,
)
from scripts.daemon import config
from scripts.daemon.manager import JobManager


def write_state(path, job, step=30, epoch=3):
    path.mkdir(parents=True)
    save_file({"weight": torch.ones(2, 2)}, path / "model.safetensors")
    for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
        torch.save({"step": step}, path / name)
    data = dict(
        schema_version=3,
        global_step=step,
        current_epoch=epoch,
        micro_batch_offset=10,
        job_id=job.id,
        root_job_id=job.root_job_id,
        config_signature="cfg",
        dataset_signature="data",
    )
    (path / "train_state.json").write_text(json.dumps(data))
    (path / "complete.marker").write_text("ok")
    return path


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    manager = JobManager()
    cfg = dict(
        network_module="networks.lora_anima",
        output_name="example",
        output_dir=str(tmp_path / "ckpt"),
        max_train_epochs=3,
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=0.2,
        mixed_precision="fp16",
        attn_mode="torch",
    )
    job = manager.submit_command(
        label="lora-gui",
        argv=["tasks.py", "lora-gui", "lora"],
        extra_env={"PRESET": "V100"},
        config_snapshot=cfg,
        start=False,
    )
    job.state = "done"
    job.config_signature = "cfg"
    job.dataset_signature = "data"
    job.persist()
    Path(job.progress_path).write_text(
        json.dumps(dict(ev="run_start", total_steps=30, total_epochs=3)) + "\n"
    )
    state = write_state(
        resolve_output_layout(cfg["output_dir"], cfg["output_name"]).state, job
    )
    return manager, job, cfg, state


def prepare(manager, job, target=4):
    return manager.continuations.prepare(
        job.root_job_id,
        dict(source_job_id=job.id, budget_key="max_train_epochs", target=target),
    )


def submit(manager, job, plan):
    return manager.continuations.submit(
        job.root_job_id, dict(token=plan["token"], idempotency_key=plan["token"])
    )


def test_extension_pins_input_and_only_changes_budget(source):
    manager, job, cfg, state = source
    original = Path(job.config_file).read_bytes()
    child = submit(manager, job, prepare(manager, job))
    assert child.root_job_id == job.root_job_id
    assert child.parent_job_id == job.id and child.attempt_index == 1
    assert child.target_steps == 40 and child.target_epochs == 4
    assert toml.load(child.config_file) == dict(cfg, max_train_epochs=4)
    assert Path(job.config_file).read_bytes() == original
    assert Path(child.recovery_state) != state
    assert state_fingerprint(
        Path(child.recovery_state), content=True
    ) == state_fingerprint(state, content=True)
    assert child.continuation["warmup_steps"] == 6
    assert manager.is_paused()
    command, env = manager._build_cmd(job)
    assert "--config_file" not in command
    command, env = manager._build_cmd(child)
    assert "--resume" in command and child.recovery_state in command
    assert "--config_file" not in command
    assert env.get("CONFIG_FILE") == job.config_file
    assert env.get("ANIMA_CONTINUATION_FILE") == str(child.dir / "continuation.json")
    assert "ANIMA_TRAIN_FRESH" not in env


@pytest.mark.parametrize("target", [3, 2, 0, 3.5, True])
def test_completed_cannot_restart_or_shrink(source, target):
    manager, job, _, _ = source
    with pytest.raises(ValueError):
        prepare(manager, job, target)


@pytest.mark.parametrize("failure", ["owner", "signature", "missing", "corrupt"])
def test_unusable_states_are_not_offered(source, failure):
    manager, job, _, state = source
    if failure == "missing":
        (state / "optimizer.bin").unlink()
    elif failure == "corrupt":
        (state / "optimizer.bin").write_bytes(b"corrupt")
    else:
        p = state / "train_state.json"
        data = json.loads(p.read_text())
        data["root_job_id" if failure == "owner" else "config_signature"] = "other"
        p.write_text(json.dumps(data))
    candidate = manager.continuations.candidates("lora")["candidates"][0]
    assert not candidate["available"] and candidate["reason"]
    with pytest.raises(ValueError):
        prepare(manager, job)


def test_changed_state_after_prepare_is_rejected(source):
    manager, job, _, state = source
    plan = prepare(manager, job)
    torch.save({"step": 22}, state / "optimizer.bin")
    with pytest.raises(ValueError, match="变化"):
        submit(manager, job, plan)
    assert len(manager.list_jobs()) == 1


def test_submit_is_idempotent_and_preserves_original(source):
    manager, job, _, state = source
    plan = prepare(manager, job)
    child = submit(manager, job, plan)
    assert submit(manager, job, plan).id == child.id
    torch.save({"changed": True}, Path(child.recovery_state) / "optimizer.bin")
    assert torch.load(state / "optimizer.bin", weights_only=True) == {"step": 30}
    assert len(manager.list_jobs()) == 2


def test_second_extension_uses_latest_state_and_original_warmup(source):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    child.state = "done"
    write_state(
        resolve_output_layout(
            child.continuation["output_dir"], cfg["output_name"]
        ).state,
        child,
        40,
        4,
    )
    grandchild = submit(manager, child, prepare(manager, child, 5))
    assert grandchild.recovery_step == 40 and grandchild.target_steps == 50
    assert grandchild.attempt_index == 2
    assert grandchild.continuation["warmup_steps"] == 6
    assert grandchild.continuation["scheduler_steps"] == 30


def test_stopped_task_can_resume_original_goal(source):
    manager, job, _, state = source
    job.state = "stopped"
    p = state / "train_state.json"
    data = json.loads(p.read_text())
    data.update(global_step=25, current_epoch=3, micro_batch_offset=5)
    p.write_text(json.dumps(data))
    child = submit(manager, job, prepare(manager, job, 3))
    assert child.recovery_step == 25 and child.target_steps == 30
    assert child.continuation["mode"] == "resume"


def test_runtime_context_cannot_load_changed_input(source, monkeypatch):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    monkeypatch.setenv("ANIMA_CONTINUATION_FILE", str(child.dir / "continuation.json"))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    monkeypatch.setenv("ANIMA_DAEMON_ROOT_JOB_ID", child.root_job_id)
    args = SimpleNamespace(**cfg, resume=child.recovery_state)
    setup_continuation(args)
    assert args.output_dir.startswith(child.continuation["output_dir"])
    torch.save({"changed": True}, Path(child.recovery_state) / "optimizer.bin")
    with pytest.raises(ValueError, match="changed"):
        setup_continuation(SimpleNamespace(**cfg, resume=child.recovery_state))


def test_fresh_gui_job_rejects_existing_output(source):
    manager, _, cfg, _ = source
    with pytest.raises(ValueError, match="输出名称"):
        manager.submit_command(
            label="lora-gui",
            argv=["tasks.py", "lora-gui", "lora"],
            config_snapshot=cfg,
            extra_env={"ANIMA_TRAIN_FRESH": "1"},
        )


def test_proportional_warmup_does_not_expand_with_training_budget():
    original = SimpleNamespace(max_train_steps=30, lr_warmup_steps=0.2)
    total, warmup = scheduler_budget(original, 1)
    resumed = SimpleNamespace(
        max_train_steps=50,
        lr_warmup_steps=0.2,
        _continuation=dict(scheduler_steps=total, warmup_steps=warmup),
    )
    assert scheduler_budget(resumed, 1) == (30, 6)
    assert scheduler_budget(
        SimpleNamespace(max_train_steps=50, lr_warmup_steps=0.2), 1
    ) == (50, 10)


@pytest.mark.parametrize("cut", [2, 6, 10])
def test_restored_optimizer_scheduler_matches_uninterrupted(cut):
    from bench.continuation.run_bench import trajectory

    expected, expected_lrs = trajectory()
    actual, actual_lrs = trajectory(cut)
    assert torch.equal(actual, expected)
    assert actual_lrs == expected_lrs


def test_parser_snapshot_does_not_overwrite_source(source, monkeypatch):
    import argparse
    from library.config.io import _write_config_snapshot

    manager, job, cfg, _ = source
    root = resolve_output_layout(cfg["output_dir"], cfg["output_name"]).root
    snapshot = root / "example.snapshot.toml"
    snapshot.write_text("original")
    child = submit(manager, job, prepare(manager, job))
    monkeypatch.setenv("ANIMA_CONTINUATION_FILE", str(child.dir / "continuation.json"))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    monkeypatch.setenv("ANIMA_DAEMON_ROOT_JOB_ID", child.root_job_id)
    args = argparse.Namespace(**cfg, resume=child.recovery_state)
    _write_config_snapshot(args, argparse.ArgumentParser(), {})
    assert snapshot.read_text() == "original"
    assert Path(args.output_dir).is_relative_to(child.continuation["output_dir"])


def test_http_contract_and_duplicate_submit(source):
    import threading
    import requests
    from scripts.daemon.server import serve

    manager, job, cfg, _ = source
    server = serve(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert requests.get(base + "/continuation-candidates?variant=lora").json()[
            "candidates"
        ][0]["available"]
        url = f"{base}/jobs/{job.id}/continuation"
        assert requests.get(url).json()["snapshot"] == cfg
        body = dict(source_job_id=job.id, budget_key="max_train_epochs", target=3)
        rejected = requests.post(url + "/prepare", json=body)
        assert rejected.status_code == 409
        # The error body carries the i18n key so the WebUI can localize it.
        assert rejected.json()["key"] == "target_reached"
        body["target"] = 4
        plan = requests.post(url + "/prepare", json=body).json()
        payload = dict(token=plan["token"], idempotency_key=plan["token"])
        first = requests.post(url, json=payload)
        second = requests.post(url, json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert (
            requests.post(
                base + "/jobs",
                json=dict(
                    kind="command",
                    label="lora-gui",
                    argv=["tasks.py", "lora-gui", "lora"],
                    config_snapshot=cfg,
                    extra_env={"ANIMA_TRAIN_FRESH": "1"},
                ),
            ).status_code
            == 409
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_failed_empty_continuation_still_offers_source(source):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    child.state = "error"
    child.persist()
    candidate = manager.continuations.candidates("lora")["candidates"][0]
    assert candidate["available"]
    assert candidate["job_id"] == job.id
    assert candidate["step"] == 30 and candidate["epoch"] == 3
    assert candidate["target"] == 3
    grandchild = submit(manager, job, prepare(manager, job, 5))
    assert grandchild.parent_job_id == child.id
    assert grandchild.attempt_index == 2
    assert grandchild.recovery_step == 30
    assert grandchild.target_epochs == 5


def test_unreadable_latest_sidecar_does_not_fall_back(source):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    child.state = "error"
    state = write_state(
        resolve_output_layout(
            child.continuation["output_dir"], cfg["output_name"]
        ).state,
        child,
        31,
        4,
    )
    (state / "train_state.json").write_text("{")
    candidate = manager.continuations.candidates("lora")["candidates"][0]
    assert not candidate["available"]
    assert "最新" in candidate["reason"]
    with pytest.raises(ValueError, match="最新"):
        prepare(manager, job, 5)


def test_dataset_blueprint_stays_in_job_dir(source, monkeypatch):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    args = SimpleNamespace(
        progress_jsonl=str(child.dir / "progress.jsonl"),
        config_file="configs/gui-methods/lora.toml",
    )
    path = dataset_blueprint_path(args)
    assert path == child.dir / "dataset.snapshot.json"
    monkeypatch.delenv("ANIMA_DAEMON_JOB_ID")
    assert dataset_blueprint_path(args) is None


def test_prepare_reuses_source_continuation_blueprint(source):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    child.state = "done"
    child.continuation = dict(
        child.continuation, dataset_blueprint={"datasets": [{"subsets": []}]}
    )
    write_state(
        resolve_output_layout(
            child.continuation["output_dir"], cfg["output_name"]
        ).state,
        child,
        40,
        4,
    )
    plan = prepare(manager, child, 5)
    stored = manager.continuations.plans[plan["token"]]
    assert stored["dataset_blueprint"] == {"datasets": [{"subsets": []}]}


def test_incomplete_latest_state_does_not_fall_back(source):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    child.state = "error"
    write_state(
        resolve_output_layout(
            child.continuation["output_dir"], cfg["output_name"]
        ).state,
        child,
        31,
        4,
    )
    (
        resolve_output_layout(
            child.continuation["output_dir"], cfg["output_name"]
        ).state
        / "optimizer.bin"
    ).unlink()
    candidate = manager.continuations.candidates("lora")["candidates"][0]
    assert not candidate["available"]
    assert "完整" in candidate["reason"] or "最新" in candidate["reason"]
    with pytest.raises(ValueError):
        prepare(manager, job, 5)


def test_resume_signature_stable_after_continuation_reroute(source, monkeypatch):
    from library.io.output_layout import layout_from_args
    from train import _resume_config_signature

    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    original = SimpleNamespace(**cfg)
    layout_from_args(original)
    expected = _resume_config_signature(original)
    monkeypatch.setenv("ANIMA_CONTINUATION_FILE", str(child.dir / "continuation.json"))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    monkeypatch.setenv("ANIMA_DAEMON_ROOT_JOB_ID", child.root_job_id)
    args = SimpleNamespace(
        **dict(cfg, max_train_epochs=99, resume=child.recovery_state)
    )
    setup_continuation(args)
    assert args.max_train_epochs == 4
    assert args.output_name == cfg["output_name"]
    assert _resume_config_signature(args) == expected
    args.config_signature = "stale"
    args.dataset_signature = child.continuation["dataset_signature"]
    validate_continuation_signatures(args)
    assert args.config_signature == child.continuation["config_signature"]


def test_continuation_ignores_current_variant_output_name(source, monkeypatch):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    monkeypatch.setenv("ANIMA_CONTINUATION_FILE", str(child.dir / "continuation.json"))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    monkeypatch.setenv("ANIMA_DAEMON_ROOT_JOB_ID", child.root_job_id)
    args = SimpleNamespace(
        **dict(
            cfg,
            output_name="Lora-darksouls3-v2",
            learning_rate=0.0001,
            max_train_epochs=99,
            resume=child.recovery_state,
        )
    )
    setup_continuation(args)
    assert args.output_name == cfg["output_name"]
    assert args.max_train_epochs == 4
    assert "darksouls3" not in str(args.output_dir)


def test_blueprint_from_source_snapshot(source):
    manager, job, cfg, _ = source
    cfg = dict(
        cfg,
        datasets=[{"subsets": [{"num_repeats": 3}]}],
        general={"caption_extension": ".txt"},
    )
    Path(job.config_file).write_text(toml.dumps(cfg))
    job.persist()
    plan = prepare(manager, job)
    stored = manager.continuations.plans[plan["token"]]
    assert stored["dataset_blueprint"]["datasets"][0]["subsets"][0]["num_repeats"] == 3
    assert blueprint_from_config(cfg)["general"]["caption_extension"] == ".txt"


def test_sparse_gui_blueprint_keeps_base_image_dir():
    args = SimpleNamespace(
        resized_image_dir="/run/resized",
        lora_cache_dir="/run/lora",
    )
    sparse = {
        "datasets": [{"subsets": [{"num_repeats": 3, "repeat_by_folder_name": True}]}],
        "general": {"caption_extension": ".txt"},
    }
    completed = complete_continuation_blueprint(sparse, args)
    subset = completed["datasets"][0]["subsets"][0]
    assert subset["image_dir"] == "/run/resized"
    assert subset["cache_dir"] == "/run/lora"
    assert subset["num_repeats"] == 3
    assert subset["repeat_by_folder_name"] is True
    assert completed["general"]["caption_extension"] == ".txt"


def test_continuation_ignores_inherited_launcher_overrides(source, monkeypatch):
    manager, job, _, _ = source
    child = submit(manager, job, prepare(manager, job))
    for key in ("GUI_PRESETS", "ARTIST", "PROFILE_STEPS", "CONFIG_FILE"):
        monkeypatch.setenv(key, "unrelated")
    _, env = manager._build_cmd(child)
    assert all(key not in env for key in ("GUI_PRESETS", "ARTIST", "PROFILE_STEPS"))
    assert env.get("CONFIG_FILE") == job.config_file
    assert env.get("CONFIG_FILE") != "unrelated"


# ── explicit max_train_steps budget ──────────────────────────────────


@pytest.fixture
def source_steps(source):
    manager, job, cfg, state = source
    steps_cfg = {k: v for k, v in cfg.items() if k != "max_train_epochs"}
    steps_cfg["max_train_steps"] = 30
    Path(job.config_file).write_text(toml.dumps(steps_cfg))
    job.persist()
    return manager, job, steps_cfg, state


def prepare_steps(manager, job, target):
    return manager.continuations.prepare(
        job.root_job_id,
        dict(source_job_id=job.id, budget_key="max_train_steps", target=target),
    )


def test_steps_budget_extension(source_steps):
    manager, job, _, _ = source_steps
    child = submit(manager, job, prepare_steps(manager, job, 40))
    assert child.target_steps == 40 and child.target_epochs is None
    assert child.continuation["budget_key"] == "max_train_steps"
    assert child.continuation["continuation_kind"] == "extend"
    assert child.continuation["source_target"] == 30


@pytest.mark.parametrize("target", [30, 29, 0, 30.5, True])
def test_steps_budget_done_cannot_replay_or_shrink(source_steps, target):
    manager, job, _, _ = source_steps
    with pytest.raises(ValueError):
        prepare_steps(manager, job, target)


def test_steps_budget_stopped_resumes_original_goal(source_steps):
    manager, job, _, state = source_steps
    job.state = "stopped"
    p = state / "train_state.json"
    data = json.loads(p.read_text())
    data.update(global_step=25)
    p.write_text(json.dumps(data))
    child = submit(manager, job, prepare_steps(manager, job, 30))
    assert child.continuation["mode"] == "resume"
    assert child.target_steps == 30 and child.recovery_step == 25


# ── scheduler gate reads the effective config, not the sparse snapshot ──


def test_scheduler_gate_reads_source_argv_over_sparse_snapshot(source):
    manager, job, _, _ = source
    job.argv = list(job.argv) + ["--lr_scheduler", "cosine"]
    job.persist()
    candidate = manager.continuations.describe(job)[0]
    assert not candidate["available"]
    assert candidate["reason_key"] == "scheduler_not_extensible"
    with pytest.raises(ValueError, match="调度器"):
        prepare(manager, job, 4)


def test_absolute_warmup_steps_in_argv_stay_absolute(source):
    manager, job, _, _ = source
    job.argv = list(job.argv) + ["--lr_warmup_steps", "100"]
    job.persist()
    plan = prepare(manager, job, 4)
    stored = manager.continuations.plans[plan["token"]]
    assert stored["warmup_steps"] == 100  # not int(100 * 30)
    assert stored["critical_params"] is None or stored["critical_params"] == {}


def test_scheduler_gate_reads_trainer_written_snapshot(source):
    manager, job, cfg, _ = source
    root = resolve_output_layout(cfg["output_dir"], cfg["output_name"]).root
    (root / "example.snapshot.toml").write_text(
        toml.dumps(dict(lr_scheduler="cosine", lr_warmup_steps=0))
    )
    candidate = manager.continuations.describe(job)[0]
    assert not candidate["available"]
    assert candidate["reason_key"] == "scheduler_not_extensible"


def test_cosine_stopped_task_resumes_original_goal(source):
    manager, job, _, state = source
    job.state = "stopped"
    job.argv = list(job.argv) + ["--lr_scheduler", "cosine"]
    job.persist()
    p = state / "train_state.json"
    data = json.loads(p.read_text())
    data.update(global_step=10)
    p.write_text(json.dumps(data))
    plan = prepare(manager, job, 3)
    assert plan["mode"] == "resume"
    child = submit(manager, job, plan)
    assert child.continuation["lr_scheduler"] == "cosine"


def test_scheduler_budget_rejects_extension_beyond_pinned_horizon():
    with pytest.raises(ValueError, match="constant"):
        scheduler_budget(
            SimpleNamespace(
                max_train_steps=50,
                lr_scheduler="cosine",
                lr_scheduler_type=None,
                _continuation=dict(scheduler_steps=30, warmup_steps=6),
            ),
            1,
        )


def test_scheduler_budget_allows_constant_extension_and_cosine_resume():
    extend = scheduler_budget(
        SimpleNamespace(
            max_train_steps=50,
            lr_scheduler="constant_with_warmup",
            lr_scheduler_type=None,
            _continuation=dict(scheduler_steps=30, warmup_steps=6),
        ),
        1,
    )
    assert extend == (30, 6)
    resume = scheduler_budget(
        SimpleNamespace(
            max_train_steps=30,
            lr_scheduler="cosine",
            lr_scheduler_type=None,
            _continuation=dict(scheduler_steps=30, warmup_steps=6),
        ),
        1,
    )
    assert resume == (30, 6)


def test_setup_continuation_pins_scheduler_identity(source, monkeypatch):
    manager, job, cfg, _ = source
    child = submit(manager, job, prepare(manager, job))
    monkeypatch.setenv("ANIMA_CONTINUATION_FILE", str(child.dir / "continuation.json"))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    monkeypatch.setenv("ANIMA_DAEMON_ROOT_JOB_ID", child.root_job_id)
    args = SimpleNamespace(
        **dict(cfg, lr_scheduler="cosine", lr_warmup_steps=0.9, resume=child.recovery_state)
    )
    setup_continuation(args)
    assert args.lr_scheduler == "constant_with_warmup"
    assert args.lr_warmup_steps == 0.2


# ── critical-parameter drift against the trainer-written snapshot ─────


def test_critical_param_drift_is_rejected(source, monkeypatch):
    manager, job, cfg, _ = source
    root = resolve_output_layout(cfg["output_dir"], cfg["output_name"]).root
    (root / "example.snapshot.toml").write_text(
        toml.dumps(dict(train_batch_size=1, network_dim=16))
    )
    child = submit(manager, job, prepare(manager, job))
    monkeypatch.setenv("ANIMA_CONTINUATION_FILE", str(child.dir / "continuation.json"))
    monkeypatch.setenv("ANIMA_DAEMON_JOB_ID", child.id)
    monkeypatch.setenv("ANIMA_DAEMON_ROOT_JOB_ID", child.root_job_id)
    args = SimpleNamespace(
        **dict(cfg, train_batch_size=2, network_dim=16, resume=child.recovery_state)
    )
    setup_continuation(args)
    assert args.train_batch_size == 2  # sparse snapshot may not carry it
    args.dataset_signature = child.continuation["dataset_signature"]
    with pytest.raises(ValueError, match="train_batch_size"):
        validate_continuation_signatures(args)
    args.train_batch_size = 1
    validate_continuation_signatures(args)


# ── auto_resume cursor for continuation (checkpoints.py branch) ───────


def test_auto_resume_sets_explicit_cursor_for_continuation(tmp_path, monkeypatch):
    from library.training.checkpoints import CheckpointSaver

    monkeypatch.delenv("ANIMA_TRAIN_FRESH", raising=False)
    saver = CheckpointSaver(
        args=SimpleNamespace(),
        accelerator=object(),
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=0),
        current_step=SimpleNamespace(value=0),
    )
    args = SimpleNamespace(
        resume=str(tmp_path),
        _continuation={"job_id": "j"},
        skip_until_initial_step=False,
        initial_epoch=2,
        initial_step=7,
    )
    saver.args = args
    saver.auto_resume()
    assert args.skip_until_initial_step is True
    assert args.initial_epoch is None and args.initial_step is None
    # A plain (non-continuation) resume keeps the caller-provided cursors.
    plain = SimpleNamespace(
        resume=str(tmp_path),
        skip_until_initial_step=False,
        initial_epoch=2,
        initial_step=7,
    )
    saver.args = plain
    saver.auto_resume()
    assert plain.skip_until_initial_step is False
    assert plain.initial_epoch == 2 and plain.initial_step == 7


# ── submit hygiene: restart replay, copy window, orphan cleanup ──────


def test_submit_after_plans_loss_returns_existing_attempt(source):
    manager, job, _, _ = source
    plan = prepare(manager, job)
    child = submit(manager, job, plan)
    # A daemon restart rebuilds jobs from disk but wipes in-memory plans.
    manager.continuations.plans.clear()
    assert submit(manager, job, plan).id == child.id


def test_failed_submit_leaves_no_orphan_directory(source, monkeypatch):
    from scripts.daemon import continuation as continuation_module

    manager, job, _, _ = source
    plan = prepare(manager, job)

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(continuation_module, "copy_state", broken)
    with pytest.raises(OSError):
        submit(manager, job, plan)
    assert sorted(p.name for p in config.JOBS_DIR.iterdir()) == [job.id]


def test_duplicate_submit_during_copy_returns_same_attempt(source, monkeypatch):
    import threading

    from scripts.daemon import continuation as continuation_module

    manager, job, _, _ = source
    entered = threading.Event()
    release = threading.Event()
    original = continuation_module.copy_state

    def slow_copy(src, dst):
        entered.set()
        release.wait(10)
        return original(src, dst)

    monkeypatch.setattr(continuation_module, "copy_state", slow_copy)
    plan = prepare(manager, job)
    results: dict[str, object] = {}

    def run(name: str) -> None:
        try:
            results[name] = submit(manager, job, plan)
        except BaseException as exc:  # pragma: no cover - asserted below
            results[name] = exc

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    threads[0].start()
    assert entered.wait(5)
    threads[1].start()
    release.set()
    for thread in threads:
        thread.join(10)
    assert not any(isinstance(value, BaseException) for value in results.values())
    assert results["a"].id == results["b"].id


def test_candidates_cache_refreshed_after_submit(source):
    manager, job, _, _ = source
    manager.continuations.candidates("lora")  # warm the TTL cache
    child = submit(manager, job, prepare(manager, job))
    child.state = "running"
    child.persist()
    candidate = manager.continuations.candidates("lora")["candidates"][0]
    assert not candidate["available"]
    assert candidate["reason_key"] == "task_active"


def test_public_payload_hides_continuation_internals(source):
    manager, job, _, _ = source
    child = submit(manager, job, prepare(manager, job))
    payload = child.public()
    assert payload["continuation"]["mode"] == "extend"
    for secret in ("token", "dataset_blueprint", "critical_params", "fingerprint"):
        assert secret not in payload["continuation"]


def test_http_unknown_task_maps_to_404(source):
    import requests
    from scripts.daemon.server import serve

    manager, _, _, _ = source
    server = serve(manager, port=0)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = requests.get(base + "/jobs/missing-task/continuation")
        assert response.status_code == 404
        assert "不存在" in response.json()["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
