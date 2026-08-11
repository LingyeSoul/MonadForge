"""The job manager: FIFO serial queue + worker thread + state table.

One worker thread drains a ``queue.Queue`` of job ids. Per job it builds the
same ``accelerate launch … train.py`` command the CLI builds, spawns it
detached (so a console ctrl-C can't reach it), points ``--progress_jsonl`` at
the job dir, then monitors by polling ``(pid, create_time)`` liveness — never
by awaiting a subprocess transport (sidesteps Windows ProactorEventLoop
subprocess bugs). On boot it reconciles ``jobs/`` so it can re-attach a
still-alive orphan or mark a dead one ``orphaned``.

Serial by design (single local GPU): exactly one job runs at a time.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import threading
import time
import json
from pathlib import Path
from typing import Optional

import toml

from . import config, gpu, proc, tail
from .jobs import (
    STATE_DONE,
    STATE_ERROR,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_STOPPED,
    TERMINAL_STATES,
    Job,
    load_all,
    new_job_id,
)

logger = logging.getLogger("anima.daemon")

_POLL_INTERVAL = 1.0  # seconds between liveness checks
_STOP_FILE_RETRIES = 8
_SENTINEL = "__stop__"

# Signal → user-actionable hint, for a process that died without writing a
# run_end event. POSIX ``Popen.poll()`` reports a signal death as a negative
# number; a shell/launcher layer (``accelerate launch``) relays it as 128+N.
_SIGNAL_HINTS = {
    9: "killed (SIGKILL) — almost always out of memory. Lower batch size, "
    "raise blocks_to_swap, or try PRESET=low_vram.",
    6: "aborted (SIGABRT) — usually a CUDA assert / illegal memory access. "
    "See the last traceback above.",
    11: "segfault (SIGSEGV) — a native crash. See the last traceback above.",
    15: "terminated (SIGTERM).",
}


def _classify_exit(rc) -> str:
    """Human-readable diagnosis for a nonzero/unknown process exit code."""
    if rc is None:
        # The adopted-orphan path (``popen is None``) can't ``poll()`` a process
        # that vanished while the daemon was down, so it lands here with no exit
        # code at all. Don't claim "crashed before finishing" + point at a
        # non-existent traceback — give the actionable "we couldn't read it"
        # diagnosis instead.
        return (
            "process exit code unavailable — the adopted job vanished before the "
            "daemon could read its exit code. Check if the process was killed "
            "externally (OOM reaper, SIGKILL from another tool), then retry."
        )
    sig = None
    if rc is not None and rc < 0:
        sig = -rc
    elif rc is not None and rc > 128:
        sig = rc - 128
    if sig in _SIGNAL_HINTS:
        return f"process exited (code={rc}): {_SIGNAL_HINTS[sig]}"
    return (
        f"process exited (code={rc}) — crashed before finishing. "
        "See the last traceback above."
    )


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._popens: dict[str, object] = {}  # job_id -> Popen (spawned only)
        # A stop request runs on a helper thread so the worker can keep serving
        # health/status requests.  The monitor nevertheless waits for this
        # event before finalizing a job: an accelerate launcher may exit as
        # soon as it receives SIGTERM while its train/DataLoader descendants
        # are still saving state and releasing the GPU.
        self._stop_events: dict[str, threading.Event] = {}
        self._adopt: list[str] = []  # running orphans to monitor before the queue
        self._subscribers: set["queue.Queue[dict]"] = set()
        self._stopping = False
        self._kill_on_shutdown = False
        # Queue run gate: set → worker launches queued jobs as the GPU frees;
        # cleared → queue paused (dequeued jobs held `queued` until `resume()`,
        # a running job left alone). Default set so non-opt-in callers run now.
        self._run_gate = threading.Event()
        self._run_gate.set()
        # Worker liveness: bumped every loop iteration and every monitor poll.
        # Exposed via /health so a wedged-or-dead worker is observable (the GUI
        # spinner otherwise looks identical to a healthy long-running job).
        self._worker_heartbeat = time.time()
        self._worker = threading.Thread(
            target=self._run, name="anima-job-worker", daemon=True
        )

    def start(self) -> None:
        config.ensure_state_dirs()
        self._reconcile()
        self._worker.start()

    def shutdown(self, *, kill_jobs: bool = False, mode: str | None = None) -> None:
        """Stop accepting work and unblock the worker. With ``kill_jobs`` the
        active job tree is torn down and the GPU freed before the daemon exits.
        """
        # ``mode`` is the durable public contract.  Keep kill_jobs as a
        # backwards-compatible alias used by older clients.
        mode = mode or ("force" if kill_jobs else "detach")
        if mode not in {"detach", "cooperative-stop", "force"}:
            raise ValueError("shutdown mode must be detach, cooperative-stop, or force")
        with self._lock:
            self._stopping = True
            self._kill_on_shutdown = mode == "force"
            current = self._current_running_locked()
        if current is not None and mode == "cooperative-stop":
            self.stop(current.id)
        elif current is not None and mode == "force":
            current.stop_requested = True
            current.forced_stop = True
            current.status_detail = "force shutdown requested; latest checkpoint not guaranteed"
            current.persist()
            self._kill_job_tree(current)
        self._run_gate.set()  # release a worker parked on a paused queue
        self._queue.put(_SENTINEL)  # wake the worker so it can exit

    def submit(
        self,
        *,
        method: str,
        preset: str,
        methods_subdir: Optional[str],
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
        from_chain: bool = False,
        root_job_id: Optional[str] = None,
        parent_job_id: Optional[str] = None,
        attempt_index: int = 0,
        start: Optional[bool] = None,
    ) -> Job:
        job = Job(
            id=new_job_id(),
            method=method,
            preset=preset,
            methods_subdir=methods_subdir,
            overrides=dict(overrides or {}),
            extra=list(extra or []),
            from_chain=from_chain,
            root_job_id=root_job_id,
            parent_job_id=parent_job_id,
            attempt_index=attempt_index,
        )
        self._attach_config_file(
            job, config_snapshot=config_snapshot, config_file=config_file
        )
        return self._register_and_queue(job, start=start)

    def submit_command(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: Optional[dict] = None,
        chain_train: Optional[dict] = None,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
        root_job_id: Optional[str] = None,
        parent_job_id: Optional[str] = None,
        attempt_index: int = 0,
        start: Optional[bool] = None,
    ) -> Job:
        """Enqueue a plain ``python <argv>`` task (preprocess / mask).

        Goes through the same serial queue as training so a cache-build and a
        training run can't fight over the single local GPU. ``label`` is the
        display name; ``argv`` is passed straight to the venv interpreter (e.g.
        ``["tasks.py", "preprocess"]``); ``extra_env`` carries the GUI's knobs
        (``CAPTION_SHUFFLE_VARIANTS``, ``RUN_SAM_MASK``, …).

        ``chain_train`` (``{method, preset, methods_subdir}``) makes this an
        auto-chain step: on successful completion the daemon enqueues that
        training job itself (see ``_finalize``), so the chain runs to the end
        even if the GUI that started it has since closed."""
        job = Job(
            id=new_job_id(),
            method=label,
            preset="",
            kind="command",
            argv=list(argv or []),
            extra_env=dict(extra_env or {}),
            chain_train=dict(chain_train) if chain_train else None,
            root_job_id=root_job_id,
            parent_job_id=parent_job_id,
            attempt_index=attempt_index,
        )
        self._attach_config_file(
            job, config_snapshot=config_snapshot, config_file=config_file
        )
        if job.config_file:
            job.extra_env["CONFIG_FILE"] = job.config_file
            if job.chain_train is not None:
                job.chain_train.setdefault("config_file", job.config_file)
        return self._register_and_queue(job, start=start)

    def _attach_config_file(
        self,
        job: Job,
        *,
        config_snapshot: Optional[dict] = None,
        config_file: Optional[str] = None,
    ) -> None:
        """Write/copy an immutable config snapshot into this job directory."""
        if not config_snapshot and not config_file:
            return
        dst = config.job_dir(job.id) / "config.snapshot.toml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if config_snapshot:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            tmp.write_text(toml.dumps(config_snapshot), encoding="utf-8")
            tmp.replace(dst)
        else:
            src = os.path.abspath(str(config_file))
            if os.path.abspath(str(dst)) != src:
                shutil.copyfile(src, dst)
        job.config_file = str(dst)
        try:
            merged = toml.load(dst)
            job.config_signature = merged.get("config_signature") or merged.get("_config_signature")
            target = merged.get("max_train_steps") or merged.get("max_train_steps_target")
            job.target_steps = int(target) if target is not None else None
            epochs = merged.get("max_train_epochs")
            job.target_epochs = int(epochs) if epochs is not None else None
            manifest = (
                merged.get("preprocess_run")
                or merged.get("dataset_manifest")
                or merged.get("staged_profile_manifest")
            )
            job.data_manifest = str(manifest) if manifest else None
        except (OSError, ValueError, TypeError):
            job.legacy = True

    def _register_and_queue(self, job: Job, *, start: Optional[bool] = None) -> Job:
        # ``start`` controls the run gate atomically with enqueue, so there's no
        # window where a "hold this one" job could slip past the worker:
        #   False → "add to queue". Hold this job for a later Start Queue ONLY
        #           when the queue is otherwise idle (so a user can stage several
        #           jobs before pressing Start). If a job is already running or
        #           queued, the queue is "playing" — leave the gate alone so this
        #           job auto-advances behind the current ones (cassette-tape
        #           behaviour). Pausing a playing queue here was the bug that
        #           stalled auto-advance the moment the running job finished.
        #   True  → enqueue, then resume (run now — flushes any held backlog);
        #   None  → leave the gate as-is (legacy: runs if not currently paused).
        job.root_job_id = job.root_job_id or job.id
        if start is False:
            with self._lock:
                queue_idle = self._queue_is_idle_locked()
            if queue_idle:
                self.pause()
        d = config.job_dir(job.id)
        job.progress_path = str(d / "progress.jsonl")
        job.stdout_path = str(d / "stdout.log")
        # Per-job sample dir — mirrors progress_path so the preview API can
        # locate the gallery even after a WebUI restart (it reads this back
        # over HTTP, not from the injected argv).
        job.sample_dir = str(d / "sample")
        if self._job_runs_train(job) and not job.config_file:
            job.legacy = True
        tokens = list(job.extra if job.kind == "train" else job.argv)
        for key, attr in (("--preprocess_run", "data_manifest"), ("--dataset_config", "data_manifest")):
            if key in tokens and tokens.index(key) + 1 < len(tokens):
                setattr(job, attr, str(tokens[tokens.index(key) + 1]))
                break
        target = (job.overrides or {}).get("max_train_steps")
        if target is None and "--max_train_steps" in tokens:
            index = tokens.index("--max_train_steps")
            target = tokens[index + 1] if index + 1 < len(tokens) else None
        if target is not None:
            try:
                job.target_steps = int(target)
            except (TypeError, ValueError):
                pass
        epochs = (job.overrides or {}).get("max_train_epochs")
        if epochs is None and "--max_train_epochs" in tokens:
            index = tokens.index("--max_train_epochs")
            epochs = tokens[index + 1] if index + 1 < len(tokens) else None
        if epochs is not None:
            try:
                job.target_epochs = int(epochs)
            except (TypeError, ValueError):
                pass
        with self._lock:
            self._jobs[job.id] = job
            job.persist()
        self._queue.put(job.id)
        if start is True:
            self.resume()
        self._broadcast({"ev": "submitted", "job_id": job.id, "state": job.state})
        return job

    def pause(self) -> None:
        """Hold the queue: queued jobs stay ``queued`` until :meth:`resume`. A
        job already running is left alone — only the next launch waits."""
        if self._run_gate.is_set():
            self._run_gate.clear()
            self._broadcast({"ev": "queue_state", "paused": True})

    def resume(self) -> None:
        """Release a paused queue so the worker launches queued jobs in order."""
        if not self._run_gate.is_set():
            self._run_gate.set()
            self._broadcast({"ev": "queue_state", "paused": False})

    def is_paused(self) -> bool:
        return not self._run_gate.is_set()

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at)

    def delete(self, job_id: str) -> bool:
        """Delete only a terminal daemon job record and its owned artifacts."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state not in TERMINAL_STATES:
                return False
            path = job.dir
            self._jobs.pop(job_id, None)
        # The job directory contains metadata/logs/progress/sample only.  State
        # and model outputs live elsewhere and are intentionally untouched.
        shutil.rmtree(path, ignore_errors=False)
        self._broadcast({"ev": "deleted", "job_id": job_id})
        return True

    @staticmethod
    def _attempt_sort_key(job: Job) -> tuple[int, float, str]:
        return (int(job.attempt_index or 0), float(job.submitted_at or 0.0), job.id)

    def lineage(self, job_or_id: Job | str) -> list[Job]:
        """Return one logical task's physical attempts in execution order."""
        with self._lock:
            job = job_or_id if isinstance(job_or_id, Job) else self._jobs.get(job_or_id)
            if job is None:
                return []
            root_id = job.root_job_id or job.id
            attempts = [
                candidate
                for candidate in self._jobs.values()
                if (candidate.root_job_id or candidate.id) == root_id
            ]
        return sorted(attempts, key=self._attempt_sort_key)

    @staticmethod
    def _attempt_public(job: Job) -> dict:
        payload = job.public()
        payload["job_id"] = job.id
        return payload

    def job_group(self, job_id: str) -> dict | None:
        attempts = self.lineage(job_id)
        if not attempts:
            return None
        first, latest = attempts[0], attempts[-1]
        payload = latest.public()
        root_id = first.root_job_id or first.id
        payload.update(
            {
                "id": root_id,
                "root_job_id": root_id,
                "current_job_id": latest.id,
                "attempt_count": len(attempts),
                "attempts": [self._attempt_public(attempt) for attempt in attempts],
                "submitted_at": first.submitted_at,
                "started_at": first.started_at or latest.started_at,
                "ended_at": latest.ended_at,
            }
        )
        return payload

    def list_job_groups_filtered(
        self,
        *,
        state: str | None = None,
        resumable: bool | None = None,
        before: float | None = None,
        after: float | None = None,
        offset: int = 0,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> tuple[list[dict], int]:
        with self._lock:
            root_ids = {
                job.root_job_id or job.id
                for job in self._jobs.values()
            }
        groups = [self.job_group(root_id) for root_id in root_ids]
        states = {part.strip() for part in (state or "").split(",") if part.strip()}
        filtered: list[dict] = []
        for group in groups:
            if group is None:
                continue
            if states and group.get("state") not in states:
                continue
            submitted_at = float(group.get("submitted_at") or 0.0)
            if before is not None and submitted_at >= before:
                continue
            if after is not None and submitted_at < after:
                continue
            if resumable is not None and bool(group.get("recovery_state")) != resumable:
                continue
            filtered.append(group)
        filtered.sort(
            key=lambda group: (float(group.get("submitted_at") or 0.0), str(group.get("id") or "")),
            reverse=newest_first,
        )
        total = len(filtered)
        start = max(0, int(offset))
        end = None if limit is None else start + max(0, int(limit))
        return filtered[start:end], total

    def delete_group(self, job_id: str) -> list[str] | None:
        """Delete all terminal attempt records for one logical task."""
        attempts = self.lineage(job_id)
        if not attempts or any(job.state not in TERMINAL_STATES for job in attempts):
            return None
        paths = [job.dir for job in attempts]
        ids = [job.id for job in attempts]
        with self._lock:
            for attempt_id in ids:
                self._jobs.pop(attempt_id, None)
        try:
            for path in paths:
                shutil.rmtree(path, ignore_errors=False)
        except OSError:
            # Keep the in-memory table consistent with the surviving records.
            with self._lock:
                for job in attempts:
                    if job.dir.exists():
                        self._jobs[job.id] = job
            raise
        self._broadcast({"ev": "deleted_group", "root_job_id": attempts[0].root_job_id or attempts[0].id, "job_ids": ids})
        return ids

    def list_jobs_filtered(self, *, state: str | None = None, resumable: bool | None = None,
                           before: float | None = None, after: float | None = None,
                           offset: int = 0, limit: int | None = None,
                           newest_first: bool = False) -> tuple[list[Job], int]:
        jobs = self.list_jobs()
        states = {part.strip() for part in (state or "").split(",") if part.strip()}
        filtered = []
        for job in jobs:
            if states and job.state not in states:
                continue
            if before is not None and job.submitted_at >= before:
                continue
            if after is not None and job.submitted_at < after:
                continue
            if resumable is not None and bool(job.recovery_state) != resumable:
                continue
            filtered.append(job)
        filtered.sort(key=lambda job: (job.submitted_at, job.id), reverse=newest_first)
        total = len(filtered)
        start = max(0, int(offset))
        end = None if limit is None else start + max(0, int(limit))
        return filtered[start:end], total

    @staticmethod
    def _read_state_sidecar(path: Path) -> dict | None:
        try:
            data = json.loads((path / "train_state.json").read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "global_step" not in data and "current_step" not in data:
                return None
            if int(data.get("schema_version", 1) or 1) >= 2 and not (path / "complete.marker").is_file():
                return None
            data["global_step"] = int(data.get("global_step", data.get("current_step", 0)) or 0)
            return data
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _state_candidates(job: Job) -> list[tuple[int, Path]]:
        """Discover complete state dirs from a pinned config/output contract."""
        try:
            from library.io.output_layout import resolve_output_layout
            import toml
            cfg = toml.load(job.config_file) if job.config_file and os.path.isfile(job.config_file) else {}
            if not cfg:
                method = job.method
                preset = job.preset or (job.extra_env or {}).get("PRESET") or "default"
                methods_subdir = job.methods_subdir or "methods"
                if job.kind == "command" and job.argv:
                    if len(job.argv) >= 3 and str(job.argv[0]).endswith("tasks.py") and job.argv[1] == "lora-gui":
                        method, methods_subdir = job.argv[2], "gui-methods"
                    elif len(job.argv) >= 3 and str(job.argv[0]).endswith("tasks.py") and job.argv[1] == "staged-train":
                        try:
                            from library.training.staged_resolution_plan import (
                                compile_runtime_config,
                                load_profile,
                            )
                            profile = str(job.argv[2])
                            cfg_path = compile_runtime_config(
                                profile, load_profile(profile), config.ROOT
                            )
                            cfg = toml.load(cfg_path)
                        except Exception:
                            cfg = {}
                    elif len(job.argv) >= 2 and str(job.argv[0]).endswith("tasks.py"):
                        method = {"lora": "lora", "easycontrol": "easycontrol"}.get(job.argv[1], method)
                try:
                    from library.config.io import load_method_preset
                    cfg = load_method_preset(method, preset, methods_subdir=methods_subdir)
                except Exception:
                    cfg = {}
            output_dir = cfg.get("output_dir", "output/ckpt")
            output_name = cfg.get("output_name") or job.method or "last"
            if job.target_steps is None and cfg.get("max_train_steps") is not None:
                try:
                    job.target_steps = int(cfg["max_train_steps"])
                except (TypeError, ValueError):
                    pass
            if job.target_epochs is None and cfg.get("max_train_epochs") is not None:
                try:
                    job.target_epochs = int(cfg["max_train_epochs"])
                except (TypeError, ValueError):
                    pass
            layout = resolve_output_layout(output_dir, output_name, cwd=config.ROOT)
            suffixes = ("-interrupted-state", "-rolling-state", "-checkpoint-state", "-state")
            raw_name = Path(str(output_name)).name
            candidate_names = [layout.name]
            if raw_name and raw_name not in candidate_names:
                candidate_names.append(raw_name)
            out = []
            for root in layout.legacy_candidates():
                paths = [
                    root / f"{name}{suffix}"
                    for name in candidate_names
                    for suffix in suffixes
                ]
                for name in candidate_names:
                    paths.extend(sorted(root.glob(f"{name}-step*-state")))
                    paths.extend(sorted(root.glob(f"{name}-[0-9]*-state")))
                for path in paths:
                    data = JobManager._read_state_sidecar(path)
                    if data is None:
                        continue
                    if job.config_signature and data.get("config_signature") != job.config_signature:
                        continue
                    if job.dataset_signature and data.get("dataset_signature") != job.dataset_signature:
                        continue
                    out.append((int(data.get("global_step", 0) or 0), path))
            def priority(path: Path) -> int:
                name = path.name
                if name.endswith("-interrupted-state"): return 0
                if name.endswith("-rolling-state"): return 1
                if name.endswith("-checkpoint-state"): return 2
                if name == f"{layout.name}-state": return 3
                return 4
            return sorted({(step, path) for step, path in out}, key=lambda item: (-item[0], priority(item[1]), str(item[1])))
        except Exception:
            logger.exception("resume state discovery failed for job %s", job.id)
            return []

    def resume_job(self, job_id: str) -> Job | None:
        source = self.get(job_id)
        if source is None or source.state not in {STATE_STOPPED, STATE_ERROR} or source.kind != "train" and not self._command_runs_train(source.argv):
            return None
        attempts = self.lineage(source)
        if not attempts:
            return None
        latest = attempts[-1]
        if latest.id != source.id:
            raise ValueError(
                f"only the latest attempt can be resumed; latest_job_id={latest.id}"
            )
        if any(attempt.state in {STATE_QUEUED, STATE_RUNNING} for attempt in attempts):
            raise ValueError("this training already has an active attempt")
        candidates = self._state_candidates(source)
        if not candidates:
            raise ValueError("no complete signature-matched training state found")
        step, state_dir = candidates[0]
        state_data = self._read_state_sidecar(state_dir) or {}
        if source.target_steps is not None and step >= source.target_steps:
            raise ValueError(f"training target already reached at global_step={step}")
        if source.target_epochs is not None:
            current_epoch = int(state_data.get("current_epoch", state_data.get("epoch", 0)) or 0)
            if current_epoch >= source.target_epochs:
                raise ValueError(
                    f"training target already reached at epoch={current_epoch}"
                )
        if source.kind == "command":
            argv = list(source.argv)
            argv += ["--resume", str(state_dir)]
            job = self.submit_command(label=source.method, argv=argv, extra_env=source.extra_env,
                                      config_file=source.config_file,
                                      root_job_id=source.root_job_id or source.id,
                                      parent_job_id=source.id,
                                      attempt_index=int(source.attempt_index or 0) + 1,
                                      start=False)
        else:
            extra = list(source.extra)
            extra += ["--resume", str(state_dir)]
            job = self.submit(method=source.method, preset=source.preset, methods_subdir=source.methods_subdir,
                              config_file=source.config_file, overrides=source.overrides, extra=extra,
                              root_job_id=source.root_job_id or source.id,
                              parent_job_id=source.id,
                              attempt_index=int(source.attempt_index or 0) + 1,
                              start=False)
        job.config_signature = source.config_signature
        job.dataset_signature = source.dataset_signature
        job.target_steps = source.target_steps
        job.target_epochs = source.target_epochs
        job.data_manifest = source.data_manifest
        job.recovery_state = str(state_dir)
        job.recovery_step = step
        job.persist()
        # Publish all recovery metadata before allowing the worker to launch;
        # otherwise a fast queue can build the command before --config_file is
        # injected for command-style training jobs.
        self.resume()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def stale_for(self, job: Job) -> Optional[float]:
        """Seconds since the job's last progress event, for a running job."""
        if job.state != STATE_RUNNING:
            return None
        ev = tail.last_event(job.progress_path)
        if not ev:
            return None
        # progress ts is relative to run start; compare wall clock instead.
        try:
            mtime = os.path.getmtime(job.progress_path)
        except OSError:
            return None
        return round(time.time() - mtime, 1)

    def stop(self, job_id: Optional[str] = None) -> Optional[Job]:
        """Abort a job. ``None`` → the running job. Queued → cancelled in place;
        running → request a cooperative save/exit, force-killing the tree only
        after the configured grace period. The daemon stays up and advances to
        the next queued job."""
        with self._lock:
            job = self._jobs.get(job_id) if job_id else self._current_running_locked()
            if job is None or job.state in TERMINAL_STATES:
                return job
            already_requested = job.stop_requested
            job.stop_requested = True
            job.stop_requested_at = job.stop_requested_at or time.time()
            job.status_detail = "stopping"
            state = job.state
            if state == STATE_QUEUED:
                # Finalize the queued job *now* (reentrant RLock) so its cancel
                # is visible immediately: the worker is blocked monitoring a
                # running job and won't reach this id, so the old lazy path left
                # a stopped-but-"queued" entry the UI couldn't clear. The worker
                # skips dequeued ids whose state isn't QUEUED → stale FIFO entry
                # is harmless.
                self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
                return job
            job.persist()
            if state == STATE_RUNNING and not already_requested:
                # Install the monitor gate before releasing the job lock.  The
                # launcher can exit immediately on SIGTERM; without this order
                # the monitor could miss the event and advance the queue while
                # descendants were still saving/releasing GPU memory.
                self._start_stop_worker(job)
        return job

    def _start_stop_worker(self, job: Job) -> threading.Event:
        """Start at most one in-memory stop worker for a running job."""
        with self._lock:
            existing = self._stop_events.get(job.id)
            if existing is not None and not existing.is_set():
                return existing
            event = threading.Event()
            self._stop_events[job.id] = event
        threading.Thread(
            target=self._stop_job_tree_worker,
            args=(job, event),
            name=f"anima-job-stop-{job.id}",
            daemon=True,
        ).start()
        return event

    def _run(self) -> None:
        # Drain re-attached orphans before touching the queue so the serial
        # GPU invariant holds across a daemon restart. Crash-guarded like the
        # main loop: a monitor that raises must not strand the queue behind it.
        for job_id in self._adopt:
            self._worker_heartbeat = time.time()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._monitor(job, popen=None)
            except Exception:  # noqa: BLE001
                logger.exception("monitor crashed for adopted job %s", job_id)
                self._fail_safely(job_id, "daemon monitor crashed; see daemon.log")
        while True:
            job_id = self._queue.get()
            self._worker_heartbeat = time.time()
            if job_id == _SENTINEL:
                break
            with self._lock:
                if self._stopping:
                    break
            # One bad job must NEVER kill the worker thread: a dead worker leaves
            # every later job stuck `queued` forever with no error and no
            # watchdog (the stall watchdog only guards *running* jobs). Catch
            # everything, fail the offending job loudly, and keep draining.
            try:
                self._process_one(job_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "worker crashed handling job %s; queue continues", job_id
                )
                self._fail_safely(
                    job_id, "daemon worker hit an unexpected error; see daemon.log"
                )

    def _process_one(self, job_id: str) -> None:
        """Launch + monitor a single dequeued job. Uses ``return`` (not the
        loop's ``continue``) so it can run under the crash guard in ``_run``."""
        job = self._jobs.get(job_id)
        if job is None or job.state != STATE_QUEUED:
            return
        if job.stop_requested:
            self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
            return
        # Hold here while the queue is paused (the GUI's "Start Queue" button
        # resumes it). Re-validate after waking: the job may have been
        # cancelled while held, or the daemon may be shutting down.
        if not self._await_run_gate(job):
            return
        with self._lock:
            if job.state != STATE_QUEUED or job.stop_requested:
                return
        # Auto-chained train steps skip the guard: the daemon just ran the
        # preceding preprocess on this same serial queue, so the only VRAM
        # in flight is that step's still-releasing allocation, which the
        # guard would needlessly wait on. Standalone jobs still guard.
        if not job.from_chain:
            self._gpu_guard(job)
        self._launch_and_monitor(job)

    def _fail_safely(self, job_id: str, error: str) -> None:
        """Finalize a job ERROR without ever propagating — the last line of
        defense so the worker survives even a finalize that itself raises."""
        job = self.get(job_id)
        if job is None or job.state in TERMINAL_STATES:
            return
        try:
            self._finalize(job, STATE_ERROR, error=error)
        except Exception:  # noqa: BLE001
            logger.exception("failed to finalize crashed job %s", job_id)

    def worker_idle_for(self) -> float:
        """Seconds since the worker last advanced. Large + a job stuck ``queued``
        ⇒ the worker is wedged or dead. Exposed via /health."""
        return round(time.time() - self._worker_heartbeat, 1)

    def worker_alive(self) -> bool:
        return self._worker.is_alive()

    def _await_run_gate(self, job: Job) -> bool:
        """Block while the queue is paused. Returns True when cleared to launch,
        False if the worker should skip this job (daemon stopping, or the job was
        cancelled while held). Polls so a stop/shutdown is noticed promptly even
        though the gate itself stays closed."""
        if self._run_gate.is_set():
            return True
        self._broadcast({"ev": "queue_held", "job_id": job.id})
        while not self._run_gate.wait(timeout=1.0):
            with self._lock:
                if self._stopping:
                    return False
                cur = self._jobs.get(job.id)
                if cur is None or cur.stop_requested or cur.state in TERMINAL_STATES:
                    return False
        return not self._stopping

    def _launch_and_monitor(self, job: Job) -> None:
        d = config.job_dir(job.id)
        try:
            # _build_cmd runs the full config merge + lazy task-runner import for
            # train jobs; keep it INSIDE the guard so a bad config / import error
            # fails just this job instead of crashing the worker.
            cmd, env = self._build_cmd(job)
            if self._job_runs_train(job):
                stop_path = self._stop_request_path(job)
                try:
                    stop_path.unlink(missing_ok=True)
                except OSError:
                    pass
                env["ANIMA_DAEMON_STOP_FILE"] = str(stop_path)
            popen = proc.spawn_detached(
                cmd,
                cwd=config.ROOT,
                stdout_path=d / "stdout.log",
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            self._finalize(job, STATE_ERROR, error=f"launch failed: {exc}")
            return
        with self._lock:
            job.state = STATE_RUNNING
            job.started_at = time.time()
            job.pid = popen.pid
            job.create_time = proc.create_time(popen.pid)
            job.persist()
            self._popens[job.id] = popen
        self._broadcast({"ev": "started", "job_id": job.id, "pid": job.pid})
        self._monitor(job, popen=popen)

    def _monitor(self, job: Job, *, popen) -> None:
        """Block until the job process exits, then finalize. Works for both a
        process we spawned (``popen`` reaps the child) and an adopted orphan
        (``popen is None`` → psutil liveness)."""
        while self._proc_running(job, popen):
            self._worker_heartbeat = time.time()
            if self._kill_on_shutdown:
                self._kill_job_tree(job)
                break
            stalled = self._stall_reason(job)
            if stalled is not None:
                logger.warning("job %s killed by stall watchdog: %s", job.id, stalled)
                self._kill_job_tree(job)
                # Finalize now so the post-loop _finalize_from_exit (which would
                # otherwise classify the SIGKILL exit) sees a terminal state and
                # no-ops, preserving the actionable stall diagnostic.
                self._finalize(job, STATE_ERROR, error=stalled)
                break
            time.sleep(_POLL_INTERVAL)
        # Reap our own child to avoid a zombie.
        if popen is not None:
            try:
                popen.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self._wait_for_stop_worker(job)
        self._popens.pop(job.id, None)
        self._finalize_from_exit(job, popen)

    def _wait_for_stop_worker(self, job: Job) -> None:
        """Do not finalize/advance the queue before tree teardown completes."""
        if not job.stop_requested:
            return
        with self._lock:
            event = self._stop_events.get(job.id)
        if event is None:
            # Adopted jobs from an older daemon may have a persisted stop flag
            # but no in-memory event.  Their reconciliation path starts a new
            # worker when possible; this fallback keeps the monitor non-blocking
            # for genuinely stale records.
            return
        timeout = max(5.0, float(config.STOP_GRACE_SECONDS) + 10.0)
        if not event.wait(timeout=timeout):
            logger.warning(
                "stop worker for job %s did not finish within %.1fs; "
                "finalizing with the available process evidence",
                job.id,
                timeout,
            )
        with self._lock:
            if self._stop_events.get(job.id) is event:
                self._stop_events.pop(job.id, None)

    @staticmethod
    def _proc_running(job: Job, popen) -> bool:
        if popen is not None:
            return popen.poll() is None
        return proc.is_alive(job.pid, job.create_time)

    @staticmethod
    def _stall_reason(job: Job) -> Optional[str]:
        """If the running job has produced no output for longer than the
        configured stall timeout, return an actionable error naming where it
        wedged; otherwise ``None``.

        Liveness is the most recent mtime of stdout.log *or* progress.jsonl, so
        both a preprocess job (tqdm-to-stdout, no progress.jsonl) and a training
        job (progress.jsonl) are covered, and any phase that still flushes the
        occasional line — including a slow download's tqdm bar — counts as
        alive. A truly wedged process (stalled socket with no bytes, a
        symlink-cycle walk, a deadlock) writes nothing, so its files freeze and
        the watchdog fires. ``TQDM_MININTERVAL`` (10s) keeps even a busy bar
        well under either budget.

        The budget is per *kind*: a command (preprocess / mask) job is tight
        (it never legitimately goes quiet for more than a model-load), while a
        train job is unwatched by default (budget 0 → skipped here) because its
        silent first-step torch.compile trace would false-positive; it can be
        opted in via ANIMA_DAEMON_JOB_STALL_TIMEOUT.
        """
        timeout = (
            config.CMD_STALL_TIMEOUT
            if job.kind == "command"
            else config.JOB_STALL_TIMEOUT
        )
        if not timeout or timeout <= 0 or job.started_at is None:
            return None
        last = job.started_at
        for path in (job.stdout_path, job.progress_path):
            if not path:
                continue
            try:
                last = max(last, os.path.getmtime(path))
            except OSError:
                continue
        idle = time.time() - last
        if idle < timeout:
            return None
        where = JobManager._last_output_line(job)
        detail = f" last output: {where!r}" if where else " (no output captured)"
        return (
            f"stalled: no output for {int(idle)}s (limit {int(timeout)}s); daemon "
            f"killed the job so the queue can advance.{detail}"
        )

    @staticmethod
    def _last_output_line(job: Job, *, max_bytes: int = 8192) -> Optional[str]:
        """Best-effort last non-empty stdout line (carriage-return aware, so a
        tqdm bar's latest redraw is returned rather than an empty fragment) —
        this is the "where did it wedge" hint folded into the stall error."""
        path = job.stdout_path
        if not path:
            return None
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                blob = f.read()
        except OSError:
            return None
        parts = [
            p.strip() for p in re.split(r"[\r\n]", blob.decode("utf-8", "replace"))
        ]
        parts = [p for p in parts if p]
        return parts[-1] if parts else None

    def _finalize_from_exit(self, job: Job, popen) -> None:
        if job.state in TERMINAL_STATES:
            return
        ev = tail.last_event(job.progress_path)
        rc = popen.poll() if popen is not None else None
        # A terminal run_end is the trainer's authoritative outcome.  The stop
        # endpoint and the monitor run on different threads, so a click that
        # lands after the trainer has flushed run_end:ok but before the daemon
        # observes process exit must not rewrite a successful run as stopped.
        # Cooperative stops still emit run_end:stopped and therefore retain the
        # stopped state and the checkpoint/state evidence.
        if ev and ev.get("ev") == "run_end":
            status = ev.get("status")
            mapped = {
                "ok": STATE_DONE,
                "stopped": STATE_STOPPED,
                "error": STATE_ERROR,
            }.get(status, STATE_ERROR)
            event_detail = None
            if mapped == STATE_STOPPED:
                event_detail = (
                    "stop timeout; process tree force-killed"
                    if job.forced_stop
                    else "cooperative stop completed"
                )
            # A run_end event is the trainer's own terminal signal — its rc
            # isn't exposed in the event, but the popen still carries the real
            # process exit code (0 on a clean ``sys.exit``). Forward it so a
            # run_end:ok job reports 0, not None.
            if status == "ok" and job.stop_requested:
                # Do not leave a misleading "stop requested" marker on a job
                # whose trainer completed normally.  Clearing it also prevents
                # the asynchronous stop worker from signalling a reused PID
                # after this job has already become terminal.
                with self._lock:
                    job.stop_requested = False
                    job.stop_requested_at = None
                    job.forced_stop = False
                    if job.status_detail == "stopping":
                        job.status_detail = None
            self._finalize(
                job,
                mapped,
                error=ev.get("error"),
                detail=event_detail,
                rc=rc,
            )
            return
        if job.stop_requested:
            # No terminal event means the process exited before it could report
            # its own outcome. Carry the kill's exit code so callers can
            # distinguish a clean SIGTERM/SIGKILL from an unknown stop (popen
            # may already be None for an adopted orphan that vanished).
            if job.forced_stop:
                detail = "stop timeout; process tree force-killed"
            else:
                detail = "stop requested; process exited"
            self._finalize(job, STATE_STOPPED, detail=detail, rc=rc)
            return
        if rc == 0:
            self._finalize(job, STATE_DONE, rc=rc)
        else:
            # No run_end + nonzero exit: the trainer died before its terminal
            # event. Classify the code — signal deaths (SIGKILL/OOM, CUDA
            # SIGABRT, segfault) leave no traceback, so it's the only signal.
            self._finalize(job, STATE_ERROR, error=_classify_exit(rc), rc=rc)

    def _finalize(
        self,
        job: Job,
        state: str,
        *,
        error: Optional[str] = None,
        detail: Optional[str] = None,
        rc: Optional[int] = None,
    ) -> None:
        with self._lock:
            job.state = state
            job.ended_at = time.time()
            if error:
                job.error = error
            if detail:
                job.status_detail = detail
            # The real subprocess exit code (``popen.poll()``); ``None`` when
            # the caller has no process to read (queued-cancel / launch-fail /
            # orphan paths). Written in the same persist() so the WebUI's
            # ``_finalize_from_daemon`` sees it atomically with the state flip.
            job.rc = rc
            job.ckpt_path = tail.last_ckpt_path(job.progress_path)
            self._refresh_recovery_metadata(job)
            # Auto-chain: a done command job with a chain_train spec enqueues its
            # follow-on train job here (survives the GUI closing). chained_job_id
            # persists in the same write that flips us to `done` → atomic for a
            # client observing this job.
            if (
                state == STATE_DONE
                and job.kind == "command"
                and job.chain_train
                and not job.chained_job_id
            ):
                ct = job.chain_train
                follow = self.submit(
                    method=ct.get("method"),
                    preset=ct.get("preset") or "default",
                    methods_subdir=ct.get("methods_subdir"),
                    config_snapshot=ct.get("config_snapshot") or None,
                    config_file=ct.get("config_file") or None,
                    overrides=ct.get("overrides") or {},
                    extra=ct.get("extra") or [],
                    from_chain=True,
                )
                job.chained_job_id = follow.id
                logger.info(
                    "auto-chain: job %s done → enqueued training %s",
                    job.id,
                    follow.id,
                )
            job.persist()
        self._broadcast({"ev": "ended", "job_id": job.id, "state": state})

    def _refresh_recovery_metadata(self, job: Job) -> None:
        """Persist the best available state/signature metadata after a run."""
        # Signatures are computed only after train.py has parsed the pinned
        # config and materialized the dataset. Persist them in run_start so the
        # daemon can distinguish this job's state from an older run that reused
        # the same output_name, even after a daemon restart.
        starts = tail.read_events(job.progress_path, events=["run_start"], last_n=1)
        if starts:
            started = starts[-1]
            if started.get("config_signature"):
                job.config_signature = started["config_signature"]
            if started.get("dataset_signature"):
                job.dataset_signature = started["dataset_signature"]
        candidates = self._state_candidates(job)
        if not candidates:
            return
        step, path = candidates[0]
        job.recovery_state = str(path)
        job.recovery_step = step
        try:
            data = self._read_state_sidecar(path) or {}
            job.config_signature = job.config_signature or data.get("config_signature")
            job.dataset_signature = job.dataset_signature or data.get("dataset_signature")
        except Exception:
            logger.debug("could not read recovery metadata for job %s", job.id)

    def _gpu_guard(
        self,
        job: Job,
        *,
        retries: int = config.GPU_GUARD_RETRIES,
        delay: float = config.GPU_GUARD_DELAY,
        busy_frac: float = config.GPU_GUARD_BUSY_FRAC,
    ) -> None:
        """Before launching, make sure the GPU is actually free.

        Busy/free is decided from **total VRAM in use**, not the process list:
        on Windows WDDM every desktop app (dwm, explorer, browser, …) shows up
        as a "compute" process, so gating on process presence stalled the queue
        on a dozen innocent renderers every launch. A real training run holds
        GBs; an idle desktop holds <1 GB — so `used/total < busy_frac` reliably
        means "go". The threshold is deliberately loose (default 0.85): the only
        thing the guard *must* catch is VRAM leaked by our own dead jobs, and
        that is reaped by pid below regardless of the fraction; the fraction only
        guesses whether some *other* process owns the card, so a partially-loaded
        ComfyUI / browser shouldn't trip it. Process enumeration is kept only to
        reap VRAM leaked by our *own* dead jobs, matched by pid (a stranger's pid
        never matches a job, so the polluted holder list is harmless on that
        path). If we can't probe memory at all we assume free rather than
        deadlock the queue. Tunable via ANIMA_DAEMON_GPU_{BUSY_FRAC,RETRIES,DELAY}.
        """
        # A resident inference server (scripts/inference_server.py) holds a warm
        # DiT on the card. Politely ask it to free VRAM before we launch — it
        # stays alive and reloads on its next request. Best-effort; if none is
        # running this is a couple of cheap stat() calls.
        self._evict_resident_inference()

        for attempt in range(retries):
            # Reap leftovers from our own (now-terminal/dead) jobs. Safe even
            # when gpu_pids() is polluted: only pids that match a known job act.
            holders = gpu.gpu_pids() or set()
            with self._lock:
                known = {j.pid: j for j in self._jobs.values() if j.pid in holders}
            reaped = False
            for pid, owner in known.items():
                if owner.id == job.id:
                    continue
                if owner.create_time is None:
                    logger.warning(
                        "gpu_guard: refusing to reap job %s pid %s without create_time",
                        owner.id,
                        pid,
                    )
                    continue
                logger.warning(
                    "gpu_guard: reaping leaked VRAM from job %s (pid %s)", owner.id, pid
                )
                proc.kill_tree(pid, expected_create_time=owner.create_time)
                reaped = True
            if reaped:
                time.sleep(0.5)  # let the killed procs release VRAM

            mem = gpu.gpu_mem()
            if mem is None:  # can't tell → don't deadlock the queue
                return
            used, total = mem
            if total <= 0 or used / total < busy_frac:
                return  # GPU effectively free → go
            logger.warning(
                "gpu_guard: GPU busy — %d/%d MiB used (attempt %d/%d)",
                used,
                total,
                attempt + 1,
                retries,
            )
            self._broadcast(
                {
                    "ev": "gpu_wait",
                    "job_id": job.id,
                    "used_mib": used,
                    "total_mib": total,
                }
            )
            time.sleep(delay)
        # Give up waiting — proceed (the OS will OOM us if there genuinely
        # isn't room; we won't kill what we didn't start).
        job.status_detail = "launched despite busy GPU"

    def _kill_job_tree(self, job: Job) -> None:
        with self._lock:
            pid = job.pid
            expected_create_time = job.create_time
        if pid is None:
            return
        if expected_create_time is None:
            logger.warning(
                "refusing to kill job %s pid %s without persisted create_time",
                job.id,
                pid,
            )
            return
        proc.kill_tree(pid, expected_create_time=expected_create_time)

    def _stop_job_tree(self, job: Job) -> None:
        """Request a checkpoint-capable stop and enforce the hard deadline."""

        with self._lock:
            # The monitor may have observed run_end:ok while this asynchronous
            # stopper was waiting to run.  Never signal a terminal job (or a
            # PID that was cleared by a later launch) after that race.
            if job.state in TERMINAL_STATES or not job.stop_requested:
                return
            pid = job.pid
            expected_create_time = job.create_time
        if pid is None:
            return
        if expected_create_time is None:
            logger.warning(
                "refusing to stop job %s pid %s without persisted create_time",
                job.id,
                pid,
            )
            return
        if not self._job_runs_train(job):
            with self._lock:
                if (
                    job.state in TERMINAL_STATES
                    or not job.stop_requested
                    or job.pid != pid
                    or job.create_time != expected_create_time
                ):
                    return
                job.forced_stop = True
                job.status_detail = "non-training job force-killed"
                job.persist()
            proc.kill_tree(pid, expected_create_time=expected_create_time)
            return

        stop_path = self._stop_request_path(job)
        try:
            self._write_stop_request(stop_path)
        except OSError as exc:
            logger.warning("job %s stop-file write failed: %s", job.id, exc)

        with self._lock:
            if (
                job.state in TERMINAL_STATES
                or not job.stop_requested
                or job.pid != pid
                or job.create_time != expected_create_time
            ):
                return
        graceful = proc.stop_tree_gracefully(
            pid,
            expected_create_time=expected_create_time,
            grace_seconds=config.STOP_GRACE_SECONDS,
        )
        if not graceful:
            with self._lock:
                if (
                    job.state in TERMINAL_STATES
                    or not job.stop_requested
                    or job.pid != pid
                    or job.create_time != expected_create_time
                ):
                    return
                job.forced_stop = True
                job.status_detail = "stop timeout; process tree force-killed"
                job.persist()

    def _stop_job_tree_worker(
        self, job: Job, event: threading.Event
    ) -> None:
        """Run tree teardown and always release the monitor's wait gate."""
        try:
            self._stop_job_tree(job)
        finally:
            event.set()

    def _evict_resident_inference(self) -> None:
        """Ask a resident inference server (if any) to free VRAM before launch.

        Discovery mirrors scripts/inference_server.py's pidfiles (in-repo +
        per-user mirror + $ANIMA_INFERENCE_PIDFILE). Done inline (no import) so
        the daemon stays decoupled from the inference server; every failure is
        swallowed — coexistence is a courtesy, and the server's own idle-TTL
        eventually frees the card anyway.
        """
        import json
        import urllib.request
        from pathlib import Path

        candidates = []
        override = os.environ.get("ANIMA_INFERENCE_PIDFILE")
        if override:
            candidates.append(Path(override))
        candidates += [
            config.ROOT / "output" / "inference" / "server.json",
            Path.home() / ".anima" / "inference.json",
        ]
        for pf in candidates:
            try:
                port = json.loads(pf.read_text()).get("port")
            except (OSError, ValueError):
                continue
            if not port:
                continue
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{port}/unload", method="POST"
                    ),
                    timeout=5,
                ).read()
                logger.info("gpu_guard: inference server (port %s) unloaded", port)
                time.sleep(1.0)  # let VRAM release before we measure
            except Exception:  # noqa: BLE001 — best-effort
                pass
            return

    @staticmethod
    def _command_runs_train(argv: list[str]) -> bool:
        """True for command jobs that eventually invoke ``train.py`` via tasks.py.

        Delegates the command-name set to
        :func:`scripts.tasks._common.command_runs_training` so this and the
        WebUI's mirror decision stay in lockstep — see that helper for the
        rationale (and the exp-* commands that also run train.py)."""
        if len(argv) < 2:
            return False
        script = os.path.basename(str(argv[0])).lower()
        if script != "tasks.py":
            return False
        from scripts.tasks._common import command_runs_training

        return command_runs_training(str(argv[1]))

    @classmethod
    def _job_runs_train(cls, job: Job) -> bool:
        return job.kind == "train" or cls._command_runs_train(job.argv)

    @staticmethod
    def _stop_request_path(job: Job):
        return config.job_dir(job.id) / "stop.requested"

    @staticmethod
    def _write_stop_request(path, *, retries: int = _STOP_FILE_RETRIES) -> None:
        """Publish the Windows cooperative-stop marker with bounded retries."""

        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: OSError | None = None
        for attempt in range(max(1, retries)):
            temporary = path.with_name(
                f".{path.name}.{os.getpid()}-{time.time_ns()}.tmp"
            )
            try:
                temporary.write_text("stop\n", encoding="utf-8")
                os.replace(temporary, path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt + 1 < max(1, retries):
                    time.sleep(min(0.1 * (attempt + 1), 0.75))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _ensure_flag(argv: list[str], flag: str, value: str) -> None:
        """Append ``flag value`` to *argv* unless the caller already set it.

        Lowest-precedence injection: the daemon only fills in per-job paths the
        caller didn't provide (e.g. ``--progress_jsonl`` / ``--sample_dir``).
        Note this is a bare-token check, so ``--flag=value`` from a caller would
        double-add — the WebUI rejects those reserved flags at the edge instead
        (``webui/api/tasks.py::_reject_forbidden_args``).
        """
        if flag not in argv:
            argv += [flag, value]

    def _build_cmd(self, job: Job) -> tuple[list[str], dict]:
        from .client import venv_python
        from library.runtime.compat import prepare_python_child_env

        env = os.environ.copy()
        prepare_python_child_env(env)
        # tqdm redraws ride "\r"; at 0.1s cadence they drown stdout.log's real
        # lines (warnings/tracebacks). 10s is plenty — the GUI tracker parses
        # only the latest line, and training has its own progress.jsonl.
        env.setdefault("TQDM_MININTERVAL", "10")

        # Command jobs: a plain task invocation under pythonw.exe (windowless).
        # A uv-venv python.exe re-execs the real interpreter and CREATE_NO_WINDOW
        # doesn't survive that, so it pops a console whose close kills the job
        # with STATUS_CONTROL_C_EXIT (0xC000013A); pythonw never allocates one
        # (stdout still lands via spawn_detached's file redirect).
        if job.kind == "command":
            env.update(job.extra_env or {})
            argv = list(job.argv)
            # WebUI training commands are still submitted through the command
            # surface for compatibility with tasks.py, but they must use the
            # daemon's per-job progress stream. Otherwise train.py falls back to
            # output/logs/<output_name>.progress.jsonl, a cross-run shared file
            # that makes the dashboard replay stale metrics at task start.
            if self._command_runs_train(argv):
                env["TQDM_MININTERVAL"] = "0.5"
                env["TQDM_MINITERS"] = "1"
                # A resumed command job uses its immutable submission snapshot;
                # the original command remains unchanged for compatibility.
                if job.recovery_state and job.config_file:
                    self._ensure_flag(argv, "--config_file", job.config_file)
                # Per-job progress + preview paths. The WebUI reads
                # job.progress_path / job.sample_dir back over HTTP (the argv
                # here is only the train process's copy, not visible to the
                # dashboard API after a restart), so these two must be set on
                # the Job record — not just injected into argv.
                self._ensure_flag(argv, "--progress_jsonl", job.progress_path or "")
                self._ensure_flag(argv, "--sample_dir", job.sample_dir or "")
            return [venv_python(windowless=True), *argv], env

        # Imported lazily so loading the daemon package never drags in the task
        # runner's transitive imports until a job actually launches.
        from scripts.tasks._common import build_launch_cmd, build_method_args

        overrides = dict(job.overrides or {})
        extra = list(job.extra or [])
        # Dict overrides → --key value (unless already in extra). NOTE: train.py
        # bools are `store_true`, so a True override emits `--flag` but a False
        # one can only be expressed by omitting it (train.py then keeps the
        # chain's value) — a caller can't force a preset-on flag back off here.
        for key, val in overrides.items():
            flag = f"--{key}"
            if flag in extra:
                continue
            if isinstance(val, bool):
                if val:
                    extra.append(flag)
            elif key == "target_res" and isinstance(val, (list, tuple)):
                extra += [flag, *[str(v) for v in val]]
            else:
                extra += [flag, str(val)]
        # Point the structured progress stream + preview dir at the job dir so
        # we always know where they are, regardless of the method's
        # output_name default. Same rationale as the command branch above.
        self._ensure_flag(extra, "--progress_jsonl", job.progress_path or "")
        self._ensure_flag(extra, "--sample_dir", job.sample_dir or "")
        if job.config_file:
            args = ["--config_file", job.config_file, *extra]
        else:
            args = build_method_args(
                job.method,
                preset=job.preset,
                methods_subdir=job.methods_subdir,
                extra=extra,
            )
        # Windowless interpreter for the same reason as command jobs above, so
        # nothing in the train tree (incl. accelerate-launched workers) pops a
        # closable console that would CTRL_CLOSE the run.
        cmd = build_launch_cmd(*args, python_exe=venv_python(windowless=True))
        return cmd, env

    def _reconcile(self) -> None:
        self._jobs = load_all()
        for job in self._jobs.values():
            if self._job_runs_train(job) and not job.config_file and not job.legacy:
                job.legacy = True
                job.persist()
            if job.state == STATE_RUNNING:
                if proc.is_alive(job.pid, job.create_time):
                    logger.info("reconcile: re-attaching live job %s", job.id)
                    self._adopt.append(job.id)
                    if job.stop_requested:
                        logger.info(
                            "reconcile: resuming persisted stop request for job %s",
                            job.id,
                        )
                        self._start_stop_worker(job)
                else:
                    logger.info("reconcile: job %s died while we were down", job.id)
                    # The progress stream is the only durable terminal proof
                    # available when the daemon did not own/reap the child.
                    # Prefer it over a synthetic orphan/error classification.
                    terminal = tail.last_event(job.progress_path)
                    if terminal and terminal.get("ev") == "run_end":
                        status = terminal.get("status")
                        mapped = {"ok": STATE_DONE, "stopped": STATE_STOPPED, "error": STATE_ERROR}.get(status)
                        if mapped:
                            self._finalize(job, mapped, error=terminal.get("error"), rc=None,
                                           detail="reconciled from progress run_end")
                            continue
                    if job.stop_requested:
                        self._finalize(
                            job,
                            STATE_STOPPED,
                            detail=(
                                "forced stop while daemon was down"
                                if job.forced_stop
                                else "cooperative stop while daemon was down"
                            ),
                        )
                    else:
                        self._finalize(
                            job,
                            STATE_ERROR,
                            error="daemon was down when the process exited",
                            detail="orphaned",
                        )
            elif job.state == STATE_QUEUED:
                self._queue.put(job.id)
            elif job.state in TERMINAL_STATES:
                previous = (job.recovery_state, job.recovery_step)
                self._refresh_recovery_metadata(job)
                if previous != (job.recovery_state, job.recovery_step):
                    job.persist()

    def _current_running_locked(self) -> Optional[Job]:
        for job in self._jobs.values():
            if job.state == STATE_RUNNING:
                return job
        return None

    def _queue_is_idle_locked(self) -> bool:
        """True when no job is running or waiting to run — the worker is parked
        on an empty queue. Used to decide whether an ``add to queue`` submission
        should hold the gate (idle → stage it) or leave the gate alone (a job is
        already playing → let the new one auto-advance behind it). The
        just-submitted job is not in ``_jobs`` yet when this is consulted."""
        return not any(
            job.state in (STATE_QUEUED, STATE_RUNNING) for job in self._jobs.values()
        )

    def active_job(self) -> Optional[Job]:
        """The currently-running job, if any (lock-safe public accessor)."""
        with self._lock:
            return self._current_running_locked()

    def subscribe(self) -> "queue.Queue[dict]":
        q: "queue.Queue[dict]" = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "queue.Queue[dict]") -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _broadcast(self, event: dict) -> None:
        event.setdefault("ts", time.time())
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow consumer; drop rather than block the worker
