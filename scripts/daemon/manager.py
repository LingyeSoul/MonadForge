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
import threading
import time
from typing import Optional

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
_SENTINEL = "__stop__"


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._popens: dict[str, object] = {}  # job_id -> Popen (spawned only)
        self._adopt: list[str] = []  # running orphans to monitor before the queue
        self._subscribers: set["queue.Queue[dict]"] = set()
        self._stopping = False
        self._kill_on_shutdown = False
        self._worker = threading.Thread(
            target=self._run, name="anima-job-worker", daemon=True
        )

    # ----- lifecycle -----

    def start(self) -> None:
        config.ensure_state_dirs()
        self._reconcile()
        self._worker.start()

    def shutdown(self, *, kill_jobs: bool) -> None:
        """Stop accepting work and unblock the worker. With ``kill_jobs`` the
        active job tree is torn down and the GPU freed before the daemon exits.
        """
        with self._lock:
            self._stopping = True
            self._kill_on_shutdown = kill_jobs
            current = self._current_running_locked()
        if kill_jobs and current is not None:
            current.stop_requested = True
            self._kill_job_tree(current)
        self._queue.put(_SENTINEL)  # wake the worker so it can exit

    # ----- submission / query -----

    def submit(
        self,
        *,
        method: str,
        preset: str,
        methods_subdir: Optional[str],
        overrides: dict,
        extra: list[str],
    ) -> Job:
        job = Job(
            id=new_job_id(),
            method=method,
            preset=preset,
            methods_subdir=methods_subdir,
            overrides=dict(overrides or {}),
            extra=list(extra or []),
        )
        return self._register_and_queue(job)

    def submit_command(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: Optional[dict] = None,
        chain_train: Optional[dict] = None,
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
        )
        return self._register_and_queue(job)

    def _register_and_queue(self, job: Job) -> Job:
        d = config.job_dir(job.id)
        job.progress_path = str(d / "progress.jsonl")
        job.stdout_path = str(d / "stdout.log")
        with self._lock:
            self._jobs[job.id] = job
            job.persist()
        self._queue.put(job.id)
        self._broadcast({"ev": "submitted", "job_id": job.id, "state": job.state})
        return job

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at)

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

    # ----- stop (job-scoped) -----

    def stop(self, job_id: Optional[str] = None) -> Optional[Job]:
        """Abort a job. ``None`` → the running job. Queued → cancelled in place;
        running → tree killed, GPU freed. The daemon stays up and advances to
        the next queued job."""
        with self._lock:
            job = self._jobs.get(job_id) if job_id else self._current_running_locked()
            if job is None or job.state in TERMINAL_STATES:
                return job
            job.stop_requested = True
            job.persist()
            state = job.state
        if state == STATE_RUNNING:
            self._kill_job_tree(job)
        # A queued job is finalized lazily when the worker dequeues it and sees
        # stop_requested — no need to surgically remove it from the FIFO.
        return job

    # ----- worker -----

    def _run(self) -> None:
        # Drain re-attached orphans before touching the queue so the serial
        # GPU invariant holds across a daemon restart.
        for job_id in self._adopt:
            job = self.get(job_id)
            if job is not None:
                self._monitor(job, popen=None)
        while True:
            job_id = self._queue.get()
            if job_id == _SENTINEL:
                break
            with self._lock:
                if self._stopping:
                    break
                job = self._jobs.get(job_id)
            if job is None or job.state != STATE_QUEUED:
                continue
            if job.stop_requested:
                self._finalize(job, STATE_STOPPED, detail="cancelled while queued")
                continue
            self._gpu_guard(job)
            self._launch_and_monitor(job)

    def _launch_and_monitor(self, job: Job) -> None:
        cmd, env = self._build_cmd(job)
        d = config.job_dir(job.id)
        try:
            popen = proc.spawn_detached(
                cmd,
                cwd=config.ROOT,
                stdout_path=d / "stdout.log",
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            self._finalize(job, STATE_ERROR, error=f"spawn failed: {exc}")
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
            if self._kill_on_shutdown:
                self._kill_job_tree(job)
                break
            time.sleep(_POLL_INTERVAL)
        # Reap our own child to avoid a zombie.
        if popen is not None:
            try:
                popen.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self._popens.pop(job.id, None)
        self._finalize_from_exit(job, popen)

    @staticmethod
    def _proc_running(job: Job, popen) -> bool:
        if popen is not None:
            return popen.poll() is None
        return proc.is_alive(job.pid, job.create_time)

    def _finalize_from_exit(self, job: Job, popen) -> None:
        if job.state in TERMINAL_STATES:
            return
        ev = tail.last_event(job.progress_path)
        rc = popen.poll() if popen is not None else None
        if job.stop_requested:
            self._finalize(job, STATE_STOPPED)
            return
        if ev and ev.get("ev") == "run_end":
            status = ev.get("status")
            mapped = {
                "ok": STATE_DONE,
                "stopped": STATE_STOPPED,
                "error": STATE_ERROR,
            }.get(status, STATE_ERROR)
            self._finalize(job, mapped, error=ev.get("error"))
            return
        if rc == 0:
            self._finalize(job, STATE_DONE)
        else:
            # No run_end and a nonzero/unknown exit — the trainer died before it
            # could write its terminal event.
            self._finalize(
                job,
                STATE_ERROR,
                error=f"process exited (code={rc}) without a run_end event",
            )

    def _finalize(
        self,
        job: Job,
        state: str,
        *,
        error: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        with self._lock:
            job.state = state
            job.ended_at = time.time()
            if error:
                job.error = error
            if detail:
                job.status_detail = detail
            job.ckpt_path = tail.last_ckpt_path(job.progress_path)
            # Daemon-managed auto-chain: a successfully finished command job that
            # carries a chain_train spec enqueues its follow-on training job
            # right here, so the chain survives the GUI closing. Recorded on this
            # job (chained_job_id) and persisted in the same write that flips us
            # to `done`, so a client observing this job sees both atomically.
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
                    overrides=ct.get("overrides") or {},
                    extra=ct.get("extra") or [],
                )
                job.chained_job_id = follow.id
                logger.info(
                    "auto-chain: job %s done → enqueued training %s",
                    job.id,
                    follow.id,
                )
            job.persist()
        self._broadcast({"ev": "ended", "job_id": job.id, "state": state})

    # ----- gpu guard -----

    def _gpu_guard(
        self, job: Job, *, retries: int = 6, delay: float = 5.0, busy_frac: float = 0.50
    ) -> None:
        """Before launching, make sure the GPU is actually free.

        Busy/free is decided from **total VRAM in use**, not the process list:
        on Windows WDDM every desktop app (dwm, explorer, browser, …) shows up
        as a "compute" process, so gating on process presence stalled the queue
        on a dozen innocent renderers every launch. A real training run holds
        GBs; an idle desktop holds <1 GB — so `used/total < busy_frac` reliably
        means "go". Process enumeration is kept only to reap VRAM leaked by our
        *own* dead jobs, matched by pid (a stranger's pid never matches a job,
        so the polluted holder list is harmless on that path). If we can't probe
        memory at all we assume free rather than deadlock the queue.
        """
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
                logger.warning(
                    "gpu_guard: reaping leaked VRAM from job %s (pid %s)", owner.id, pid
                )
                proc.kill_tree(pid)
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
        if job.pid is not None:
            proc.kill_tree(job.pid)

    # ----- command building -----

    def _build_cmd(self, job: Job) -> tuple[list[str], dict]:
        from .client import venv_python

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        # Command jobs (preprocess / mask) are a plain task invocation. Launch
        # under pythonw.exe (windowless): a uv-venv python.exe is a trampoline
        # that re-execs the real interpreter, and CREATE_NO_WINDOW doesn't
        # survive that re-exec — so a python.exe child pops a console window
        # that, when closed (or torn down with the GUI), kills the job with
        # STATUS_CONTROL_C_EXIT (0xC000013A). pythonw.exe never allocates a
        # console; the tqdm progress the GUI tails from stdout.log still lands
        # because spawn_detached redirects the child's stdout/stderr to that
        # file (a real handle, not an inherited console). No --progress_jsonl
        # injection — these emit tqdm to stdout and the monitor finalizes them
        # on exit code (no run_end event).
        if job.kind == "command":
            env.update(job.extra_env or {})
            return [venv_python(windowless=True), *job.argv], env

        # Imported lazily so loading the daemon package never drags in the task
        # runner's transitive imports until a job actually launches.
        from scripts.tasks._common import build_launch_cmd, build_method_args

        overrides = dict(job.overrides or {})
        extra = list(job.extra or [])
        # Translate dict overrides into --key value pairs unless already present.
        # NOTE: most train.py bool flags are `store_true`, so a True override
        # emits `--flag` but a False one can only be expressed by *omitting* it —
        # train.py then keeps whatever the base→preset→method chain set. That's
        # fine for the ComfyUI node's overrides (the only preset that turns these
        # bools on is low_vram, which it never pairs with a False override), but
        # a caller can't force a preset-on flag back off through this path.
        for key, val in overrides.items():
            flag = f"--{key}"
            if flag in extra:
                continue
            if isinstance(val, bool):
                if val:
                    extra.append(flag)
            else:
                extra += [flag, str(val)]
        # Point the structured progress stream at the job dir so we always know
        # where it is, regardless of the method's output_name default.
        if "--progress_jsonl" not in extra:
            extra += ["--progress_jsonl", job.progress_path or ""]
        args = build_method_args(
            job.method,
            preset=job.preset,
            methods_subdir=job.methods_subdir,
            extra=extra,
        )
        # Windowless interpreter for the same reason as command jobs above:
        # accelerate's `python -m accelerate_cli launch` parent and the train.py
        # workers it spawns (via sys.executable) all inherit pythonw.exe, so
        # nothing pops a closable console that would CTRL_CLOSE the run.
        cmd = build_launch_cmd(*args, python_exe=venv_python(windowless=True))
        return cmd, env

    # ----- reconciliation (boot) -----

    def _reconcile(self) -> None:
        self._jobs = load_all()
        for job in self._jobs.values():
            if job.state == STATE_RUNNING:
                if proc.is_alive(job.pid, job.create_time):
                    logger.info("reconcile: re-attaching live job %s", job.id)
                    self._adopt.append(job.id)
                else:
                    logger.info("reconcile: job %s died while we were down", job.id)
                    job.stop_requested = False
                    self._finalize(
                        job,
                        STATE_ERROR,
                        error="daemon was down when the process exited",
                        detail="orphaned",
                    )
            elif job.state == STATE_QUEUED:
                self._queue.put(job.id)

    # ----- helpers -----

    def _current_running_locked(self) -> Optional[Job]:
        for job in self._jobs.values():
            if job.state == STATE_RUNNING:
                return job
        return None

    def active_job(self) -> Optional[Job]:
        """The currently-running job, if any (lock-safe public accessor)."""
        with self._lock:
            return self._current_running_locked()

    # ----- pub/sub for SSE -----

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
