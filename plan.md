# Plan — Local Training Daemon + Structured Progress + MCP

Status: Phase 0 + Phase 1 + Phase 2 implemented (2026-05-20); Phase 3 (MCP) not started. Owner: @sorryhyun. Drafted 2026-05-20.

Phase 1 lives in `scripts/daemon/` (`config`, `proc`, `gpu`, `jobs`, `tail`,
`manager`, `server`, `client`, `__main__`), CLI verbs in `scripts/tasks/daemon.py`
(`make daemon[-attach|-kill|-terminate]`), the shared `build_method_args`
extracted into `scripts/tasks/_common.py`, and the ComfyUI trainer-node cutover
in `custom_nodes/comfyui-anima-trainer/nodes.py`. Tests: `tests/test_daemon.py`.

## Goal

Unify the three training frontends (PySide6 GUI, ComfyUI trainer node, `tasks.py`
CLI) behind a single **local job-queue daemon**, add a **structured progress
event stream**, and expose an **MCP server** so Claude/agents can drive
train→eval→retune loops.

Motivations (confirmed): (1) sweep / job queue, (2) agentic iteration via MCP,
(3) fix the ComfyUI node blocking the whole UI with no progress.

## Explicit non-goals

Single local GPU, one job at a time. Therefore **do not build**:

- auth / TLS / remote network binding (bind `127.0.0.1` only)
- multi-GPU scheduler or concurrent jobs (queue is FIFO/serial *by design*)
- distributed / multi-machine anything

If we ever go multi-machine, revisit — but do not pre-pay for it now.

## Current state (why this is mostly plumbing around existing code)

| Frontend | Launch mechanism | Progress | Lifecycle |
|---|---|---|---|
| PySide6 GUI | subprocess → `tasks.py` → `accelerate launch` | regex-parses tqdm stdout (`gui/progress.py:28`) | dies if GUI closes |
| ComfyUI trainer node | **in-process** `AnimaTrainer.train()` (`custom_nodes/comfyui-anima-trainer/nodes.py:87`) | none — blocks ComfyUI | blocks the whole UI |
| `tasks.py` CLI | subprocess → `accelerate launch` (`scripts/tasks/_common.py:444` `accelerate_launch`) | stdout | tied to terminal |

Three launch paths, **zero shared state**. The ComfyUI node is the worst: it
runs training synchronously inside the ComfyUI process with no progress.

## Architecture

```
                    ┌─────────────────────────────┐
   ComfyUI node ───▶│  anima trainer daemon        │
   GUI          ───▶│  (127.0.0.1, localhost HTTP) │──▶ spawns subprocess:
   CLI (--queue)───▶│  • FIFO job queue (serial)   │   accelerate launch
   MCP server   ───▶│  • spawns subprocess per job │   → train.py
                    │  • relays progress.jsonl     │   (reuses existing path)
                    └─────────────────────────────┘
```

Two load-bearing design calls:

1. **Daemon spawns subprocesses; it does NOT run training in-process.** A CUDA
   OOM / segfault then kills the *job*, not the daemon + queue. It also reuses
   the `accelerate_launch()` path that already exists. (This is the opposite of
   today's ComfyUI node — which is exactly why that node is fragile.)
2. **Localhost only, no auth.** No away-from-desk control requirement.

---

## Phase 0 — Structured progress sink *(DONE — 2026-05-20)*

The keystone. Everything downstream reads this. Independently useful: the GUI
can drop its brittle tqdm regex *today*, before any daemon exists.

**Implemented:** `library/training/progress.py` (`ProgressSink` +
`run_scope` lifecycle context manager), wired into `train.py` (`run_start` at
sink construction, `run_scope` emits the matching `run_end` ok/stopped/error on
loop exit, `step`/`val` via `dispatch_logs`, `ckpt` via `CheckpointSaver.save`).
Gated by `--progress_jsonl` (default on → `<output_dir>/<output_name>.progress.jsonl`;
empty/`none`/`off` disables). GUI cutover via `gui/progress.py`
`JsonlProgressReader` + a 400 ms `QTimer` poll in `config_tab.py`, with the tqdm
regex retained as fallback until the file appears. Tests:
`tests/test_progress_sink.py`, `tests/test_gui_jsonl_progress.py`.

The metrics fan-out was extracted off the trainer in the same pass: the body of
the former `AnimaTrainer.accelerator_logging` now lives as the free function
`dispatch_logs` in `library/training/log_dispatch.py` (distinct from
`library/log.py`, which is stdlib console logging). `AnimaTrainer` keeps thin
`step_logging`/`epoch_logging`/`val_logging` wrappers that call it with the
trainer's `progress_sink`, so loop.py / validation call sites are unchanged.

### Hook point

`dispatch_logs` (`library/training/log_dispatch.py`) is the single chokepoint —
it receives `logs` (loss, lr, `vr/*`, CMMD, …), `global_step`, `epoch`,
`val_step`, fans out to tensorboard/wandb trackers, and forwards the same dict
to the progress sink on the main process. The sink is **one more sink appended
here**, not an accelerate `GeneralTracker` (it needs lifecycle events the
tracker protocol doesn't model).

### Event schema (JSONL, one event per line)

Write to `output/ckpt/<run_name>/progress.jsonl` (next to the checkpoint).

```jsonc
// run lifecycle
{"ev": "run_start", "ts": 0.0, "run": "<name>", "method": "...", "preset": "...",
 "total_steps": 1234, "total_epochs": 64, "pid": 12345}
// per-log-interval (mirrors the existing `logs` dict, flattened)
{"ev": "step", "ts": ..., "global_step": 100, "epoch": 2, "loss": 0.0123,
 "lr": 1e-4, "vr/lambda_ema": -0.72}
// validation pass
{"ev": "val", "ts": ..., "global_step": 100, "cmmd": 0.0345}
// checkpoint written
{"ev": "ckpt", "ts": ..., "global_step": 100, "path": "output/ckpt/....safetensors"}
// terminal
{"ev": "run_end", "ts": ..., "status": "ok|error|stopped", "final_step": 1234,
 "error": null}
```

Append-only, fsync-light (line-buffered). A reader tails the file; missing file
= not started; last line `run_end` = done.

### Implementation sketch

- Add a tiny `ProgressTracker` (an accelerate `GeneralTracker` subclass, or a
  plain object) that opens the JSONL and writes on `.log()`. Register it in the
  same place the other trackers are set up so `accelerator_logging` fans out to
  it automatically.
- Emit `run_start` / `run_end` at train-loop entry/exit (`train.py:1857`
  `def train`, final save at `train.py:2257` `saver.save_final`).
- Gate behind `--progress_jsonl <path>` (default on, derived from `output_name`)
  so it is zero-config for the daemon and CLI alike.

### Client cutover (free win)

GUI `TqdmProgressTracker` (`gui/progress.py:28`) reads the JSONL instead of
regex-parsing stdout. Keep stdout fallback for one release.

### Acceptance

- A normal `make lora` run produces a well-formed `progress.jsonl` with
  `run_start … step* … val* … ckpt* … run_end:ok`.
- Killing the run yields `run_end:stopped` (or absence handled gracefully).
- GUI progress bar driven from JSONL matches tqdm within one log interval.

---

## Phase 1 — The daemon + migrate the ComfyUI node *(DONE — 2026-05-20)*

Smallest thing that fixes the ComfyUI pain and delivers the queue.

**Implemented as designed.** `scripts/daemon/` is a single localhost
`ThreadingHTTPServer` (zero new deps) with a FIFO `queue.Queue` + one worker
thread. Each job spawns `accelerate launch … train.py` detached
(`proc.spawn_detached`: `start_new_session` POSIX / `CREATE_NO_WINDOW |
DETACHED_PROCESS` Windows), with `--progress_jsonl` pointed at
`output/daemon/jobs/<id>/progress.jsonl` so the daemon follows the run by
file-tail (no subprocess transport — Windows-safe) and never needs a pipe. Jobs
are identified by `(pid, create_time)` and torn down via a psutil tree walk
(`proc.kill_tree`, the `Popen`-flavored port of `gui/process.py`). Boot
reconciliation (`manager._reconcile`) re-attaches a live orphan, marks a dead
one `error`/`orphaned`, and re-enqueues `queued` jobs; a GPU guard
(`gpu.gpu_pids` via pynvml→nvidia-smi) reaps VRAM leaked by our own dead jobs
and waits-then-warns on an unknown holder rather than blind-killing it. The
single-daemon lock is the pidfile `(pid, create_time)` + `EADDRINUSE`. SSE is
hand-rolled (flush-per-event, BrokenPipe-guarded). The ComfyUI node now submits
to the daemon and drives a `ProgressBar` from the progress stream instead of
running training in-process.

Deferred to later phases (unchanged from original plan): CLI `--queue`
submission and the GUI-as-client cutover are **Phase 2**; the MCP wrapper is
**Phase 3**. Phase 1 ships no CLI job *producer* — the ComfyUI node and direct
HTTP clients are the producers; `make daemon*` only manages the daemon.

### Daemon

- New package `server/` (or `scripts/daemon/`). Single process: FIFO
  `queue.Queue` + worker thread + localhost HTTP via stdlib
  `http.ThreadingHTTPServer` (no framework, **zero new deps**). Thread-per-
  connection is fine at this scale — one ComfyUI node, maybe one attached
  terminal, the MCP client. A parked SSE stream just holds a thread blocking on
  the queue.
- Routing is a hand-written `(method, path)` dispatch on a
  `BaseHTTPRequestHandler` subclass (~6–8 endpoints); request bodies are
  `json.loads`'d into dicts — no Pydantic, since the only client is a trusted
  localhost caller with no hostile input.
- Per job: build the same arg list `train()` builds
  (`scripts/tasks/_common.py:480`), spawn via the existing `accelerate_launch`
  command (refactor it to *return* the cmd list so daemon + `tasks.py` share
  one builder), capture stdout to a log file, tail the run's `progress.jsonl`.
- State: in-memory job table + a small on-disk `jobs/` dir (one JSON per job)
  so the daemon survives restart and can show history.

### HTTP API

```
POST /jobs            {method, preset, methods_subdir, overrides:{}} → {job_id}
GET  /jobs            → [{job_id, state, run, submitted_at}]
GET  /jobs/{id}       → {state, latest progress event, ckpt path, pid, stale_for}
POST /jobs/{id}/stop  → killpg the job tree (reuse gui/process.py tree-kill)
GET  /jobs/{id}/logs  → tail stdout + progress.jsonl (SSE or chunked)
GET  /events          → daemon-level event stream (queue changes, job lifecycle)
GET  /health          → {ok, pid, active_job}
POST /shutdown        → {kill_jobs: bool} graceful daemon exit (terminate path)
```

States: `queued → running → {done|error|stopped}`.

**SSE is hand-rolled** (no framework helper): set `Content-Type:
text/event-stream`, write `data: <json>\n\n` per event, and **flush after each**
(`BaseHTTPRequestHandler.wfile` buffers — without the flush the client sees
nothing until the buffer fills). Wrap the write loop to catch
`BrokenPipeError`/`ConnectionResetError` and break, so a client ctrl-C doesn't
leak the handler thread.

### ComfyUI node cutover (the headline fix)

Convert `comfyui-anima-trainer/nodes.py:87` `_train_and_save` from in-process
blocking to a **daemon client**: `POST /jobs`, poll `GET /jobs/{id}`, surface
progress via ComfyUI's `comfy.utils.ProgressBar`, return the ckpt path on
`done`. Auto-start the daemon if not running (spawn + wait for health).

### CLI surface

Four new `COMMANDS` entries (`tasks.py:39`) with bodies in a new
`scripts/tasks/daemon.py`:

| Target | Daemon | Training | Meaning |
|---|---|---|---|
| `make daemon` | **start** (idempotent — no-op if up) | — | spawn detached, write pidfile, wait `/health` |
| `make daemon-attach [JOB=<id>]` | — | — | read-only viewer; ctrl-C detaches only |
| `make daemon-kill [JOB=<id>]` | **stays alive** | kills the running (or `JOB`) job, frees GPU | "abort this run, keep serving" |
| `make daemon-terminate` | **stops** | active job dies too | "shut the whole thing down" |

**Why ctrl-C can't kill the daemon (two independent guarantees):**

1. `make daemon` starts it **detached from the console** — `start_new_session=True`
   on POSIX (setsid; the terminal's `SIGINT` goes to the foreground process group
   only, so it can't reach the daemon), `CREATE_NO_WINDOW | DETACHED_PROCESS` on
   Windows (console `CTRL_C_EVENT` doesn't reach a detached process). It redirects
   stdout to a log file (mandatory on Windows — a detached process has no inherited
   stdio) and writes a `(pid, create_time)` pidfile, then `make daemon` polls
   `/health` and returns.
2. `make daemon-attach` is a **non-owning client** — it streams `/events` (or
   `/jobs/<id>/logs`) and is the parent of nothing. Ctrl-C catches `SIGINT`,
   closes the socket, exits 0; the accelerate spawn is a child of the *daemon*,
   not this terminal, so it is untouched. Attaching N terminals and ctrl-C-ing
   all of them leaves training unaffected. `daemon-attach` does NOT auto-start
   the daemon (prints "no daemon; `make daemon` to start").

**Teardown semantics** — both verify `(pid, create_time)` from the pidfile first
so neither touches a PID-reused stranger:

- `daemon-kill` is **job-scoped**: killpg the running job's tree
  (SIGTERM→SIGKILL escalation via `gui/process.py`), free GPU, mark it
  `stopped`. The daemon stays up and **immediately advances to the next queued
  job** — to stop everything use `terminate`, not `kill`. Single-GPU/serial
  makes "the running job" unambiguous with no arg; `JOB=<id>` cancels a specific
  queued/running one.
- `daemon-terminate` is **daemon-scoped**: `POST /shutdown {kill_jobs:true}` —
  stop accepting, kill the active job tree, free GPU, exit the daemon, clear
  pidfile. Queue is discarded.

### Process lifecycle & stale-process handling

Every job is a **process tree** (`accelerate launch → train.py → workers`), not a
single PID — that's the root of most staleness. Load-bearing rules:

- **Identify jobs by `(pid, create_time)`, never PID alone.** Record
  `psutil.Process(pid).create_time()` at spawn; liveness = PID exists *and*
  create_time matches. This is the sole defense against PID reuse and is what
  makes crash recovery safe.
- **Kill the whole tree via psutil, not `os.killpg`.** Snapshot descendants
  up-front (`parent.children(recursive=True)`) → `terminate()` → `kill()`
  survivors after a grace period. This is exactly `gui/process.py:57`
  `kill_process_tree` — port it to a `Popen`-flavored copy (the GUI one is
  `QProcess`-bound) rather than reinventing. setsid / `start_new_session=True`
  is a **Unix-only optimization**, not a requirement — psutil's tree walk works
  on Windows with no process group (see Cross-platform notes).
- **Boot-time reconciliation sweep** (crash recovery core). For each job in
  `jobs/`:
  - `running` + alive → **re-attach**: resume tailing its `progress.jsonl`, keep
    monitoring. No pipe needed — this is the payoff of file-based progress
    (Phase 0); the daemon can adopt an orphan it didn't spawn.
  - `running` + dead → mark `error` (`status: "orphaned"`); read last
    `progress.jsonl` line for the death point.
  - `queued` → re-enqueue.
- **GPU guard before dequeuing the next job.** Serial = exactly zero anima
  training procs should hold VRAM between jobs. Query GPU procs (pynvml /
  `nvidia-smi`): free → launch; held by a known-dead job's PID → kill then
  launch; held by an unknown proc → refuse and surface in status (never blind-kill
  what we didn't start).
- **Don't auto-kill on silence.** A `running` job whose `progress.jsonl` stalled
  is ambiguous (hung step vs. a normal-but-quiet CMMD val pass / checkpoint
  save). PID dead → unambiguous → auto-reap to `error`. PID alive but silent →
  *warn only* (`stale_for` in `/jobs/{id}`), never auto-kill.
- **Single-daemon lock.** Pidfile `(pid, create_time)` + fixed localhost port.
  Startup: live daemon → refuse/attach; dead PID → take over. `EADDRINUSE` is a
  second free signal.

### Cross-platform / Windows notes

The daemon must run on Windows (`python tasks.py daemon[-attach|-kill|-terminate]`
— `make` is the Unix alias). All Windows risk is in the process-control layer;
the codebase already solved most of it.

- **psutil is the cross-platform process abstraction.** `create_time()`,
  `children(recursive=True)`, `terminate()`/`kill()`, `pid_exists()` all work on
  both OSes. Route every spawn/kill/liveness check through it — never call
  `os.killpg`/setsid/`SIGKILL` directly.
- **Detached spawn** branches on `sys.platform`: `start_new_session=True` (POSIX)
  vs `CREATE_NO_WINDOW | DETACHED_PROCESS` (Windows). `_common.py` already carries
  the `CREATE_NO_WINDOW` half for the no-console GUI path — extend it.
- **No asyncio subprocess transport** (sidesteps Windows `ProactorEventLoop`
  subprocess bugs). The daemon `Popen`s the job detached and monitors via
  file-tail (`progress.jsonl`) + psutil polling — it never awaits a child
  transport. This is a payoff of the Phase 0 file-based-progress decision.
- **Orphan liveness is platform-agnostic** — Windows doesn't reparent orphans to
  init, but reconciliation uses the pidfile's `(pid, create_time)` + psutil, not
  parent-child, so the boot sweep works identically.
- **Test item, not a blocker:** concurrent read of `progress.jsonl` while the
  training process appends. Python `open()` on Windows defaults to shared-read so
  tail-while-write generally works, but Windows file locking is stricter than
  POSIX — add an explicit smoke test.

### Acceptance

- Submitting two jobs runs them serially; second shows `queued` then `running`.
- ComfyUI node no longer freezes the UI; shows a moving progress bar.
- `make daemon` then ctrl-C in an attached terminal: training keeps running.
- `make daemon-kill` aborts the active job and frees VRAM; daemon stays up and
  starts the next queued job. `make daemon-terminate` takes everything down.
- Kill the daemon mid-job, restart: reconciliation re-attaches the still-alive
  job (or marks a dead one `orphaned`); job history preserved from `jobs/`.
- A job holding VRAM after a crash is detected by the GPU guard before the next
  launch instead of OOM-ing it.

---

## Phase 2 — GUI + CLI as clients *(DONE — 2026-05-20)*

**Implemented.**

- **CLI `--queue`** (`scripts/tasks/_common.py`): `train()` detects `--queue`
  anywhere in `extra`, strips it, and routes to `_queue_submit` instead of
  `accelerate_launch`. `_queue_submit` auto-starts the daemon
  (`client.ensure_daemon`), folds `ARTIST` / `PROFILE_STEPS` into `extra` as
  explicit flags (the daemon's own `build_method_args` doesn't read env vars),
  POSTs the job, and prints the job id + `make daemon-*` follow/cancel hints.
  Works for both `make lora --queue` and `make lora-gui <variant> --queue`
  (both go through `train()`). This is the overnight sweep: submit ×N, drains
  serially.
- **GUI as daemon client** (`gui/daemon.py` + `gui/tabs/config_tab.py`):
  `_launch_training` submits to the daemon instead of owning a `QProcess`, so
  training survives the GUI closing. The tab *observes* the job by polling the
  per-job files the daemon already writes locally — `job.json` (state),
  `progress.jsonl` (bar, via the existing `JsonlProgressReader`), `stdout.log`
  (log, via a new `FileTailer`) — off a single 400 ms `QTimer`. No SSE
  consumer thread: the daemon is localhost-only so the files are right there
  (chosen over the plan's literal "subscribe to `/jobs/{id}/logs`"; functionally
  equivalent, thread-free). `_try_reattach` (on tab construction) re-binds to a
  job still running from a previous session or one the CLI / ComfyUI node
  submitted. Stop aborts via the daemon (job-scoped); window-close
  (`cleanup_subprocess`) deliberately leaves the job alive. **Test and
  preprocess stay on the QProcess path** — they aren't daemon jobs.

Tests: `tests/test_daemon.py::test_cli_queue_*`.

- GUI: submit to daemon instead of owning a `QProcess`; training survives GUI
  close. Subscribe to `/jobs/{id}/logs`.
- `tasks.py lora --queue`: enqueue instead of running inline. This is the
  **overnight sweep** — `make lora --queue` ×N drains serially.

### Acceptance

- Close GUI mid-train → training continues; reopening GUI re-attaches to the
  running job.
- `for v in tlora ortholora fera; do make lora-gui GUI_PRESETS=$v --queue; done`
  enqueues 3 jobs that run back-to-back unattended.

---

## Phase 3 — MCP server *(the agentic payoff)*

Thin wrapper over the daemon HTTP API. This is the part that is genuinely novel
for this project rather than plumbing, because the eval signal already exists.

### Tools

- `submit_training(variant, preset, overrides)` → job_id
- `get_status(job_id)` → state + latest progress event
- `list_jobs()` / `stop_job(job_id)` / `tail_logs(job_id)`
- `read_progress(job_id)` → parsed `progress.jsonl` (loss/CMMD curve)
- `read_bench_result(method, run)` → the `bench/<method>/results/.../result.json`
  envelope

### Why this fits Anima specifically

The agentic loop has real numbers to read, not just "training finished":
- CMMD is already the live val signal (`project_cmmd_val_signal` memory) and lands
  in `progress.jsonl` via Phase 0.
- `bench/<method>/results/<ts>/result.json` is a standardized envelope
  (`bench/_common.py`).

So *"train a tlora at rank 32, watch CMMD, stop and try rank 16 if it plateaus
before epoch 40"* becomes a conversation the agent can actually execute and
judge.

### Acceptance

- From a Claude session: submit a job, poll until `done`, read the CMMD curve,
  submit a follow-up with a changed override — all via MCP tools, no shell.

---

## Cost / benefit

- Daemon itself is small: a queue + subprocess spawn + ~5 HTTP endpoints.
- Real work hides in **Phase 0 instrumentation** (touches the training loop) and
  the **client refactors** (×3).
- **Phase 0 + Phase 1 alone** fix the ComfyUI wart and deliver the sweep queue —
  that is most of the value. Phases 2–3 are incremental.
- Recommended order: **0 → 1 → 3 → 2** (MCP before GUI polish, since agentic
  iteration is a stated motivation and the GUI already works).

## Open questions

- HTTP framework: **resolved** — stdlib `http.ThreadingHTTPServer`, **no new
  deps**. FastAPI was considered and dropped: its strengths (Pydantic
  validation, OpenAPI/`/docs`, async-at-scale) all target problems the non-goals
  rule out (no auth, localhost-only, single user, ~6 endpoints). The "reuse for
  MCP" rationale doesn't hold either — the MCP server is just an HTTP *client* to
  the daemon, so the reuse is at the client boundary regardless of framework.
- `daemon-attach` transport: **resolved** — SSE, hand-rolled on the stdlib
  server (see Phase 1 HTTP API note). Plain `text/event-stream`; reusable by a
  browser dashboard later. Not a Unix domain socket — Windows AF_UNIX support is
  flaky and localhost TCP is the cleaner cross-platform IPC for the ComfyUI/MCP
  clients.
- Daemon lifecycle: **resolved** — explicit `make daemon` (idempotent) *and*
  auto-spawn-on-first-submit (ComfyUI node / CLI). Teardown via
  `daemon-kill` (job) / `daemon-terminate` (daemon). See CLI surface above.
- Launch command builder: **resolved + done** — `build_launch_cmd(*args)`
  extracted from `accelerate_launch` (`scripts/tasks/_common.py`). Pure, no
  side effects; the daemon `Popen`s it directly while `accelerate_launch` keeps
  the blocking `run` + nsys-wrap CLI path. Still TODO: a shared *method/preset
  arg* builder (the `["--method", m, "--preset", p, ...]` assembly currently in
  `train()`), so the daemon doesn't duplicate ARTIST/PROFILE_STEPS handling.
