# Anima training daemon — REST + programmatic interface

A single localhost process that owns a **serial job queue**: submit a training
run (or a plain command), one job runs at a time, state survives restarts, and
observers poll or stream rather than holding the run open. The GUI Train button,
the ComfyUI trainer node, and `make … --queue` all submit here; this doc
describes the same surface for **direct use** — a script, an MCP server, or an
agent driving training without going through `make`.

Design in one line: `http://127.0.0.1:8765`, JSON in / JSON out, no auth, no
remote (localhost only, by design — see `config.py`). All state is on disk under
`output/daemon/`, so anything that can read files can observe a run even with the
HTTP port down.

**Self-describing.** The daemon serves its own docs: `GET /` returns this file
(markdown), `GET /tools` returns a machine-readable manifest — one entry per
operation with a JSON-Schema `input_schema`, the HTTP `method`+`path`, and a
description. An agent (or a thin MCP bridge) can discover the whole surface with
`curl 127.0.0.1:8765/tools` and needs nothing else.

## Start / discover the daemon

The daemon auto-starts on first submit — you rarely start it by hand. But:

```bash
python tasks.py daemon            # start it, detached, wait for /health
python -m scripts.daemon          # equivalent (what the spawner runs)
python tasks.py daemon-status     # one JSON object: health + resolved base_url
                                  # + compact job summaries (--full for raw
                                  # records); passive, exit 1 when down
curl -s 127.0.0.1:8765/health     # {"ok":true,"pid":…,"active_job":…,"paused":…}
```

`daemon-status` is the one-shot answer to "is anything running and where do I
talk to it" — scripts and agents should start there instead of assuming a port.

The port is **not** guaranteed to be 8765: if a stranger holds that port the
daemon falls back to an OS-chosen one and records it in the pidfile. Always
resolve the real port from the pidfile rather than hardcoding — the Python
client does this for you (`DaemonClient()` with no arg), and `client._resolve_port()`
reads `output/daemon/daemon.json` → `port`.

## Two job kinds

| kind | what it runs | how it finalizes |
|------|--------------|------------------|
| `train` (default) | a `train.py` run built from `method` + `preset` + `overrides` + `extra` | `progress.jsonl` stream + exit code |
| `command` | a plain `python <argv>` task (preprocess, mask, a distill loop) | exit code only |

A `command` job can carry a **`chain_train`** spec — `{method, preset,
methods_subdir, overrides}`. When the command finishes successfully the daemon
auto-enqueues that training job. This is how "preprocess → train" survives the
submitter closing: the chain lives in the daemon, not the caller. The follow-on
job's id lands in the command job's `chained_job_id`.

## REST endpoints

Bodies are plain JSON dicts; there's no schema validation (trusted localhost
callers only). Field reference for the `Job` record is in `jobs.py`.

### `GET /` · `GET /tools` — self-description

`GET /` (alias `/readme`) serves this README as markdown. `GET /tools` returns
the operation manifest (`[{name, description, method, path, input_schema}, …]`)
— the same catalog an MCP bridge would register. Neither needs the rest of this
doc to be useful; they're the entry point for an agent discovering the daemon.

### `POST /jobs` — submit

Training job:
```json
{
  "method": "lora",
  "preset": "default",
  "methods_subdir": "gui-methods",
  "overrides": {"network_dim": 32, "max_train_epochs": 64},
  "extra": ["--some_flag"],
  "config_snapshot": null,
  "config_file": null,
  "start": true
}
```
Only `method` is required. `overrides` become `--key value` CLI args; `extra` is
appended verbatim. `config_snapshot` (a merged config dict) or `config_file` (a
path) pin the exact config instead of re-resolving the merge chain.

Command job:
```json
{
  "kind": "command",
  "label": "preprocess",
  "argv": ["tasks.py", "preprocess-config", "..."],
  "extra_env": {"FOO": "bar"},
  "chain_train": {"method": "lora", "preset": "default", "methods_subdir": "gui-methods", "overrides": {}},
  "start": true
}
```
`argv` is required (non-empty list). `chain_train` is optional.

`start` controls the queue gate: `true` → run now (resume queue), `false` → add
but hold the queue paused, omitted/`null` → leave the gate as-is.

Response: `201 {"job_id": "20260611-142233-a1b2c3", "state": "queued"}`.

### `GET /jobs` — list

Returns `[job, …]` (full records, submission order). Each job has `state` ∈
`queued | running | done | error | stopped`.

### `GET /jobs/{id}` — status

The job record plus two live fields:
- `latest` — last event from `progress.jsonl` (training progress; `null` for command jobs)
- `stale_for` — seconds since the last progress tick (heartbeat staleness)

### `GET /jobs/{id}/progress` — query the structured progress stream

Filtered view of the job's `progress.jsonl` (train jobs only) — the surface to
**debug or analyze a run from**: loss/lr/metric curves (`step`), validation CMMD
(`val`), checkpoint saves (`ckpt`), mirrored WARNING+ log records (`log`), and
the run outcome (`run_start`/`run_end`). Query params, all optional:

| param | meaning |
|-------|---------|
| `events` | comma-separated `ev` kinds to keep, e.g. `events=step,val` or `events=log,run_end` |
| `since_step` | keep events at/after this `global_step` (step-less events inherit the preceding step) |
| `every_nth` | thin `step` events to every n-th (the latest step is always kept) |
| `last_n` | trailing cap on returned events (default 200) |

Returns `{job_id, state, progress_path, count, events}`. Example — the loss
curve at 1-in-50 resolution plus any warnings:

```
GET /jobs/{id}/progress?events=step,log&every_nth=50
```

### `POST /jobs/{id}/stop` — abort

Stops a running or queued job (tree-kills the process). Returns `{job_id, state}`.
The Python client's `stop()` with no id resolves the active job from `/health`.

### `POST /queue/pause` · `POST /queue/start`

Hold / resume the queue gate. A paused queue keeps accepting submissions but
launches nothing until started.

### `GET /jobs/{id}/logs` — SSE log tail

Server-Sent Events; each `data:` line is a line of the job's combined
stdout+stderr, from the start of the file. Emits a final `{"ev":"eof","state":…}`
once the job is terminal and the log is drained.

### `GET /events` — SSE daemon lifecycle

Daemon-level events (job start/finish, etc.), plus `: keepalive` comments while idle.

### `GET /health`

`{"ok", "pid", "port", "root", "active_job", "paused", "worker_alive",
"worker_idle_for"}`. `root` is the checkout the daemon belongs to — useful to
confirm you're talking to *this* repo's daemon and not another checkout's (see
`daemon_matches_root` in `client.py`). `worker_idle_for` is seconds since the job
worker thread last advanced; a large value while a job sits `queued` means the
worker is wedged behind a long-running job (normal) or — with `worker_alive`
false — has died (a bug worth a report).

### `POST /shutdown`

`{"kill_jobs": true}` → stop the daemon, optionally killing the running job.

## Python client (`scripts.daemon.client`)

Pure stdlib (`urllib`) — imports without dragging in `library.*`/torch, so it's
safe to call from anywhere.

```python
from scripts.daemon.client import DaemonClient, ensure_daemon

client = ensure_daemon()          # start-if-needed, returns a live client
# or: client = DaemonClient()     # attach only; assumes one is up

# submit a training run
r = client.submit(
    method="lora",
    preset="default",
    methods_subdir="gui-methods",
    overrides={"network_dim": 32, "max_train_epochs": 64},
    start=True,
)
job_id = r["job_id"]

# poll to completion
import time
while True:
    job = client.get(job_id)
    if job["state"] in ("done", "error", "stopped"):
        break
    time.sleep(2.0)
print(job["state"], job.get("error"), job.get("ckpt_path"))

# stream logs instead of polling
for line in client.stream_logs(job_id):
    print(line)

# control
client.pause_queue(); client.start_queue()
client.stop(job_id)               # or client.stop() for the active job
client.list_jobs()
```

`submit_command(label=…, argv=[…], chain_train=…)` submits a command job. All
methods map 1:1 onto the endpoints above.

`ensure_daemon(expected_root=…)` refuses to attach to a daemon belonging to a
different checkout if that daemon still has live jobs — pass your repo root when
correctness across checkouts matters.

## Observing without HTTP

Everything is mirrored to disk, so a reader can skip the port entirely:

```
output/daemon/
  daemon.json            pidfile: {pid, create_time, port, root}
  daemon.log             the detached daemon's own stdout/stderr
  jobs/<id>/
    job.json             the full Job record (atomic-replaced on each change)
    stdout.log           the subprocess's captured stdout+stderr
    progress.jsonl       structured training progress (train jobs only)
```

`job.json` → `state` is the fast, dependency-free way to check a job; the GUI
reads these files directly (`gui/daemon.py`) rather than polling HTTP in the Qt
thread.

## Environment

| var | default | effect |
|-----|---------|--------|
| `ANIMA_DAEMON_PORT` | `8765` | preferred bind port |
| `ANIMA_DAEMON_PIDFILE` | `~/.anima/daemon.json` | per-user pidfile mirror (cross-checkout discovery) |
| `ANIMA_LORA_ROOT` | — | explicit repo root for pidfile discovery |
| `ANIMA_DAEMON_GPU_BUSY_FRAC` | `0.85` | pre-launch GPU guard: card treated as busy above this used/total fraction |
| `ANIMA_DAEMON_GPU_RETRIES` / `_DELAY` | `1` / `2.0` | guard wait before launching anyway |

## Gotchas

- **Localhost only.** No remote, no auth — the caller must run on the same machine.
- **Serial queue.** One job runs at a time; submitting while one runs enqueues.
- **No blocking wait.** Completion is poll-based (`GET /jobs/{id}`) or stream-based
  (`/jobs/{id}/logs` ends with an `eof` event); there is no "submit and block" call.
- **Port drift.** Resolve from the pidfile, not a constant — the daemon may bind
  an ephemeral port. `DaemonClient()` and `ensure_daemon()` handle this.
- **`config_snapshot` vs re-resolve.** Without a snapshot/file the daemon
  re-runs the `base → preset → method → overrides` merge at launch; pin a
  snapshot when you need bit-stable config across a queued delay.
- **Command-job progress.** `latest`/`progress.jsonl` are training-only; a
  command job exposes only `state` + `stdout.log` until it exits.

## MCP bridge (`mcp.py`)

A stdio MCP server over this same surface — pure stdlib, newline-delimited
JSON-RPC, no new deps. Register it with any MCP client (Claude Code, Claude
Desktop, OpenClaw, …) as a **command, never an address**: the bridge resolves
the daemon itself via the pidfile, so it survives port drift and daemon
restarts without reconfiguration.

```bash
# Claude Code (use your checkout's absolute paths; any cwd works)
claude mcp add anima-daemon -- <repo>/.venv/Scripts/python.exe <repo>/scripts/daemon/mcp.py
```

For other clients, the equivalent JSON config:

```json
{"mcpServers": {"anima-daemon": {
  "command": "<repo>/.venv/Scripts/python.exe",
  "args": ["<repo>/scripts/daemon/mcp.py"]
}}}
```

The tool catalog **is** the `GET /tools` manifest (`server.TOOLS`, registered
verbatim — one source of truth), with two deviations:

- `tail_logs` (SSE) is replaced by **`tail_log`** `{id, lines=80}` — last N
  lines + current state in one call; it reads the on-disk `job.json` +
  `stdout.log` as fallback, so it answers even with the daemon down. tqdm
  `\r`-redraws are collapsed to their final rendering, so a progress bar
  counts as one line.
- **`get_progress`** is served from the on-disk `progress.jsonl` (same filters
  as the HTTP endpoint above), so it too answers even with the daemon down.
- Only `submit_training` / `submit_command` auto-start the daemon; every other
  tool is passive, so an agent asking "is anything running?" never boots a
  daemon as a side effect (`health` returns `{"up": false}` instead of erroring).
