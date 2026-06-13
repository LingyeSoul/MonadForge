# SQLite run/job ledger — a derived sidecar over the flat-file truth

Status: **proposal, nothing built**. This consolidates two adjacent
itches — (1) the daemon's per-job state is a fan of `job.json` files that
multiple readers race over, and (2) the `.snapshot.toml` config provenance is
scattered across three locations, one of which (`output/logs/<run_TS>/`)
accumulates a near-duplicate TOML per run. Both fold into one small SQLite
ledger that is **derived, never authoritative** — the flat files stay the
source of truth; the DB is a rebuildable index plus a home for run history.

## The one load-bearing principle

The DB is a **cache you can `rm` and rebuild**. Every fact it holds also lives
in a flat file that survives a corrupt or deleted DB:

- `job.json` (`scripts/daemon/jobs.py:100`) stays the truth for job state —
  atomic via `.tmp` swap, human-readable, `cat`-debuggable.
- `output/ckpt/<name>.snapshot.toml` (`library/config/io.py:695`) stays the
  truth for "the config that produced the checkpoint on disk right now."

If the DB is missing on daemon boot, `load_all()` (`jobs.py:120`) already walks
every job dir — that same pass repopulates the DB. No migration, no dual-write
consistency contract, fully reversible. **The moment any field lives *only* in
SQLite, we've signed up for a real migration and a consistency bug surface — so
we don't.**

## What's actually messy today (and what isn't)

There are three `.snapshot.toml` copies per run:

| # | Path | Keying | Lifetime |
|---|------|--------|----------|
| 1 | `output/ckpt/<name>.snapshot.toml` | `output_name` only | clobbered each rerun |
| 2 | `output/logs/<run_TS>/<name>.snapshot.toml` | timestamped dir | accumulates forever |
| 3 | `output/daemon/jobs/<id>/config.snapshot.toml` | job id | per-job |

**#1's clobber is correct — leave it.** It sits next to
`output/ckpt/<name>.safetensors`, which *also* clobbers on rerun with the same
`output_name`. The sibling pair stays in sync: the `.snapshot.toml` describes
the `.safetensors` beside it. That is exactly the artifact you want for "what
produced this checkpoint" — keep it a plain file.

**#2 is the actual mess.** It's run *history* expressed as a sprawl of
timestamped directories, each carrying a near-duplicate of #1. Cross-run
questions ("every config that ran `easycontrol`", "diff this run vs the last
run of the same `output_name`") mean globbing `output/logs/<run_TS>/` and
parsing TOML by hand. This is what the ledger retires.

**#3 can't be virtualized away** — it's passed back as `--config_file` when the
daemon spawns `train.py` (`scripts/daemon/manager.py:603`), so the path must
exist on disk at spawn time. Keep writing it; it self-cleans with the job dir.

## Schema

One DB, two tables. Proposed home: `output/daemon/ledger.db` (WAL mode — the
whole point is concurrent readers/writers without `.tmp`-swap races).

### `jobs` — mirror of the `Job` dataclass (`jobs.py:35`)

```sql
CREATE TABLE jobs (
    id            TEXT PRIMARY KEY,   -- YYYYmmdd-HHMMSS-<6hex>
    method        TEXT,
    preset        TEXT,
    kind          TEXT,               -- "train" | "command"
    state         TEXT,               -- queued|running|done|error|stopped
    submitted_at  REAL,
    started_at    REAL,
    ended_at      REAL,
    pid           INTEGER,
    ckpt_path     TEXT,
    error         TEXT,
    chained_job_id TEXT,
    job_dir       TEXT,               -- back-pointer to the authoritative job.json
    job_json      TEXT                -- full serialized record (faithful blob)
);
CREATE INDEX jobs_state ON jobs(state);
CREATE INDEX jobs_method ON jobs(method);
CREATE INDEX jobs_submitted ON jobs(submitted_at);
```

`job_json` is the whole record verbatim, so the table is a faithful mirror even
as the dataclass grows; the exploded columns exist only to be indexed/queried.
`daemon-status`, `GET /tools`, the MCP bridge, and the GUI read **here** instead
of scanning + parsing N dirs.

### `runs` — one row per training run, carries the config

```sql
CREATE TABLE runs (
    run_id       TEXT PRIMARY KEY,    -- the run_TS log-dir stem (already unique)
    output_name  TEXT,
    method       TEXT,
    preset       TEXT,
    git_sha      TEXT,
    started_at   REAL,
    ended_at     REAL,
    status       TEXT,                -- ok | error (from run_end event)
    final_step   INTEGER,
    final_loss   REAL,
    cmmd         REAL,
    ckpt_path    TEXT,
    job_id       TEXT,                -- FK to jobs.id when daemon-spawned, else NULL
    config_toml  TEXT                 -- exact _render_merged_toml() output
);
CREATE INDEX runs_output ON runs(output_name);
CREATE INDEX runs_method ON runs(method);
```

`config_toml` is the **same string** `library/config/io.py` already renders for
#1/#2 — no new serialization path. The exploded columns are the queryable index;
the blob is the faithful artifact, re-renderable to a file any time. This is
what replaces the `output/logs/<run_TS>/` snapshot mirror: history lives in
rows, not in a fan of directories.

The clean split after this lands:

- **`output/ckpt/<name>.snapshot.toml`** = "config for the checkpoint on disk
  *now*" (one file, clobbers with its checkpoint — kept).
- **`runs` rows** = "every run ever" (replaces the timestamped log mirror).

## Write hooks

Minimal, all additive:

1. **`Job.persist()`** (`jobs.py:100`) — after the `.tmp` swap, `UPSERT` the
   row. job.json remains the write that *must* succeed; the DB write is
   best-effort (wrap in try/except, log-and-continue — a DB hiccup must never
   fail a job state transition).
2. **Daemon boot** — in/after `load_all()` (`jobs.py:120`), `CREATE TABLE IF NOT
   EXISTS` then bulk-upsert every loaded job. This *is* the rebuild path; a
   deleted `ledger.db` heals here.
3. **`runs` row at run start** — where the snapshot is rendered
   (`io.py:695`–`718`), insert the `runs` row with `config_toml` + run metadata.
   Best-effort, same as above.
4. **`runs` row finalize at `run_end`** — the `ProgressSink` already emits
   `run_end` with `status`/`final_step`/`error`
   (`library/training/progress.py`); the same callsite updates `ended_at`,
   `status`, `final_loss`, `cmmd`. One write per run, not per step.

Turbo's bespoke loop renders its own snapshot (`scripts/distill_turbo/
distill.py:631`/`:653`) and won't get the `runs` row for free — it must mirror
the hook explicitly, the standing pattern for the bespoke loops
([[project_daemon_wiring_pattern]]). Out of scope for v1; note it so it's a
known gap, not a silent one.

## Explicit non-goals

- **`progress.jsonl` stays exactly as is.** It's line-buffered and *tailed*
  (`scripts/daemon/tail.py` `read_events`); per-step `INSERT`s would put a write
  dependency on the training hot path and replace clean append-tail with table
  polling. The ledger reads the *summary* at `run_end`, never the per-step
  stream.
- **TensorBoard event files untouched** — separate UI ecosystem, no overlap.
- **#1 ckpt snapshot sibling untouched** — correct as a clobbering file.
- **No field becomes SQLite-only.** Re-stating the principle because it's the
  whole safety story.

## Migration / rollout

1. Land tables + the four write hooks, all best-effort. Nothing reads the DB
   yet → zero behavior change, pure shadow-write to validate the writes.
2. Point `daemon-status` / `GET /tools` / MCP at the `jobs` table; keep the
   dir-scan as a `--no-db` fallback for one release.
3. Once `runs` rows are trusted, **stop writing #2** (the `output/logs/
   <run_TS>/` snapshot mirror). Optional `scripts/` one-shot to backfill `runs`
   from existing `<run_TS>/<name>.snapshot.toml` + `progress.jsonl` so history
   predating the ledger isn't lost. The timestamped *dirs* stay (TensorBoard
   lives there) — only the duplicate TOML stops being written.

## Why now / why not

**For:** the daemon is the agent/MCP/GUI surface, and that's where concurrent
reads multiply — WAL + an indexed table is the right tool, and it retires the
`output/logs/` snapshot sprawl as a side effect of the same `runs` table.

**Against (honest):** at current scale — ~23 jobs, ~44 runs — the dir scan is
not a measured bottleneck. The justification is **concurrency correctness +
query surface + history tidiness**, not speed. If the MCP/agent surface isn't
where effort is going, this is a defer-able nice-to-have. The snapshot-sprawl
annoyance alone probably doesn't justify it; the daemon job index is what tips
it over, and the snapshot history rides along for nearly free once the `runs`
table exists.

## Open questions

- One DB or two (`output/daemon/ledger.db` for both vs. a separate
  `output/logs/runs.db`)? Leaning one — the `runs.job_id` FK wants them
  colocated, and inference/non-daemon runs still write `runs` (just with
  `job_id = NULL`).
- Does anything want to *read* `runs.config_toml` programmatically (e.g. "re-run
  with the exact config of run X")? If yes, a `render-config --run <id>` helper
  falls out trivially; if no, the blob is provenance-only and we don't build it.
- Retention: `jobs` rows for deleted job dirs — prune on boot to match
  `load_all()`, or keep as tombstones for history? Probably prune (DB mirrors
  live dirs); `runs` is the permanent history, `jobs` is operational.
