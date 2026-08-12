"""run-analyzer — standalone FastAPI server + static SPA.

Usage::

    python -m scripts.run_analyzer.server [--port 8320] [--host 127.0.0.1] [--open]

Endpoints:
    GET  /api/runs                    run index
    GET  /api/runs/{id}               full analysis payload
    GET  /api/runs/{id}/overview      lightweight overview
    GET  /api/runs/{id}/live?since=ts incremental events since ts (seconds)
    GET  /api/runs/{id}/stdout?tail=&q= stdout.log tail / filtered
    POST /api/compare                 {ids: [...]} → normalized overlay payload
    GET  /api/runs/{id}/samples/{fn}  preview image file
    GET  /                            static SPA
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scripts.run_analyzer import analyze
from scripts.run_analyzer.discovery import (
    JOBS_DIR,
    LOGS_DIR,
    Run,
    apply_jsonl_metadata,
    discover,
    merge_daemon_attempts,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="run-analyzer")

_index_lock = threading.Lock()
_index: list[Run] = []
_index_cache_key: Optional[str] = None


_INDEX_WATCH_FILES = ("progress.jsonl", "stdout.log", "job.json", "snapshot.toml")


def _dir_mtime_key() -> str:
    """所有 job 目录 + 日志目录的 mtime 指纹；未变化时跳过全量重解析。

    仅取顶层目录 mtime 不够：progress.jsonl/tfevents 是内容追加写入，
    文件 mtime 变化但目录 mtime 不变，会导致索引冻结在旧解析。
    因此对每个 job/日志目录额外取关键文件自身的 mtime。
    """
    parts = []
    for base in (JOBS_DIR, LOGS_DIR):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            try:
                stat = os.stat(p)
                parts.append(f"{base}/{name}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                continue
            if not os.path.isdir(p):
                continue
            for sub in _INDEX_WATCH_FILES:
                fp = os.path.join(p, sub)
                try:
                    stat = os.stat(fp)
                    parts.append(
                        f"{base}/{name}/{sub}:{stat.st_mtime_ns}:{stat.st_size}"
                    )
                except OSError:
                    continue
            tb = os.path.join(p, "network_train")
            if os.path.isdir(tb):
                for root, _dirs, files in os.walk(tb):
                    for filename in sorted(files):
                        if not filename.startswith("events.out.tfevents."):
                            continue
                        event_path = os.path.join(root, filename)
                        try:
                            stat = os.stat(event_path)
                            parts.append(
                                f"{event_path}:{stat.st_mtime_ns}:{stat.st_size}"
                            )
                        except OSError:
                            continue
    return "|".join(parts)


def _refresh_index(force: bool = False) -> None:
    global _index, _index_cache_key
    key = _dir_mtime_key()
    if not force and key == _index_cache_key and _index:
        return
    with _index_lock:
        _index = discover()
        _index_cache_key = key


def _find(run_id: str) -> Run:
    # A resumed attempt creates a new job directory while the browser may stay
    # on the existing root run. Refresh first so that poll immediately adopts
    # the new physical attempt into the same logical run.
    _refresh_index()
    with _index_lock:
        runs = _index
    for r in runs:
        if r.id == run_id:
            return r
    raise HTTPException(status_code=404, detail=f"run {run_id} not found")


def _freshen_physical_attempt(run: Run) -> None:
    """Reload one physical attempt from its durable files."""
    from scripts.run_analyzer.sources.jsonl import parse as parse_jsonl
    from scripts.run_analyzer.sources.stdout_log import parse as parse_stdout

    jl = parse_jsonl(run.jsonl_path or os.path.join(run.dir, "progress.jsonl"))
    if jl is not None:
        apply_jsonl_metadata(run, jl)
    job_path = os.path.join(run.dir, "job.json")
    try:
        with open(job_path, encoding="utf-8") as handle:
            job = json.load(handle)
    except (OSError, json.JSONDecodeError):
        job = None
    if job is not None:
        run.job = job
        run.state = job.get("state") or run.state
        run.submitted_at = job.get("submitted_at")
        run.started_at = job.get("started_at")
        run.ended_at = job.get("ended_at")
        run.ckpt_path = job.get("ckpt_path")
        run.sample_dir = job.get("sample_dir") or run.sample_dir
        run.error = job.get("error")
        run.stop_requested = bool(job.get("stop_requested"))
        if jl is not None:
            apply_jsonl_metadata(run, jl)
    stdout = parse_stdout(os.path.join(run.dir, "stdout.log"))
    if stdout is not None:
        run.stdout = stdout


def _freshen_run(run: Run) -> None:
    """按需重解析逻辑 run 的所有 attempt，不触发全量 discover。

    监控 2s 轮询调 /api/runs/{id}：index 里的 Run.jsonl 是上次 discover 的
    快照，progress.jsonl 是内容追加写入（文件 mtime 变、目录 mtime 不变），
    必须按请求重读才能拿到最新 step。index 列表仍由 _dir_mtime_key 控制。
    """
    try:
        if run.kind == "daemon":
            if run.attempts:
                for attempt in run.attempts:
                    _freshen_physical_attempt(attempt)
                refreshed = merge_daemon_attempts(run.attempts)
                run.__dict__.update(refreshed.__dict__)
            else:
                _freshen_physical_attempt(run)
        run.sources = {
            "jsonl": run.jsonl is not None,
            "tensorboard": run.tb is not None,
            "snapshot": run.snapshot is not None,
            "stdout": run.stdout is not None,
        }
    except Exception:
        pass


@app.on_event("startup")
def _startup() -> None:
    _refresh_index()


@app.get("/api/runs")
def api_runs():
    _refresh_index()
    with _index_lock:
        runs = _index
    return {"runs": analyze.index_payload(runs)}


@app.get("/api/runs/{run_id}/overview")
def api_run_overview(run_id: str):
    r = _find(run_id)
    _freshen_run(r)
    return analyze.overview_payload(r)


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    r = _find(run_id)
    _freshen_run(r)
    return analyze.full_payload(r)


@app.get("/api/runs/{run_id}/live")
def api_live(run_id: str, since: float = 0.0):
    """Incremental events since ``since`` (seconds of run-relative ts)."""
    r = _find(run_id)
    _freshen_run(r)
    jl = r.jsonl
    if jl is None:
        return {"events": []}
    events = []
    for kind, evs in (("ckpt", jl.ckpts), ("sample", jl.samples), ("val", jl.vals), ("log", jl.logs)):
        for ev in evs:
            ts = ev.get("ts")
            if ts is not None and ts > since:
                events.append({"kind": kind, **ev})
    return {"events": events}


@app.get("/api/runs/{run_id}/stdout")
def api_stdout(run_id: str, tail: int = Query(400, ge=1, le=20000), q: str = ""):
    r = _find(run_id)
    _freshen_run(r)
    so = r.stdout
    if so is None:
        return {"lines": [], "warnings": [], "errors": []}
    lines = so.lines
    if q:
        lines = [line for line in lines if q.lower() in line.lower()]
    return {
        "lines": lines[-tail:],
        "warnings": so.warnings[-200:],
        "errors": so.errors[-200:],
    }


@app.post("/api/compare")
def api_compare(ids: list[str] = Body(..., embed=True)):
    with _index_lock:
        runs = _index
    by_id = {r.id: r for r in runs}
    chosen = []
    for rid in ids[:8]:
        r = by_id.get(rid)
        if r is not None:
            chosen.append(r)
    return analyze.compare_payload(chosen)


@app.get("/api/runs/{run_id}/samples/{fn}")
def api_sample_file(run_id: str, fn: str, attempt_id: Optional[str] = None):
    r = _find(run_id)
    if os.path.basename(fn) != fn or fn in {".", ".."}:
        raise HTTPException(status_code=404, detail="not found")
    attempts = r.attempts or [r]
    if attempt_id:
        attempts = [attempt for attempt in attempts if attempt.id == attempt_id]
        if not attempts:
            raise HTTPException(status_code=404, detail="attempt not found")
    for attempt in reversed(attempts):
        sdir = attempt.sample_dir
        if not sdir or not os.path.isdir(sdir):
            continue
        root = os.path.realpath(sdir)
        path = os.path.realpath(os.path.join(root, fn))
        try:
            if os.path.commonpath([root, path]) != root:
                continue
        except ValueError:
            continue
        if os.path.isfile(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="not found")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="run-analyzer training analysis dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8320)
    parser.add_argument("--open", action="store_true", help="open browser on start")
    args = parser.parse_args()
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}/")).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
