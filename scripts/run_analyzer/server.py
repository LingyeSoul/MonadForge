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
import time
import webbrowser
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scripts.run_analyzer import analyze
from scripts.run_analyzer.discovery import Run, discover

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="run-analyzer")

_index_lock = threading.Lock()
_index: list[Run] = []
_index_cache_key: Optional[str] = None


def _dir_mtime_key() -> str:
    """所有 job 目录 + 日志目录的 mtime 指纹；未变化时跳过全量重解析。"""
    jobs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "output", "daemon", "jobs")
    logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "output", "logs")
    parts = []
    for base in (jobs, logs):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            try:
                parts.append(f"{name}:{os.path.getmtime(p)}")
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
    with _index_lock:
        runs = _index
    for r in runs:
        if r.id == run_id:
            return r
    raise HTTPException(status_code=404, detail=f"run {run_id} not found")


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
    return analyze.overview_payload(r)


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    r = _find(run_id)
    return analyze.full_payload(r)


@app.get("/api/runs/{run_id}/live")
def api_live(run_id: str, since: float = 0.0):
    """Incremental events since ``since`` (seconds of run-relative ts)."""
    r = _find(run_id)
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
    so = r.stdout
    if so is None:
        return {"lines": [], "warnings": [], "errors": []}
    lines = so.lines
    if q:
        lines = [l for l in lines if q.lower() in l.lower()]
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
def api_sample_file(run_id: str, fn: str):
    r = _find(run_id)
    sdir = r.sample_dir
    if not sdir or not os.path.isdir(sdir):
        raise HTTPException(status_code=404, detail="no sample dir")
    p = os.path.normpath(os.path.join(sdir, fn))
    if not p.startswith(os.path.normpath(sdir)) or not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p)


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
