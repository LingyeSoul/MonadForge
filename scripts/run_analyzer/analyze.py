"""Per-run analysis aggregation — turns discovery sources into API payloads."""

from __future__ import annotations

import json
import os
import re
from statistics import fmean, pstdev
from typing import Optional

from scripts.run_analyzer.discovery import Run, MONADFORGE_ROOT
from scripts.run_analyzer.sources.jsonl import JsonlRun
from scripts.run_analyzer.sources.snapshot import keyline as snapshot_keyline

_LOSS_TAGS = ("loss/current", "loss/average")
_LR_TAG_PREFIXES = ("lr/",)
_NORM_TAGS = ("norm/", "max_norm/", "vr/")

_EVAL_TAGS = ("loss/current", "loss/average", "lr/unet")


def _resolve_path(run: Run, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(MONADFORGE_ROOT, p))


def sparkline(run: Run, n: int = 60) -> list:
    """loss/average 降采样为 n 点（用于索引行内迷你曲线）。

    均匀分桶取中位数，缺失区间以 None 占位（前端断线）。
    """
    series = run.jsonl.series if run.jsonl else {}
    pts = series.get("loss/average") or []
    if not pts and run.tb is not None:
        pts = run.tb.series.get("loss/average") or []
    if not pts:
        return []
    xs = [p[0] for p in pts]
    lo, hi = xs[0], xs[-1]
    if hi <= lo:
        return [[lo, pts[0][1]]]
    out: list = []
    for i in range(n):
        a = lo + (hi - lo) * i / n
        b = lo + (hi - lo) * (i + 1) / n
        bucket = [v for x, v in pts if a <= x < b]
        if bucket:
            bucket.sort()
            out.append([round((a + b) / 2, 1), bucket[len(bucket) // 2]])
        else:
            out.append(None)
    return out


def epoch_stats(run: Run) -> dict:
    """Per-epoch min/max/mean/std over loss/current (jsonl-primary)."""
    jl = run.jsonl
    series = jl.series if jl else {}
    current = series.get("loss/current") or []
    if not current:
        return {}
    buckets: dict[int, list[float]] = {}
    if jl is not None:
        for gs, v in current:
            ep = jl.gs_epoch.get(int(gs))
            if ep is not None:
                buckets.setdefault(ep, []).append(v)
    out: dict[int, dict] = {}
    for ep in sorted(buckets):
        vals = buckets[ep]
        out[ep] = {
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "mean": round(fmean(vals), 6),
            "std": round(pstdev(vals), 6),
            "n": len(vals),
        }
    return out


def compute_kpis(run: Run) -> dict:
    jl = run.jsonl
    series = jl.series if jl else {}
    steps = jl.last_step if jl is not None else None
    if steps is None and run.tb is not None and run.tb.steps:
        steps = run.tb.steps[-1]
    avg = series.get("loss/average") or []
    cur = series.get("loss/current") or []
    lr = series.get("lr/unet") or []
    epoch_avg = series.get("loss/epoch_average") or []
    if not avg and run.tb is not None:
        avg = run.tb.series.get("loss/average") or []
        cur = run.tb.series.get("loss/current") or []
        lr = run.tb.series.get("lr/unet") or []
        epoch_avg = run.tb.series.get("loss/epoch_average") or []
    final_avg = avg[-1][1] if avg else None
    min_loss = min((v for _, v in cur), default=None)
    max_loss = max((v for _, v in cur), default=None)
    lr_final = lr[-1][1] if lr else None
    lr_max = max((v for _, v in lr), default=None) if lr else None
    eavg_series = [(gs, v) for gs, v in epoch_avg]
    first_ts = jl.first_step_ts if jl is not None else None
    last_ts = jl.last_step_ts if jl is not None else None
    duration = None
    if first_ts is not None and last_ts is not None:
        duration = last_ts - first_ts
    elif run.started_at and run.ended_at:
        duration = run.ended_at - run.started_at
    return {
        "final_avr_loss": final_avg,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "lr_final": lr_final,
        "lr_max": lr_max,
        "steps": steps,
        "actual_epochs": max((jl.gs_epoch or {}).values(), default=None) if jl else None,
        "epoch_avg_last": eavg_series[-1][1] if eavg_series else None,
        "epoch_average": eavg_series,
        "duration_s": round(duration, 1) if duration is not None else None,
        "ckpt_count": len(jl.ckpts) if jl else 0,
        "sample_count": len(jl.samples) if jl else 0,
        "val_count": len(jl.vals) if jl else 0,
        "warn_count": len(jl.logs) if jl else 0,
        "error_count": jl.error_count if jl else 0,
    }


def series_payload(run: Run, source: str = "jsonl", limit: int = 0) -> dict:
    """All scalar series for charting. ``limit``>0 → stride downsampling."""
    if source == "jsonl" and run.jsonl is not None:
        series = dict(run.jsonl.series)
    else:
        series = dict(run.tb.series) if run.tb is not None else {}
    out = {}
    for tag, pts in series.items():
        if limit and len(pts) > limit:
            stride = max(1, len(pts) // limit)
            pts = pts[::stride]
        out[tag] = pts
    return out


def classify_tags(series: dict) -> dict:
    groups = {"loss": [], "lr": [], "norm": [], "other": []}
    for tag in series:
        t = tag.lower()
        if t.startswith("loss/"):
            groups["loss"].append(tag)
        elif t.startswith(_LR_TAG_PREFIXES):
            groups["lr"].append(tag)
        elif t.startswith(_NORM_TAGS):
            groups["norm"].append(tag)
        else:
            groups["other"].append(tag)
    return groups


_SAMPLE_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def _sample_size(path: Optional[str]) -> Optional[tuple[int, int]]:
    if not path:
        return None
    try:
        return _SAMPLE_SIZE_CACHE[path]
    except KeyError:
        pass
    try:
        from PIL import Image

        with Image.open(path) as im:
            size = im.size
    except Exception:
        return None
    if len(_SAMPLE_SIZE_CACHE) > 4096:
        _SAMPLE_SIZE_CACHE.clear()
    _SAMPLE_SIZE_CACHE[path] = size
    return size


def sample_list(run: Run) -> list[dict]:
    jl = run.jsonl
    items: list[dict] = []
    if jl is not None:
        for s in jl.samples:
            p = s.get("path")
            path = _resolve_path(run, p)
            size = _sample_size(path)
            items.append(
                {
                    "global_step": s.get("global_step"),
                    "epoch": s.get("epoch"),
                    "path": path,
                    "prompt": s.get("prompt"),
                    "ts": s.get("ts"),
                    "w": size[0] if size else None,
                    "h": size[1] if size else None,
                }
            )
    # PNG files in sample_dir not covered by events (defensive)
    sdir = run.sample_dir
    if sdir and os.path.isdir(sdir):
        have = {os.path.basename(i["path"]) for i in items if i["path"]}
        for fn in sorted(os.listdir(sdir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            if fn in have:
                continue
            m = re.match(r".*_e(\d{6})_(\d{2})_", fn)
            path = os.path.join(sdir, fn)
            size = _sample_size(path)
            items.append(
                {
                    "global_step": None,
                    "epoch": int(m.group(1)) if m else None,
                    "path": path,
                    "prompt": None,
                    "ts": None,
                    "w": size[0] if size else None,
                    "h": size[1] if size else None,
                }
            )
    return items


def overview_payload(run: Run) -> dict:
    jl = run.jsonl
    job = run.job or {}
    return {
        "id": run.id,
        "kind": run.kind,
        "run_name": run.run_name or run.id,
        "method": run.method or "",
        "preset": run.preset or "",
        "state": run.state,
        "job_state": job.get("state"),
        "stop_requested": run.stop_requested,
        "error": run.error or (jl.run_end_error if jl else None),
        "run_end": {
            "status": jl.run_end_status if jl else None,
            "final_step": jl.run_end_final_step if jl else None,
            "error": jl.run_end_error if jl else None,
            "extra": jl.run_end_extra if jl else {},
        },
        "total_steps": run.total_steps,
        "total_epochs": run.total_epochs,
        "submitted_at": run.submitted_at,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "argv": job.get("argv"),
        "extra_env": job.get("extra_env"),
        "ckpt_path": _resolve_path(run, job.get("ckpt_path")),
        "rc": job.get("rc"),
        "kpis": compute_kpis(run),
        "sources": run.sources,
        "paths": {
            "dir": run.dir,
            "log_dir": run.log_dir,
            "sample_dir": run.sample_dir,
            "progress": os.path.join(run.dir, "progress.jsonl") if run.kind == "daemon" else None,
            "stdout": os.path.join(run.dir, "stdout.log") if run.kind == "daemon" else None,
        },
    }


def epoch_spans(run: Run) -> dict:
    """Per-epoch [gs_start, gs_end] from loss/current points."""
    jl = run.jsonl
    if jl is None:
        return {}
    current = jl.series.get("loss/current") or []
    spans: dict[int, list] = {}
    for gs, _v in current:
        ep = jl.gs_epoch.get(int(gs))
        if ep is None:
            continue
        span = spans.get(ep)
        if span is None:
            spans[ep] = [int(gs), int(gs)]
        else:
            span[0] = min(span[0], int(gs))
            span[1] = max(span[1], int(gs))
    return spans


def full_payload(run: Run) -> dict:
    ov = overview_payload(run)
    series = series_payload(run, source="jsonl")
    if not series and run.tb is not None:
        series = series_payload(run, source="tb")
    jl = run.jsonl
    return {
        **ov,
        "series": series,
        "tags": classify_tags(series),
        "epochs": epoch_stats(run),
        "epoch_spans": epoch_spans(run),
        "ckpts": [dict(c, path=_resolve_path(run, c.get("path"))) for c in (jl.ckpts if jl else [])],
        "samples": sample_list(run),
        "vals": [dict(v) for v in (jl.vals if jl else [])],
        "logs": [dict(l) for l in (jl.logs if jl else [])],
        "params": {
            "sections": run.snapshot.sections if run.snapshot else [],
            "keyline": snapshot_keyline(run.snapshot) if run.snapshot else [],
            "error": run.snapshot.parse_error if run.snapshot else None,
        },
    }


def index_payload(runs: list[Run]) -> list[dict]:
    out = []
    for r in runs:
        kp = compute_kpis(r)
        out.append(
            {
                "id": r.id,
                "kind": r.kind,
                "run_name": r.run_name or r.id,
                "method": r.method or "",
                "preset": r.preset or "",
                "state": r.state,
                "stop_requested": r.stop_requested,
                "error": r.error,
                "total_steps": r.total_steps,
                "total_epochs": r.total_epochs,
                "steps": kp["steps"],
                "actual_epochs": kp["actual_epochs"],
                "final_avr_loss": kp["final_avr_loss"],
                "min_loss": kp["min_loss"],
                "duration_s": kp["duration_s"],
                "ckpt_count": kp["ckpt_count"],
                "sample_count": kp["sample_count"],
                "error_count": kp["error_count"],
                "submitted_at": r.submitted_at,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "sources": r.sources,
                "spark": sparkline(r, 60),
            }
        )
    return out


def compare_payload(runs: list[Run]) -> dict:
    rows = []
    for r in runs:
        series = r.jsonl.series if r.jsonl else {}
        avg = series.get("loss/average") or []
        if not avg and r.tb is not None:
            avg = r.tb.series.get("loss/average") or []
        ep_avg = series.get("loss/epoch_average") or []
        if not ep_avg and r.tb is not None:
            ep_avg = r.tb.series.get("loss/epoch_average") or []
        kp = compute_kpis(r)
        rows.append(
            {
                "id": r.id,
                "run_name": r.run_name or r.id,
                "method": r.method or "",
                "preset": r.preset or "",
                "state": r.state,
                "loss_average": avg,
                "epoch_average": ep_avg,
                "epoch_spans": epoch_spans(r),
                "final_avr_loss": kp["final_avr_loss"],
                "steps": kp["steps"],
                "actual_epochs": kp["actual_epochs"],
                "total_epochs": r.total_epochs,
            }
        )
    return {"runs": rows}
