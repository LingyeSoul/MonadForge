#!/usr/bin/env python3
"""Sweep ``blocks_to_swap`` × ``activation_memory_budget`` → peak VRAM surface.

Motivation
----------
A single static ``activation_memory_budget`` / ``blocks_to_swap`` in a preset is
wrong across methods (easycontrol and lora have very different footprints). Before
wiring a ``use_autotune`` flag we need the actual tradeoff surface: for each
(swap, budget) rung, what is the steady-state **peak VRAM**, the **step time**,
and does the swap+budget combo stay **numerically correct** (the block-swap
offloader vs the compiled-backward partitioner recompute is unaudited).

Approach
--------
One real ``train.py`` subprocess **per grid cell** — the DCW idiom
(``bench/dcw/sweep_buckets.py``). A fresh process per cell keeps torch.compile /
dynamo state clean across the per-budget recompiles, and runs the genuine load →
apply → block-swap → compile → train path, so the measured peak is exactly what a
real run would use (including the inductor compile-context overhead).

Peak VRAM is sampled **GPU-side via pynvml**, NOT ``torch.cuda.max_memory_allocated``:
the multi-graph inductor compile context adds ~2 GB that is invisible to the
caching allocator (the "mid-run climb that isn't a leak"). Allocator-only
measurement undercounts and would make an autotuner pick rungs that OOM.
Token-family warmup pins the peak to the first few steps, so a short run captures
steady state.

Each cell records: peak_used_mib, peak_delta_mib (vs pre-launch baseline), median
steady-state s/it, final avr_loss (for parity vs the budget=1.0 column), and an
OOM/timeout status. Output is the standard bench envelope plus surface.md / surface.csv.

Usage
-----
    python bench/autotune/sweep.py                       # full 3×4 lora grid, backgroundable
    python bench/autotune/sweep.py --swaps 0,20 --budgets 1.0,0.7
    python bench/autotune/sweep.py --dry-run             # print the grid + argv, run nothing
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # anima_lora/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

# --------------------------------------------------------------------------- #
# GPU peak sampler — pynvml primary (whole-GPU used, sees compile context),
# nvidia-smi fallback. We poll the *aggregate* GPU memory and report both the
# absolute peak and the delta over a baseline sampled right before launch, so a
# few-hundred-MiB idle desktop doesn't inflate the run's apparent footprint.
# --------------------------------------------------------------------------- #


def _make_nvml_reader():
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
    except Exception:
        return None

    def read() -> tuple[int, int] | None:
        try:
            used = total = 0
            for h in handles:
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                used += int(info.used)
                total += int(info.total)
            return used // (1024 * 1024), total // (1024 * 1024)
        except Exception:
            return None

    def close() -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return read, close


def _make_smi_reader():
    import shutil

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None

    def read() -> tuple[int, int] | None:
        try:
            out = subprocess.run(
                [
                    smi,
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:
            return None
        used = total = 0
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                used += int(parts[0])
                total += int(parts[1])
        return (used, total) if total else None

    def close() -> None:
        pass

    return read, close


def _make_reader():
    r = _make_nvml_reader()
    if r is not None:
        return (*r, "pynvml")
    r = _make_smi_reader()
    if r is not None:
        return (*r, "nvidia-smi")
    return None


class GPUPeakSampler:
    """Background thread tracking the max aggregate GPU-used (MiB)."""

    def __init__(self, read, poll: float):
        self._read = read
        self._poll = poll
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_used = 0
        self.total = 0
        self.samples = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            r = self._read()
            if r is not None:
                used, total = r
                self.peak_used = max(self.peak_used, used)
                self.total = total
                self.samples += 1
            self._stop.wait(self._poll)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# tqdm log parsing — train.py drives a tqdm bar with set_postfix(avr_loss=...).
# Lines are \r-delimited; pull the s/it (or it/s) cadence and final avr_loss.
# --------------------------------------------------------------------------- #

_LOSS = re.compile(r"avr_loss=([0-9.eE+\-]+)")
# Anchor to the *training* tqdm bar (desc="steps:") — the preamble has its own
# bars (dataset scan, "read caption:", bucketing at N/3058) that would otherwise
# trip the step counter and we'd kill the run before training even starts.
_STEP = re.compile(r"steps:\s*\d+%\|[^|]*\|\s*(\d+)/\d+")
# Same bar, also capturing tqdm's cumulative elapsed "[MM:SS<". We derive
# steady-state s/it from elapsed deltas across a post-warmup window rather than
# tqdm's displayed rate: the displayed rate is a *cumulative* average that the
# compile-heavy first step drags down for the entire (short) run.
_STEPLINE = re.compile(r"steps:\s*\d+%\|[^|]*\|\s*(\d+)/\d+\s*\[([0-9:]+)<")
_OOM = re.compile(r"out of memory|OutOfMemoryError|CUDA error|CUBLAS", re.IGNORECASE)


def _elapsed_to_s(t: str) -> int:
    """tqdm elapsed 'MM:SS' / 'H:MM:SS' → seconds."""
    sec = 0
    for p in t.split(":"):
        sec = sec * 60 + int(p)
    return sec


def _last_step(log_path: Path) -> int:
    """Highest *training* step index observed so far (0 if none yet)."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return 0
    steps = _STEP.findall(text)
    return max((int(s) for s in steps), default=0)


def _parse_log(text: str, skip_steps: int) -> dict:
    """Steady-state s/it from post-warmup elapsed window + final avr_loss.

    Only the training bar (lines with ``steps:``) is parsed. s/it is
    ``(elapsed[last] - elapsed[anchor]) / (last - anchor)`` where ``anchor`` is
    the first step ≥ ``skip_steps`` — excluding the compile-heavy first step(s)
    and averaging over the window to beat tqdm's integer-second resolution.
    """
    step_elapsed: dict[int, int] = {}
    losses: list[float] = []
    for chunk in re.split(r"[\r\n]", text):
        if "steps:" not in chunk:
            continue
        m = _STEPLINE.search(chunk)
        if m:
            step_elapsed[int(m.group(1))] = _elapsed_to_s(m.group(2))
        ml = _LOSS.search(chunk)
        if ml:
            losses.append(float(ml.group(1)))
    s_per_it = None
    if len(step_elapsed) >= 2:
        steps_sorted = sorted(step_elapsed)
        anchor = next((s for s in steps_sorted if s >= skip_steps), steps_sorted[0])
        last = steps_sorted[-1]
        if last > anchor:
            s_per_it = (step_elapsed[last] - step_elapsed[anchor]) / (last - anchor)
    avr_loss = losses[-1] if losses else None
    return {"s_per_it": s_per_it, "avr_loss": avr_loss, "n_steps": len(step_elapsed)}


# --------------------------------------------------------------------------- #
# Cell runner
# --------------------------------------------------------------------------- #


def _build_argv(args, swap: int, budget: float) -> list[str]:
    argv = [
        sys.executable,
        "train.py",
        "--method",
        args.method,
        "--preset",
        args.preset,
        "--activation_memory_budget",
        str(budget),
        "--sample_ratio",
        str(args.sample_ratio),
        "--seed",
        str(args.seed),
        "--output_name",
        "autotune_probe",
    ]
    if swap > 0:
        argv += ["--blocks_to_swap", str(swap)]
    return argv


def _drain_wait(reader, threshold_mib: int, max_wait: float) -> int:
    """Block until GPU-used falls to ~idle (prior SIGKILL'd cell drained).

    nvidia-smi lags process exit by several seconds; without this the next
    cell's peak (and any delta) is contaminated by the previous cell's ~15 GB
    that hasn't been reclaimed yet. Returns the used-MiB we launched at.
    """
    t0 = time.time()
    last = 0
    while time.time() - t0 < max_wait:
        r = reader[0]()
        if r is None:
            break
        last = r[0]
        if last <= threshold_mib:
            return last
        time.sleep(1.0)
    return last


def run_cell(args, swap: int, budget: float, reader, run_dir: Path, baseline_used: int) -> dict:
    argv = _build_argv(args, swap, budget)
    log_path = run_dir / f"cell_s{swap}_b{budget}.log"

    # Wait for the previous cell's VRAM to actually drain before launching, so
    # this cell's peak isn't inflated by undrained residual.
    launch_at = _drain_wait(reader, baseline_used + args.drain_margin, args.drain_max)

    sampler = GPUPeakSampler(reader[0], args.poll)
    env = dict(os.environ)
    env.pop("ANIMA_ACCELERATE_LAUNCH", None)  # force the direct single-GPU path

    # Step-aware kill: max_train_epochs in the method TOML overrides
    # --max_train_steps (train.py recomputes it), so we can't rely on the run
    # stopping itself. Instead watch the tqdm step counter and kill once the
    # cell has run `args.steps` steps — peak is pinned to warmup (steps 1-4 via
    # _largest_bucket_first) and s/it has settled well before step 50.
    t0 = time.time()
    status = "ok"
    killed_by_us = False
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        sampler.start()
        try:
            while True:
                if proc.poll() is not None:  # exited on its own (error / OOM)
                    break
                if time.time() - t0 > args.timeout:
                    proc.kill()
                    proc.wait()
                    status, killed_by_us = "timeout", True
                    break
                if _last_step(log_path) >= args.steps:
                    proc.kill()
                    proc.wait()
                    killed_by_us = True
                    break
                time.sleep(args.step_poll)
        finally:
            sampler.stop()
    wall = time.time() - t0

    text = log_path.read_text(errors="replace")
    parsed = _parse_log(text, args.skip_steps)
    rc = proc.returncode
    if not killed_by_us and rc != 0:  # died before reaching the step cap
        status = "OOM" if _OOM.search(text) else f"exit{rc}"

    peak_used = sampler.peak_used
    cell = {
        "swap": swap,
        "budget": budget,
        "status": status,
        "returncode": rc,
        "peak_used_mib": peak_used or None,
        "peak_delta_mib": (peak_used - baseline_used) if peak_used else None,
        "baseline_used_mib": baseline_used,
        "launch_at_mib": launch_at,
        "total_mib": sampler.total or None,
        "s_per_it": parsed["s_per_it"],
        "avr_loss": parsed["avr_loss"],
        "wall_s": round(wall, 1),
        "samples": sampler.samples,
        "log": log_path.name,
    }
    flag = {"ok": "✓", "timeout": "⏱", "OOM": "✗OOM"}.get(status, "✗")
    print(
        f"  [{flag}] swap={swap:>2} budget={budget:<4} "
        f"peak={_fmt(peak_used)} Δ={_fmt(cell['peak_delta_mib'])} (launch@{launch_at}) "
        f"s/it={_fmt(parsed['s_per_it'], '.2f')} loss={_fmt(parsed['avr_loss'], '.4f')} "
        f"({wall:.0f}s)",
        flush=True,
    )
    return cell


def _fmt(v, spec: str = "d") -> str:
    if v is None:
        return "—"
    return format(v, spec) if spec != "d" else f"{int(v)}"


# --------------------------------------------------------------------------- #
# Surface tables
# --------------------------------------------------------------------------- #


def _grid_table(cells: list[dict], swaps, budgets, key, fmt) -> list[list[str]]:
    by = {(c["swap"], c["budget"]): c for c in cells}
    header = ["swap \\ budget"] + [str(b) for b in budgets]
    rows = [header]
    for s in swaps:
        row = [str(s)]
        for b in budgets:
            c = by.get((s, b))
            if c is None or c.get(key) is None or c["status"] not in ("ok",):
                row.append(c["status"] if c else "—")
            else:
                row.append(fmt(c[key]))
        rows.append(row)
    return rows


def _md_table(rows: list[list[str]]) -> str:
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for ri, r in enumerate(rows):
        out.append("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
        if ri == 0:
            out.append("| " + " | ".join("-" * w[i] for i in range(len(r))) + " |")
    return "\n".join(out)


def write_surface(run_dir: Path, cells, swaps, budgets) -> list[str]:
    peak_rows = _grid_table(cells, swaps, budgets, "peak_used_mib", lambda v: f"{int(v)}")
    spi_rows = _grid_table(cells, swaps, budgets, "s_per_it", lambda v: f"{v:.2f}")
    loss_rows = _grid_table(cells, swaps, budgets, "avr_loss", lambda v: f"{v:.4f}")

    total = next((c["total_mib"] for c in cells if c.get("total_mib")), None)
    baseline = next((c["baseline_used_mib"] for c in cells if c.get("baseline_used_mib") is not None), 0)

    md = [
        "# Autotune surface: blocks_to_swap × activation_memory_budget\n",
        f"Peak VRAM (MiB used, whole-GPU incl. compile context; OOM cap ≈ {total}, "
        f"idle baseline {baseline}):\n",
        _md_table(peak_rows),
        "\n\nStep time (s/it, steady-state median):\n",
        _md_table(spi_rows),
        "\n\nFinal avr_loss (parity — compare each row across budget columns vs 1.0):\n",
        _md_table(loss_rows),
        "\n",
    ]
    (run_dir / "surface.md").write_text("\n".join(md))

    import csv

    with open(run_dir / "surface.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            ["swap", "budget", "status", "peak_used_mib", "peak_delta_mib",
             "launch_at_mib", "total_mib", "s_per_it", "avr_loss", "wall_s"]
        )
        for c in cells:
            wr.writerow(
                [c["swap"], c["budget"], c["status"], c["peak_used_mib"],
                 c["peak_delta_mib"], c.get("launch_at_mib"), c["total_mib"],
                 c["s_per_it"], c["avr_loss"], c["wall_s"]]
            )
    return ["surface.md", "surface.csv"]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="lora")
    p.add_argument("--preset", default="default")
    p.add_argument("--swaps", default="0,10,20", help="comma list of blocks_to_swap")
    p.add_argument("--budgets", default="1.0,0.99,0.85,0.7,0.6", help="comma list")
    p.add_argument("--steps", type=int, default=50, help="steps to run per cell before kill")
    p.add_argument("--skip_steps", type=int, default=2,
                   help="exclude the first N (compile-heavy) steps from the s/it window")
    p.add_argument("--sample_ratio", type=float, default=0.01,
                   help="dataset fraction per cell — small keeps cells fast")
    p.add_argument("--timeout", type=float, default=600, help="per-cell wall backstop (s)")
    p.add_argument("--poll", type=float, default=0.2, help="GPU sample interval (s)")
    p.add_argument("--step_poll", type=float, default=1.0, help="step-counter check interval (s)")
    p.add_argument("--drain_margin", type=int, default=1200,
                   help="MiB above baseline under which the GPU counts as drained")
    p.add_argument("--drain_max", type=float, default=45.0,
                   help="max wait for prior cell's VRAM to drain (s)")
    p.add_argument("--label", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    swaps = [int(x) for x in args.swaps.split(",") if x.strip() != ""]
    budgets = [float(x) for x in args.budgets.split(",") if x.strip() != ""]

    grid = [(s, b) for s in swaps for b in budgets]
    print(f"Grid: {len(grid)} cells — swaps={swaps} budgets={budgets} "
          f"method={args.method} preset={args.preset} steps={args.steps}")
    if args.dry_run:
        for s, b in grid:
            print("  " + " ".join(_build_argv(args, s, b)))
        return

    reader = _make_reader()
    if reader is None:
        print("ERROR: no GPU memory reader (pynvml / nvidia-smi). Cannot measure peak.",
              file=sys.stderr)
        sys.exit(1)
    print(f"GPU memory reader: {reader[2]}")

    label = args.label or f"{args.method}-{args.preset}"
    run_dir = make_run_dir("autotune", label=label)
    print(f"Run dir: {run_dir}")

    # One clean baseline up front (idle desktop), reused for every cell's delta —
    # a per-cell baseline is contaminated by the prior SIGKILL'd cell's VRAM.
    baseline = reader[0]()
    baseline_used = baseline[0] if baseline else 0
    print(f"Clean baseline GPU used: {baseline_used} MiB")

    cells: list[dict] = []
    try:
        for s, b in grid:
            cells.append(run_cell(args, s, b, reader, run_dir, baseline_used))
    finally:
        reader[1]()  # close nvml

    artifacts = write_surface(run_dir, cells, swaps, budgets)
    artifacts += [c["log"] for c in cells]
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "grid": {"swaps": swaps, "budgets": budgets},
            "steps": args.steps,
            "cells": cells,
        },
        label=label,
        artifacts=artifacts,
        device="cuda:0",
    )
    print(f"\nSurface written:\n  {run_dir / 'surface.md'}")
    print((run_dir / "surface.md").read_text())


if __name__ == "__main__":
    main()
