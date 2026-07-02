"""Issue #58 — freefit VRAM regression: A/B the zero-perf-candidate levers.

Freefit + dynamic-seq compile + the 2026-06-10 custom-autograd removal grew
no-grad-ckpt training VRAM (reported +6-7 GB at the 1536 tier on a 4090;
docs/findings/custom_autograd_removal_partitioner_oom.md measured the
partitioner share at +0.8 GB @ 4200 tokens). Two candidate levers that should
cost ~zero step time, benched here against the shipping config on the real
default `make lora` job (1024 tier, batch 1, no grad ckpt):

  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — targets reserved-pool
     fragmentation from freefit's per-step varying seq_len.
  2. --partitioner_recompute_views --partitioner_aggressive_recomputation —
     targets the min-cut partitioner's default saved-for-backward set,
     without the activation_memory_budget knapsack (whose cheap-recompute
     pool exhausts below ~0.85 and starts recomputing GEMMs).

Arms (TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 everywhere so a cached partition
from another arm can never replay — see the compile-cache guard-poisoning
memory):

  b100_base       budget=1.0, both levers off   — regression control
                  (expected OOM per the finding)
  b100_exp        budget=1.0 + expandable_segments
  b100_flags      budget=1.0 + partitioner flags
  b100_exp_flags  budget=1.0 + both
  ship_base       budget=0.99 (shipping), levers off, long run
  ship_exp_flags  budget=0.99 + both, long run  — the candidate new default

Metrics per arm: outcome (ok/oom/timeout), peak nvidia-smi used (1 s poll,
timeseries CSV artifact for the fragmentation curve), steady-state s/it from
step timestamps (excludes compile). Runs are bounded by a step watchdog
(SIGTERM at target_steps), with --max_train_epochs 1 as the outer bound.

Usage:
    uv run python bench/freefit_vram/run_bench.py [--short-steps 30]
        [--long-steps 150] [--label mylabel] [--arms b100_exp,ship_base]
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import REPO_ROOT, make_run_dir, write_result  # noqa: E402

# Training bar only: library/training/loop.py builds it with desc="steps",
# so anchor on that prefix — the startup bars (image-size scan, cache checks)
# are un-prefixed and must not trip the step watchdog (they did, 2026-07-02:
# every arm SIGTERM'd at "676/676" of the image-size scan in 5 s).
STEP_RE = re.compile(r"steps:.*?\s(\d+)/(\d+)\s*\[")
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory")


def _nvidia_smi_used() -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return int(out.stdout.strip().splitlines()[0])


class VramPoller(threading.Thread):
    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                used = _nvidia_smi_used()
            except Exception:
                used = -1
            if used >= 0:
                self.samples.append((time.monotonic() - t0, used))
                self.peak = max(self.peak, used)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class OutputReader(threading.Thread):
    """Drain the child's merged stdout on the raw fd (os.read returns what's
    available; a buffered text read(n) would block until n chars and stall
    the step watchdog), tee to a log file, track step counter + OOM."""

    def __init__(self, proc: subprocess.Popen, log_path: Path):
        super().__init__(daemon=True)
        self.proc = proc
        self.log_path = log_path
        self.last_step = 0
        self.step_times: list[tuple[int, float]] = []
        self.oom = False

    def run(self):
        fd = self.proc.stdout.fileno()
        buf = ""
        with open(self.log_path, "w", errors="replace") as log_f:
            while True:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                log_f.write(text)
                log_f.flush()
                buf += text
                *lines, buf = re.split(r"[\r\n]", buf)
                for line in lines:
                    if any(m in line for m in OOM_MARKERS):
                        self.oom = True
                    m = STEP_RE.search(line)
                    if m:
                        step = int(m.group(1))
                        if step > self.last_step:
                            self.last_step = step
                            self.step_times.append((step, time.monotonic()))
            if buf and any(m in buf for m in OOM_MARKERS):
                self.oom = True


def run_arm(
    name: str,
    *,
    budget: float,
    expandable: bool,
    views: bool,
    aggressive: bool,
    target_steps: int,
    run_dir: Path,
    timeout_s: float = 3600.0,
) -> dict:
    log_path = run_dir / f"{name}.log"
    csv_path = run_dir / f"{name}_vram.csv"

    # Wait for the previous arm's trainer to fully release the GPU — without
    # this the idle baseline (and early peak samples) include the dying
    # process (seen 2026-07-02: idle_mib_before=11592 on arm 3).
    drain_t0 = time.monotonic()
    while time.monotonic() - drain_t0 < 120:
        if _nvidia_smi_used() <= 1500:
            break
        time.sleep(2)

    cmd = [
        sys.executable,
        "train.py",
        "--method",
        "lora",
        "--preset",
        "default",
        "--seed",
        "42",
        "--output_name",
        f"vram_bench_{name}",
        "--max_train_epochs",
        "1",
        "--activation_memory_budget",
        str(budget),
    ]
    if views:
        cmd += ["--partitioner_recompute_views"]
    if aggressive:
        cmd += ["--partitioner_aggressive_recomputation"]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Cold compile every arm: persistent AOT/inductor caches key on config,
    # but a replayed stale partition would silently invalidate the A/B.
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    if expandable:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env.pop("ANIMA_EXPANDABLE_SEGMENTS", None)
    else:
        env["ANIMA_EXPANDABLE_SEGMENTS"] = "0"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    idle_mib = _nvidia_smi_used()
    poller = VramPoller()
    poller.start()
    t_start = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    reader = OutputReader(proc, log_path)
    reader.start()

    killed_reason = None
    last_beat = t_start
    while proc.poll() is None:
        if reader.last_step >= target_steps and killed_reason is None:
            killed_reason = "target_steps"
            proc.send_signal(signal.SIGTERM)
        elif time.monotonic() - t_start > timeout_s and killed_reason is None:
            killed_reason = "timeout"
            proc.send_signal(signal.SIGTERM)
        elif killed_reason and time.monotonic() - t_start > timeout_s + 60:
            proc.kill()
        if time.monotonic() - last_beat >= 60:
            last_beat = time.monotonic()
            print(
                f"[{name}] heartbeat: step {reader.last_step}/{target_steps}, "
                f"peak {poller.peak} MiB, t+{time.monotonic() - t_start:.0f}s",
                flush=True,
            )
        time.sleep(0.5)

    rc = proc.wait()
    reader.join(timeout=10)
    poller.stop()
    poller.join(timeout=5)
    wall_s = time.monotonic() - t_start

    with open(csv_path, "w") as f:
        f.write("t_s,used_mib\n")
        for t, used in poller.samples:
            f.write(f"{t:.1f},{used}\n")

    # Steady-state s/it from step timestamps, skipping the first 5 steps
    # (compile + warmup pollute tqdm's own running average).
    s_per_it = None
    settled = [(s, t) for s, t in reader.step_times if s >= 5]
    if len(settled) >= 2:
        (s0, t0), (s1, t1) = settled[0], settled[-1]
        if s1 > s0:
            s_per_it = (t1 - t0) / (s1 - s0)

    if reader.oom:
        outcome = "oom"
    elif killed_reason == "timeout":
        outcome = "timeout"
    elif rc != 0 and killed_reason is None and reader.last_step == 0:
        outcome = f"failed_rc{rc}"
    else:
        outcome = "ok"

    result = {
        "outcome": outcome,
        "steps_reached": reader.last_step,
        "s_per_it": round(s_per_it, 3) if s_per_it else None,
        "peak_mib": poller.peak,
        "idle_mib_before": idle_mib,
        "peak_delta_mib": poller.peak - idle_mib,
        "wall_s": round(wall_s, 1),
        "killed": killed_reason,
        "returncode": rc,
    }
    print(f"[{name}] {result}", flush=True)
    return result


ARMS = {
    # name: (budget, expandable, views, aggressive, long_run)
    "b100_base": (1.0, False, False, False, False),
    "b100_exp": (1.0, True, False, False, False),
    "b100_flags": (1.0, False, True, True, False),
    "b100_exp_flags": (1.0, True, True, True, False),
    "b100_views": (1.0, False, True, False, False),
    "ship_base": (0.99, False, False, False, True),
    "ship_views": (0.99, False, True, False, True),
    "ship_aggr": (0.99, False, False, True, True),
    "ship_exp_flags": (0.99, True, True, True, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-steps", type=int, default=30)
    ap.add_argument("--long-steps", type=int, default=150)
    ap.add_argument("--label", default=None)
    ap.add_argument("--arms", default=None, help="comma-separated subset of arm names")
    args = ap.parse_args()

    names = list(ARMS) if not args.arms else args.arms.split(",")
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        ap.error(f"unknown arms: {unknown}; available: {list(ARMS)}")

    run_dir = make_run_dir("freefit_vram", args.label)
    print(f"run dir: {run_dir}", flush=True)

    metrics = {}
    for name in names:
        budget, expandable, views, aggressive, long_run = ARMS[name]
        target = args.long_steps if long_run else args.short_steps
        metrics[name] = run_arm(
            name,
            budget=budget,
            expandable=expandable,
            views=views,
            aggressive=aggressive,
            target_steps=target,
            run_dir=run_dir,
        )

    # Bench byproducts, not real checkpoints (SIGTERM normally precludes a
    # save; sweep any stragglers).
    for ckpt in (REPO_ROOT / "output" / "ckpt").glob("vram_bench_*"):
        if ckpt.is_file():
            ckpt.unlink()

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=sorted(p.name for p in run_dir.glob("*.log"))
        + sorted(p.name for p in run_dir.glob("*_vram.csv")),
    )
    print(f"result written: {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
