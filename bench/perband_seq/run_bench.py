"""Per-band dynamic-seq compile (--compile_seq_bands) — union-range A/B.

Phase 0 bench for _archive/proposals/perband_dynamic_seq.md. Both arms run the
same short `train.py --method lora` job on a multi-band pool — by default a
`--sigma_lowres` combolate run (native 1024 counts + demoted 896-route
counts), the real gap-spanning constituency — differing only in the flag:

  union     --compile_dynamic_seq only (today's single (min,max) mark range)
  perband   + --compile_seq_bands     (one tight mark band per tier cluster)

Metrics per arm: steady-state s/it (steps >= settle), step-0 compile wall
(process start -> first training step), peak nvidia-smi VRAM (1 s poll, CSV
artifact), outcome (ok/oom/constraint_violation/timeout). Both arms run
seed-matched with cold compile caches (TORCHINDUCTOR_FORCE_DISABLE_CACHES=1)
so a replayed partition can never blur the A/B; per-step avr_loss traces are
captured and the report includes the max |A-B| divergence — per-band is NOT
expected to be bit-exact (different autotune configs can reorder reductions),
so the gate is trajectory tolerance, not equality.

Ship gate (proposal): perband s/it >= union, no ConstraintViolationError,
compile-wall increase documented.

Usage (through the daemon — agent-launched GPU work must not run inline):
    make daemon-run ARGS="bench/perband_seq/run_bench.py [--steps 60]
        [--label mylabel] [--arms union,perband] [--no-sigma-lowres]
        [--extra-train-arg=--foo ...]"
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench._common import REPO_ROOT, make_run_dir, write_result  # noqa: E402

# Training bar only (desc="steps" from library/training/loop.py) — startup
# bars are un-prefixed and must not trip the step watchdog (freefit_vram
# 2026-07-02 gotcha).
STEP_RE = re.compile(r"steps:.*?\s(\d+)/(\d+)\s*\[")
LOSS_RE = re.compile(r"avr_loss=([0-9.eE+-]+)")
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory")
CONSTRAINT_MARKERS = ("ConstraintViolationError",)


def _nvidia_smi_used() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
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
    """Drain the child's merged stdout on the raw fd (a buffered text read(n)
    would block and stall the watchdog), tee to a log, track step timestamps,
    per-step avr_loss, and failure markers."""

    def __init__(self, proc: subprocess.Popen, log_path: Path):
        super().__init__(daemon=True)
        self.proc = proc
        self.log_path = log_path
        self.last_step = 0
        self.step_times: list[tuple[int, float]] = []
        self.losses: dict[int, float] = {}
        self.oom = False
        self.constraint_violation = False

    def _scan(self, line: str) -> None:
        if any(m in line for m in OOM_MARKERS):
            self.oom = True
        if any(m in line for m in CONSTRAINT_MARKERS):
            self.constraint_violation = True
        m = STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            if step > self.last_step:
                self.last_step = step
                self.step_times.append((step, time.monotonic()))
            lm = LOSS_RE.search(line)
            if lm and step > 0:
                try:
                    self.losses[step] = float(lm.group(1))
                except ValueError:
                    pass

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
                    self._scan(line)
            if buf:
                self._scan(buf)


def run_arm(
    name: str,
    *,
    seq_bands: bool,
    sigma_lowres: bool,
    target_steps: int,
    seed: int,
    run_dir: Path,
    extra_train_args: list[str],
    timeout_s: float,
) -> dict:
    log_path = run_dir / f"{name}.log"
    csv_path = run_dir / f"{name}_vram.csv"

    # Wait for the previous arm's trainer to fully release the GPU.
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
        str(seed),
        "--deterministic",
        "--output_name",
        f"perband_bench_{name}",
        "--max_train_epochs",
        "1",
        # NB: no --sample_prompts — sampling off keeps the compile count set
        # to the pool itself (train.py:1830 gates on `is not None`, so even
        # an empty-string override would trip load_prompts).
    ]
    if sigma_lowres:
        cmd += ["--sigma_lowres"]
    if seq_bands:
        cmd += ["--compile_seq_bands"]
    cmd += extra_train_args

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Cold compile both arms: compile wall is a primary metric, and a replayed
    # partition from the other arm would silently invalidate the A/B.
    env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

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

    # Step-0 compile wall: process start -> first training step. Steady-state
    # s/it over the SECOND HALF of the reached steps: per-band arms compile
    # each remaining band on its first hit — early in the shuffle — so a
    # fixed low settle point (first cut: settle=10) swallowed those stalls
    # into the "steady" window and read as a phantom +13% s/it (the tail
    # rates were identical). Halfway is safely past every band's first hit.
    compile_wall_s = None
    if reader.step_times:
        compile_wall_s = reader.step_times[0][1] - t_start
    s_per_it = None
    settle = max(10, reader.last_step // 2)
    settled = [(s, t) for s, t in reader.step_times if s >= settle]
    if len(settled) >= 2:
        (s0, t0), (s1, t1) = settled[0], settled[-1]
        if s1 > s0:
            s_per_it = (t1 - t0) / (s1 - s0)

    if reader.constraint_violation:
        outcome = "constraint_violation"
    elif reader.oom:
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
        "compile_wall_s": round(compile_wall_s, 1) if compile_wall_s else None,
        "s_per_it": round(s_per_it, 3) if s_per_it else None,
        "peak_mib": poller.peak,
        "idle_mib_before": idle_mib,
        "peak_delta_mib": poller.peak - idle_mib,
        "wall_s": round(wall_s, 1),
        "killed": killed_reason,
        "returncode": rc,
        "losses": reader.losses,
    }
    print(
        f"[{name}] { {k: v for k, v in result.items() if k != 'losses'} }", flush=True
    )
    return result


def loss_divergence(a: dict, b: dict, settle: int = 5) -> dict:
    """Max/mean |A-B| over the common settled steps of two arms' avr_loss
    traces. Per-band is not bit-exact (autotune reorders reductions); the
    invariant is trajectory tolerance."""
    common = sorted(s for s in set(a["losses"]) & set(b["losses"]) if int(s) >= settle)
    if not common:
        return {"n_common_steps": 0}
    diffs = [abs(a["losses"][s] - b["losses"][s]) for s in common]
    return {
        "n_common_steps": len(common),
        "max_abs_diff": round(max(diffs), 6),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default=None)
    ap.add_argument("--arms", default="union,perband")
    ap.add_argument(
        "--no-sigma-lowres",
        action="store_true",
        help="drop --sigma_lowres (e.g. to bench a genuinely multi-tier pool "
        "or an 896-only pool for the mix_order_reduction secondary)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=5400.0,
        help="per-arm wall timeout in seconds",
    )
    ap.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="extra flag forwarded to both arms' train.py (repeatable)",
    )
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ("union", "perband")]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown} (valid: union, perband)")

    run_dir = make_run_dir("perband_seq", args.label)
    print(f"run dir: {run_dir}", flush=True)

    results: dict[str, dict] = {}
    for name in arms:
        results[name] = run_arm(
            name,
            seq_bands=(name == "perband"),
            sigma_lowres=not args.no_sigma_lowres,
            target_steps=args.steps,
            seed=args.seed,
            run_dir=run_dir,
            extra_train_args=args.extra_train_arg,
            timeout_s=args.timeout,
        )

    metrics: dict = {
        name: {k: v for k, v in r.items() if k != "losses"}
        for name, r in results.items()
    }
    if "union" in results and "perband" in results:
        metrics["loss_divergence"] = loss_divergence(
            results["union"], results["perband"]
        )
        a, b = results["union"], results["perband"]
        if a.get("s_per_it") and b.get("s_per_it"):
            metrics["s_per_it_ratio_perband_over_union"] = round(
                b["s_per_it"] / a["s_per_it"], 4
            )

    # Full loss traces as an artifact, not in the envelope.
    losses_path = run_dir / "losses.json"
    losses_path.write_text(
        json.dumps({n: r["losses"] for n, r in results.items()}, indent=1)
    )

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[losses_path, *sorted(run_dir.glob("*_vram.csv"))],
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
