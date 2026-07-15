"""Shared helpers for ``scripts/tasks/*`` command modules.

Centralizes:
- ``ROOT`` (project root, regardless of where the calling module lives)
- ``PY`` resolution (venv-aware, pythonw.exe-safe)
- ``run`` / ``build_launch_cmd`` / ``accelerate_launch`` / ``train`` subprocess helpers
- ``latest_output`` / ``latest_lora`` / ``latest_hydra`` checkpoint pickers
- ``INFERENCE_BASE`` — shared inference.py argv prefix
- ``_path`` / ``_preset`` config-overlay helpers
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Shared so the shipped and experimental inference modules agree on what counts
# as a REF_IMAGE for the test-* commands.
_REF_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _python_exe() -> str:
    """Resolve the venv's ``python.exe`` even if this process runs as pythonw.exe.

    Why python.exe and not just ``sys.executable``: when the GUI is launched
    via the desktop shortcut, sys.executable is pythonw.exe. pythonw children
    don't surface a working ``sys.stdout``/``sys.stderr`` to inherited pipes
    the way python.exe does — tqdm progress (which writes to stderr) silently
    drops, breaking the GUI's progress bar. python.exe + ``CREATE_NO_WINDOW``
    (set in ``run()`` when this process has no console) gives us both no
    console popup AND working stdio for grandchildren.
    """
    if sys.platform == "win32":
        cand = Path(sys.executable).with_name("python.exe")
        if cand.exists():
            return str(cand)
    return sys.executable


PY = _python_exe()


def _preset(default: str = "default") -> str:
    return os.environ.get("PRESET", default)


_PATH_OVERRIDES_CACHE: dict | None = None


def _path_overrides() -> dict:
    """Top-level path scalars from base.toml → preset → method file (cached).

    Reads ``METHOD`` and ``METHODS_SUBDIR`` env vars so the GUI can point
    preprocess at the same variant file training will use (e.g.
    ``METHOD=lora METHODS_SUBDIR=gui-methods`` honors overrides written from
    ``ConfigTab``). Missing env vars → just base + preset.

    Defers the import of ``library.config.io`` so commands that don't touch
    preprocess (e.g. ``test-merge``) keep the module-load surface small.
    """
    global _PATH_OVERRIDES_CACHE
    if _PATH_OVERRIDES_CACHE is not None:
        return _PATH_OVERRIDES_CACHE
    try:
        from library.config.io import (
            load_path_overrides,
            load_path_overrides_from_config,
        )

        config_file = os.environ.get("CONFIG_FILE")
        if config_file:
            _PATH_OVERRIDES_CACHE = load_path_overrides_from_config(config_file)
        else:
            _PATH_OVERRIDES_CACHE = load_path_overrides(
                preset=_preset(),
                method=os.environ.get("METHOD") or None,
                methods_subdir=os.environ.get("METHODS_SUBDIR") or "methods",
            )
    except Exception as e:  # noqa: BLE001 — fall back silently to defaults
        print(f"warn: could not read base.toml path overrides: {e}", file=sys.stderr)
        _PATH_OVERRIDES_CACHE = {}
    return _PATH_OVERRIDES_CACHE


def _path(key: str, default: str) -> str:
    """Fetch a path key from base.toml/preset overrides, with hardcoded fallback."""
    val = _path_overrides().get(key, default)
    return str(val) if val is not None else default


def bespoke_preset_flags(preset: str) -> list[str]:
    """Translate ``configs/presets.toml[<preset>]`` into CLI flags for the
    bespoke distillation loops (``scripts/distill_mod/distill.py`` / ``scripts/distill_turbo/distill.py``)
    that bypass ``train.py``'s config merge chain.

    Honored keys:
      - ``blocks_to_swap`` → ``--blocks_to_swap N``
      - ``gradient_checkpointing`` (bool) → ``--grad_ckpt`` / ``--no_grad_ckpt``
      - ``sample_ratio`` → ``--sample_ratio R`` (per-bucket subsample; makes
        ``PRESET=debug/half/quarter/tenth`` actually run on a small slice).

    When the preset omits ``gradient_checkpointing`` we default to
    ``--no_grad_ckpt`` (the trainable footprints here are tiny; ckpt is a perf
    loss when VRAM isn't tight). Other preset keys are silently dropped.
    """
    try:
        from library.config.io import load_preset_section
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not import preset loader: {e}", file=sys.stderr)
        return ["--no_grad_ckpt"]

    try:
        section = load_preset_section(preset)
    except (FileNotFoundError, KeyError) as e:
        print(
            f"warn: preset '{preset}' not found ({e}); using bespoke-loop defaults",
            file=sys.stderr,
        )
        return ["--no_grad_ckpt"]

    flags: list[str] = []
    if "blocks_to_swap" in section:
        flags += ["--blocks_to_swap", str(int(section["blocks_to_swap"]))]
    if "gradient_checkpointing" in section:
        flags.append(
            "--grad_ckpt" if section["gradient_checkpointing"] else "--no_grad_ckpt"
        )
    else:
        flags.append("--no_grad_ckpt")
    if "sample_ratio" in section:
        flags += ["--sample_ratio", str(float(section["sample_ratio"]))]
    return flags


def latest_output(prefix: str = "", exclude: str | None = None) -> Path:
    """Return the most recently modified .safetensors file in output/ckpt/ matching prefix.

    If `exclude` is given, any filename containing that substring is skipped. Useful
    to disambiguate overlapping prefixes (e.g. anima_postfix vs anima_postfix_exp).
    HydraLoRA multi-head sibling files (`*_moe.safetensors`) and backup files
    (containing `.bak.`) are always excluded.
    """
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("*.safetensors")
            if f.name.startswith(prefix)
            and not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and (exclude is None or exclude not in f.name)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        label = f"'{prefix}*.safetensors'" if prefix else "*.safetensors"
        print(f"No {label} files found in output/ckpt/", file=sys.stderr)
        sys.exit(1)
    return outputs[0]


def latest_lora() -> Path:
    # Exclude pooled_text_proj heads: not LoRAs (resolved separately by MOD=1),
    # but a blind newest-`.safetensors` pick would grab them after `make distill-mod`.
    return latest_output(exclude="pooled_text_proj")


def latest_hydra() -> Path:
    """Latest HydraLoRA multi-head file (`anima_hydra*_moe.safetensors`)."""
    outputs = sorted(
        (
            f
            for f in (ROOT / "output" / "ckpt").glob("anima_hydra*_moe.safetensors")
            if ".bak." not in f.name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not outputs:
        print(
            "No 'anima_hydra*_moe.safetensors' files found in output/ckpt/ "
            "(enable the HydraLoRA block in configs/methods/lora.toml and run `make lora`)",
            file=sys.stderr,
        )
        sys.exit(1)
    return outputs[0]


def _random_ref_image(directory: Path) -> str | None:
    """Pick a random image under ``directory`` (recursive), or ``None`` if empty.

    Source layouts (``post_image_dataset/resized/``, ``easycontrol-dataset/``)
    nest images under per-artist subdirs, so recurse rather than only scanning
    top-level files. Shared by the EasyControl / IP-Adapter / DirectEdit test
    commands as the fallback reference when no REF_IMAGE is supplied.
    """
    if not directory.is_dir():
        return None
    pool = [p for p in directory.rglob("*") if p.suffix.lower() in _REF_IMAGE_EXTS]
    if not pool:
        return None
    pick = random.choice(pool)
    print(f"  > Random ref: {pick}")
    return str(pick)


def _has_console() -> bool:
    """True if this process is attached to a Windows console (or is non-Windows).

    Used to decide whether to suppress new console popups for child processes.
    A pythonw.exe-launched process (e.g. desktop GUI shortcut) has no console.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:  # noqa: BLE001 — err on the safe side: keep output visible
        return True


def _stdout_is_regular_file() -> bool:
    """True iff ``sys.stdout`` is a regular file (not a pipe / TTY / DEVNULL).

    The training daemon launches this process with stdout redirected to
    ``<job_dir>/stdout.log`` (``spawn_detached``). In that case the
    grandchild's stdout is a regular file, not a pipe — so tqdm's
    high-volume redraws won't fill an OS pipe buffer and stall the
    trainer. The legacy pipe-based callers (QProcess / WebUI direct
    subprocess) keep their stdout as a pipe; for those we still want
    to swallow the grandchild's stdout to avoid the stall.
    """
    import os
    import stat

    try:
        st = os.fstat(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return False
    return stat.S_ISREG(st.st_mode)


def run(cmd: list[str], **kwargs):
    """Run a subprocess, exit on failure.

    Prepends the venv's Scripts/bin directory to PATH (in both the child env
    and our own lookup) so venv-installed CLIs (``accelerate``, ``hf``, ...)
    resolve even when this process was started via a desktop shortcut that
    invokes ``pythonw.exe`` directly, bypassing venv activation.

    On Windows, ``subprocess.run`` uses the parent's PATH to locate the exe —
    setting ``env["PATH"]`` only affects the *child's* environment, not the
    lookup. We resolve the first arg to an absolute path with ``shutil.which``
    against the boosted PATH so the lookup works regardless.

    When this process has no console (pythonw.exe), Windows would allocate a
    new console for any console-subsystem child (python.exe, hf.exe, ...).
    We pass ``CREATE_NO_WINDOW`` to suppress that popup so GUI users don't
    see a terminal flash for every subprocess.
    """
    print(f"  > {' '.join(cmd)}")
    env = kwargs.pop("env", None)
    if env is None:
        env = os.environ.copy()
    # Bound `hf` CLI socket timeouts so a stalled download can't hang the serial
    # daemon queue. The task route shells out to the `hf` CLI (downloads.py + the
    # preprocess tagger-vocab auto-fetch), which — unlike library.runtime.hf_download —
    # is otherwise unbounded; a hung-not-failed fetch inside the daemon wedges every
    # job queued behind it (training included). Reuse the library's single source of
    # truth; setdefault means an explicit ANIMA_HF_TIMEOUT / HF_HUB_* still wins.
    if cmd and os.path.basename(str(cmd[0])) in ("hf", "hf.exe"):
        from library.runtime.hf_download import ensure_hf_timeouts

        ensure_hf_timeouts()
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.environ["HF_HUB_DOWNLOAD_TIMEOUT"])
        env.setdefault("HF_HUB_ETAG_TIMEOUT", os.environ["HF_HUB_ETAG_TIMEOUT"])
    venv_bin = str(Path(PY).parent)
    if venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    # Un-buffer children's stdio so the GUI sees tqdm/training output live
    # instead of in pipe-buffered chunks. Inherited by grandchildren too.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Force UTF-8 stdio regardless of console locale: on non-UTF-8 consoles
    # (e.g. Korean Windows cp949) a child printing an em-dash crashes with
    # UnicodeEncodeError. PYTHONIOENCODING is the belt-and-suspenders fallback.
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = list(cmd)
    if cmd and not Path(cmd[0]).is_absolute():
        resolved = shutil.which(cmd[0], path=env["PATH"])
        if resolved:
            cmd[0] = resolved
    if sys.platform == "win32" and not _has_console():
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        # Stdout is a pipe to the parent (WebUI / QProcess / GUI) — no
        # one is actively draining the read end at terminal speed.  High-
        # volume tqdm output fills the OS pipe buffer and stalls the
        # training loop.  Redirect the grandchild's stdout to DEVNULL;
        # the caller already reads structured progress from the JSONL
        # sink.  stderr stays live for low-volume diagnostic output.
        #
        # Skip the redirect when our own stdout is already a regular file
        # — that's the case under the training daemon, which redirects
        # the subprocess's stdout to ``<job_dir>/stdout.log`` via
        # ``spawn_detached``. Throwing the grandchild's tqdm output to
        # DEVNULL there silently strands the WebUI's live-log view: the
        # daemon keeps reading the file, but only stderr (Python
        # ``logging``) lands in it, and the dashboard shows a frozen
        # single log line instead of the streaming tqdm redraws.
        # ``fileno()`` raises if ``sys.stdout`` was replaced (e.g. by
        # ``contextlib.redirect_stdout``) — treat that as "not a regular
        # file" and fall back to the pipe-safe behavior.
        if not _stdout_is_regular_file():
            kwargs.setdefault("stdout", subprocess.DEVNULL)
        if sys.stderr is not None:
            kwargs.setdefault("stderr", sys.stderr)
    result = subprocess.run(cmd, cwd=kwargs.pop("cwd", ROOT), env=env, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _nsys_wrapper() -> tuple[list[str], Path] | tuple[None, None]:
    """Build an ``nsys profile`` prefix when PROFILE_STEPS is set.

    Returns ``(prefix, out_path)`` when active so the caller can both wrap the
    launch AND run ``nsys stats`` against the resulting report afterward.
    Returns ``(None, None)`` when PROFILE_STEPS is unset. Honors NSYS_OUT for
    the report path (default ``output/nsys/profile.nsys-rep``).

    Why ``--capture-range-end=stop`` (not ``stop-shutdown``) and ``--wait=primary``:
    the wrapped tree is ``nsys → accelerate launcher → train.py worker``. With
    ``stop-shutdown`` nsys SIGTERMs the launcher the moment ``cuProfilerStop``
    fires, the launcher dies before reaping the worker, the worker gets
    reparented to init, and the default ``--wait=all`` blocks forever waiting
    for it. Instead: the worker calls ``cuProfilerStop`` and then voluntarily
    ``sys.exit(0)`` (see ``library/training/loop.py`` ``_profiler_step_end``),
    the launcher exits naturally, and ``--wait=primary`` lets nsys finalize
    the report as soon as the launcher (its primary target) is gone — no
    leftover ``/tmp/*.qdstrm``.
    """
    if not os.environ.get("PROFILE_STEPS"):
        return None, None
    nsys = shutil.which("nsys")
    if nsys is None:
        print(
            "warn: PROFILE_STEPS set but `nsys` not found on PATH; "
            "running without profiler wrapper",
            file=sys.stderr,
        )
        return None, None
    out = os.environ.get("NSYS_OUT", "output/nsys/profile.nsys-rep")
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  > nsys report -> {out_path}")
    # Profile config tuned for kernel + bottleneck analysis (GPU perf counters,
    # CUDA-graph node timing, mem tracking, Python sampling, sqlite stats).
    # Symbol-resolution is OFF (--resolve-symbols=false + the *=none flags):
    # otherwise nsys finalize stalls for minutes reaching NVIDIA's symbol servers
    # while VRAM stays reserved. None of the enabled data needs symbol resolution.
    metrics_set = os.environ.get("NSYS_GPU_METRICS_SET")
    cmd = [
        nsys,
        "profile",
        "-o",
        str(out_path.with_suffix("")),  # nsys appends .nsys-rep
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--wait=primary",
        "--trace=cuda,nvtx,cudnn,cublas",
        "--cuda-graph-trace=node",
        "--cuda-memory-usage=true",
        "--python-sampling=true",
        "--python-sampling-frequency=1000",
        "--stats=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--cudabacktrace=none",
        "--resolve-symbols=false",
    ]
    if _nsys_gpu_metrics_available(nsys):
        cmd += [
            "--gpu-metrics-devices=cuda-visible",
            "--gpu-metrics-frequency=10000",
        ]
        if metrics_set:
            cmd.append(f"--gpu-metrics-set={metrics_set}")
    else:
        print(
            "  > nsys: GPU metrics disabled (perf counters restricted to admin). "
            "To enable SM occupancy / tensor-core / memory-bandwidth counters:\n"
            "      sudo tee /etc/modprobe.d/nvidia-perf.conf <<<'options nvidia "
            '"NVreg_RestrictProfilingToAdminUsers=0"\'\n'
            "      sudo update-initramfs -u && sudo reboot\n"
            "    See https://developer.nvidia.com/ERR_NVGPUCTRPERM",
            file=sys.stderr,
        )
    return cmd, out_path


def _nsys_gpu_metrics_available(nsys: str) -> bool:
    """Probe whether nsys can collect GPU metrics on this host.

    nsys validates ``--gpu-metrics-devices`` at argv-parse time and aborts the
    whole run if the perf-counter ioctl is restricted to root (the default on
    most distros — see ERR_NVGPUCTRPERM). Probing first lets us silently skip
    the flag instead of crashing the training task.
    """
    try:
        out = subprocess.run(
            [nsys, "profile", "--gpu-metrics-devices=help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    blob = (out.stdout or "") + (out.stderr or "")
    return (
        "Insufficient privilege" not in blob
        and "None of the installed GPUs" not in blob
    )


# nsys stats reports auto-generated after profiling, tuned for kernel +
# bottleneck analysis on a per-step NVTX trace.
_NSYS_STATS_REPORTS = (
    "cuda_gpu_kern_sum",
    "nvtx_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_api_sum",
    "cuda_kern_exec_sum",
)


def _nsys_run_stats(rep_path: Path) -> None:
    """Generate textual ``nsys stats`` reports next to the .nsys-rep.

    Writes one ``<stem>_<report>.txt`` per report into the same directory as
    the .nsys-rep. Best-effort: if the .nsys-rep didn't materialize (e.g. nsys
    aborted before finalizing) or stats fails, prints a warning and returns —
    a missing summary shouldn't fail the training task itself.
    """
    if not rep_path.exists():
        print(
            f"warn: nsys report not found at {rep_path}; skipping stats",
            file=sys.stderr,
        )
        return
    nsys = shutil.which("nsys")
    if nsys is None:
        return
    out_prefix = rep_path.with_suffix("")
    cmd = [
        nsys,
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=column",
        "--output",
        str(out_prefix),
    ]
    for report in _NSYS_STATS_REPORTS:
        cmd += ["--report", report]
    cmd.append(str(rep_path))
    print(f"  > nsys stats -> {out_prefix.parent}/")
    # Best-effort: the .nsys-rep is the canonical artifact, so don't sys.exit.
    try:
        subprocess.run(cmd, cwd=ROOT, check=False)
    except OSError as e:
        print(f"warn: nsys stats failed: {e}", file=sys.stderr)


def build_launch_cmd(*args: str, python_exe: str | None = None) -> list[str]:
    """Build the ``train.py`` launch command list (no side effects).

    Pure command construction — the returned list is exactly what launches
    training. Extracted from ``accelerate_launch`` so other spawners (the
    training daemon under ``scripts/daemon/``) can ``Popen`` the same command
    themselves — detached, with their own stdio redirection and process-tree
    monitoring — instead of going through ``run()``'s blocking
    ``subprocess.run`` + ``sys.exit``-on-failure path. The nsys profiling
    wrapper stays in ``accelerate_launch``: it's a CLI-only concern, not
    something the daemon ever applies.

    Single-GPU fast path (default): invoke ``train.py`` directly. The
    ``accelerate launch`` wrapper is a *second* full Python bootstrap — the
    launcher process imports accelerate (which drags in torch, ~5s) only to
    spawn one local worker, pure overhead on a single-GPU box. ``train.py``
    builds its own ``Accelerator()``, which defaults to single-process, and
    ``prepare_accelerator`` reads ``mixed_precision`` from the config chain
    (``base.toml`` → ``bf16``), so the launcher's ``--mixed_precision bf16`` was
    redundant. The dropped ``--num_cpu_threads_per_process 3`` only capped CPU
    intra-op threads to avoid multi-process oversubscription — irrelevant for a
    single GPU-bound process.

    Set ``ANIMA_ACCELERATE_LAUNCH=1`` to force the accelerate launcher, which
    multi-GPU / distributed runs genuinely need (it sets rank / world-size env
    and re-execs one worker per device).

    ``python_exe`` overrides the launching interpreter (default ``PY`` =
    python.exe). The detached daemon passes ``pythonw.exe`` here: a uv-venv
    python.exe is a trampoline that re-execs the real interpreter, and
    ``CREATE_NO_WINDOW`` doesn't survive that re-exec — so a python.exe worker
    pops a console window that, when closed, kills the job with
    ``STATUS_CONTROL_C_EXIT``. pythonw.exe never allocates a console; stdio
    still works because the daemon redirects the child's stdout/stderr to a
    file (not an inherited console). With the direct path train.py runs as that
    windowless interpreter itself; under the accelerate launcher the workers it
    re-spawns inherit it via ``sys.executable``.

    The accelerate path is invoked as
    ``python -m accelerate.commands.accelerate_cli launch`` rather than the bare
    ``accelerate`` console-script, so the launching interpreter propagates to
    accelerate's workers via ``sys.executable`` — the accelerate.exe shim
    hardcodes python.exe as the worker interpreter, defeating that.
    """
    py = python_exe or PY
    if not os.environ.get("ANIMA_ACCELERATE_LAUNCH"):
        return [py, "train.py", *args]
    # Forward --mixed_precision to the launcher so it matches the Accelerator()
    # train.py builds (defaults to bf16 when unset on the CLI).
    mixed_precision = "bf16"
    for i, a in enumerate(args):
        if a == "--mixed_precision" and i + 1 < len(args):
            mixed_precision = args[i + 1]
        elif a.startswith("--mixed_precision="):
            mixed_precision = a.split("=", 1)[1]
    return [
        py,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_cpu_threads_per_process",
        "3",
        "--mixed_precision",
        mixed_precision,
        "train.py",
        *args,
    ]


def accelerate_launch(*args: str):
    """Launch training via accelerate with extra CLI args forwarded (blocking).

    Builds the command via ``build_launch_cmd`` then runs it through ``run``
    (blocking, exits on failure). When PROFILE_STEPS is set, wraps the launch
    with ``nsys profile`` so ``make <method> PROFILE_STEPS=3-5`` produces a
    navigable Nsight report at ``output/nsys/profile.nsys-rep`` (override with
    NSYS_OUT). After the run, generates per-report textual summaries via
    ``nsys stats`` next to the .nsys-rep.
    """
    cmd = build_launch_cmd(*args)
    nsys_prefix, nsys_out = _nsys_wrapper()
    if nsys_prefix is not None:
        cmd = nsys_prefix + ["--"] + cmd
    run(cmd)
    if nsys_out is not None:
        _nsys_run_stats(nsys_out)


def build_method_args(
    method: str,
    *,
    preset: str,
    methods_subdir: str | None = None,
    extra=None,
    artist: str | None = None,
    profile_steps: str | None = None,
) -> list[str]:
    """Assemble the ``["--method", m, "--preset", p, ...]`` train.py arg list.

    Pure — no env reads, no subprocess. Shared by the CLI ``train()`` path and
    the training daemon (``scripts/daemon``) so the daemon doesn't duplicate the
    ARTIST / PROFILE_STEPS handling. ``extra`` is appended verbatim; ``artist`` /
    ``profile_steps`` add their flags only when the caller didn't already pass
    them in ``extra``.
    """
    extra = list(extra or [])
    args = ["--method", method, "--preset", preset]
    if methods_subdir:
        args += ["--methods_subdir", methods_subdir]
    if artist and not any(a == "--artist_filter" for a in extra):
        args += ["--artist_filter", artist]
    if profile_steps and not any(a == "--profile_steps" for a in extra):
        args += ["--profile_steps", profile_steps]
    return [*args, *extra]


def _queue_submit(
    method: str,
    *,
    preset: str,
    methods_subdir: str | None,
    extra: list[str],
    artist: str | None,
    profile_steps: str | None,
) -> None:
    """Enqueue a training job on the local daemon instead of running it inline.

    The ``--queue`` path (``make lora --queue``, ``make lora-gui <v> --queue``)
    turns the CLI into a job *producer*: it auto-starts the daemon if needed
    and POSTs the same method/preset/methods_subdir the inline path would have
    built, then returns immediately. Submit ×N to drain an overnight sweep
    serially.

    ARTIST / PROFILE_STEPS are folded into ``extra`` as explicit flags here
    because the daemon's own ``build_method_args`` call doesn't read those env
    vars — without folding, a queued artist run would silently train the full
    dataset.
    """
    extra = list(extra)
    if artist and "--artist_filter" not in extra:
        extra += ["--artist_filter", artist]
    if profile_steps and "--profile_steps" not in extra:
        extra += ["--profile_steps", profile_steps]

    from scripts.daemon import client as _daemon_client

    cl = _daemon_client.ensure_daemon()
    resp = cl.submit(
        method=method,
        preset=preset,
        methods_subdir=methods_subdir,
        extra=extra,
    )
    job_id = resp.get("job_id")
    print(
        f"queued job {job_id} (method={method}, preset={preset}). "
        f"daemon: {cl.base}\n"
        f"  python tasks.py daemon-attach JOB={job_id}   # follow this job's output\n"
        f"  python tasks.py daemon-attach                # follow queue/lifecycle events\n"
        f"  python tasks.py daemon-kill JOB={job_id}     # cancel it\n"
        f"  python tasks.py daemon-terminate             # stop the daemon + discard queue"
    )


def queue_command(label: str, argv: list[str]) -> None:
    """Enqueue a bespoke-loop distillation command on the local daemon.

    The training daemon is generic over "run this argv" via its ``kind="command"``
    job path (the same one preprocess/mask use). The bespoke loops
    (``scripts/distill_turbo`` / ``scripts/distill_spd``) bypass ``train.py``, so
    they can't ride the train-job ``_queue_submit`` path — they submit a plain
    command job instead. ``argv`` is run by the daemon as
    ``[venv_python, *argv]`` from the repo root, so pass the module form
    (``["-m", "scripts.distill_turbo.distill", ...]``). Preset/CLI flags must be
    baked into ``argv`` here: the command-job path does no config merging.
    """
    from scripts.daemon import client as _daemon_client

    cl = _daemon_client.ensure_daemon()
    resp = cl.submit_command(label=label, argv=list(argv))
    job_id = resp.get("job_id")
    print(
        f"queued job {job_id} (label={label}). daemon: {cl.base}\n"
        f"  python tasks.py daemon-attach JOB={job_id}   # follow this job's output\n"
        f"  python tasks.py daemon-attach                # follow queue/lifecycle events\n"
        f"  python tasks.py daemon-kill JOB={job_id}     # cancel it\n"
        f"  python tasks.py daemon-terminate             # stop the daemon + discard queue"
    )


# ``tasks.py`` command names whose body is ``train(...)`` — i.e. they run
# ``train.py`` (via ``accelerate_launch`` / ``build_launch_cmd``) and therefore
# honor ``--progress_jsonl``. Bespoke-loop commands (``turbo``, ``exp-spd``,
# ``distill-*``) bypass ``train.py`` and never read that flag, so they're
# excluded. Single source of truth shared by:
#   - scripts/daemon/manager.py::_command_runs_train  (injects --progress_jsonl)
#   - webui/services/task_service.py::_commandRunsTraining (watches the jsonl)
# Keep both delegating here so the two surfaces can't drift — a prior drift
# (only ``lora``/``lora-gui``/``easycontrol``) left exp-chimera / exp-soft-tokens
# / exp-byg falling back to the shared cross-run progress file.
_TRAIN_COMMANDS = frozenset(
    {
        "lora",
        "lora-gui",
        "staged-train",
        "easycontrol",
        "exp-chimera",
        "exp-soft-tokens",
        "exp-byg",
    }
)


def command_runs_training(command: str | None) -> bool:
    """True iff the ``tasks.py`` command ``command`` runs through ``train.py``.

    See :data:`_TRAIN_COMMANDS` for the contract + the two call sites.
    """
    return command in _TRAIN_COMMANDS


def train(
    method: str, extra, preset: str | None = None, methods_subdir: str | None = None
):
    """Launch training for a given method + preset (PRESET env overrides default).

    `methods_subdir` selects the folder under `configs/` that holds the method
    file (default ``"methods"``; pass ``"gui-methods"`` for the clean per-variant
    files used by the `lora-gui` path).

    ARTIST env var trains an artist-only LoRA — equivalent to passing
    `--artist_filter <name>` (filters dataset to `@<name>`-tagged captions and
    redirects output to `output/ckpt-artist/`).

    ``--queue`` anywhere in ``extra`` enqueues the job on the local training
    daemon and returns immediately instead of running it inline (the overnight
    sweep path — see ``_queue_submit``).
    """
    preset = preset or _preset()
    extra = list(extra or [])
    artist = os.environ.get("ARTIST")
    profile_steps = os.environ.get("PROFILE_STEPS")

    if "--queue" in extra:
        extra.remove("--queue")
        _queue_submit(
            method,
            preset=preset,
            methods_subdir=methods_subdir,
            extra=extra,
            artist=artist,
            profile_steps=profile_steps,
        )
        return

    args = build_method_args(
        method,
        preset=preset,
        methods_subdir=methods_subdir,
        extra=extra,
        artist=artist,
        profile_steps=profile_steps,
    )
    accelerate_launch(*args)


INFERENCE_BASE = [
    PY,
    "inference.py",
    "--dit",
    _path("pretrained_model_name_or_path", "models/diffusion_models/anima-base-v1.0.safetensors"),
    "--text_encoder",
    _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
    "--vae",
    _path("vae", "models/vae/qwen_image_vae.safetensors"),
    "--vae_chunk_size",
    "64",
    "--vae_disable_cache",
    "--attn_mode",
    "flash",  # flash4 not supported yet (flash-attention-sm120 disabled)
    "--lora_multiplier",
    "1.0",
    "--prompt",
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds.",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size",
    "1024",
    "1024",
    "--infer_steps",
    "28",
    "--flow_shift",
    "3.0",
    "--sampler",
    "euler",
    "--guidance_scale",
    "4.0",
    "--seed",
    "42",
    "--save_path",
    "output/tests",
]


def override_arg(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a ``--flag VALUE`` (or ``--flag V1 V2``) pair in argv with a
    fresh ``--flag value`` pair. Used to retarget ``INFERENCE_BASE`` defaults
    (e.g. the turbo contract: 4 steps, cfg=1.0) without rewriting the whole list.
    """
    if flag not in argv:
        return argv + [flag, value]
    i = argv.index(flag)
    return argv[:i] + [flag, value] + argv[i + 2 :]
