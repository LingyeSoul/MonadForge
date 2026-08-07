"""``make sr-*`` — ResShift super-resolution sidecar dispatch.

The SR sidecar runs in the root Anima venv. ``sr-setup`` keeps two install paths:
standard environments use the locked ``sr`` dependency group, while the local V100
stack uses an additive ``uv pip install`` path so it cannot replace its Torch or
V100-specific FlashAttention build. See ``sr/README.md`` for details.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from ._common import queue_command, run

ROOT = Path(__file__).resolve().parents[2]
SR = ROOT / "sr"
VENV_PY = ROOT / ".venv" / "bin" / "python"

V100_TORCH = "2.10.0+cu129"
V100_TORCHVISION = "0.25.0+cu129"
V100_CUDA = "12.9"
V100_ORT = "1.26.0"
V100_FLASH = "26.6"
V100_SR_PACKAGES = [
    "scipy",
    "timm",
    "pandas",
    "scikit-image",
    "lpips",
    "loguru",
    "omegaconf",
    "six",
    "imageio",
    "loralib",
    "pyiqa",
    "albumentations",
    "einops",
    "matplotlib",
    "opencv-python",
    "scikit-learn",
    "tqdm",
    "onnxruntime-gpu==1.26.0",
]


def _venv_py() -> str:
    """Root venv python — falls back to the invoking interpreter if absent."""
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _run(cmd, cwd=ROOT):
    print("RUN:", " ".join(str(c) for c in cmd), f"(cwd={cwd})")
    run(list(cmd), cwd=cwd)


def _module_for_script(script: Path) -> str:
    """Return the importable module name for a repository Python script."""
    rel = script.resolve().relative_to(ROOT)
    return ".".join(rel.with_suffix("").parts)


def _run_module(script: Path, extra: list[str], cwd=ROOT) -> None:
    _run([_venv_py(), "-m", _module_for_script(script), *extra], cwd=cwd)


def _probe_runtime() -> dict:
    """Probe the target venv without importing it into the task dispatcher."""
    code = r'''
import importlib.metadata as md
import importlib.util
import json
import torch

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

out = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": version("torchvision"),
    "onnxruntime": version("onnxruntime-gpu") or version("onnxruntime"),
    "cuda_available": bool(torch.cuda.is_available()),
    "capability": None,
    "device": None,
    "flash_attn": version("flash-attn"),
    "flash_attn_v100": version("flash-attn-v100"),
    "flash_v100_module": bool(importlib.util.find_spec("flash_attn_v100")),
}
if out["cuda_available"]:
    cap = torch.cuda.get_device_capability(0)
    out["capability"] = [int(cap[0]), int(cap[1])]
    out["device"] = torch.cuda.get_device_name(0)
print(json.dumps(out, sort_keys=True))
'''
    result = subprocess.run(
        [_venv_py(), "-c", code], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": f"invalid runtime probe output: {result.stdout[-500:]}"}


def _is_v100_runtime(probe: dict) -> bool:
    """Recognize the pinned local V100 stack, including hidden-CUDA probes."""
    if probe.get("error"):
        return False
    cap = tuple(probe.get("capability") or ())
    device = str(probe.get("device") or "")
    if cap == (7, 0) and "V100" in device:
        return True
    exact_stack = (
        probe.get("torch") == V100_TORCH
        and probe.get("torch_cuda") == V100_CUDA
        and probe.get("torchvision") == V100_TORCHVISION
    )
    # CUDA can be hidden in a shell/CI probe; the exact requirements-v100 stack is
    # still safer to treat as V100 than to let uv sync replace it.
    return exact_stack and (not probe.get("cuda_available") or probe.get("flash_v100_module"))


def _parse_setup_args(extra: list[str]) -> tuple[str, bool, list[str]]:
    profile = os.environ.get("SR_SETUP_PROFILE", "auto").lower()
    dry_run = os.environ.get("SR_SETUP_DRY_RUN", "").lower() in {"1", "true", "yes"}
    forwarded = []
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--v100":
            profile = "v100"
        elif arg == "--standard":
            profile = "standard"
        elif arg == "--profile":
            if i + 1 >= len(extra):
                raise SystemExit("--profile requires auto, v100, or standard")
            profile = extra[i + 1].lower()
            i += 1
        elif arg.startswith("--profile="):
            profile = arg.split("=", 1)[1].lower()
        else:
            forwarded.append(arg)
        i += 1
    if profile not in {"auto", "v100", "standard"}:
        raise SystemExit(f"invalid SR_SETUP_PROFILE={profile!r}; expected auto|v100|standard")
    return profile, dry_run, forwarded


def _check_v100_before_install(probe: dict) -> None:
    expected = {
        "torch": V100_TORCH,
        "torchvision": V100_TORCHVISION,
        "torch_cuda": V100_CUDA,
        "onnxruntime": V100_ORT,
        "flash_attn_v100": V100_FLASH,
        "flash_v100_module": True,
    }
    bad = {key: (probe.get(key), value) for key, value in expected.items()
           if probe.get(key) != value}
    if bad:
        raise SystemExit(
            "V100 SR setup requires the existing local stack "
            f"torch={V100_TORCH}, torchvision={V100_TORCHVISION}, cuda={V100_CUDA}; "
            f"got {bad}. Use SR_SETUP_PROFILE=standard for a normal environment."
        )


def _check_v100_after_install(before: dict, after: dict) -> None:
    keys = (
        "torch", "torchvision", "torch_cuda", "onnxruntime",
        "flash_attn_v100", "flash_v100_module",
    )
    changed = {key: (before.get(key), after.get(key)) for key in keys
               if before.get(key) != after.get(key)}
    if changed:
        raise SystemExit(f"V100 SR setup changed protected runtime packages: {changed}")
    if after.get("onnxruntime") != V100_ORT:
        raise SystemExit(
            f"V100 SR setup requires onnxruntime {V100_ORT}, got {after.get('onnxruntime')}"
        )
    if after.get("flash_attn_v100") != V100_FLASH or not after.get("flash_v100_module"):
        raise SystemExit(
            f"V100 SR setup requires flash-attn-v100 {V100_FLASH}, got "
            f"version={after.get('flash_attn_v100')} module={after.get('flash_v100_module')}"
        )


def _gpu_run(label: str, script: Path, extra: list[str]) -> None:
    """Run an SR GPU script inline, or enqueue it with the existing daemon.

    The daemon command-job path supplies the project interpreter. Use module names
    in both paths so ``library`` resolves from the repository root.
    """
    argv = list(extra)
    queued = False
    for flag in ("--queue", "--detach"):
        while flag in argv:
            argv.remove(flag)
            queued = True
    for flag in ("--inline", "--attach"):
        while flag in argv:
            argv.remove(flag)
    module = _module_for_script(script)
    if queued:
        queue_command(label, ["-m", module, *argv])
        return
    _run([_venv_py(), "-m", module, *argv])


def cmd_sr_setup(extra):
    """Install SR deps using auto, V100-safe, or standard profile."""
    requested, dry_run, forwarded = _parse_setup_args(list(extra))
    before = _probe_runtime()
    profile = "v100" if requested == "auto" and _is_v100_runtime(before) else requested
    print(f"[sr-setup] profile={profile} requested={requested} runtime={before}")
    if profile == "v100":
        _check_v100_before_install(before)
        cmd = ["uv", "pip", "install", "--python", _venv_py(), *V100_SR_PACKAGES]
        if dry_run:
            cmd.insert(3, "--dry-run")
        _run([*cmd, *forwarded])
    else:
        cmd = ["uv", "sync", "--group", "sr", "--inexact"]
        if dry_run:
            cmd.append("--dry-run")
        _run([*cmd, *forwarded])
    if dry_run:
        print(f"[sr-setup] {profile} dry-run complete; setup_env skipped")
        return
    after = _probe_runtime()
    if profile == "v100":
        _check_v100_after_install(before, after)
    print(f"[sr-setup] completed profile={profile} runtime={after}")
    # Sanity-check the vendored, basicsr-free ResShift import resolves.
    _run([str(SR / "scripts" / "setup_env.sh")], cwd=SR)


def cmd_sr_prep(extra):
    """Build the frozen synthetic-LR eval set from image_dataset/ (anima env OK)."""
    # build_eval_set only needs PIL/numpy — run under whatever python invoked us.
    _run_module(SR / "scripts" / "build_eval_set.py", extra)


def cmd_sr_phase0(extra):
    """Phase-0 sanity: released ResShift x4 on the eval set + metrics + montage."""
    _run_module(SR / "scripts" / "run_phase0.py", extra)


def cmd_sr_build_hr_pool(extra):
    """Filter gelcrawl/retrieved + image_dataset into a sharp HR pool (anima env OK)."""
    # PIL/numpy only — run under whatever python invoked us (no SR venv needed).
    _run_module(SR / "scripts" / "build_hr_pool.py", extra)


def cmd_sr_detect_text(extra):
    """Precompute CTD text-region boxes for SR crop sampling (not OCR)."""
    _run_module(SR / "scripts" / "detect_text_boxes.py", extra)


def cmd_sr_train(extra):
    """ResShift domain-finetune on our HR pool, warm-started from a released teacher.

    VERSION picks the scale/schedule family (default x2):
      x2   — 2x finetune off the released x4 v2 (the shipped x2 line).
      x4   — 15-step x4 teacher finetune; its checkpoint feeds `sr-rsd-train --version x4ft`.
      x4s4 — 4-step (v3-schedule) x4 finetune; distilling it needs an explicit --config.
    Output lands in output/sr/{x2_lpips_30k,x4_art,x4_s4_art}.

    Defaults --src to sr/data/hr_pool when it exists (build it with `make sr-build-hr-pool`);
    otherwise train.py falls back to image_dataset. Pass ARGS=\"--iters … --bs … --amp\".

    Runs directly by default, matching this branch's single-GPU task behavior;
    ``--queue``/``--detach`` submits a command job, and ``--inline`` is accepted
    as an explicit direct-run spelling.
    """
    argv = list(extra)
    version = os.environ.get("VERSION", "")
    if version and not any(a == "--version" or a.startswith("--version=") for a in argv):
        argv = ["--version", version, *argv]
    _gpu_run("sr-train", SR / "train_sr" / "train.py", argv)


def cmd_sr_rsd_train(extra):
    """RSD distillation: distill the v2 15-step teacher -> 1-step student on our art.

    Defaults --src to the 4096-capped HR cache (sr/data/rsd_hr_cap4096) when it exists
    and the caller didn't pass their own --src — that cache carries native ~4096-scale
    detail at bounded decode cost (NOT the downsized rsd_hr_1024). Falls back to
    train.py's own image_dataset default if the cache is absent.

    VERSION picks the teacher: x4 (default) = released 15-step v2; x2 = our sr-train x2
    finetune (sr/weights/resshift_x2_final.pth); x4ft = our sr-train x4 finetune
    (output/sr/x4_art/resshift_x4_final.pth). Output lands in output/sr/rsd[_<version>].
    """
    argv = list(extra)
    version = os.environ.get("VERSION", "")
    if version and not any(a == "--version" or a.startswith("--version=") for a in argv):
        argv = ["--version", version, *argv]
    if not any(a == "--src" or a.startswith("--src=") for a in argv):
        cache = SR / "data" / "rsd_hr_cap4096"
        if cache.is_dir():
            print(f"[sr-rsd-train] defaulting --src to {cache} (pass --src to override)")
            argv = ["--src", str(cache), *argv]
        else:
            print(f"[sr-rsd-train] {cache} absent — train.py will fall back to image_dataset")
    _gpu_run("sr-rsd-train", SR / "distill_rsd" / "train.py", argv)


def cmd_sr_rsd_dryrun(extra):
    """RSD VRAM feasibility dry-run (build all nets + 1 fake/gen step)."""
    _run_module(SR / "distill_rsd" / "dry_run.py", extra)


def cmd_sr_rsd_infer(extra):
    """Single-step RSD student inference + MUSIQ. CKPT=… picks a ckpt; unset = most recent.

    VERSION=x2 loads the x2 config (sf=2) and defaults ckpt_dir/out_dir to output/sr/rsd_x2.
    """
    ckpt = os.environ.get("CKPT", "")
    version = os.environ.get("VERSION", "")
    argv = (["--version", version] if version else []) \
        + (["--ckpt", ckpt] if ckpt else []) + list(extra)
    _run_module(SR / "distill_rsd" / "infer.py", argv)


def cmd_sr_test(extra):
    """Tiled SR (released x4 or one of our art finetunes) on a folder/image:
    make sr-test IN=<path> [OUT=… VERSION=v3|v2|x2|x4ft|x4s4 CHOP=512 CKPT=…].

    Thin pass-through to the vendored, basicsr-free sr/scripts/sr_infer.py. Output is
    rsd-infer-style: per-image PNGs + infer_summary.json (MUSIQ) + contact_sheet.png.
    OUT unset -> script default (output/sr/<version>/infer for local versions,
    sr/data/results for released x4). ARGS="--no_musiq --no_sheet --sheet_max N" to trim
    the extras.
    """
    in_path = os.environ.get("IN", "")
    if not in_path:
        sys.exit("set IN=<input image or dir>  (e.g. make sr-test IN=foo.png)")
    version = os.environ.get("VERSION", "v3")
    chop = os.environ.get("CHOP", "512")
    cmd = ["-i", in_path, "--version", version, "--chop_size", chop]
    out = os.environ.get("OUT", "")
    if out:
        cmd += ["-o", out]
    ckpt = os.environ.get("CKPT", "")  # local versions only: pick a make-sr-train checkpoint
    if ckpt:
        cmd += ["--ckpt", ckpt]
    cmd += list(extra)
    _run_module(SR / "scripts" / "sr_infer.py", cmd)
