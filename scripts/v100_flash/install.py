"""Build and install the pinned flash-attention-v100 main commit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from scripts.v100_flash import PINNED_COMMIT, PINNED_REPOSITORY, ROOT

EXPECTED_PYTHON = (3, 13)
EXPECTED_TORCH = "2.10.0+cu129"
EXPECTED_TORCH_CUDA = "12.9"
EXPECTED_TOOLKIT = "12.9"
EXPECTED_WHEEL_PARTS = ("cp313", "cp313", "linux_x86_64")
FORBIDDEN_BUILD_ENV = ("MMA_NATIVE", "MMA_884", "ATTENTION_DEBUG")


def executable_path(path: Path) -> Path:
    """Make an interpreter path absolute without escaping its virtualenv symlink."""
    return Path(os.path.abspath(path.expanduser()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_string(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {command_string(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"\n[exit_code={result.returncode}]\n")
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def capture_command(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({command_string(command)}):\n{result.stdout}"
        )
    return result.stdout.strip()


def build_environment(
    base: dict[str, str], *, cuda_home: Path, gcc: Path, gxx: Path
) -> dict[str, str]:
    env = dict(base)
    for name in FORBIDDEN_BUILD_ENV:
        env.pop(name, None)
    env.update(
        {
            "CUDA_HOME": str(cuda_home),
            "CC": str(gcc),
            "CXX": str(gxx),
            "CUDAHOSTCXX": str(gxx),
            "MAX_JOBS": "2",
            "NVCC_THREADS": "2",
            "PATH": f"{cuda_home / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": (
                f"{cuda_home / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
            ),
        }
    )
    return env


def validate_wheel_name(path: Path) -> None:
    name = path.name
    if not name.startswith("flash_attn_v100-26.6-") or not name.endswith(".whl"):
        raise RuntimeError(f"unexpected wheel name: {name}")
    missing = [part for part in EXPECTED_WHEEL_PARTS if part not in name]
    if missing:
        raise RuntimeError(f"wheel is not the required cp313 Linux build: {name}")


def installed_flash_provider() -> dict[str, Any]:
    providers: dict[str, Any] = {"modules": {}, "distributions": {}}
    for name in ("flash_attn", "flash_attn_v100"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            providers["modules"][name] = spec.origin
    for name in ("flash-attn", "flash-attn-v100", "flash_attn_v100"):
        try:
            providers["distributions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return providers


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} not found: {path}")


def preflight(
    *, python: Path, cuda_home: Path, gcc: Path, gxx: Path, allow_reinstall: bool
) -> dict[str, Any]:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("flash-attention-v100 landing supports Linux x86_64 only")
    if Path(sys.executable).resolve() != python.resolve():
        raise RuntimeError(
            f"installer must run with {python}; current interpreter is {sys.executable}"
        )
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"got {platform.python_version()}"
        )
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"expected torch {EXPECTED_TORCH} / CUDA {EXPECTED_TORCH_CUDA}, "
            f"got {torch.__version__} / {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to the build interpreter")
    capability = torch.cuda.get_device_capability(0)
    gpu = torch.cuda.get_device_name(0)
    if capability != (7, 0) or "V100" not in gpu:
        raise RuntimeError(f"expected Tesla V100 SM70, got {gpu} SM{capability}")

    nvcc = cuda_home / "bin" / "nvcc"
    _require_file(nvcc, "CUDA nvcc")
    _require_file(gcc, "GCC")
    _require_file(gxx, "G++")
    nvcc_version = capture_command([str(nvcc), "--version"])
    if f"release {EXPECTED_TOOLKIT}" not in nvcc_version:
        raise RuntimeError(f"expected CUDA toolkit {EXPECTED_TOOLKIT}:\n{nvcc_version}")

    provider = installed_flash_provider()
    if (provider["modules"] or provider["distributions"]) and not allow_reinstall:
        raise RuntimeError(
            "an existing flash_attn provider would be overwritten; remove it first "
            f"or pass --allow-reinstall after checking it: {provider}"
        )
    return {
        "python": platform.python_version(),
        "python_executable": str(python),
        "python_resolved": str(python.resolve()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver": capture_command(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ).splitlines()[0],
        "cuda_home": str(cuda_home.resolve()),
        "nvcc": nvcc_version.splitlines()[-1],
        "gcc": capture_command([str(gcc), "--version"]).splitlines()[0],
        "gxx": capture_command([str(gxx), "--version"]).splitlines()[0],
        "existing_provider": provider,
    }


def prepare_source(
    *, shared_repo: Path, source_dir: Path, logs_dir: Path
) -> dict[str, str]:
    if not (shared_repo / ".git").is_dir():
        shared_repo.parent.mkdir(parents=True, exist_ok=True)
        run_logged(
            ["git", "clone", PINNED_REPOSITORY, str(shared_repo)],
            log_path=logs_dir / "source.log",
        )
    run_logged(
        ["git", "fetch", "origin"],
        cwd=shared_repo,
        log_path=logs_dir / "source.log",
    )
    remote_head = capture_command(["git", "rev-parse", "origin/main"], cwd=shared_repo)
    if remote_head != PINNED_COMMIT:
        raise RuntimeError(
            f"upstream main moved to {remote_head}; expected pinned {PINNED_COMMIT}. "
            "Review and re-run the full matrix before changing the pin."
        )
    run_logged(
        ["git", "worktree", "add", "--detach", str(source_dir), PINNED_COMMIT],
        cwd=shared_repo,
        log_path=logs_dir / "source.log",
    )
    source_head = capture_command(["git", "rev-parse", "HEAD"], cwd=source_dir)
    source_tree = capture_command(["git", "rev-parse", "HEAD^{tree}"], cwd=source_dir)
    dirty = capture_command(["git", "status", "--porcelain"], cwd=source_dir)
    if source_head != PINNED_COMMIT or dirty:
        raise RuntimeError(
            f"prepared source is not the clean pinned commit: head={source_head} dirty={dirty!r}"
        )
    return {"commit": source_head, "tree": source_tree, "remote_main": remote_head}


def build_wheel(
    *,
    python: Path,
    source_dir: Path,
    wheels_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    run_logged(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "ninja==1.13.0",
            "wheel==0.46.3",
        ],
        log_path=logs_dir / "build-dependencies.log",
        env=env,
    )
    wheels_dir.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            uv,
            "build",
            "--wheel",
            "--no-build-isolation",
            "--python",
            str(python),
            "--out-dir",
            str(wheels_dir),
            str(source_dir),
        ],
        log_path=logs_dir / "build.log",
        env=env,
    )
    wheels = sorted(wheels_dir.glob("flash_attn_v100-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one built wheel, found {wheels}")
    validate_wheel_name(wheels[0])
    return wheels[0]


def install_and_probe(
    *, python: Path, wheel: Path, logs_dir: Path, env: dict[str, str]
) -> dict[str, Any]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    run_logged(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        log_path=logs_dir / "install.log",
        env=env,
    )
    probe_code = r"""
import json
from pathlib import Path
import torch
import flash_attn
import flash_attn_v100
import flash_attn_v100_cuda
from flash_attn import flash_attn_func, flash_attn_varlen_func

assert torch.cuda.get_device_capability(0) == (7, 0)
assert callable(flash_attn_func) and callable(flash_attn_varlen_func)
assert flash_attn_func.__module__.startswith("flash_attn_v100.")
q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, q, q, dropout_p=0.0)
torch.cuda.synchronize()
assert bool(torch.isfinite(out).all().item())
print(json.dumps({
    "flash_attn_version": getattr(flash_attn, "__version__", None),
    "flash_attn_doc": getattr(flash_attn, "__doc__", None),
    "flash_attn_path": str(Path(flash_attn.__file__).resolve()),
    "flash_attn_v100_version": getattr(flash_attn_v100, "__version__", None),
    "flash_attn_v100_path": str(Path(flash_attn_v100.__file__).resolve()),
    "extension_path": str(Path(flash_attn_v100_cuda.__file__).resolve()),
    "public_function_module": flash_attn_func.__module__,
    "probe_shape": list(out.shape),
    "probe_finite": True,
}))
"""
    output = capture_command([str(python), "-c", probe_code])
    probe = json.loads(output.splitlines()[-1])
    extension = Path(probe["extension_path"])
    _require_file(extension, "installed V100 extension")
    probe["extension_sha256"] = sha256_file(extension)
    return probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--cuda-home",
        type=Path,
        required=True,
        help="CUDA 12.9 toolkit root containing bin/nvcc and lib64/.",
    )
    parser.add_argument("--gcc", type=Path, default=Path("/usr/bin/gcc-14"))
    parser.add_argument("--gxx", type=Path, default=Path("/usr/bin/g++-14"))
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "output/v100-flash-install"
    )
    parser.add_argument("--allow-reinstall", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    python = executable_path(args.python)
    cuda_home = args.cuda_home.resolve()
    output_root = args.output_root.resolve()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "runs" / run_id
    logs_dir = run_dir / "logs"
    source_dir = run_dir / "source"
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "building",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "invocation": command_string([sys.executable, *sys.argv]),
        "repository": PINNED_REPOSITORY,
        "expected_source_commit": PINNED_COMMIT,
        "build_policy": {
            "max_jobs": 2,
            "nvcc_threads": 2,
            "mma_native": False,
            "mma_884": False,
            "attention_debug": False,
        },
    }
    write_json(manifest_path, manifest)

    try:
        manifest["environment"] = preflight(
            python=python,
            cuda_home=cuda_home,
            gcc=args.gcc,
            gxx=args.gxx,
            allow_reinstall=args.allow_reinstall,
        )
        env = build_environment(
            os.environ,
            cuda_home=cuda_home,
            gcc=args.gcc,
            gxx=args.gxx,
        )
        manifest["source"] = prepare_source(
            shared_repo=output_root / "source/upstream",
            source_dir=source_dir,
            logs_dir=logs_dir,
        )
        wheel = build_wheel(
            python=python,
            source_dir=source_dir,
            wheels_dir=run_dir / "wheel",
            logs_dir=logs_dir,
            env=env,
        )
        manifest["wheel"] = {
            "path": str(wheel),
            "filename": wheel.name,
            "sha256": sha256_file(wheel),
            "size": wheel.stat().st_size,
        }
        manifest["installed"] = install_and_probe(
            python=python,
            wheel=wheel,
            logs_dir=logs_dir,
            env=env,
        )
        manifest["status"] = "installed_unvalidated"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)
        write_json(output_root / "current.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        write_json(manifest_path, manifest)
        raise

    print(f"installed wheel: {manifest['wheel']['filename']}")
    print(f"wheel sha256: {manifest['wheel']['sha256']}")
    print(f"manifest: {output_root / 'current.json'}")
    print("status: installed_unvalidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
