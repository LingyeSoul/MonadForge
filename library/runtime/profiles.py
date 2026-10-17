"""Resolve locally registered Python runtimes without importing torch.

The daemon accepts a profile name, never an interpreter supplied by a browser.
Local operators register interpreters in ``configs/custom/runtime.toml``::

    [runtimes.default]
    python = ".venv/bin/python"
    [runtimes.curation]
    python = "/path/to/curation/bin/python"
    env = { HF_HUB_OFFLINE = "1" }
    [runtimes.v100]
    python = "/path/to/v100/bin/python"

Relative paths are anchored to the repository, not the daemon's working dir.
Interpreter symlinks are deliberately not resolved: doing so escapes a venv.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path


class RuntimeProfileError(ValueError):
    """A requested local runtime is missing, invalid or incompatible."""


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    python: Path
    env: dict[str, str] = field(default_factory=dict)


def _python_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    # abspath normalizes '..' without dereferencing a virtualenv's symlink.
    path = Path(os.path.abspath(path))
    if not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK)):
        raise RuntimeProfileError(f"Python interpreter is missing or not executable: {path}")
    return path


def torch_version(python: str | Path | None = None) -> str | None:
    """Read distribution metadata; this never imports torch or probes a GPU."""
    selected = Path(os.path.abspath(python or sys.executable))
    if selected == Path(os.path.abspath(sys.executable)):
        try:
            return metadata.version("torch")
        except metadata.PackageNotFoundError:
            return None
    code = (
        "from importlib import metadata\n"
        "try: print(metadata.version('torch'))\n"
        "except metadata.PackageNotFoundError: print('')\n"
    )
    try:
        result = subprocess.run(
            [str(selected), "-I", "-c", code],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeProfileError(f"Cannot inspect Python runtime {selected}: {exc}") from exc
    return result.stdout.strip() or None


def is_v100_runtime(python: str | Path | None = None) -> bool:
    """Recognize the protected V100 stack from its package metadata."""
    version = torch_version(python)
    return bool(version and version.startswith("2.10.") and "+cu129" in version)


def resolve_runtime(
    profile: str = "default", configs_dir: Path | None = None,
) -> RuntimeSpec:
    """Resolve ``default``, ``curation`` or ``v100`` from local configuration.

    Default uses the current interpreter; curation inherits default until it
    is registered separately. V100 never silently falls back to another stack.
    A registered interpreter is checked for existence here; model/backend
    capability checks belong to the task running in that interpreter.
    """
    if profile not in {"default", "curation", "v100"}:
        raise RuntimeProfileError(f"Unknown runtime profile: {profile!r}")
    directory = Path(configs_dir) if configs_dir is not None else Path(__file__).parents[2] / "configs"
    directory = Path(os.path.abspath(directory.expanduser()))
    path = directory / "custom" / "runtime.toml"
    root = directory.parent
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeProfileError(f"Cannot read runtime configuration {path}: {exc}") from exc
    registrations = data.get("runtimes", {})
    if not isinstance(registrations, dict):
        raise RuntimeProfileError(f"{path}: runtimes must be a TOML table")

    def registration(name: str) -> tuple[str | None, dict[str, str]]:
        entry = registrations.get(name, {})
        if not isinstance(entry, dict):
            raise RuntimeProfileError(f"{path}: runtimes.{name} must be a TOML table")
        python = entry.get("python")
        if python is not None and (not isinstance(python, str) or not python.strip()):
            raise RuntimeProfileError(f"{path}: runtimes.{name}.python must be a nonempty path")
        env = entry.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not k or "=" in k or "\0" in k
            or not isinstance(v, str) or "\0" in v
            for k, v in env.items()
        ):
            raise RuntimeProfileError(f"{path}: runtimes.{name}.env must map names to strings")
        return python, dict(env)

    default_python, default_env = registration("default")
    default_python = default_python or sys.executable
    if profile == "default":
        return RuntimeSpec(profile, _python_path(default_python, root), default_env)
    python, env = registration(profile)
    if profile == "curation":
        return RuntimeSpec(profile, _python_path(python or default_python, root), default_env | env)
    if python is None:
        if not is_v100_runtime():
            raise RuntimeProfileError(
                f"V100 runtime is not registered in {path}; set [runtimes.v100] "
                "python to the existing Torch 2.10/CUDA 12.9 interpreter. "
                "The current interpreter is not that stack."
            )
        python = sys.executable
    return RuntimeSpec(profile, _python_path(python, root), env)


def list_runtime_profiles(configs_dir: Path | None = None) -> list[dict]:
    """Describe the fixed profiles without importing torch or exposing env.

    Availability means the existing resolver accepts the local registration;
    it does not claim model/backend compatibility or initialize a GPU.
    One broken registration is reported on its row instead of hiding the rest.
    """
    result = []
    for name in ("default", "curation", "v100"):
        try:
            spec = resolve_runtime(name, configs_dir)
        except (ValueError, OSError) as exc:
            result.append({"name": name, "available": False, "python": None, "error": str(exc)})
        else:
            result.append({"name": name, "available": True, "python": str(spec.python), "error": None})
    return result
