"""Runtime compatibility policy for project and child Python processes."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, MutableMapping
from importlib import metadata, util
from pathlib import Path


_TORCH_CUDA_BUILD_RE = re.compile(r"(?:\+|[._-])cu(?P<version>\d+)", re.IGNORECASE)
_BNB_CUDA_BINARY_RE = re.compile(r"libbitsandbytes_cuda(?P<version>\d+)")


def select_bitsandbytes_cuda_fallback(
    torch_version: str | None, available_versions: Iterable[str]
) -> str | None:
    """Choose the newest older bitsandbytes binary in the same CUDA major."""
    match = _TORCH_CUDA_BUILD_RE.search(torch_version or "")
    if match is None:
        return None

    target_text = match.group("version")
    target = int(target_text)
    available = {int(value) for value in available_versions if str(value).isdigit()}
    if target in available:
        return None

    same_major = [
        value for value in available if value // 10 == target // 10 and value < target
    ]
    return str(max(same_major)) if same_major else None


def _installed_bitsandbytes_cuda_versions() -> set[str]:
    try:
        spec = util.find_spec("bitsandbytes")
    except (ImportError, ValueError):
        return set()
    if spec is None or not spec.submodule_search_locations:
        return set()

    versions: set[str] = set()
    for location in spec.submodule_search_locations:
        try:
            for entry in Path(location).iterdir():
                match = _BNB_CUDA_BINARY_RE.search(entry.name)
                if match is not None:
                    versions.add(match.group("version"))
        except OSError:
            continue
    return versions


def installed_bitsandbytes_cuda_fallback() -> str | None:
    """Return the fallback needed by the installed Torch/bitsandbytes pair."""
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None
    return select_bitsandbytes_cuda_fallback(
        torch_version, _installed_bitsandbytes_cuda_versions()
    )


def configure_bitsandbytes_cuda_override(
    env: MutableMapping[str, str] | None = None,
) -> str | None:
    """Set BNB_CUDA_VERSION only when the exact bundled CUDA binary is absent."""
    target_env = os.environ if env is None else env
    if "BNB_CUDA_VERSION" in target_env:
        return None

    fallback = installed_bitsandbytes_cuda_fallback()
    if fallback is not None:
        target_env["BNB_CUDA_VERSION"] = fallback
    return fallback


def prepare_python_child_env(
    env: MutableMapping[str, str], *, platform: str | None = None
) -> MutableMapping[str, str]:
    """Prepare stable Python stdio and nested subprocess decoding for a child."""
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # PYTHONUTF8 also changes subprocess.run(text=True)'s implicit decoder.
    # On Windows, native tools write the active ANSI code page (for example
    # cp936), so keep locale decoding while forcing only Python stdio to UTF-8.
    if (platform or sys.platform) == "win32":
        env["PYTHONUTF8"] = "0"
    else:
        env.setdefault("PYTHONUTF8", "1")

    configure_bitsandbytes_cuda_override(env)
    return env
