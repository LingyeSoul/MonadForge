"""Task-runner entries for the V100 FlashAttention source landing."""

from pathlib import Path

from ._common import ROOT, run


def _venv_python() -> str:
    python = Path(ROOT) / ".venv/bin/python"
    if not python.is_file():
        raise SystemExit(f"V100 environment not found: {python}")
    return str(python)


def cmd_install(extra):
    """Build and install the pinned V100 FlashAttention cp313 wheel."""
    run([_venv_python(), "-m", "scripts.v100_flash.install", *extra])


def cmd_validate(extra):
    """Run the strict V100 FlashAttention acceptance suite."""
    run([_venv_python(), "-m", "scripts.v100_flash.validate", *extra])
