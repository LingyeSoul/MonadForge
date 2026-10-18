"""``library.training``'s facade must stay lazy (and keep resolving).

Most ``library.training.*`` submodules import torch/accelerate, and importing any
submodule of the package executes its ``__init__``. An eager facade therefore
made every WebUI process load the training stack just to read
``library.training.staged_resolution_plan`` at startup. These tests pin both
halves of the contract: the names still resolve, and a torch-free consumer stays
torch-free.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import library.training as lt


def test_every_facade_name_resolves():
    assert lt.__all__, "the facade must export something"
    missing = [name for name in lt.__all__ if getattr(lt, name, None) is None]
    assert not missing, f"facade names failed to resolve: {missing}"


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        lt.NotARealTrainingName  # noqa: B018 — attribute access is the assertion


def test_export_map_covers_every_public_name():
    assert set(lt.__all__) == set(lt._EXPORTS)


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_a_submodule_does_not_load_torch():
    out = _run(
        "import sys; import library.training.staged_resolution_plan; "
        "print('torch' in sys.modules)"
    )
    assert out == "False", "library.training.* must not drag torch in"


def test_webui_server_starts_without_torch():
    """The WebUI process never trains; it must not pay for the training stack."""
    out = _run("import sys; import webui.server; print('torch' in sys.modules)")
    assert out == "False"
