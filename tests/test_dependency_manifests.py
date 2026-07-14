"""Dependency parity checks for alternate installation paths."""

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v100_requirements_include_lycoris_backend() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lycoris_dependencies = [
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("lycoris-lora")
    ]
    requirements = {
        line.strip()
        for line in (ROOT / "requirements-v100.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert len(lycoris_dependencies) == 1
    assert lycoris_dependencies[0] in requirements
