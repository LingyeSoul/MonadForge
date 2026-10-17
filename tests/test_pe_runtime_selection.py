"""PE backend selection with mock modules; never construct or load a model."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from library.runtime import profiles


def _load_selector(monkeypatch, *, v100=False, failure=None):
    upstream = ModuleType("anime_tools.vision.pe")
    legacy = ModuleType("library.models._legacy_pe")
    calls = []

    def load(name):
        calls.append(name)
        if name == "anime_tools.vision.pe":
            if failure is not None:
                raise failure
            return upstream
        assert name == "library.models._legacy_pe"
        return legacy

    monkeypatch.setattr(profiles, "is_v100_runtime", lambda: v100)
    monkeypatch.setattr(importlib, "import_module", load)
    path = Path(__file__).resolve().parents[1] / "library/models/pe.py"
    spec = importlib.util.spec_from_file_location("_pe_selector_fixture", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return sys.modules[spec.name], upstream, legacy, calls


def test_supported_runtime_keeps_upstream_module_identity(monkeypatch):
    chosen, upstream, _, calls = _load_selector(monkeypatch)
    assert chosen is upstream and calls == ["anime_tools.vision.pe"]


def test_v100_uses_local_model_without_importing_anime_tools(monkeypatch):
    chosen, _, legacy, calls = _load_selector(monkeypatch, v100=True)
    assert chosen is legacy and calls == ["library.models._legacy_pe"]


@pytest.mark.parametrize(
    "name", ["anime_tools", "anime_tools.vision", "anime_tools.vision.pe"]
)
def test_missing_optional_package_uses_local_model(monkeypatch, name):
    chosen, _, legacy, calls = _load_selector(
        monkeypatch, failure=ModuleNotFoundError(name=name)
    )
    assert chosen is legacy
    assert calls == ["anime_tools.vision.pe", "library.models._legacy_pe"]


@pytest.mark.parametrize("name", ["timm", "einops", "anime_tools.internal_missing"])
def test_real_dependency_errors_are_not_hidden(monkeypatch, name):
    failure = ModuleNotFoundError(name=name)
    with pytest.raises(ModuleNotFoundError) as caught:
        _load_selector(monkeypatch, failure=failure)
    assert caught.value is failure
