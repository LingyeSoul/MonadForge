"""PE vision tower, using anime_tools when its runtime is supported.

V100 and installations without anime_tools retain the complete local PE model.
The selected implementation remains the same module object, including when
callers monkeypatch it. Missing dependencies inside an installed package are
reported directly rather than silently selecting another implementation.
"""

from __future__ import annotations

import importlib
import sys

from library.runtime.profiles import is_v100_runtime

if is_v100_runtime():
    _implementation = importlib.import_module("library.models._legacy_pe")
else:
    try:
        _implementation = importlib.import_module("anime_tools.vision.pe")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "anime_tools",
            "anime_tools.vision",
            "anime_tools.vision.pe",
        }:
            raise
        _implementation = importlib.import_module("library.models._legacy_pe")

sys.modules[__name__] = _implementation
