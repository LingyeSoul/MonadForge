"""Shared PEP 562 machinery for the façade's lazy re-export modules."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import ModuleType


def attach(
    module_globals: dict,
    attr_to_module: Mapping[str, str | Callable[[], ModuleType]],
) -> None:
    """Install lazy ``__getattr__`` / ``__dir__`` / ``__all__`` on a module.

    *attr_to_module* maps each exported name to the dotted module defining it
    (imported on first access), or to a zero-arg callable returning the module
    (for sources that need custom loading, e.g. the repo-root ``train.py``,
    which is not an installed package).

    A pre-set ``__all__`` in *module_globals* is preserved, so callers can
    export extra non-lazy names (``ROOT``, eagerly imported submodules).
    """
    module_name = module_globals["__name__"]

    def __getattr__(name: str):
        target = attr_to_module.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        module = target() if callable(target) else importlib.import_module(target)
        return getattr(module, name)

    def __dir__() -> list[str]:
        return sorted(module_globals["__all__"])

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
    module_globals.setdefault("__all__", sorted(attr_to_module))
