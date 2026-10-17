"""Compatibility alias for :mod:`anima_daemon.mcp`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.mcp")
sys.modules[__name__] = _module

if __name__ == "__main__":
    raise SystemExit(_module.main())
