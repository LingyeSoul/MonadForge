"""Compatibility alias for :mod:`anima_daemon.continuation`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.continuation")
sys.modules[__name__] = _module
