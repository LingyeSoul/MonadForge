"""Compatibility alias for :mod:`anima_daemon.proc`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.proc")
sys.modules[__name__] = _module
