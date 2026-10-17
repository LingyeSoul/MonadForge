"""Compatibility alias for :mod:`anima_daemon.tail`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.tail")
sys.modules[__name__] = _module
