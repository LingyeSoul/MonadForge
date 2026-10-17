"""Compatibility alias for :mod:`anima_daemon.manager`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.manager")
sys.modules[__name__] = _module
