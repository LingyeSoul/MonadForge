"""Compatibility alias for :mod:`anima_daemon.server`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.server")
sys.modules[__name__] = _module
