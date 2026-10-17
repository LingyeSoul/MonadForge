"""Compatibility alias for :mod:`anima_daemon.jobs`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.jobs")
sys.modules[__name__] = _module
