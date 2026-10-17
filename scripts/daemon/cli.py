"""Compatibility alias for :mod:`anima_daemon.cli`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.cli")
sys.modules[__name__] = _module
