"""Compatibility alias for :mod:`anima_daemon.config`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.config")
sys.modules[__name__] = _module
