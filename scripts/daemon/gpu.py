"""Compatibility alias for :mod:`anima_daemon.gpu`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.gpu")
sys.modules[__name__] = _module
