"""Compatibility alias for :mod:`anima_daemon.client`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.client")
sys.modules[__name__] = _module
