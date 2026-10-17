"""Compatibility alias for :mod:`anima_daemon.webui_sidecar`."""

import importlib
import sys

_module = importlib.import_module("anima_daemon.webui_sidecar")
sys.modules[__name__] = _module
