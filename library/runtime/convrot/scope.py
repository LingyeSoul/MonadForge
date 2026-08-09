"""Scope helpers for ConvRot — thin wrap of int8_linear classifiers."""

from __future__ import annotations

from library.runtime.int8_linear import (
    ATTENTION_LINEAR_MODULES,
    INT8_LINEAR_SCOPE_MODULES,
    MLP_LINEAR_MODULES,
    classify_frozen_linear_module,
    selected_int8_linear_modules,
)

# Re-export under convrot names so call sites do not import int8_linear for scope.
CONVROT_SCOPE_MODULES = INT8_LINEAR_SCOPE_MODULES
CONVROT_MLP_MODULES = MLP_LINEAR_MODULES
CONVROT_ATTENTION_MODULES = ATTENTION_LINEAR_MODULES

selected_convrot_modules = selected_int8_linear_modules
classify_convrot_linear_module = classify_frozen_linear_module

__all__ = [
    "CONVROT_ATTENTION_MODULES",
    "CONVROT_MLP_MODULES",
    "CONVROT_SCOPE_MODULES",
    "classify_convrot_linear_module",
    "selected_convrot_modules",
]
