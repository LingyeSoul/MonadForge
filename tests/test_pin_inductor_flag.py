"""pin_inductor_flag must hold across execution contexts.

Inductor config ``user_override``s are thread-local ContextVars (torch 2.12):
a plain ``config.triton.mix_order_reduction = False`` only exists in the thread
that ran it, and the grad-enabled step-0 compile schedules in a context where
the read falls back to the entry's env-derived default (True) — the v1.14.0
adaln crash (ConstraintViolationError on the dynamic-seq marks; see
docs/optimizations/for_compile.md §2.6). The fixture threads here stand in for
that compile context: a fresh thread has no override, so it reads whatever the
pin left in ``.default``.
"""

import threading

import torch._inductor.config as icfg
from torch.utils._config_module import _UNSET_SENTINEL

from library.runtime.dynamo import pin_inductor_flag

KEY = "triton.mix_order_reduction"


def _read_in_fresh_thread() -> bool:
    out = {}

    def worker():
        out["value"] = icfg.triton.mix_order_reduction

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return out["value"]


def test_pin_holds_in_fresh_thread_context():
    entry = icfg._config[KEY]
    orig_default = entry.default
    orig_override = entry.user_override.get(_UNSET_SENTINEL)
    try:
        # Simulate the shipped wheel state: env-derived default True, no override.
        entry.default = True
        entry.user_override.set(_UNSET_SENTINEL)

        # The trap a plain assignment falls into: an off-context read sees the
        # default, not the main-thread override.
        icfg.triton.mix_order_reduction = False
        assert _read_in_fresh_thread() is True

        pin_inductor_flag(KEY, False)
        assert icfg.triton.mix_order_reduction is False  # same-context read
        assert _read_in_fresh_thread() is False  # compile-context read — the regression
    finally:
        entry.default = orig_default
        entry.user_override.set(orig_override)
