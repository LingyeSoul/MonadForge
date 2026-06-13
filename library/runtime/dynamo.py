"""Dynamo compile-budget helpers.

The one knob worth a shared home: raising a ``torch._dynamo`` recompile budget
so it survives into the *backward* compile context. A plain
``config.recompile_limit = N`` assignment is context-local (see
``pin_dynamo_limit``), so every call site that pre-raises the budget before a
``torch.compile`` would otherwise re-discover the same ContextVar trap and spill
to eager mid-warmup. Keep this dependency-free (torch + logging only) so any
layer — ``library.anima`` models, ``library.runtime`` harness, ``networks/``
adapters, ``scripts/`` distill loops — can import it without a cycle.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pin_dynamo_limit(name: str, value: int) -> int:
    """Raise a dynamo recompile budget so it holds in EVERY execution context.

    ``torch._dynamo.config.<name>`` is backed by a ``ContextVar`` (``user_override``),
    so a plain ``config.<name> = value`` assignment only takes effect in the thread
    /context that ran it. Dynamo compiles the grad-bearing block ``_forward`` in a
    *different* context (the AOTAutograd / backward compile path), where the override
    is absent and the read falls back to the config entry's ``default`` (8) — so the
    budget silently reverts and the loop spills to eager at the first grad forward,
    despite a correct setup-time raise (verified: the override reads 64 in the main
    thread but 8 in a worker thread). Pinning the canonical entry's ``.default``
    makes the raise context-independent. We set both: the override (same-context
    reads + log visibility) and the default (compile-/backward-thread reads).

    ``name`` may be an alias (e.g. ``cache_size_limit`` aliases ``recompile_limit``);
    the canonical entry is followed so the right ``.default`` is pinned. Returns the
    effective (max of current and requested) value.
    """
    import torch._dynamo as _dynamo

    cfg_mod = _dynamo.config
    target = max(getattr(cfg_mod, name), value)
    setattr(cfg_mod, name, target)  # context-local override (main thread + logs)
    try:
        entry = cfg_mod._config[name]
        # An alias (e.g. cache_size_limit) stores the canonical name fully
        # qualified; follow it to the real entry whose ``.default`` is the
        # cross-context fallback every thread reads.
        canon = (entry.alias or name).rsplit(".", 1)[-1]
        cfg_mod._config[canon].default = target  # global fallback (all contexts)
    except Exception as e:  # noqa: BLE001 - defensive against torch internals
        logger.warning(
            f"could not pin dynamo {name} default ({e}); budget may revert to 8 "
            "in the backward-compile context and spill to eager"
        )
    return target
