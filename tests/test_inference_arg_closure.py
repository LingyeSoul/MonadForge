"""H2 guard: ``getattr(args, …)`` read keys must be declared argparse flags.

``docs/findings/entanglement_audit_high_severity.md`` §H2 — the inference
long-tail (DCW / SMC-CFG / spectrum / soft-tokens) and the training loop read
config off ``args`` by name via ``getattr(args, "k", default)``. Rename or drop
the flag and the read silently falls back to its default — feature off, all
static tests still green.

These tests pin every such read to a declared flag, so a rename/drop fails
loudly here instead. Same closure pattern as H1 (``test_network_registry.py``),
shared via ``tests/config_closure.py``.
"""

from __future__ import annotations

from pathlib import Path

from tests.config_closure import assert_closed, declared_argparse_keys, getattr_keys

_REPO = Path(__file__).resolve().parent.parent


def test_inference_getattr_reads_are_declared_flags():
    """``getattr(args, …)`` in generation.py ⊆ inference argparse flags."""
    from library.inference import args as iargs

    declared = declared_argparse_keys(iargs.build_parser())
    reads = getattr_keys(_REPO / "library/inference/generation.py", receiver="args")
    assert_closed(reads, declared, what="inference args (generation.py)")


def test_train_getattr_reads_are_declared_flags():
    """``getattr(args, …)`` in train.py ⊆ train argparse flags.

    Two intentional non-flag reads are allowlisted:

    * ``_network_kwargs`` — private cache set by ``resolve_network_kwargs``,
      never an argparse flag.
    * ``sampler`` — the *training-time* M1 noise-sampler registry key. There is
      deliberately no ``--sampler`` train flag (the inference ``--sampler`` is a
      different concept), so this read always resolves to ``"default"``. Pinned
      here so the fallback stays a conscious choice, not silent drift.
    """
    import train as trainmod

    declared = declared_argparse_keys(trainmod.setup_parser())
    reads = getattr_keys(_REPO / "train.py", receiver="args")
    assert_closed(
        reads,
        declared,
        what="train args (train.py)",
        allow_unregistered=frozenset({"_network_kwargs", "sampler"}),
    )
