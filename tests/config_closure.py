"""Shared helpers for *config-key closure* tests.

H1 and H2 in ``docs/findings/entanglement_audit_high_severity.md`` are the same
smell: a config key is threaded by *string* from a declaration site (a kwarg
allowlist, an argparse flag) to a read site (``kwargs.get("k")`` /
``getattr(args, "k")``) with no link between the two — so renaming or dropping
the declaration leaves the read returning a default, and the feature silently
no-ops. No runtime error, tests stay green.

These helpers turn that silent drift into a loud, targeted CI failure: AST-scan
the read sites for literal keys and assert they are a subset of the declared
keys (a small explicit allowlist covers intentional non-declared reads — private
caches, deliberate always-default fallbacks, back-compat aliases).

Used by:
* ``test_network_registry.py``      — H1 (``kwargs.get`` ⊆ ``NETWORK_KWARGS``)
* ``test_inference_arg_closure.py``  — H2 (``getattr(args, …)`` ⊆ inference flags
                                       and ⊆ train flags)
"""

from __future__ import annotations

import ast
from pathlib import Path


def attr_get_keys(path: Path, receiver: str = "kwargs") -> set[str]:
    """Literal string keys read via ``<receiver>.get("literal"[, default])``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == receiver
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def getattr_keys(path: Path, receiver: str = "args") -> set[str]:
    """Literal string keys read via ``getattr(<receiver>, "literal"[, default])``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == receiver
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            keys.add(node.args[1].value)
    return keys


def declared_argparse_keys(parser) -> set[str]:
    """Every flag name + dest an ``ArgumentParser`` exposes on its namespace."""
    declared: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                declared.add(opt[2:].replace("-", "_"))
        if action.dest:
            declared.add(action.dest)
    return declared


def assert_closed(
    reads: set[str],
    declared: set[str],
    *,
    what: str,
    allow_unregistered: frozenset[str] = frozenset(),
    check_dead: bool = False,
    allow_dead: frozenset[str] = frozenset(),
) -> None:
    """Assert ``reads ⊆ declared`` (and optionally ``declared ⊆ reads``).

    Args:
        reads: literal keys a consumer reads by string.
        declared: keys some declaration site exposes.
        what: short label for the failure message.
        allow_unregistered: read keys intentionally not declared (private
            caches, deliberate always-default fallbacks, aliases).
        check_dead: also assert no declared key goes unread.
        allow_dead: declared keys intentionally unread (e.g. write-only).
    """
    unregistered = reads - declared - allow_unregistered
    assert not unregistered, (
        f"[{what}] keys read by string but not declared "
        f"(silently inert / always-default on drift): {sorted(unregistered)}"
    )
    if check_dead:
        dead = declared - reads - allow_dead
        assert not dead, (
            f"[{what}] declared keys no consumer reads (dead): {sorted(dead)}"
        )
