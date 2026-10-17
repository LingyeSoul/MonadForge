"""Configuration and construction contracts across the Turbo module split.

Only argparse/dataclasses and AST are used; no model, tensors or training run.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from scripts.distill_turbo.config import (
    build_argparser,
    resolve_config,
    snapshot_toml_text,
)


@pytest.mark.parametrize(
    "flag,field,value",
    [
        ("--repa_weight", "repa_weight", "0.05"),
        ("--mean_var_weight", "mean_var_weight", "0.01"),
        ("--softrank_weight", "softrank_weight", "0.05"),
        ("--cdm_weight", "cdm_weight", "0.1"),
        ("--dmad", "dmad", None),
        ("--dual_pool", "dual_pool", None),
    ],
)
def test_local_and_upstream_options_resolve_and_snapshot(flag, field, value):
    argv = [flag] if value is None else [flag, value]
    resolved = resolve_config(build_argparser().parse_args(argv), {})
    assert getattr(resolved, field) == (True if value is None else float(value))
    assert field in snapshot_toml_text(resolved)


def test_ortho_init_survives_config_and_keeps_conflict_guard():
    args = build_argparser().parse_args([])
    resolved = resolve_config(args, {"network": {"student_ortho_init": True}})
    assert resolved.student_ortho_init
    with pytest.raises(ValueError, match="student_ortho_init"):
        resolve_config(
            args,
            {
                "network": {
                    "student_ortho_init": True,
                    "student_down_init": "weight_svd",
                }
            },
        )


def test_split_modules_have_complete_config_and_context_contracts():
    cfg = resolve_config(build_argparser().parse_args([]), {})
    fields = {field.name for field in dataclasses.fields(cfg)}
    root = Path(__file__).resolve().parents[1] / "scripts/distill_turbo"
    trees = {path.name: ast.parse(path.read_text()) for path in root.glob("*.py")}
    unresolved = [
        (name, node.lineno, node.attr)
        for name, tree in trees.items()
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cfg"
        and node.attr not in fields
    ]
    assert not unresolved
    context = next(
        node
        for node in trees["setup.py"].body
        if isinstance(node, ast.ClassDef) and node.name == "RunContext"
    )
    required = {
        node.target.id
        for node in context.body
        if isinstance(node, ast.AnnAssign) and node.value is None
    }
    calls = [
        node
        for node in ast.walk(trees["setup.py"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RunContext"
    ]
    assert len(calls) == 1
    assert required <= {kw.arg for kw in calls[0].keywords}
