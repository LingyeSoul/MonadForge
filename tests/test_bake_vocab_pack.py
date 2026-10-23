"""``scripts/toolkits/bake_vocab_pack.py`` — folding a wake-line ext-row delta
into a vocab pack.

Invariants:

* **Bake equals hook** (the baked pack serves what the base pack + the
  ``ExtDelta`` hook serve) is checked beside that hook, in
  ``project/cjk_anima_reseed/tests/test_bake_equals_hook.py``.
* **Untouched rows are byte-identical**, the routing maps are unchanged, and
  only the summed rows are re-tiered in ``provenance``.
* An id outside the base table refuses instead of growing the table.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.anima.ext_vocab import T5_TABLE_SIZE, pack_digest


from scripts.toolkits import bake_vocab_pack as bvp

REPO = Path(__file__).resolve().parents[1]

DIM = 16
ROWS = 8


def _base():
    g = torch.Generator().manual_seed(0)
    table = torch.randn(ROWS, DIM, generator=g) * 200.0
    mapping = {
        "rows": ROWS,
        "qwen": {str(100 + i): i for i in range(ROWS)},
        "route": {"ranges": [[0x3040, 0x30FF]]},
        "provenance": ["mapped"] * ROWS,
    }
    return table, mapping


def _delta(ids, row_scale=197.0):
    g = torch.Generator().manual_seed(1)
    return {
        "path": "synthetic",
        "arm": "rows",
        "ext_ids": list(ids),
        "raw": torch.randn(len(ids), DIM, generator=g),
        "row_scale": row_scale,
        "arm_tag": "t",
        "data_tag": "d",
        "train_steps": 1,
        "units": ["kana"],
        "init_rows": None,
    }


def _adapter():
    torch.manual_seed(2)
    return SimpleNamespace(embed=torch.nn.Embedding(T5_TABLE_SIZE, DIM))


def test_bake_formula_and_provenance():
    table, mapping = _base()
    delta = _delta([1, 5])
    baked, m, summary = bvp.bake(table, mapping, delta, scale=0.5)
    idx = torch.tensor([1, 5])
    expect = table[idx] + delta["raw"] * (197.0 * 0.5)
    assert torch.equal(baked[idx], expect)
    others = torch.tensor([0, 2, 3, 4, 6, 7])
    assert torch.equal(baked[others], table[others])
    assert m["provenance"] == [
        "mapped",
        "render",
        "mapped",
        "mapped",
        "mapped",
        "render",
        "mapped",
        "mapped",
    ]
    assert m["qwen"] == mapping["qwen"] and m["route"] == mapping["route"]
    assert mapping["provenance"] == ["mapped"] * ROWS, "base mapping mutated"
    assert summary["rows"] == 2 and summary["scale"] == 0.5
    assert pack_digest(baked, m) != pack_digest(table, mapping)


def test_bake_refuses_ids_outside_table():
    table, mapping = _base()
    with pytest.raises(ValueError, match="outside"):
        bvp.bake(table, mapping, _delta([2, ROWS]))
