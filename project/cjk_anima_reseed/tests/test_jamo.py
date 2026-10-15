"""The jamo factor rows (``reseed/jamo.py``): the syllable code, KS X 1001,
and the composition the trainer writes into ``Rows.delta.raw``."""

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from reseed import jamo as J  # noqa: E402


def test_code_and_sets():
    assert J.decompose("가") == (0, 0, 0)
    assert J.decompose("없") == (11, 4, 18)  # ㅇ ㅓ ㅄ
    assert all(J.compose(*J.decompose(s)) == s for s in J.ALL)
    assert len(J.ALL) == 11172 and len(J.KSX1001) == 2350
    assert "쀼" in J.KSX1001 and "뷁" not in J.KSX1001
    assert Counter(map(J.layout, J.KSX1001)) == {
        "VF": 1069,
        "HF": 585,
        "CF": 347,
        "V": 149,
        "C": 109,
        "H": 91,
    }
    assert len(set().union(*map(J.cells, J.KSX1001))) == 112
    assert [J.layout(s) for s in "가고과각곡곽"] == ["V", "H", "C", "VF", "HF", "CF"]


def _rows(ids, frozen, d=8):
    delta = SimpleNamespace(
        ext_ids=ids, raw=torch.nn.Parameter(torch.randn(len(ids), d))
    )
    return SimpleNamespace(
        delta=delta, frozen_mask=torch.tensor([i in frozen for i in ids]), params=None
    )


def test_compose_and_grad():
    syl = {10: "가", 11: "각", 12: "고"}
    rows = _rows([5, 10, 11, 12], frozen={5})
    held = rows.delta.raw.detach()[0].clone()
    jm = J.Jamo(rows, syl, lr=1e-3)
    assert rows.params[0]["params"] == [jm.b, jm.C, jm.V, jm.F]
    assert J.Jamo.N_VECTORS == 106
    with torch.no_grad():
        for t in (jm.b, jm.C, jm.V, jm.F):
            t.normal_()
    jm.apply()
    raw = rows.delta.raw
    assert torch.equal(raw[0], held)  # the frozen context row kept
    F = torch.cat([torch.zeros_like(jm.F[:1]), jm.F])
    # 가 = ㄱ (vertical) + ㅏ + no final; 각 adds the final ㄱ; 고 is horizontal
    want = jm.b[0] + jm.C[0, 0] + jm.V[0]
    assert torch.allclose(raw[1], want)
    assert torch.allclose(raw[2], want + F[1])
    assert torch.allclose(raw[3], jm.b[0] + jm.C[0, 1] + jm.V[8])
    raw[1:].sum().backward()
    assert jm.b.grad is not None and float(jm.b.grad.abs().sum()) > 0
    assert float(jm.C.grad[0, 2].abs().sum()) == 0  # no compound-vowel syllable
    assert rows.delta.raw is not jm.free and not jm.free.requires_grad


def test_partial_forms():
    # 어 less ㅇ → the medial ㅓ; 언 less ㅇ ㅓ → the final ㄴ (cjk_ko, 10-12)
    assert len(J.PART_JUNG) == 21 and len(J.PART_JONG) == 27
    assert J.PART_SPELL["\u1165"] == "\u115f\u1165"
    assert J.PART_SPELL["\u11ab"] == "\u115f\u1160\u11ab"
    assert J.is_part("\u1165") and not J.is_part("ㅓ") and not J.is_part("어")
    assert J.codes(["\u1165", "\u11ab"]).tolist() == [
        [J.N_CHO, 0, 4, 0],
        [J.N_CHO, 0, J.N_JUNG, 4],
    ]
    syl = {10: "어", 11: "\u1165", 12: "\u11ab", 13: "언"}
    rows = _rows([5, 10, 11, 12, 13], frozen={5})
    jm = J.Jamo(rows, syl, lr=1e-3)
    with torch.no_grad():
        for t in (jm.b, jm.C, jm.V, jm.F):
            t.normal_()
    jm.apply()
    raw = rows.delta.raw
    b, C, V, F = jm.b[0], jm.C, jm.V, jm.F
    assert torch.allclose(raw[1], b + C[11, 0] + V[4])  # 어
    assert torch.allclose(raw[2], b + V[4])  # ᅥ: no initial
    assert torch.allclose(raw[3], b + F[3])  # ᆫ: no initial, no vowel
    assert torch.allclose(raw[4], raw[1] + F[3])  # 언
    raw[2:4].sum().backward()
    assert float(jm.C.grad.abs().sum()) == 0  # a partial form never reaches C
    assert float(jm.V.grad[4].abs().sum()) > 0 and float(jm.F.grad[3].abs().sum()) > 0


def test_merge_all():
    syl = {10: "가", 11: "각"}
    rows = _rows([5, 10, 11], frozen={5})
    jm = J.Jamo(rows, syl, lr=1e-3)
    with torch.no_grad():
        jm.V.normal_()
        jm.apply()
    sd = {
        "delta": {
            "ext_ids": [5, 10, 11],
            "raw": rows.delta.raw.detach().clone(),
            "row_scale": 1.0,
        }
    }
    n = jm.merge_all(sd, {"가": 10, "각": 11, "나": 20, "갸": 3})
    assert n == 2 and sd["delta"]["ext_ids"] == [3, 5, 10, 11, 20]
    raw = sd["delta"]["raw"]
    assert torch.allclose(raw[4], raw[2])  # 나 = 가 (C[ㄴ] still 0)
    assert torch.allclose(raw[0], jm.V[2].detach())  # 갸: ㅑ


def test_residual_and_init_state():
    syl = {10: "가", 11: "각"}
    rows = _rows([5, 10, 11], frozen={5})
    rows.row_scale = 2.0
    src = {"b": torch.ones(1, 8), "C": torch.zeros(19, 3, 8), "V": torch.zeros(21, 8)}
    src["F"] = torch.zeros(27, 8)
    jm = J.Jamo(rows, syl, lr=1e-3, residual=True, pull=0.5)
    assert rows.params[0]["params"] == [jm.b, jm.C, jm.V, jm.F, jm.r]
    jm.init_state(src, row_scale=4.0)  # rescaled into this run's row_scale
    assert torch.allclose(jm.b.detach(), torch.full((1, 8), 2.0))
    with torch.no_grad():
        jm.r.normal_()
    jm.apply()
    raw = rows.delta.raw
    assert torch.allclose(raw[1], jm.b[0] + jm.r[0])  # composition + r
    assert torch.allclose(jm.penalty(), 0.5 * (jm.r**2).sum(1).mean())
    sd = {
        "delta": {"ext_ids": [5, 10, 11], "raw": raw.detach().clone(), "row_scale": 2.0}
    }
    jm.merge_all(sd, {"가": 10, "각": 11, "나": 20})
    got = sd["delta"]["raw"]
    assert torch.allclose(got[1], raw[1].detach())  # a trained row keeps its r
    assert torch.allclose(got[3], jm.b[0].detach())  # 나: the composition alone


def test_own_rows():
    syl = {10: "가", 11: "각"}
    rows = _rows([5, 10, 11, 12], frozen={5})  # 12: no syllable (a compat jamo)
    held = rows.delta.raw.detach().clone()
    jm = J.Jamo(rows, syl, lr=1e-3)
    assert jm.n_own == 1 and rows.params[0]["params"][-1] is jm.free
    with torch.no_grad():
        jm.V.normal_()
    jm.apply()
    raw = rows.delta.raw
    assert torch.equal(raw[0], held[0]) and torch.equal(raw[3], held[3])
    raw.sum().backward()
    g = jm.free.grad
    assert float(g[:3].abs().sum()) == 0 and float(g[3].abs().sum()) > 0
    assert float(jm.V.grad.abs().sum()) > 0


def test_counterfactual_plan():
    # 언 less ㅇ → ㅓ ㄴ, less ㅓ → ㅇ ㄴ, less ㄴ → 어 (a row unless held out)
    from reseed.counterfactual import plan

    extra, forms = plan("언안각", held="아")
    assert extra == ["어", "가"]
    by = {(f["drop"], tuple(f["code"])): f["src"] for f in forms.values()}
    assert by["cho", (J.N_CHO, 0, 4, 4)] == ["언"]
    assert by["jung", (11, 0, J.N_JUNG, 4)] == ["언", "안"]  # ㅇ (vertical) + ㄴ
    assert len(forms) == 5 and not set(forms) & set(J.KSX1001)
    syl = {10: "언", **{20 + k: tuple(f["code"]) for k, f in enumerate(forms.values())}}
    rows = _rows(sorted(syl), frozen=set())
    jm = J.Jamo(rows, syl, lr=1e-3)
    with torch.no_grad():
        for t in (jm.b, jm.C, jm.V, jm.F):
            t.normal_()
    jm.apply()
    raw = dict(zip(sorted(syl), rows.delta.raw))
    host = next(20 + k for k, f in enumerate(forms.values()) if f["src"] == ["언", "안"])
    assert torch.allclose(raw[10] - raw[host], jm.V[4])  # 언 − (ㅇ ㄴ) = ㅓ
