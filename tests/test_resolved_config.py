"""library/config/resolved.py — CLI/TOML/default precedence + snapshot rendering.

Pins the precedence rule the bespoke distill schemas share (a precedence bug is a
DX bug: ``ARGS=``/``PRESET=`` must mean the same thing across every loop) and the
provenance-snapshot / TB-summary rendering.
"""

from dataclasses import dataclass

import pytest

from library.config.resolved import (
    dataclass_snapshot_toml,
    dataclass_tb_text,
    load_toml,
    pick,
)

CFG = {
    "data_dir": "from_toml",
    "optim": {"lr": 1e-4, "warmup": 0.02},
    "network": {"rank": 48},
}


def test_pick_cli_override_wins():
    assert pick("from_cli", CFG, "data_dir", "default") == "from_cli"
    assert pick(2e-5, CFG, "optim.lr", 1.0) == 2e-5


@pytest.mark.parametrize("sentinel", [None, -1, -1.0])
def test_pick_sentinel_falls_to_toml(sentinel):
    assert pick(sentinel, CFG, "optim.lr", 1.0) == 1e-4


def test_pick_missing_toml_key_uses_default():
    assert pick(None, CFG, "optim.nonexistent", 7) == 7
    assert pick(-1, CFG, "no.such.path", 3) == 3


def test_pick_real_zero_overrides_toml():
    # 0 / 0.0 are NOT sentinels — an explicit 0 must beat the TOML value.
    assert pick(0.0, CFG, "optim.lr", 1.0) == 0.0
    assert pick(0, CFG, "network.rank", 99) == 0


def test_pick_bool_flags_pass_through():
    # store_true/false land True/False; neither equals the -1 sentinels.
    assert pick(True, CFG, "network.rank", 99) is True
    assert pick(False, CFG, "network.rank", 99) is False


def test_pick_custom_sentinels():
    assert pick(-2, CFG, "network.rank", 0, sentinels=(-2,)) == 48
    assert pick(5, CFG, "network.rank", 0, sentinels=(-2,)) == 5


def test_load_toml_roundtrip(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text('data_dir = "x"\n[optim]\nlr = 0.001\n', encoding="utf-8")
    cfg = load_toml(str(p))
    assert cfg["data_dir"] == "x"
    assert cfg["optim"]["lr"] == 0.001


@dataclass(frozen=True)
class _Cfg:
    a: int
    b: str
    c: float
    opt: "str | None"


def test_snapshot_toml_dumps_all_fields_and_source():
    text = dataclass_snapshot_toml(
        _Cfg(1, "x", 2.5, None), title="title here", source_config="some.toml"
    )
    assert "# title here" in text
    assert "source config: some.toml" in text
    assert "a = 1" in text
    assert 'b = "x"' in text
    assert "c = 2.5" in text
    # None has no TOML null → recorded as an (unset) comment, not dropped.
    assert "opt = (unset)" in text
    assert "\nopt =" not in text  # not emitted as a real key


def test_snapshot_toml_reparses():
    # the rendered snapshot is valid TOML and recovers the (non-None) fields.
    import tomllib

    text = dataclass_snapshot_toml(_Cfg(1, "x", 2.5, "set"), title="t")
    assert tomllib.loads(text) == {"a": 1, "b": "x", "c": 2.5, "opt": "set"}


def test_tb_text_full_is_declaration_order():
    txt = dataclass_tb_text(_Cfg(1, "x", 2.5, None))
    assert txt == "a: 1  \nb: x  \nc: 2.5  \nopt: None"


def test_tb_text_include_selects_subset_in_order():
    txt = dataclass_tb_text(_Cfg(1, "x", 2.5, None), include=["c", "a"])
    assert txt == "c: 2.5  \na: 1"


def test_tb_text_exclude_drops_keys():
    txt = dataclass_tb_text(_Cfg(1, "x", 2.5, None), exclude=["b", "opt"])
    assert txt == "a: 1  \nc: 2.5"
