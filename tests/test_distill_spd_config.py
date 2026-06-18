"""scripts/distill_spd_config.py — CLI/TOML precedence + schedule sanity.

The resolution logic moved out of the distill_spd.py training loop into a
testable resolve_config; pin the precedence + schedule invariants here.
"""

import pytest

from scripts.distill_spd_config import build_argparser, resolve_config


def _resolve(argv, cfg):
    args = build_argparser().parse_args(argv)
    return resolve_config(args, cfg)


def test_defaults_when_cfg_empty():
    c = _resolve([], {})
    assert c.stages == [0.5, 1.0]
    assert c.transition_sigmas == [0.5]
    assert c.lr == 1e-4
    assert c.rank == 48
    assert c.val_split == 0.0


def test_toml_values_used_when_cli_sentinel():
    cfg = {"optim": {"lr": 0.002}, "network": {"rank": 64}, "iterations": 5000}
    c = _resolve([], cfg)
    assert c.lr == 0.002
    assert c.rank == 64
    assert c.iterations == 5000


def test_cli_overrides_toml():
    cfg = {"optim": {"lr": 0.002}, "network": {"rank": 64}}
    c = _resolve(["--lr", "0.01", "--rank", "16"], cfg)
    assert c.lr == 0.01
    assert c.rank == 16


def test_alpha_defaults_to_rank():
    # network.alpha sentinel -1.0 → falls to TOML network.alpha, default = rank.
    c = _resolve(["--rank", "32"], {})
    assert c.alpha == 32.0


def test_single_prompt_disables_validation():
    c = _resolve(["--single_prompt_idx", "0", "--val_split", "0.1"], {})
    assert c.val_split == 0.0


def test_boolean_optional_action_dynamic_seq():
    assert _resolve(["--no-compile_dynamic_seq"], {}).compile_dynamic_seq is False
    assert _resolve(["--compile_dynamic_seq"], {}).compile_dynamic_seq is True
    # sentinel (unset) → TOML default True
    assert _resolve([], {}).compile_dynamic_seq is True
    assert _resolve([], {"compile_dynamic_seq": False}).compile_dynamic_seq is False


def test_stages_must_end_at_one():
    with pytest.raises(ValueError, match="must end at 1.0"):
        _resolve(["--stages", "0.5", "0.8"], {})


def test_stages_must_ascend():
    with pytest.raises(ValueError, match="ascending"):
        _resolve(["--stages", "0.8", "0.5", "1.0"], {})


def test_transition_sigmas_length_checked():
    with pytest.raises(ValueError, match="must be len\\(stages\\)-1"):
        _resolve(["--stages", "0.5", "1.0", "--transition_sigmas", "0.3", "0.6"], {})


def test_stage_weights_length_checked():
    with pytest.raises(ValueError, match="must match"):
        _resolve(["--stage_weights", "1.0"], {})  # 1 weight vs 2 default stages
