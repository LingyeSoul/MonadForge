"""Inductor flags must survive fresh backward-compiler contexts."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import Context

import pytest
import torch._inductor.config as config

from library.runtime.dynamo import pin_inductor_flag


NAME = "triton.mix_order_reduction"


@pytest.fixture
def flag_entry():
    entry = config._config[NAME]
    previous_default = entry.default
    token = entry.user_override.set(True)
    entry.default = True
    try:
        yield entry
    finally:
        entry.default = previous_default
        entry.user_override.reset(token)


@pytest.mark.parametrize("value", [False, True])
def test_pin_reaches_fresh_context_and_thread(flag_entry, value):
    assert pin_inductor_flag(NAME, value) is value
    assert getattr(config, NAME) is value
    assert flag_entry.default is value
    assert Context().run(lambda: getattr(config, NAME)) is value
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(lambda: getattr(config, NAME)).result() is value


def test_unknown_flag_fails_explicitly():
    with pytest.raises(KeyError):
        pin_inductor_flag("triton.nonexistent_test_flag", False)


def test_anima_dynamic_compile_initialization_pins_flag(flag_entry):
    from tests.test_native_flatten import _tiny_anima

    model = _tiny_anima()
    model.compile_blocks(backend="eager", dynamic_seq=True, seq_range=(1024, 2209))
    assert config.triton.mix_order_reduction is False
    assert Context().run(lambda: config.triton.mix_order_reduction) is False
