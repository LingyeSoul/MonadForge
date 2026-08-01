"""Training preview decode timing regression guards."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import train
from library.anima.training import _should_decode_inline


@pytest.mark.parametrize(
    ("explicit", "blocks_to_swap", "expected"),
    [
        (None, 0, False),
        (None, 20, False),
        ("auto", 0, False),
        ("auto", 20, False),
        (True, 20, True),
        (False, 0, False),
        ("true", 20, True),
        ("false", 0, False),
    ],
)
def test_sample_decode_inline_tri_state(explicit, blocks_to_swap, expected):
    args = argparse.Namespace(
        sample_decode_inline=explicit,
        blocks_to_swap=blocks_to_swap,
    )
    assert _should_decode_inline(args) is expected


def test_deferred_sample_decode_cleanup_is_best_effort(monkeypatch):
    calls = []

    class FakeUnet:
        def to(self, device):
            calls.append(("unet", device))

    class FakeAccelerator:
        device = "cuda"

        @staticmethod
        def unwrap_model(model):
            return model

    def fake_decode(accelerator, args, vae, *, progress_sink):
        calls.append(("decode", accelerator, args, vae, progress_sink))

    monkeypatch.setattr(train, "clean_memory_on_device", lambda device: None)
    monkeypatch.setattr(train.anima_train_utils, "decode_pending_samples", fake_decode)
    accelerator = FakeAccelerator()
    args = SimpleNamespace(sample_prompts="prompts.txt")
    vae = object()
    sink = object()

    train._decode_pending_samples_at_exit(
        accelerator,
        args,
        vae,
        FakeUnet(),
        is_main_process=True,
        progress_sink=sink,
    )

    assert calls[0] == ("unet", "cpu")
    assert calls[1] == ("decode", accelerator, args, vae, sink)


def test_deferred_sample_decode_continues_when_unet_offload_fails(monkeypatch):
    class BrokenUnet:
        @staticmethod
        def to(_device):
            raise RuntimeError("offload failed")

    class FakeAccelerator:
        device = "cuda"

        @staticmethod
        def unwrap_model(model):
            return model

    decoded = []
    monkeypatch.setattr(train, "clean_memory_on_device", lambda device: None)
    monkeypatch.setattr(
        train.anima_train_utils,
        "decode_pending_samples",
        lambda *args, **kwargs: decoded.append((args, kwargs)),
    )

    train._decode_pending_samples_at_exit(
        FakeAccelerator(),
        SimpleNamespace(sample_prompts="prompts.txt"),
        object(),
        BrokenUnet(),
        is_main_process=True,
    )

    assert len(decoded) == 1
