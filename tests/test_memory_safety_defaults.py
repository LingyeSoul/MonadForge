"""Regression guards for the training process memory defaults."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import toml
import torch

import train
from library.training.loop import release_text_encoder_handles


def test_parser_uses_single_process_loader_by_default():
    args = train.setup_parser().parse_args([])
    assert args.max_data_loader_n_workers == 0


def test_base_config_defers_preview_decode_and_loader_workers():
    config = toml.load("configs/base.toml")
    assert config["max_data_loader_n_workers"] == 0
    assert config["preview"]["sample_decode_inline"] == "false"


def test_release_text_encoder_handles_clears_aliasable_list():
    encoder = object()
    handles = [encoder]

    scalar, released = release_text_encoder_handles(encoder, handles)

    assert scalar is None
    assert released == []
    assert handles == []


def test_accelerator_bundle_does_not_retain_cached_only_text_encoder():
    class FakeAccelerator:
        device = torch.device("cpu")

        @staticmethod
        def prepare(*objects):
            return objects[0] if len(objects) == 1 else objects

        @staticmethod
        def unwrap_model(model):
            return model

    class FakeNetwork(torch.nn.Module):
        @staticmethod
        def prepare_grad_etc(_text_encoder, _unet):
            return None

    class FakeTextEncoder(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    trainer = train.AnimaTrainer()
    trainer.is_swapping_blocks = False
    trainer._use_unsloth_offload_checkpointing = False
    encoder = FakeTextEncoder(2, 2)
    encoder_ref = weakref.ref(encoder)
    handles = [encoder]

    bundle = trainer._prepare_with_accelerator(
        SimpleNamespace(
            cache_text_encoder_outputs=True,
            gradient_checkpointing=False,
        ),
        FakeAccelerator(),
        FakeNetwork(),
        object(),
        object(),
        object(),
        object(),
        torch.nn.Linear(2, 2),
        handles,
        handles,
        None,
        torch.float32,
        torch.float32,
        True,
        False,
        True,
    )

    assert handles == []
    assert bundle.text_encoders == []
    assert bundle.text_encoder is None
    del encoder
    gc.collect()
    assert encoder_ref() is None
