"""Pre-step memory budget checks for non-zero DiT block swapping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.runtime import block_swap_budget as budget_mod


class TinyNetwork(torch.nn.Module):
    def __init__(self, count: int = 16):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.zeros(count, dtype=torch.float16))


def _args(**overrides):
    values = {
        "blocks_to_swap": 2,
        "mixed_precision": "fp16",
        "train_batch_size": 2,
        "gradient_checkpointing": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_estimate_accounts_for_adapter_optimizer_and_max_tokens(monkeypatch):
    monkeypatch.setattr(budget_mod, "_free_total", lambda _device: (4 << 30, 16 << 30))
    network = TinyNetwork(16)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)

    budget = budget_mod.estimate_block_swap_budget(
        _args(),
        model=SimpleNamespace(model_channels=64),
        network=network,
        optimizer=optimizer,
        token_budget=(3, (100, 240)),
        device="cuda:0",
    )

    assert budget.max_tokens == 240
    assert budget.trainable_params == 16
    assert budget.adapter_param_bytes == 16 * 2
    assert budget.gradient_bytes == 16 * 4
    assert budget.optimizer_state_bytes == 16 * 8
    assert budget.activation_workspace_bytes == 240 * 64 * 2 * 2 * 10
    assert budget.free_bytes == 4 << 30
    assert budget.total_bytes == 16 << 30


def test_check_rejects_over_budget_before_first_step(monkeypatch):
    monkeypatch.setattr(budget_mod, "_free_total", lambda _device: (128, 1024))
    args = _args()

    with pytest.raises(RuntimeError, match="first optimizer step was not run"):
        budget_mod.check_block_swap_budget(
            args,
            model=SimpleNamespace(model_channels=64),
            network=TinyNetwork(),
            optimizer=SimpleNamespace(),
            token_budget=(1, (64, 64)),
            device="cuda:0",
        )

    assert args.block_swap_budget["max_tokens"] == 64
    assert args.block_swap_budget["estimated_required_bytes"] > 128


def test_zero_swap_does_not_change_existing_path(monkeypatch):
    monkeypatch.setattr(budget_mod, "_free_total", lambda _device: (1, 1))
    budget = budget_mod.check_block_swap_budget(
        _args(blocks_to_swap=0),
        network=TinyNetwork(),
        token_budget=4096,
        device="cuda:0",
    )
    assert budget.blocks_to_swap == 0


def test_strict_budget_can_be_disabled_for_diagnostics(monkeypatch):
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_BUDGET_STRICT", "0")
    monkeypatch.setattr(budget_mod, "_free_total", lambda _device: (1, 1))
    budget = budget_mod.check_block_swap_budget(
        _args(),
        network=TinyNetwork(),
        token_budget=4096,
        device="cuda:0",
    )
    assert budget.strict is False


def test_trainer_checks_budget_after_optimizer_before_prepare(monkeypatch):
    import train

    parameter = torch.nn.Parameter(torch.zeros(1))

    class FakeNetwork:
        def prepare_optimizer_params(self, *_args):
            return [parameter]

    class FakeDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

        def set_current_strategies(self):
            pass

        def set_max_train_steps(self, _steps):
            pass

    optimizer = object()
    captured = {}
    monkeypatch.setattr(train, "get_optimizer", lambda *_a: ("Fake", {}, optimizer))
    monkeypatch.setattr(
        train, "get_optimizer_train_eval_fn", lambda *_a: (lambda: None, lambda: None)
    )
    monkeypatch.setattr(train, "prepare_stage_runtime", lambda *_a: None)
    monkeypatch.setattr(train, "get_scheduler_fix", lambda *_a: object())

    def fake_check(args, **kwargs):
        captured.update(kwargs)
        args.block_swap_budget = {"checked": True}

    monkeypatch.setattr(train, "check_block_swap_budget", fake_check)
    args = SimpleNamespace(
        text_encoder_lr=None,
        unet_lr=1e-4,
        learning_rate=1e-4,
        blocks_to_swap=1,
        max_data_loader_n_workers=0,
        persistent_data_loader_workers=False,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
        max_train_epochs=None,
        max_train_steps=1,
        gradient_accumulation_steps=1,
        seed=1,
    )
    accelerator = SimpleNamespace(print=lambda *_a: None, num_processes=1, device="cpu")
    trainer = train.AnimaTrainer()
    trainer._compile_token_budget = (2, (100, 200))
    model = SimpleNamespace(model_channels=64)
    network = FakeNetwork()

    trainer._setup_optimizer_and_dataloader(
        args,
        accelerator,
        network,
        model,
        FakeDataset(),
        None,
        None,
    )

    assert captured == {
        "model": model,
        "network": network,
        "optimizer": optimizer,
        "token_budget": (2, (100, 200)),
        "device": "cpu",
    }

