from __future__ import annotations

import threading
import time

import pytest
import torch
from torch import nn

from library.runtime.offloading import ModelOffloader, swap_weight_devices_no_cuda


def _blocks(n: int = 3):
    return nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(n)])


def test_wait_for_all_drains_pending_transfers(monkeypatch):
    blocks = _blocks()
    offloader = ModelOffloader(blocks, blocks_to_swap=1, device=torch.device("cpu"))
    completed: list[int] = []
    lock = threading.Lock()

    def fake_swap(block_to_cpu, block_to_cuda):
        time.sleep(0.01)
        with lock:
            completed.append(id(block_to_cuda))

    monkeypatch.setattr(offloader, "swap_weight_devices", fake_swap)
    offloader._submit_move_blocks(blocks, 0, 2)
    offloader._submit_move_blocks(blocks, 1, 0)
    offloader.wait_for_all()

    assert offloader.futures == {}
    assert len(completed) == 2
    offloader.thread_pool.shutdown(wait=True)


def test_wait_for_all_propagates_worker_exception(monkeypatch):
    blocks = _blocks()
    offloader = ModelOffloader(blocks, blocks_to_swap=1, device=torch.device("cpu"))

    def fail_swap(*_args):
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(offloader, "swap_weight_devices", fail_swap)
    offloader._submit_move_blocks(blocks, 0, 2)
    with pytest.raises(RuntimeError, match="transfer failed"):
        offloader.wait_for_all()
    assert offloader.futures == {}
    offloader.thread_pool.shutdown(wait=True)


def test_cpu_swap_skips_trainable_adapter_parameters():
    left = nn.Linear(2, 2, bias=False)
    right = nn.Linear(2, 2, bias=False)
    # The paired module has one frozen base weight and one trainable adapter-like
    # parameter with the same shape; the latter must never be moved by swapping.
    left.weight.requires_grad_(False)
    right.weight.requires_grad_(True)
    before = right.weight.detach().clone()

    swap_weight_devices_no_cuda(torch.device("cpu"), left, right)

    assert right.weight.requires_grad is True
    assert torch.equal(right.weight.detach(), before)
