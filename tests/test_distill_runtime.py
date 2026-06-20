"""library/training/distill_runtime.py — the shared distill run-shell helpers.

Device/dtype, single-prompt slicing, the free-fit dynamic-seq guard, and snapshot
writing are correctness policies the three bespoke loops share; pin them here so a
change lands once instead of drifting across copies.
"""

import logging

import pytest

from library.training.distill_runtime import (
    apply_single_prompt_slice,
    create_tb_writer,
    ensure_dynamic_seq_for_freefit,
    resolve_device_dtype,
    write_config_snapshot,
)

log = logging.getLogger("test")


def test_resolve_device_dtype_default_is_cuda_bf16():
    import torch

    dev, dt = resolve_device_dtype()
    assert dev.type == "cuda"
    assert dt is torch.bfloat16


def test_resolve_device_dtype_overrides():
    import torch

    dev, dt = resolve_device_dtype(dtype="fp32", device="cpu")
    assert dev.type == "cpu"
    assert dt is torch.float32


def test_resolve_device_dtype_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_device_dtype(dtype="int8")


class _FakeDataset:
    def __init__(self, n):
        # samples are (latent_path, te_path) tuples; only [0] is read for logging.
        self.samples = [
            (f"/cache/img{i}_anima.npz", f"/cache/img{i}_te.st") for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)


def test_single_prompt_slice_pins_one_sample():
    ds = _FakeDataset(5)
    only = apply_single_prompt_slice(ds, 2, logger=log)
    assert len(ds.samples) == 1
    assert ds.samples[0] == only
    assert "img2" in only[0]


def test_single_prompt_slice_wraps_modulo():
    ds = _FakeDataset(3)
    only = apply_single_prompt_slice(ds, 7, logger=log)  # 7 % 3 == 1
    assert "img1" in only[0]
    assert len(ds.samples) == 1


# Canonical 1024-tier token counts (4032/4200).
_ON_TABLE = {4032, 4200}


def test_freefit_guard_noop_when_already_dynamic():
    assert ensure_dynamic_seq_for_freefit({999999}, True, logger=log) is True


def test_freefit_guard_forces_dynamic_for_on_table_counts():
    # Free-fit is the only resize mode now, so the guard forces dynamic_seq
    # unconditionally — even when every count happens to land on the old table.
    assert ensure_dynamic_seq_for_freefit(_ON_TABLE, False, logger=log) is True


def test_freefit_guard_forces_dynamic_for_offtable_counts():
    assert ensure_dynamic_seq_for_freefit({999999}, False, logger=log) is True


def test_freefit_guard_empty_pool_forces_dynamic():
    assert ensure_dynamic_seq_for_freefit(set(), False, logger=log) is True


def test_write_config_snapshot_success(tmp_path):
    p = tmp_path / "run" / "cfg.snapshot.toml"
    p.parent.mkdir()
    assert write_config_snapshot(p, "k = 1\n", logger=log) is True
    assert p.read_text() == "k = 1\n"


def test_write_config_snapshot_failure_is_warned_not_raised(tmp_path):
    # parent dir does not exist → OSError is swallowed and reported as False
    p = tmp_path / "missing" / "cfg.toml"
    assert write_config_snapshot(p, "x", logger=log) is False


def test_create_tb_writer_disabled_returns_none():
    w, run_log = create_tb_writer("/tmp/whatever", "cfg", enabled=False, logger=log)
    assert w is None and run_log is None


def test_create_tb_writer_enabled_makes_timestamped_dir(tmp_path):
    pytest.importorskip("torch.utils.tensorboard")
    w, run_log = create_tb_writer(
        str(tmp_path / "logs"), "a: 1", enabled=True, logger=log
    )
    try:
        assert run_log is not None and run_log.exists()
        assert run_log.parent.name == "logs"
    finally:
        if w is not None:
            w.close()
