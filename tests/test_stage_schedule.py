"""Tests for in-process staged-resolution curriculum training."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from types import SimpleNamespace

import pytest
import torch

from library.config import schema as config_schema
from library.config.io import _flatten_toml
from library.datasets.base import BaseDataset
from library.datasets.collator import collator_class
from library.datasets.group import DatasetGroup
from library.training.checkpoints import CheckpointSaver
from library.training.metadata import add_dataset_metadata
from library.training.stage_schedule import (
    StageSpec,
    active_subset_indices_for_step,
    apply_active_subsets_to_dataset,
    count_stage_targets,
    normalize_stage_dicts,
    parse_stage_specs,
    prepare_stage_runtime,
    resolve_stage_index,
    snapshot_full_image_data,
    stage_epoch_upper_bound,
    validate_stage_specs,
    validate_stage_target_sides,
)
from library.training.staged_resolution import (
    build_stage_schedule,
    configure_staged_resolution,
)


def test_normalize_and_resolve_stage_boundaries():
    stages = normalize_stage_dicts(
        [
            {"subset_index": 0, "start_pct": 0, "end_pct": 20},
            {"subset_index": 1, "start_pct": 0.2, "end_pct": 50},
            {"subset_index": 2, "start_pct": 0.5, "end_pct": 1.0},
        ]
    )
    specs = parse_stage_specs(stages)
    assert validate_stage_specs(specs, subset_count=3) == []
    assert resolve_stage_index(specs, 0.1999) == 0
    assert resolve_stage_index(specs, 0.2) == 1
    assert resolve_stage_index(specs, 0.5) == 2
    assert resolve_stage_index(specs, 1.0) == 2


def test_validate_rejects_gap_and_bad_dataset_index():
    specs = [
        StageSpec(0, 0.0, 0.2),
        StageSpec(3, 0.3, 1.0),
    ]
    problems = validate_stage_specs(specs, subset_count=2)
    assert any("gap" in problem for problem in problems)
    assert any("subset_index=3" in problem for problem in problems)


@pytest.mark.parametrize(
    "stage",
    [
        {"subset_index": -1, "start_pct": 0, "end_pct": 1},
        {"subset_index": 0, "start_pct": "oops", "end_pct": 1},
        {"subset_index": 0, "start_pct": 0, "end_pct": 250},
    ],
)
def test_normalize_rejects_invalid_stage_values(stage):
    with pytest.raises(ValueError):
        normalize_stage_dicts([stage])


def test_enabled_empty_schedule_fails_instead_of_falling_back_to_full_dataset():
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        stage_schedule=[],
        max_train_steps=100,
    )
    with pytest.raises(ValueError, match="requires at least one stage"):
        active_subset_indices_for_step(args, 0)


def test_staged_resolution_shorthand_builds_one_continuous_schedule():
    args = SimpleNamespace(
        staged_resolution=True,
        staged_resolution_ratios="20,30,50",
        staged_resolution_base_sides="512,768,1024",
        stage_schedule_enabled=False,
        stage_schedule=None,
        max_train_steps=6000,
    )
    configure_staged_resolution(args)
    assert args.stage_schedule_enabled is True
    assert [stage["subset_index"] for stage in args.stage_schedule] == [0, 1, 2]
    assert [stage["start_pct"] for stage in args.stage_schedule] == [0.0, 0.2, 0.5]
    assert [stage["end_pct"] for stage in args.stage_schedule] == [0.2, 0.5, 1.0]
    assert active_subset_indices_for_step(args, 1199) == {0}
    assert active_subset_indices_for_step(args, 1200) == {1}
    assert active_subset_indices_for_step(args, 3000) == {2}


def test_staged_resolution_rejects_invalid_ratios_and_conflicting_schedule():
    with pytest.raises(ValueError, match="sum to 100"):
        build_stage_schedule([20, 20, 20], [512, 768, 1024])

    args = SimpleNamespace(
        staged_resolution=True,
        staged_resolution_ratios=None,
        staged_resolution_base_sides=None,
        stage_schedule_enabled=True,
        stage_schedule=[{"subset_index": 0, "start_pct": 0, "end_pct": 1}],
    )
    with pytest.raises(ValueError, match="cannot both be enabled"):
        configure_staged_resolution(args)


class _Info:
    def __init__(self, key: str, *, is_reg: bool = False):
        self.image_key = key
        self.num_repeats = 1
        self.is_reg = is_reg


class _Subset:
    def __init__(self, name: str):
        self.name = name
        self.img_count = 1
        self.num_repeats = 1
        self.color_aug = False
        self.flip_aug = False
        self.random_crop = False
        self.keep_tokens_separator = ""
        self.secondary_separator = ""
        self.enable_wildcard = False
        self.caption_prefix = None
        self.caption_suffix = None
        self.resize_interpolation = "lanczos"
        self.image_dir = name
        self.metadata_file = None
        self.class_tokens = None
        self.is_reg = False


class _BucketManager:
    def __init__(self, resos):
        self.resos = list(resos)
        self.shuffle_count = 0

    def shuffle(self):
        self.shuffle_count += 1


class _Leaf(torch.utils.data.Dataset):
    snapshot_full_image_data = BaseDataset.snapshot_full_image_data
    has_full_image_data_snapshot = BaseDataset.has_full_image_data_snapshot
    rebuild_buckets_for_subsets = BaseDataset.rebuild_buckets_for_subsets

    def __init__(self, name: str, keys: list[str], resos=((512, 512),)):
        self.subsets = [_Subset(name)]
        self.image_data = {key: _Info(key) for key in keys}
        self.image_to_subset = {key: self.subsets[0] for key in keys}
        self.num_train_images = len(keys)
        self.num_reg_images = 0
        self.batch_size = 1
        self.tag_frequency = {}
        self.bucket_info = {"buckets": {}}
        self.resize_interpolation = "lanczos"
        self._length = len(keys)
        self._target_res = None
        self._warmup_bucket_indices = None
        self._stage_active = True
        self.current_epoch = 0
        self.current_step = 0
        self.seed = 0
        self.bucket_manager = _BucketManager(resos)
        self.buckets_indices = list(keys)

    def make_buckets(self, target_res=None):
        self._target_res = target_res
        self._length = len(self.image_data)
        self.buckets_indices = list(self.image_data)
        if self.image_data and self.bucket_manager is None:
            self.bucket_manager = _BucketManager([(512, 512)])

    def set_current_epoch(self, epoch):
        return BaseDataset.set_current_epoch(self, epoch)

    def set_current_step(self, step):
        return BaseDataset.set_current_step(self, step)

    def shuffle_buckets(self):
        self.bucket_manager.shuffle()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self.buckets_indices[index]


def test_group_stage_switch_restores_full_snapshot_and_concat_length():
    members = [
        _Leaf("low", ["a1", "a2"]),
        _Leaf("mid", ["b1"]),
        _Leaf("high", ["c1", "c2", "c3"]),
    ]
    group = DatasetGroup(members)
    snapshot_full_image_data(group, force=True)
    assert count_stage_targets(group) == 3

    assert apply_active_subsets_to_dataset(group, {0})
    assert len(group) == 2
    assert set(group.image_data) == {"a1", "a2"}

    assert apply_active_subsets_to_dataset(group, {2})
    assert len(group) == 3
    assert set(group.image_data) == {"c1", "c2", "c3"}

    assert apply_active_subsets_to_dataset(group, None)
    assert len(group) == 6
    assert set(group.image_data) == {"a1", "a2", "b1", "c1", "c2", "c3"}


def test_inactive_group_member_survives_real_collator_epoch_update():
    members = [_Leaf("low", ["a"]), _Leaf("high", ["b"])]
    group = DatasetGroup(members)
    snapshot_full_image_data(group, force=True)
    assert apply_active_subsets_to_dataset(group, {0})

    collate = collator_class(SimpleNamespace(value=1), SimpleNamespace(value=0), group)
    assert collate([{"key": "a"}]) == {"key": "a"}
    assert members[0].bucket_manager.shuffle_count == 1
    assert members[1]._stage_active is False
    assert members[1].bucket_manager is not None
    assert members[1].bucket_manager.shuffle_count == 0


def test_shorthand_validates_each_dataset_row_resolution_band():
    group = DatasetGroup(
        [
            _Leaf("low", ["a"], resos=[(512, 512)]),
            _Leaf("mid", ["b"], resos=[(768, 720)]),
            _Leaf("high", ["c"], resos=[(1024, 1024)]),
        ]
    )
    stages = parse_stage_specs(build_stage_schedule())
    assert validate_stage_target_sides(group, stages, [512, 768, 1024]) == []

    group.datasets[1].bucket_manager.resos = [(512, 512)]
    problems = validate_stage_target_sides(group, stages, [512, 768, 1024])
    assert any("expects the 768px tier" in problem for problem in problems)


def test_stage_epoch_budget_accounts_for_short_later_datasets():
    stages = parse_stage_specs(build_stage_schedule([20, 30, 50], [512, 768, 1024]))
    assert stage_epoch_upper_bound(stages, 1000, [100, 50, 25]) == 28

    half_stages = parse_stage_specs(build_stage_schedule([50, 50], [512, 1024]))
    assert stage_epoch_upper_bound(half_stages, 3, [1, 100]) == 3


def test_runtime_plan_rejects_referenced_row_without_complete_batches():
    empty_batches = _Leaf("high", ["b"])
    empty_batches._length = 0
    empty_batches.buckets_indices = []
    group = DatasetGroup([_Leaf("low", ["a"]), empty_batches])
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        stage_schedule=[
            {"subset_index": 0, "start_pct": 0, "end_pct": 0.5},
            {"subset_index": 1, "start_pct": 0.5, "end_pct": 1},
        ],
        staged_resolution=False,
    )
    with pytest.raises(ValueError, match="no complete batches: 1"):
        prepare_stage_runtime(args, group)


def test_deprecated_shorthand_reports_required_dataset_rows():
    args = SimpleNamespace(
        staged_resolution=True,
        stage_schedule_enabled=True,
        stage_schedule=build_stage_schedule(),
        _stage_expected_sides=(512, 768, 1024),
    )
    group = DatasetGroup([_Leaf("only-row", ["a"])])
    with pytest.raises(ValueError, match="one fully preprocessed"):
        prepare_stage_runtime(args, group)


def test_loop_switch_rebuilds_loader_through_accelerator():
    from library.training.loop import _maybe_apply_stage_schedule

    group = DatasetGroup([_Leaf("low", ["a"]), _Leaf("high", ["b", "c"])])
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        stage_schedule=[
            {"name": "low", "subset_index": 0, "start_pct": 0, "end_pct": 0.5},
            {"name": "high", "subset_index": 1, "start_pct": 0.5, "end_pct": 1},
        ],
        staged_resolution=False,
        max_train_steps=100,
    )
    plan = prepare_stage_runtime(args, group)
    plan.dataloader_kwargs = {"batch_size": 1, "num_workers": 0}

    class _Accelerator:
        def __init__(self):
            self.prepared = []

        def prepare_data_loader(self, loader):
            self.prepared.append(loader)
            return loader

        def print(self, *args, **kwargs):
            return None

    accelerator = _Accelerator()
    state = SimpleNamespace(
        args=args,
        accelerator=accelerator,
        stage_plan=plan,
        train_dataloader=None,
        stage_index=0,
        stage_batch_cursor=7,
        global_step=50,
        initial_step=7,
        metadata={},
    )
    _maybe_apply_stage_schedule(state)
    assert state.stage_index == 1
    assert state.initial_step == 7
    assert state.stage_batch_cursor == 7
    assert len(accelerator.prepared) == 1
    assert set(group.image_data) == {"b", "c"}


def test_checkpoint_state_persists_stage_cursor(tmp_path):
    from library.training.loop import LoopState

    class _Accelerator:
        is_main_process = True

        def unwrap_model(self, model):
            return model

        def register_save_state_pre_hook(self, hook):
            self.save_hook = hook

        def register_load_state_pre_hook(self, hook):
            self.load_hook = hook

    accelerator = _Accelerator()
    network = object()
    saver = CheckpointSaver(
        args=SimpleNamespace(),
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda args: {},
        current_epoch=SimpleNamespace(value=4),
        current_step=SimpleNamespace(value=59),
    )
    generator_state = torch.Generator().manual_seed(42).get_state()
    runtime_owner = SimpleNamespace(
        stage_plan=object(),
        stage_index=1,
        stage_batch_cursor=12,
        outer_epoch_index=3,
        stage_loader_generator_state=generator_state,
    )
    saver.set_runtime_state_provider(
        lambda: LoopState.checkpoint_runtime_state(runtime_owner)
    )
    saver.register_hooks(network)
    accelerator.save_hook([network], [], str(tmp_path))

    state = json.loads((tmp_path / "train_state.json").read_text(encoding="utf-8"))
    assert state == {
        "current_epoch": 4,
        "current_step": 60,
        "stage_index": 1,
        "stage_batch_cursor": 12,
        "stage_outer_epoch": 3,
        "stage_loader_generator_state": generator_state.tolist(),
    }


def test_resumed_stage_skips_saved_batch_cursor(monkeypatch):
    import library.training.loop as loop_module

    group = DatasetGroup([_Leaf("low", ["a", "b", "c"])])
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        stage_schedule=[{"subset_index": 0, "start_pct": 0, "end_pct": 1}],
        staged_resolution=False,
        max_train_steps=1,
    )
    plan = prepare_stage_runtime(args, group)
    plan.dataloader_kwargs = {"batch_size": 1, "num_workers": 0}
    processed = []

    monkeypatch.setattr(
        loop_module,
        "_run_step",
        lambda trainer, state, batch: processed.append(batch) or torch.tensor(1.0),
    )
    monkeypatch.setattr(loop_module, "_profiler_step_begin", lambda state: None)
    monkeypatch.setattr(loop_module, "_profiler_step_end", lambda state: None)
    monkeypatch.setattr(
        loop_module, "_maybe_scale_norm", lambda state: (None, None, None, {})
    )
    monkeypatch.setattr(loop_module, "_sample_at_step", lambda *args: None)
    monkeypatch.setattr(loop_module, "_log_checkpoint_artifact", lambda *args: None)
    monkeypatch.setattr(loop_module, "_log_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(loop_module, "_maybe_run_step_validation", lambda *args: None)

    class _Accelerator:
        sync_gradients = True

        def skip_first_batches(self, loader, count):
            return list(loader)[count:]

    state = SimpleNamespace(
        args=args,
        accelerator=_Accelerator(),
        initial_step=2,
        train_dataloader=["a", "b", "c"],
        stage_plan=plan,
        stage_index=0,
        stage_batch_cursor=2,
        global_step=0,
        current_step=SimpleNamespace(value=0),
        progress_bar=SimpleNamespace(update=lambda amount: None),
        saver=SimpleNamespace(maybe_save_step=lambda *args: None),
        network=object(),
        optimizer_train_fn=lambda: None,
    )
    loop_module._run_epoch_steps(object(), state, epoch=0)
    assert processed == ["c"]
    assert state.stage_batch_cursor == 3
    assert state.global_step == 1


def test_loader_generator_state_and_cursor_reproduce_remaining_sequence():
    dataset = _Leaf("row", ["a", "b", "c", "d", "e"])
    generator = torch.Generator().manual_seed(1234)
    epoch_start_state = generator.get_state()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=lambda examples: examples[0],
    )
    iterator = iter(loader)
    consumed = [next(iterator), next(iterator)]
    uninterrupted_remaining = list(iterator)

    generator.set_state(epoch_start_state)
    resumed_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=lambda examples: examples[0],
    )
    resumed_remaining = list(islice(resumed_loader, len(consumed), None))
    assert resumed_remaining == uninterrupted_remaining


def test_dataset_metadata_uses_full_counts_after_stage_filter():
    group = DatasetGroup([_Leaf("low", ["a"]), _Leaf("high", ["b", "c"])])
    args = SimpleNamespace(
        stage_schedule_enabled=True,
        stage_schedule=[
            {"subset_index": 0, "start_pct": 0, "end_pct": 0.5},
            {"subset_index": 1, "start_pct": 0.5, "end_pct": 1},
        ],
        staged_resolution=False,
    )
    plan = prepare_stage_runtime(args, group)
    metadata = {}
    add_dataset_metadata(
        metadata,
        group,
        SimpleNamespace(train_batch_size=1),
        use_user_config=True,
        use_dreambooth_method=True,
        total_batch_size=1,
        dataset_counts=list(plan.full_dataset_counts),
    )
    rows = json.loads(metadata["ss_datasets"])
    assert [row["num_train_images"] for row in rows] == [1, 2]


def test_stage_schedule_toml_values_survive_flat_config_validation():
    config_schema.populate_schema(argparse.ArgumentParser())
    stages = build_stage_schedule()
    flattened = _flatten_toml(
        {"stage_schedule_enabled": True, "stage_schedule": stages}, strict=True
    )
    assert flattened["stage_schedule_enabled"] is True
    assert flattened["stage_schedule"] == stages
