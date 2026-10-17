"""All Turbo save dispatches retain preflight identity without saving tensors."""

from types import SimpleNamespace

import pytest

from library.anima.checkpoint import AnimaCheckpointLayout
from networks.methods.turbo_dmd import TurboDMDNetwork


@pytest.mark.parametrize("layout_kind", ["plain", "dual", "expert"])
def test_student_save_dispatch_preserves_authoritative_base_identity(layout_kind):
    layout = AnimaCheckpointLayout("anima", "fixture", 40, 2048, 16, "")
    captured = []
    student = SimpleNamespace(
        state_dict=lambda: {},
        save_weights=lambda file, dtype, metadata: captured.append(metadata),
    )
    fake = SimpleNamespace(
        unet=SimpleNamespace(anima_checkpoint_layout=layout, anima_base_sha256="abc"),
        student=student,
        student_div=object() if layout_kind == "dual" else None,
        student_step_expert_K=4 if layout_kind == "expert" else 0,
        _save_dual_pool=lambda file, dtype, metadata, scale: captured.append(metadata),
        _save_student_step_expert=lambda sd, file, dtype, metadata: captured.append(
            metadata
        ),
    )
    original = {"ss_new_sd_model_hash": "wrong", "custom": "kept"}
    TurboDMDNetwork.save_student(
        fake, "/not-written/fixture.safetensors", metadata=original
    )
    assert len(captured) == 1
    assert captured[0]["ss_new_sd_model_hash"] == "abc"
    assert captured[0]["ss_anima_num_blocks"] == captured[0]["ss_num_blocks"] == "40"
    assert captured[0]["ss_anima_model_channels"] == "2048"
    assert captured[0]["custom"] == "kept"
    assert original["ss_new_sd_model_hash"] == "wrong"
