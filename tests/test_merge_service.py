from __future__ import annotations

import torch

from library.anima.merge import pick_latest_adapter
from webui.services import merge_service


def test_scan_adapter_reads_torch_weight_formats(tmp_path):
    for extension in (".ckpt", ".pt"):
        path = tmp_path / f"adapter{extension}"
        torch.save(
            {
                "layer.lora_down.weight": torch.ones(2, 4),
                "layer.lora_up.weight": torch.ones(4, 2),
            },
            path,
        )

        result = merge_service.scan_adapter(str(path))

        assert result["verdict"] == "ok"
        assert result["counts"]["lora_down"] == 1


def test_pick_latest_adapter_uses_manifest_selected_torch_weight(tmp_path):
    run = tmp_path / "adapter"
    run.mkdir()
    selected = run / "adapter.pt"
    selected.write_bytes(b"placeholder")
    (run / "adapter-step00000001.ckpt").write_bytes(b"old")
    (run / "run_manifest.json").write_text(
        '{"final_weight": "adapter.pt"}', encoding="utf-8"
    )

    assert pick_latest_adapter(tmp_path) == selected
