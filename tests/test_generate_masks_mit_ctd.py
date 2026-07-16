from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.preprocess import generate_masks_mit


class _FakeCvNet:
    def __init__(self) -> None:
        self.input = None

    def getUnconnectedOutLayersNames(self):
        return ["blocks", "segmentation"]

    def setInput(self, value) -> None:
        self.input = value

    def forward(self, output_names):
        assert output_names == ["blocks", "segmentation"]
        return (np.ones((1, 1, 7)), np.ones((1, 1, 2, 2)))


def test_load_ctd_explicit_cpu_does_not_import_onnxruntime(monkeypatch):
    net = _FakeCvNet()
    monkeypatch.setattr(generate_masks_mit.cv2.dnn, "readNetFromONNX", lambda path: net)

    class ImportTrap:
        def find_spec(self, fullname, path, target=None):
            if fullname == "onnxruntime":
                raise AssertionError("CPU mode must not import onnxruntime")
            return None

    trap = ImportTrap()
    sys.meta_path.insert(0, trap)
    sys.modules.pop("onnxruntime", None)
    try:
        forward = generate_masks_mit._load_ctd("detector.onnx", device="cpu")
    finally:
        sys.meta_path.remove(trap)

    outputs = forward(np.zeros((1024, 1024, 3), dtype=np.uint8))

    assert net.input is not None
    assert len(outputs) == 2


def test_load_ctd_uses_cuda_provider(monkeypatch):
    class FakeSession:
        def __init__(self, path, providers):
            assert path == "detector.onnx"
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.feed = None

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def run(self, output_names, feed):
            assert output_names is None
            self.feed = feed
            return [np.ones((1, 1, 7))]

    fake_ort = SimpleNamespace(
        preload_dlls=lambda: None,
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        generate_masks_mit.cv2.dnn,
        "readNetFromONNX",
        lambda path: (_ for _ in ()).throw(AssertionError("unexpected CPU fallback")),
    )

    forward = generate_masks_mit._load_ctd("detector.onnx", device="cuda")
    outputs = forward(np.full((1024, 1024, 3), 255, dtype=np.uint8))

    assert len(outputs) == 1


def test_load_ctd_falls_back_when_cuda_provider_is_unavailable(monkeypatch, capsys):
    class FakeSession:
        def __init__(self, path, providers):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        preload_dlls=lambda: None,
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    net = _FakeCvNet()
    monkeypatch.setattr(generate_masks_mit.cv2.dnn, "readNetFromONNX", lambda path: net)

    forward = generate_masks_mit._load_ctd("detector.onnx", device="cuda")
    outputs = forward(np.zeros((1024, 1024, 3), dtype=np.uint8))

    assert len(outputs) == 2
    assert "CUDAExecutionProvider unavailable" in capsys.readouterr().out


def test_ctd_text_boxes_maps_canvas_coordinates_to_image(monkeypatch):
    blocks = np.array([[[512.0, 256.0, 512.0, 256.0, 1.0, 0.9, 0.1]]], dtype=np.float32)
    segmentation = np.ones((1, 1, 1024, 1024), dtype=np.float32)
    monkeypatch.setattr(
        generate_masks_mit.cv2.dnn,
        "NMSBoxes",
        lambda boxes, scores, conf_th, nms_th: np.array([0]),
    )

    boxes = generate_masks_mit._ctd_text_boxes(
        lambda canvas: [blocks, segmentation],
        np.zeros((100, 200, 3), dtype=np.uint8),
    )

    assert boxes == [(50, 25, 150, 75)]


def test_ctd_gate_keeps_whole_overlapping_components():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[5:7, 5:7] = 1

    gated = generate_masks_mit._keep_mask_components_in_boxes(mask, [(2, 2, 3, 3)])

    expected = np.zeros_like(mask)
    expected[1:4, 1:4] = 1
    np.testing.assert_array_equal(gated, expected)


def test_ctd_gate_drops_every_component_when_no_boxes_match():
    mask = np.ones((4, 4), dtype=np.uint8)

    gated = generate_masks_mit._keep_mask_components_in_boxes(mask, [])

    assert not gated.any()


def test_masking_task_enables_ctd_gate_by_default(monkeypatch):
    from scripts.tasks import masking

    calls = []
    monkeypatch.delenv("MIT_CTD_GATE", raising=False)
    monkeypatch.setattr(masking, "run", lambda command: calls.append(command))

    masking._run_mit(Path("images"), Path("masks"), [])

    assert "--ctd-gate" in calls[0]
    assert "--no-ctd-gate" not in calls[0]


def test_masking_task_can_disable_ctd_gate(monkeypatch):
    from scripts.tasks import masking

    calls = []
    monkeypatch.setenv("MIT_CTD_GATE", "0")
    monkeypatch.setattr(masking, "run", lambda command: calls.append(command))

    masking._run_mit(Path("images"), Path("masks"), [])

    assert "--no-ctd-gate" in calls[0]
    assert "--ctd-gate" not in calls[0]
