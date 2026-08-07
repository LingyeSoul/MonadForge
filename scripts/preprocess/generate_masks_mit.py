#!/usr/bin/env python3
"""Generate text segmentation masks for training images.

Model: https://huggingface.co/a-b-c-x-y-z/Manga-Text-Segmentation-2025

By default each mask is gated by comictextdetector's text-block head
(``--ctd-gate``): only UNet++ mask components overlapping a detected text block
survive. The extra detector filters decorative line-art false positives; use
``--no-ctd-gate`` to retain the raw UNet++ mask behavior.
"""

import argparse
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm


from library.preprocess import walk_images

_ENCODER = "tu-efficientnetv2_rw_m"
_HF_REPO = "a-b-c-x-y-z/Manga-Text-Segmentation-2025"
_HF_FILENAME = "model.pth"
_CtdForward = Callable[[np.ndarray], list[np.ndarray]]


def _convert_batchnorm_to_groupnorm(module: nn.Module) -> None:
    """Replace BatchNorm2d with GroupNorm in decoder (matches training setup)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            )
        else:
            _convert_batchnorm_to_groupnorm(child)


def _load_model(model_path: str | None = None, device: str = "cuda") -> nn.Module:
    model = smp.UnetPlusPlus(
        encoder_name=_ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    _convert_batchnorm_to_groupnorm(model.decoder)

    if model_path is None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


_transform = Compose(
    [
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


@torch.no_grad()
def _detect_mask(
    model: nn.Module,
    image: np.ndarray,
    device: str = "cuda",
    text_threshold: float | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]

    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    tensor = _transform(image=image)["image"].unsqueeze(0).to(device)

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        with torch.amp.autocast("cuda"):
            logits = model(tensor)
    else:
        logits = model(tensor)

    prob_map = logits.sigmoid()[0, 0, :h, :w].cpu().numpy()

    if text_threshold is not None:
        prob_map = (prob_map > text_threshold).astype(np.float32)

    mask = (prob_map * 255).astype(np.uint8)
    return mask


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    Image.fromarray(alpha_mask, mode="L").save(path)


def _preload_nvidia_cuda_libs() -> None:
    """dlopen pip-installed nvidia CUDA runtime libs so onnxruntime's CUDA
    ExecutionProvider can resolve them.

    Torch wheels carry an RPATH into ``nvidia/*/lib`` under site-packages, but
    onnxruntime's provider library does not — on a host without a system CUDA
    install (all CUDA libs come from the ``nvidia-*-cu12`` pip packages) ORT
    fails to dlopen ``libcublasLt.so.12`` etc. and silently falls back to CPU.
    Loading the libs with ``RTLD_GLOBAL`` puts them in the loader's namespace,
    so ORT's dlopen-by-soname succeeds. No-op when CUDA libs are already
    reachable (system CUDA / LD_LIBRARY_PATH) or absent.
    """
    import ctypes
    import site

    prefixes = (
        "libcudart",
        "libcublas",
        "libcudnn",
        "libcusparse",
        "libcusolver",
        "libcurand",
        "libcufft",
        "libnvrtc",
        "libnvjitlink",
        "libcusparselt",
    )
    seen: set[str] = set()
    for base in (Path(p) for p in site.getsitepackages()):
        for so in sorted(base.glob("nvidia/*/lib/*.so*")):
            name = so.name
            if name in seen or not name.startswith(prefixes):
                continue
            seen.add(name)
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def _load_ctd(onnx_path: str, device: str = "cuda") -> _CtdForward:
    """Return a CTD forward function, preferring ONNX Runtime CUDA.

    Explicit CPU use and any unavailable or failed CUDA provider fall back to
    OpenCV DNN. Keeping the fallback independent of ONNX Runtime lets the CLI
    remain usable on non-CUDA hosts.
    """
    if device != "cpu":
        _preload_nvidia_cuda_libs()
        try:
            import onnxruntime as ort

            ort.preload_dlls()
            session = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" in session.get_providers():
                input_name = session.get_inputs()[0].name

                def forward(canvas: np.ndarray) -> list[np.ndarray]:
                    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                    return session.run(None, {input_name: blob})

                return forward
            print(
                "WARNING: onnxruntime CUDAExecutionProvider unavailable — "
                "CTD gate falls back to cv2.dnn CPU"
            )
        except Exception as exc:  # noqa: BLE001 - optional acceleration fallback
            print(
                f"WARNING: onnxruntime CUDA init failed ({exc}) — "
                "CTD gate falls back to cv2.dnn CPU"
            )

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    output_names = net.getUnconnectedOutLayersNames()

    def forward(canvas: np.ndarray) -> list[np.ndarray]:
        net.setInput(
            cv2.dnn.blobFromImage(
                canvas,
                scalefactor=1 / 255.0,
                size=(1024, 1024),
            )
        )
        return list(net.forward(output_names))

    return forward


def _ctd_text_boxes(
    ctd_forward: _CtdForward,
    image: np.ndarray,
    conf_th: float = 0.4,
    nms_th: float = 0.35,
    seg_th: float = 0.3,
    seg_cov: float = 0.03,
) -> list[tuple[int, int, int, int]]:
    """Return CTD text-block boxes in the original image coordinates."""
    h0, w0 = image.shape[:2]
    ratio = min(1024 / h0, 1024 / w0)
    nw, nh = int(round(w0 * ratio)), int(round(h0 * ratio))
    canvas = np.zeros((1024, 1024, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    outputs = ctd_forward(canvas)
    blocks = next(output for output in outputs if output.ndim == 3)[0]
    segmentation = next(
        output for output in outputs if output.ndim == 4 and output.shape[1] == 1
    )[0, 0]
    confidence = blocks[:, 4] * blocks[:, 5:].max(axis=1)
    keep = confidence > conf_th
    if not keep.any():
        return []

    blocks, confidence = blocks[keep], confidence[keep]
    xywh = np.concatenate([blocks[:, :2] - blocks[:, 2:4] / 2, blocks[:, 2:4]], axis=1)
    boxes = []
    selected = np.array(
        cv2.dnn.NMSBoxes(xywh.tolist(), confidence.tolist(), conf_th, nms_th)
    ).flatten()
    for index in selected:
        x, y, width, height = xywh[index]
        cx0, cy0 = max(int(x), 0), max(int(y), 0)
        cx1, cy1 = min(int(x + width), 1024), min(int(y + height), 1024)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (segmentation[cy0:cy1, cx0:cx1] > seg_th).mean() < seg_cov:
            continue
        boxes.append(
            (
                max(int(x / ratio), 0),
                max(int(y / ratio), 0),
                min(int((x + width) / ratio), w0),
                min(int((y + height) / ratio), h0),
            )
        )
    return boxes


def _keep_mask_components_in_boxes(
    mask: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Keep complete connected components that overlap at least one CTD box."""
    box_mask = np.zeros(mask.shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        box_mask[y0:y1, x0:x1] = True
    _, labels = cv2.connectedComponents(mask)
    keep_ids = np.unique(labels[box_mask])
    return np.isin(labels, keep_ids[keep_ids != 0]).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument(
        "--mask-dir", type=str, required=True, help="Output mask directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pth (downloads from HuggingFace if not specified)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing masks"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.8,
        help="Text segmentation threshold (default: 0.7)",
    )
    parser.add_argument(
        "--dilate", type=int, default=3, help="Mask dilation in pixels (default: 5)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="I/O workers (default: 4)"
    )
    parser.add_argument(
        "--ctd-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only mask components overlapping a comictextdetector text "
            "block (--no-ctd-gate restores raw UNet++ masks)"
        ),
    )
    parser.add_argument(
        "--ctd-onnx",
        type=str,
        default="models/mit/comictextdetector.pt.onnx",
        help="comictextdetector ONNX model used by --ctd-gate",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --image-dir. Mask output mirrors the source "
            "subdir structure under --mask-dir."
        ),
    )
    parser.add_argument(
        "--path-pattern",
        type=str,
        default=None,
        help=(
            "fnmatch glob (| to OR-combine) on each image's path relative to "
            "--image-dir, restricting which images get masked. Same semantics "
            "as the training path_pattern."
        ),
    )
    args = parser.parse_args()

    dilate_kernel = (
        np.ones((args.dilate, args.dilate), dtype=np.uint8) if args.dilate > 0 else None
    )

    print("Loading text segmentation model...")
    model = _load_model(args.model_path, device=args.device)

    ctd = None
    if args.ctd_gate:
        if Path(args.ctd_onnx).exists():
            ctd = _load_ctd(args.ctd_onnx, device=args.device)
        else:
            print(
                f"WARNING: --ctd-gate on but {args.ctd_onnx} is missing — "
                "gating disabled (pass --no-ctd-gate to silence this warning)"
            )

    image_dir = Path(args.image_dir)
    masks_dir = Path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # walk_images enforces per-subfolder stem uniqueness (same-folder stem
    # collisions would overwrite each other's mask); same stem across folders
    # is fine — the nested output layout disambiguates by subdir.
    image_files = walk_images(
        image_dir, recursive=args.recursive, pattern=args.path_pattern
    )

    work_items = []
    for image_path in image_files:
        try:
            rel = image_path.parent.relative_to(image_dir)
        except ValueError:
            rel = Path("")
        rel_str = str(rel)
        target_dir = masks_dir / rel if rel_str not in ("", ".") else masks_dir
        mask_path = target_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists() and not args.force:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        work_items.append((image_path, mask_path))

    total = len(work_items)
    if total == 0:
        print("No images to process.")
        return

    pool = ThreadPoolExecutor(max_workers=args.workers)

    pbar = tqdm(total=total, desc="Generating masks", ascii=True)
    for image_path, mask_path in work_items:
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)

        mask = _detect_mask(
            model,
            img_np,
            device=args.device,
            text_threshold=args.text_threshold,
        )

        pbar.update(1)

        if mask is None or not mask.any():
            pbar.set_postfix_str(f"{image_path.name}: skipped")
            continue

        combined_mask = (mask > 127).astype(np.uint8)

        if ctd is not None and combined_mask.any():
            boxes = _ctd_text_boxes(ctd, img_np)
            combined_mask = _keep_mask_components_in_boxes(combined_mask, boxes)
            if not combined_mask.any():
                pbar.set_postfix_str(f"{image_path.name}: skipped (ctd-gated)")
                continue

        if dilate_kernel is not None:
            combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)

        # Invert: detected=1 → alpha=0 (ignore), no detection → alpha=255 (train)
        alpha_mask = ((1 - combined_mask) * 255).astype(np.uint8)

        pool.submit(save_mask, mask_path, alpha_mask)

        h, w = img_np.shape[:2]
        masked_pct = 100 * np.count_nonzero(combined_mask) / (w * h)
        pbar.set_postfix_str(f"{image_path.name}: {masked_pct:.1f}%")

    pbar.close()
    pool.shutdown(wait=True)
    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    main()
