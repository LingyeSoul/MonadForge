"""Phase A0 — tagger-verified boy negatives for the SAM3 soft prompt.

The Phase-0 keeper (`docs/experimental/soft_prompt_for_sam.md`) boxes boys: its
training pool had no boy negatives and the count metric only scores pure-girl
images. Boy-only images are useless as a source (the corpus has ~7), so the
negatives come from the ``1boy,1girl`` images:

``build`` — run a prompt (the keeper) on every ``1boy,1girl`` image, crop each
NMS survivor (mask-blanked, as `caption-position` does) and tag the crop with
the Anima Tagger (dbv4 backend). A crop tagged ``1boy`` and not ``1girl`` is a
**boy box**, ``1girl`` and not ``1boy`` a **girl box**. Images with exactly one
of each become training rows whose target is the girl box only — the boy box
then has no Hungarian match and the focal term pushes its query down. The
tagger is the label source, so this is independent of SAM3's self-labels.
``--holdout N`` rows are held out for the gate and never trained on.

``eval`` — score a prompt on the held-out rows: boy-box rate (a survivor
overlaps the boy box, IoU ≥ ``--match_iou``) and girl recall (one overlaps the
girl box). Gate A0: boy-box ≤ 0.1, girl recall ≥ 0.95.

    make daemon-run ARGS="bench/sam3_soft_prompt/pair_negatives.py build"
    make daemon-run ARGS="bench/sam3_soft_prompt/train_soft_prompt.py --init 'anime girl' \\
        --extra_manifest post_image_dataset/captions/sam3_soft_prompt/pairs/manifest.json --label pairneg"
    make daemon-run ARGS="bench/sam3_soft_prompt/pair_negatives.py eval --prompt <soft_prompt.safetensors>"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.sam3_soft_prompt.build_targets import survivors  # noqa: E402
from bench.sam3_soft_prompt.common import (  # noqa: E402
    encode_image,
    encode_text,
    ground,
    install_prompt,
    iou_xyxy,
    load_sam3,
    load_soft_prompt,
    proposals,
)
from library.env import resolve_under_home  # noqa: E402
from library.preprocess._dataset import walk_images  # noqa: E402
from library.preprocess.instance_detection import Detection, crop_instance  # noqa: E402

PAIR_COUNT = ("1boy", "1girl")
KEEPER = "bench/sam3_soft_prompt/results/20260826-2310-animegirl-init/soft_prompt.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("build", "eval"))
    p.add_argument(
        "--prompt", default=KEEPER, help="text, or a .safetensors soft prompt"
    )
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument("--path_pattern", default="*")
    p.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    p.add_argument(
        "--out", default="post_image_dataset/captions/sam3_soft_prompt/pairs"
    )
    p.add_argument(
        "--count_filter",
        choices=("pair", "any_boy"),
        default="pair",
        help="pair = count exactly 1boy,1girl; any_boy = every image with a boy "
        "count tag (pairs included)",
    )
    p.add_argument("--holdout", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--retry_floor", type=float, default=0.35)
    p.add_argument("--iou_threshold", type=float, default=0.65)
    p.add_argument("--min_area_frac", type=float, default=0.005)
    p.add_argument("--min_fill", type=float, default=0.2)
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument(
        "--tagger_dir", default=None, help="AnimaTagger ckpt dir (default dbv4)"
    )
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default=None)
    return p.parse_args()


def load_prompt(model, spec: str, device: str) -> dict:
    if spec.endswith(".safetensors"):
        return load_soft_prompt(resolve_under_home(spec), device)
    return encode_text(model, spec, device)


def pair_images(
    index: dict, dst: Path, pattern: str, count_filter: str = "pair"
) -> list[Path]:
    images = walk_images(dst, recursive=True, pattern=pattern)
    keep = []
    for path in images:
        key = str(path.relative_to(dst).with_suffix(""))
        meta = index["image_meta"].get(key)
        if meta is None:
            continue
        count = tuple(sorted(meta.get("count", [])))
        if count_filter == "pair":
            ok = count == PAIR_COUNT
        else:
            ok = any("boy" in t for t in count)
        if ok:
            keep.append(path)
    return keep


def detect(model, processor, prompt, image: Image.Image, a) -> list[dict]:
    bo = install_prompt(encode_image(model, processor, image), prompt)
    props = proposals(ground(model, processor, bo), a.retry_floor)
    return survivors(props, a.floor, a.iou_threshold, a.min_area_frac)


def to_detection(row: dict, size: tuple[int, int]) -> Detection:
    w, h = size
    x0, y0, x1, y1 = row["box"]
    mask = (
        F.interpolate(row["mask"][None, None].float(), size=(h, w), mode="bilinear")[
            0, 0
        ]
        > 0.5
    )
    return Detection(
        box=(x0 * w, y0 * h, x1 * w, y1 * h),
        score=row["score"],
        mask=mask.cpu().numpy(),
    )


def classify(kept: dict) -> str:
    boy, girl = "1boy" in kept, "1girl" in kept
    if boy and not girl:
        return "boy"
    if girl and not boy:
        return "girl"
    return "ambiguous"


def build(a) -> None:
    from library.captioning.anima_tagger import AnimaTagger

    dst = resolve_under_home(a.dst)
    out = resolve_under_home(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(resolve_under_home(a.caption_index).read_text(encoding="utf-8"))
    images = pair_images(index, dst, a.path_pattern, a.count_filter)
    print(f"{len(images)} pair images", flush=True)

    model, processor = load_sam3(resolve_under_home(a.checkpoint), a.device)
    prompt = load_prompt(model, a.prompt, a.device)
    tagger = AnimaTagger(**({"ckpt_dir": a.tagger_dir} if a.tagger_dir else {}))

    rows: list[dict] = []
    for n, path in enumerate(images, 1):
        if n % 100 == 0:
            print(f"  [{n}/{len(images)}]", flush=True)
        rel = str(path.relative_to(dst))
        with Image.open(path) as h:
            image = h.convert("RGB")
        dets = detect(model, processor, prompt, image, a)
        boxes = []
        for r in dets:
            crop = crop_instance(image, to_detection(r, image.size), blank=True)
            pred = tagger.predict(crop)
            kept = pred.get("kept") or {}
            boxes.append(
                {
                    "box": r["box"],
                    "score": r["score"],
                    "fill": r["fill"],
                    "cls": classify(kept),
                    "p_boy": float(pred["scores"].get("1boy", 0.0)),
                    "p_girl": float(pred["scores"].get("1girl", 0.0)),
                }
            )
        cls = [b["cls"] for b in boxes]
        usable = cls.count("girl") == 1 and cls.count("boy") == 1 and len(cls) == 2
        count = index["image_meta"][rel.rsplit(".", 1)[0]].get("count", [])
        row = {
            "image": rel,
            "count": sorted(count),
            "boxes": boxes,
            "usable": usable,
        }
        if usable:
            gi = cls.index("girl")
            bi = cls.index("boy")
            if dets[gi]["fill"] < a.min_fill:
                row["usable"] = False
            else:
                tpath = out / (rel.replace("/", "__") + ".npz")
                np.savez_compressed(
                    tpath,
                    boxes=np.asarray([dets[gi]["box"]], dtype=np.float32),
                    masks=dets[gi]["mask"].cpu().numpy()[None].astype(np.bool_),
                    scores=np.asarray([dets[gi]["score"]], dtype=np.float32),
                )
                row["target"] = str(tpath.relative_to(out))
                row["girl_box"] = dets[gi]["box"]
                row["boy_box"] = dets[bi]["box"]
        rows.append(row)

    usable = [r for r in rows if r["usable"]]
    random.Random(a.seed).shuffle(usable)
    held = set(id(r) for r in usable[: a.holdout])
    for r in rows:
        r["holdout"] = id(r) in held
        r["train"] = r["usable"] and not r["holdout"]

    from collections import Counter

    per_image = Counter(
        "|".join(sorted(b["cls"] for b in r["boxes"])) or "none" for r in rows
    )
    summary = {
        "images": len(rows),
        "usable": len(usable),
        "train": sum(r["train"] for r in rows),
        "holdout": sum(r["holdout"] for r in rows),
        "boxes_per_class": dict(Counter(b["cls"] for r in rows for b in r["boxes"])),
        "image_patterns": dict(per_image.most_common(12)),
        "prompt": a.prompt,
        "args": vars(a),
    }
    (out / "manifest.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out / 'manifest.json'}")


def evaluate(a) -> None:
    dst = resolve_under_home(a.dst)
    out = resolve_under_home(a.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if r.get("holdout")]
    print(f"{len(rows)} holdout pair images", flush=True)

    model, processor = load_sam3(resolve_under_home(a.checkpoint), a.device)
    prompt = load_prompt(model, a.prompt, a.device)
    run_dir = make_run_dir("sam3_soft_prompt", a.label or "pair-eval")

    boy_hit = girl_hit = 0
    n_surv = 0
    per: list[dict] = []
    for r in rows:
        with Image.open(dst / r["image"]) as h:
            image = h.convert("RGB")
        dets = detect(model, processor, prompt, image, a)
        n_surv += len(dets)
        b = any(iou_xyxy(d["box"], r["boy_box"]) >= a.match_iou for d in dets)
        g = any(iou_xyxy(d["box"], r["girl_box"]) >= a.match_iou for d in dets)
        boy_hit += b
        girl_hit += g
        per.append({"image": r["image"], "survivors": len(dets), "boy": b, "girl": g})
    n = max(1, len(rows))
    metrics = {
        "holdout": len(rows),
        "boy_box_rate": boy_hit / n,
        "girl_recall": girl_hit / n,
        "survivors_per_image": n_surv / n,
        "gate": boy_hit / n <= 0.1 and girl_hit / n >= 0.95,
        "prompt": a.prompt,
    }
    (run_dir / "per_image.json").write_text(json.dumps(per, indent=1), encoding="utf-8")
    write_result(run_dir, script=__file__, args=a, metrics=metrics, label=a.label)
    print(json.dumps(metrics, indent=2))
    print(f"wrote: {run_dir}")


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    if a.mode == "build":
        build(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
