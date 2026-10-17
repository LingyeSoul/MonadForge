"""phash_edit × masked-anchor compose — can the archived recipe rescue small edits?

The directedit_ec Phase-1a/1b winner (archived 2026-08-19) is a zero-training
in-place edit recipe: DirectEdit inversion with the Δz anchor masked inside
the edit region (`--mask`), plus EasyControl conditioning on the source image
with a gray hole punched over the same region (`--easycontrol_mask`). With the
stock inpaint adapter it landed 3/3 ADD/REMOVE/REPLACE edits at b_offset 0.

The edit-probe (`run_edit_probe.py`) showed the trained phash_edit adapter
copy-locks on small tag deltas — the same boundary twin_edit hit, and the
archive's verdict is that only the mask recipe crosses it. This bench swaps
the phash_edit checkpoint into the recipe and asks whether the trained
edit-adapter beats the generic inpaint prior inside the hole:

    inpaint_full   baseline recipe: inpaint adapter, ψ_src = cond caption,
                   ψ_tar = caption ± delta            (the archived winner)
    phash_full     same prompts, phash_edit adapter   (adapter swap only)
    phash_delta    phash_edit adapter, ψ_src = "", ψ_tar = the delta caption
                   (the adapter's native training grammar; src="" is the
                   archived in-place configuration)
    phash_nomask   phash_delta without either mask    (control — expected to
                   copy-lock per the probe)

Metrics: MSE vs source split outside/inside the hole. Outside = preservation
(small is good); inside = movement (big is expected). Verdict is render-judged
off the contact sheet on top of the numbers.

Usage:
    uv run python bench/phash_edit/run_masked_compose.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import default_checkpoints  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

DATASET_DIR = REPO_ROOT / "post_image_dataset" / "easycontrol" / "phash_edit"
PAIRS_JSON = DATASET_DIR / "pairs.json"
RESIZED_DIR = DATASET_DIR / "resized"
PHASH_WEIGHT = (
    REPO_ROOT / "output" / "ckpt" / "anima_easycontrol_phash_edit.safetensors"
)
INPAINT_WEIGHT = REPO_ROOT / "output" / "ckpt" / "methods" / "anima_inpaint.safetensors"

# Edit-region boxes (fractional x0, y0, x1, y1), placed by eye on the resized
# cond image — the "user box" mask source, same as the archived Phase-1a set.
# 5829102: the girl's face (the delta turns her faceless). 6951514: hands at
# chest through the space above — "heart hands → arms up" is the geometry-ish
# stress case the position-locked prior is expected to struggle with.
CASES: dict[str, tuple[float, float, float, float]] = {
    "5829102__5829103": (0.05, 0.12, 0.58, 0.45),
    "6951514__6951512": (0.15, 0.05, 0.85, 0.55),
}


def load_cases() -> list[dict]:
    manifest = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    by_id = {p["pair_id"]: p for p in manifest["pairs"]}
    out = []
    for pid, box in CASES.items():
        p = by_id[pid]
        image = RESIZED_DIR / f"{p['cond']}.png"
        if not image.is_file():
            raise SystemExit(f"missing resized cond image: {image}")
        out.append(dict(p, image=image, box=box))
    return out


def apply_delta(cond_caption: str, delta_caption: str) -> str:
    """cond caption ± the mined delta → the full-caption ψ_tar.

    Both strings are miner artifacts: flat comma-joined booru tag bags with
    `-` marking removals (no position clauses — those exist only in the
    curated caption master, not in the gelcrawl pool).
    """
    tags = [t.strip() for t in cond_caption.split(",") if t.strip()]
    adds, drops = [], set()
    for t in (s.strip() for s in delta_caption.split(",")):
        if not t:
            continue
        if t.startswith("-"):
            drops.add(t[1:].strip())
        else:
            adds.append(t)
    kept = [t for t in tags if t not in drops]
    return ", ".join(kept + [a for a in adds if a not in kept])


def write_box_mask(image: Path, box, out_path: Path) -> Path:
    import numpy as np
    from PIL import Image

    with Image.open(image) as im:
        w, h = im.size
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 255
    Image.fromarray(m).save(out_path)
    return out_path


def masked_cond_preview(image: Path, mask_png: Path, out_path: Path) -> Path:
    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(image).convert("RGB")).copy()
    hole = np.asarray(Image.open(mask_png).convert("L")) > 127
    img[hole] = 128
    Image.fromarray(img).save(out_path)
    return out_path


def mse_vs_source(out_png: Path, source_png: Path, mask_png: Path) -> dict:
    import numpy as np
    from PIL import Image

    out = Image.open(out_png).convert("RGB")
    src = Image.open(source_png).convert("RGB").resize(out.size, Image.LANCZOS)
    a = np.asarray(out, dtype=np.float64) / 255.0
    b = np.asarray(src, dtype=np.float64) / 255.0
    err = (a - b) ** 2
    hole = (
        np.asarray(Image.open(mask_png).convert("L").resize(out.size, Image.NEAREST))
        > 127
    )
    return {
        "full": float(err.mean()),
        "outside": float(err[~hole].mean()),
        "inside": float(err[hole].mean()),
    }


def arm_argv(
    image: Path,
    prompt_src: str,
    prompt_tar: str,
    ec_weight: Path,
    masked: bool,
    mask_png: Path,
    out_dir: Path,
    args,
    ck,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "edit.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--image",
        str(image),
        # `=` form: delta-grammar ψ_tar can start with a removal `-tag`.
        f"--prompt_src={prompt_src}",
        f"--prompt_tar={prompt_tar}",
        "--save_path",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--infer_steps",
        str(args.infer_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--t_inj",
        "0",
        "--no_compile_blocks",
        "--easycontrol_weight",
        str(ec_weight),
        "--easycontrol_scale",
        "1.0",
    ]
    if masked:
        argv += ["--easycontrol_mask", str(mask_png), "--mask", str(mask_png)]
    return argv


def make_sheet(rows: list[tuple[str, list[tuple[str, Optional[Path]]]]], out: Path):
    from PIL import Image, ImageDraw

    thumb_h, label_h = 384, 22
    cols = max(len(r[1]) for r in rows)
    thumbs = []
    for _, cells in rows:
        row = []
        for label, p in cells:
            if p is not None and Path(p).is_file():
                im = Image.open(p).convert("RGB")
                w = int(im.size[0] * thumb_h / im.size[1])
                row.append((label, im.resize((w, thumb_h), Image.LANCZOS)))
            else:
                row.append((label, None))
        thumbs.append(row)
    cell_w = max(
        (im.size[0] for row in thumbs for _, im in row if im is not None),
        default=thumb_h,
    )
    canvas = Image.new(
        "RGB", (cell_w * cols, (thumb_h + label_h) * len(rows)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(thumbs):
        for c, (label, im) in enumerate(row):
            x, y = c * cell_w, r * (thumb_h + label_h)
            if im is not None:
                canvas.paste(im, (x + (cell_w - im.size[0]) // 2, y + label_h))
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255))
    canvas.save(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--label", default="masked-compose")
    args = p.parse_args()

    for w in (PHASH_WEIGHT, INPAINT_WEIGHT):
        if not w.is_file():
            raise SystemExit(f"checkpoint not found: {w}")
    start_heartbeat()
    ck = default_checkpoints()
    cases = load_cases()

    run_dir = make_run_dir("phash_edit", label=args.label)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    per_case = []
    sheet_rows = []
    for case in cases:
        image: Path = case["image"]
        pid = case["pair_id"]
        full_tar = apply_delta(case["cond_caption"], case["delta_caption"])
        mask_png = write_box_mask(image, case["box"], run_dir / f"{pid}_mask.png")
        preview = masked_cond_preview(image, mask_png, run_dir / f"{pid}_cond.png")

        # (name, ec_weight, prompt_src, prompt_tar, masked)
        arms = [
            ("inpaint_full", INPAINT_WEIGHT, case["cond_caption"], full_tar, True),
            ("phash_full", PHASH_WEIGHT, case["cond_caption"], full_tar, True),
            ("phash_delta", PHASH_WEIGHT, "", case["delta_caption"], True),
            ("phash_nomask", PHASH_WEIGHT, "", case["delta_caption"], False),
        ]
        rec = {
            "pair_id": pid,
            "delta_caption": case["delta_caption"],
            "box": case["box"],
            "image": str(image),
            "arms": {},
        }
        cells: list[tuple[str, Optional[Path]]] = [
            ("source", image),
            ("cond w/ hole", preview),
        ]
        for name, weight, src, tar, masked in arms:
            out_dir = run_dir / "renders" / pid / name
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = arm_argv(
                image, src, tar, weight, masked, mask_png, out_dir, args, ck
            )
            log_path = logs_dir / f"{pid}_{name}.log"
            logger.info("[%s/%s] running", pid, name)
            t0 = time.time()
            try:
                with log_path.open("w") as lf:
                    lf.write(" ".join(argv) + "\n\n")
                    lf.flush()
                    proc = subprocess.run(
                        argv,
                        cwd=REPO_ROOT,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
                logger.error("[%s/%s] TIMEOUT", pid, name)
            pngs = sorted(q for q in out_dir.glob("*.png"))
            out_png = pngs[-1] if (ok and pngs) else None
            rec["arms"][name] = {
                "ok": out_png is not None,
                "wall_s": round(time.time() - t0, 1),
                "png": str(out_png.relative_to(run_dir)) if out_png else None,
                "mse": mse_vs_source(out_png, image, mask_png) if out_png else None,
            }
            cells.append((name, out_png))
            logger.info(
                "[%s/%s] %s wall=%.0fs mse=%s",
                pid,
                name,
                "ok" if out_png else "FAILED",
                time.time() - t0,
                rec["arms"][name]["mse"],
            )
        per_case.append(rec)
        sheet_rows.append((pid, cells))

    sheet = run_dir / "grid.png"
    make_sheet(sheet_rows, sheet)
    write_result(
        run_dir,
        script=str(Path(__file__).relative_to(REPO_ROOT)),
        args=args,
        metrics={
            "phash_weight": str(PHASH_WEIGHT),
            "inpaint_weight": str(INPAINT_WEIGHT),
            "per_case": per_case,
            "note": "outside = preservation (small good); inside = movement "
            "(big expected). Render-judge the grid on top of the numbers.",
        },
        artifacts=["grid.png"],
    )
    logger.info("Human verdict artifact: %s", sheet)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
