"""phash_edit — does the edit adapter apply instructions, or copy-lock?

Non-aligned probe adapted from the archived directedit_ec subject probe
(`_archive/directedit_ec/bench/run_subject_probe.py`). Plain generation, no
DirectEdit composition: pairs come from the miner's own manifest, so each arm
replays the adapter's training task exactly —

    cond   = resized source image A          (--easycontrol_image)
    prompt = the mined delta_caption A→B     (the edit instruction)

Train-set pairs are an upper bound: a failure here is decisive (the task was
never learned), a success still needs a held-out check. Twin_edit lesson:
render-judging on aligned sources passes even a closed-gate arm, so the copy
metric (MSE vs cond) is computed per arm, not eyeballed.

Arms per pair:
    noec        prompt only, no adapter      control — what the instruction
                                             alone does without the cond stream
    noop_b0     EC + EMPTY prompt            identity-stability probe — the
                                             identity arm's training config;
                                             a near-copy of cond is EXPECTED
    ec_b<off>   EC + instruction prompt      one arm per --b_offsets entry;
                                             b_cond never trains (gates sit at
                                             init −4), so the offset is the
                                             live cond-mass dial: each −1 is
                                             ~e× less cond attention mass

Copy-lock signature: ec_b0 MSE-vs-cond ≈ noop_b0 MSE-vs-cond (instruction
ignored). A healthy arm shows ec MSE-vs-cond well above noop while staying far
below noec (identity retained, edit landed).

Usage:
    uv run python bench/phash_edit/run_edit_probe.py
    uv run python bench/phash_edit/run_edit_probe.py --b_offsets 0,-1,-2,-3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
DEFAULT_EC_WEIGHT = (
    REPO_ROOT / "output" / "ckpt" / "anima_easycontrol_phash_edit.safetensors"
)

# tag_delta strata: twin_edit showed a "big edits land, small edits suppress"
# boundary, so the probe must cover both ends, not the corpus median.
STRATA = (("small", 1, 3), ("medium", 4, 12), ("large", 13, 10**9))


def resolve(rel: str) -> Path:
    """'artist/stem' → the resized pool PNG (bucket-sized, what training saw)."""
    return RESIZED_DIR / f"{rel}.png"


def _stratum(tag_delta: int) -> str:
    for name, lo, hi in STRATA:
        if lo <= tag_delta <= hi:
            return name
    return "?"


def pick_pairs(n_per_stratum: int, seed: int, pair_ids: str = "") -> list[dict]:
    manifest = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    edits = [p for p in manifest["pairs"] if p["kind"] == "edit"]
    ok = [
        p
        for p in edits
        if resolve(p["cond"]).is_file()
        and resolve(p["target"]).is_file()
        and p["delta_caption"].strip()
    ]
    if not ok:
        raise SystemExit(f"no resolvable edit pairs in {PAIRS_JSON}")
    if pair_ids.strip():
        wanted = [s.strip() for s in pair_ids.split(",") if s.strip()]
        by_id = {p["pair_id"]: p for p in ok}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit(f"pair_ids not resolvable: {missing}")
        return [dict(by_id[w], stratum=_stratum(by_id[w]["tag_delta"])) for w in wanted]
    rng = random.Random(seed)
    chosen: list[dict] = []
    used_artists: set[str] = set()
    for name, lo, hi in STRATA:
        pool = [p for p in ok if lo <= p["tag_delta"] <= hi]
        rng.shuffle(pool)
        took = 0
        for p in pool:
            if took >= n_per_stratum:
                break
            if p["artist"] in used_artists and any(
                q["artist"] not in used_artists for q in pool
            ):
                continue
            p = dict(p, stratum=name)
            chosen.append(p)
            used_artists.add(p["artist"])
            took += 1
        if took < n_per_stratum:
            logger.warning("stratum %s: only %d/%d pairs", name, took, n_per_stratum)
    return chosen


def arm_argv(
    prompt: str,
    cond: Optional[Path],
    b_offset: Optional[float],
    size: tuple[int, int],
    out_dir: Path,
    args,
    ck,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "inference.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        "flash",
        # `=` form: delta-grammar prompts can start with a removal `-tag`,
        # which bare `--prompt -cum` misparses as a flag.
        f"--prompt={prompt}",
        "--negative_prompt=worst quality, low quality, score_1, score_2, "
        "score_3, blurry, jpeg artifacts, sepia",
        "--image_size",
        str(size[1]),
        str(size[0]),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift",
        "3.0",
        "--sampler",
        "euler",
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--save_path",
        str(out_dir),
    ]
    if cond is not None:
        argv += [
            "--easycontrol_weight",
            str(args.ec_weight),
            "--easycontrol_image",
            str(cond),
            "--easycontrol_image_match_size",
        ]
        if b_offset:
            argv += ["--easycontrol_b_offset", str(b_offset)]
    return argv


def mse(out_png: Path, ref_png: Path) -> float:
    import numpy as np
    from PIL import Image

    out = Image.open(out_png).convert("RGB")
    ref = Image.open(ref_png).convert("RGB").resize(out.size, Image.LANCZOS)
    a = np.asarray(out, dtype=np.float64) / 255.0
    b = np.asarray(ref, dtype=np.float64) / 255.0
    return float(((a - b) ** 2).mean())


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
    p.add_argument("--n_per_stratum", type=int, default=2)
    p.add_argument(
        "--pair_ids",
        default="",
        help="Comma-separated pair_ids to probe instead of stratified sampling.",
    )
    p.add_argument("--ec_weight", default=str(DEFAULT_EC_WEIGHT))
    p.add_argument(
        "--b_offsets",
        default="0,-2",
        help="Comma-separated b_cond offsets; one ec_b<off> arm each. Negative "
        "= less cond mass (gates trained at -4; -4 offset ≈ starved).",
    )
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--label", default="edit-probe")
    args = p.parse_args()

    if not Path(args.ec_weight).is_file():
        raise SystemExit(f"EC checkpoint not found: {args.ec_weight}")
    start_heartbeat()
    ck = default_checkpoints()
    offsets = [float(s) for s in args.b_offsets.split(",") if s.strip()]
    pairs = pick_pairs(args.n_per_stratum, args.seed, args.pair_ids)

    run_dir = make_run_dir("phash_edit", label=args.label)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.info("Run dir: %s", run_dir)
    logger.info(
        "Pairs: %s",
        [(q["pair_id"], q["stratum"], q["tag_delta"]) for q in pairs],
    )

    # (name, prompt_kind, ec, offset): prompt_kind ∈ {delta, empty}
    arms: list[tuple[str, str, bool, Optional[float]]] = [
        ("noec", "delta", False, None),
        ("noop_b0", "empty", True, None),
    ]
    arms += [(f"ec_b{o:g}", "delta", True, o) for o in offsets]

    from PIL import Image

    per_pair = []
    sheet_rows = []
    for pair in pairs:
        cond = resolve(pair["cond"])
        target = resolve(pair["target"])
        size = Image.open(cond).size  # (W, H) — render at the cond's bucket size
        prompt = pair["delta_caption"].strip()
        key = f"{pair['stratum']}_{pair['pair_id']}"
        rec = {
            "pair_id": pair["pair_id"],
            "stratum": pair["stratum"],
            "tag_delta": pair["tag_delta"],
            "cond": str(cond),
            "target": str(target),
            "prompt": prompt,
            "arms": {},
        }
        cells: list[tuple[str, Optional[Path]]] = [
            ("cond (A)", cond),
            ("real target (B)", target),
        ]
        for name, pkind, ec, off in arms:
            out_dir = run_dir / "renders" / key / name
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = arm_argv(
                "" if pkind == "empty" else prompt,
                cond if ec else None,
                off,
                size,
                out_dir,
                args,
                ck,
            )
            log_path = logs_dir / f"{key}_{name}.log"
            logger.info("[%s/%s] running", key, name)
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
                logger.error("[%s/%s] TIMEOUT", key, name)
            pngs = sorted(q for q in out_dir.glob("*.png"))
            out_png = pngs[-1] if (ok and pngs) else None
            rec["arms"][name] = {
                "ok": out_png is not None,
                "wall_s": round(time.time() - t0, 1),
                "png": str(out_png.relative_to(run_dir)) if out_png else None,
                "mse_vs_cond": mse(out_png, cond) if out_png else None,
                "mse_vs_target": mse(out_png, target) if out_png else None,
            }
            cells.append((name, out_png))
            logger.info(
                "[%s/%s] %s wall=%.0fs mse_cond=%s",
                key,
                name,
                "ok" if out_png else "FAILED",
                time.time() - t0,
                rec["arms"][name]["mse_vs_cond"],
            )
        per_pair.append(rec)
        sheet_rows.append((key, cells))

    sheet = run_dir / "grid.png"
    make_sheet(sheet_rows, sheet)
    write_result(
        run_dir,
        script=str(Path(__file__).relative_to(REPO_ROOT)),
        args=args,
        metrics={
            "ec_weight": args.ec_weight,
            "arms": [a for a, *_ in arms],
            "per_pair": per_pair,
            "note": "train-set pairs = upper bound; copy-lock signature = "
            "ec_b0 mse_vs_cond ≈ noop_b0 mse_vs_cond",
        },
        artifacts=["grid.png"],
    )
    logger.info("Human verdict artifact: %s", sheet)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
