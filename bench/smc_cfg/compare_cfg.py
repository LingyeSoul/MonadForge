#!/usr/bin/env python3
"""SMC-CFG A/B harness.

For each (prompt × seed × aspect) cell, runs inference.py twice — once with
vanilla CFG, once with --smc_cfg at the chosen (λ, α) — keeping every other
flag identical. Same seed within a cell, so the two outputs are directly
comparable.

Produces:
    <run_dir>/baseline/<prompt-idx>_<aspect>_<seed>.png
    <run_dir>/smc/<prompt-idx>_<aspect>_<seed>.png
    <run_dir>/pairs.csv           per-cell pixel-space divergence (L1, L2)
    <run_dir>/result.json         bench envelope

Pixel-space L1/L2 is a sanity divergence metric only — it tells you SMC-CFG
is moving the output, not whether the move is in a good direction.
Perceptual comparison (CMMD via PE-Core / ImageReward) is a follow-up.

Default prompt set is the spectrum bench's prompts.example.txt (vetted for
failure-mode coverage). Default aspects cover one square (1024×1024) and
one non-square (832×1248) — the latter is the (CFG × aspect) cell where
DCW's bias direction flips per project_dcw_cfg_aspect_signflip.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from scripts.tasks._common import INFERENCE_BASE  # noqa: E402


DEFAULT_PROMPTS_FILE = REPO_ROOT / "bench" / "spectrum" / "prompts.example.txt"


def load_prompts(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|||" in line:
            pos, neg = (s.strip() for s in line.split("|||", 1))
        else:
            pos, neg = line, ""
        out.append((pos, neg))
    return out


def latest_lora_path() -> Path:
    """Mirror scripts/tasks/_common.latest_lora() without importing it
    (avoids dragging tasks plumbing into a bench script)."""
    ckpt = REPO_ROOT / "output" / "ckpt"
    if not ckpt.is_dir():
        raise SystemExit(f"no LoRA dir: {ckpt}")
    cands = [
        p for p in ckpt.glob("*.safetensors")
        if "_moe" not in p.name and "fusion_head" not in p.name
    ]
    if not cands:
        raise SystemExit(f"no LoRA *.safetensors under {ckpt}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_inference(
    *,
    prompt: str,
    negative: str,
    seed: int,
    width: int,
    height: int,
    save_dir: Path,
    rename_to: Path,
    lora_weight: Path | None,
    extra_flags: list[str],
    infer_steps: int,
    cfg: float,
) -> None:
    """Run inference.py and rename the produced PNG to ``rename_to``.

    inference.py uses ``--save_path <dir>`` and auto-names files
    ``<timestamp>_<seed>.png`` — so we run into a per-call scratch dir and
    then move the single PNG to a deterministic name.
    """
    import subprocess

    save_dir.mkdir(parents=True, exist_ok=True)
    before = set(save_dir.glob("*.png"))

    # INFERENCE_BASE supplies --dit / --text_encoder / --vae / --sampler /
    # --flow_shift / etc.; appending overrides --prompt / --image_size / …
    # (argparse keeps the last value for repeated flags).
    cmd = [
        *INFERENCE_BASE,
        "--prompt", prompt,
        "--negative_prompt", negative,
        "--seed", str(seed),
        "--image_size", str(height), str(width),  # H W per inference.py help
        "--infer_steps", str(infer_steps),
        "--guidance_scale", str(cfg),
        *(["--lora_weight", str(lora_weight)] if lora_weight is not None else []),
        "--save_path", str(save_dir),
        "--no_metadata",
        "--compile_blocks",  # inductor cache persists across subprocesses
        *extra_flags,
    ]
    print("  $", " ".join(cmd[:6]), "...")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise SystemExit(f"inference.py exit {res.returncode} for {rename_to.name}")

    new = sorted(set(save_dir.glob("*.png")) - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise SystemExit(f"no new PNG produced under {save_dir} for {rename_to.name}")
    new[-1].rename(rename_to)


def pixel_divergence(a_path: Path, b_path: Path) -> tuple[float, float]:
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = a - b
    return float(np.abs(d).mean()), float(np.sqrt((d * d).mean()))


def parse_aspect(spec: str) -> tuple[int, int]:
    w, h = spec.lower().split("x")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_file", default=str(DEFAULT_PROMPTS_FILE))
    ap.add_argument("--aspects", nargs="+", default=["1024x1024", "832x1248"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--lambda", dest="lam", type=float, default=5.0,
                    help="SMC-CFG λ (default 5.0, paper-best on SD3.5/Flux/Qwen)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="SMC-CFG α (default 0.2, production setting on Anima — "
                    "k_t := α·mean(|e_t|) per step)")
    ap.add_argument("--lora_weight", default=None,
                    help="LoRA safetensors path (default: latest under output/ckpt/). "
                    "Pass --no_lora to disable LoRA loading entirely (base DiT only).")
    ap.add_argument("--no_lora", action="store_true",
                    help="Skip LoRA — run inference on the base DiT.")
    ap.add_argument(
        "--baseline_dir", type=str, default=None,
        help="Path to an existing baseline image dir (PNGs named "
        "<idx>_<W>x<H>_s<seed>.png — same tag scheme as this script's "
        "baseline/ output). When set, skip the baseline subprocess and "
        "pixel-compare against the existing images. When unset, only "
        "smc/ is produced and no divergence metric is computed.",
    )
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--extra_flag", action="append", default=[],
        help="Extra flag forwarded to BOTH inference runs (repeatable). "
        "Example: --extra_flag --dcw --extra_flag --dcw_lambda --extra_flag 0.01",
    )
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts_file))
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts_file}")

    if args.no_lora:
        lora = None
    else:
        lora = Path(args.lora_weight) if args.lora_weight else latest_lora_path()
        if not lora.exists():
            raise SystemExit(f"LoRA not found: {lora}")

    run_dir = make_run_dir("smc_cfg", label=args.label)
    (run_dir / "smc").mkdir(exist_ok=True)

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    if baseline_dir is not None and not baseline_dir.is_dir():
        raise SystemExit(f"--baseline_dir not found: {baseline_dir}")

    aspects = [parse_aspect(a) for a in args.aspects]
    rows: list[dict] = []

    for p_idx, (pos, neg) in enumerate(prompts):
        for asp_w, asp_h in aspects:
            for seed in args.seeds:
                tag = f"{p_idx:02d}_{asp_w}x{asp_h}_s{seed}"
                smc_png = run_dir / "smc" / f"{tag}.png"

                print(f"[{tag}] smc-cfg (λ={args.lam}, α={args.alpha}) …")
                run_inference(
                    prompt=pos, negative=neg, seed=seed,
                    width=asp_w, height=asp_h,
                    save_dir=run_dir / "smc", rename_to=smc_png,
                    lora_weight=lora,
                    extra_flags=[
                        *args.extra_flag,
                        "--smc_cfg",
                        "--smc_cfg_lambda", str(args.lam),
                        "--smc_cfg_alpha", str(args.alpha),
                    ],
                    infer_steps=args.steps, cfg=args.cfg,
                )

                row = {
                    "tag": tag, "prompt_idx": p_idx, "prompt": pos,
                    "aspect": f"{asp_w}x{asp_h}", "seed": seed,
                    "smc_png": str(smc_png.relative_to(run_dir)),
                }
                if baseline_dir is not None:
                    base_png = baseline_dir / f"{tag}.png"
                    if base_png.exists():
                        l1, l2 = pixel_divergence(base_png, smc_png)
                        row.update({
                            "l1": l1, "l2": l2,
                            "baseline_png": str(base_png),
                        })
                        print(f"  L1={l1:.4f}  L2={l2:.4f}")
                    else:
                        print(f"  ! no baseline image at {base_png}")
                        row.update({"l1": None, "l2": None,
                                    "baseline_png": str(base_png)})
                rows.append(row)

    csv_path = run_dir / "pairs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["tag"])
        w.writeheader()
        w.writerows(rows)

    l1s = [r["l1"] for r in rows if r.get("l1") is not None]
    l2s = [r["l2"] for r in rows if r.get("l2") is not None]
    metrics = {
        "n_pairs": len(rows),
        "lambda": args.lam, "alpha": args.alpha,
        "cfg": args.cfg, "steps": args.steps,
        "lora_weight": str(lora) if lora is not None else None,
        "baseline_dir": str(baseline_dir) if baseline_dir is not None else None,
        "l1_mean": float(np.mean(l1s)) if l1s else None,
        "l1_max": float(np.max(l1s)) if l1s else None,
        "l2_mean": float(np.mean(l2s)) if l2s else None,
        "l2_max": float(np.max(l2s)) if l2s else None,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["pairs.csv", "smc/"],
    )
    print(f"\nwrote {run_dir / 'result.json'}")
    if l1s:
        print(f"  pairs: {len(rows)}   L1 mean {metrics['l1_mean']:.4f}   "
              f"L2 mean {metrics['l2_mean']:.4f}")
    else:
        print(f"  smc images: {len(rows)}   (no baseline_dir — no divergence metric)")


if __name__ == "__main__":
    main()
