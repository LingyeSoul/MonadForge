#!/usr/bin/env python
"""FEI(z_t) along the live 28-step CFG=4 denoising trajectory, native bucket per stem.

Companion to ``probe_fei_artist.py``. That probe measures FEI on
training-time mixtures ``z_t = (1−t)·z_0 + t·ε`` — the router's
*training* input. This probe measures FEI on the *inference* trajectory
``z_T → z_0`` produced by the actual Anima sampler, which is the router's
*inference* input. The two need not look identical: training-time z_t is
a linear interpolation; inference z_t is whatever the (CFG-steered)
velocity field hands you.

Each stem's generation runs at its **native cache bucket** (W×H parsed
from ``{stem}_{W}x{H}_anima.npz`` under ``post_image_dataset/lora/``) so
the captured trajectory matches the resolution distribution the trainer
actually sees. Stems without a cache file are skipped.

For each artist (one stem per artist from ``caption_index.json``) we:

  1. Read the caption file as the positive prompt.
  2. Run a 28-step Euler CFG=4 trajectory at a fixed seed at the stem's
     native pixel bucket.
  3. Capture ``compute_fei_2band(z_t)`` *before* each velocity prediction.
     The engine already calls this every step at ``generation.py:725``
     to feed Hydra routers; we monkey-patch it to log instead of (only)
     setting it on the model. Base DiT has no router so the original is
     a no-op; the patched function always computes and always returns.
  4. Capture the sampler's per-step ``t`` (FM σ ∈ [0,1]) via a paired
     patch on ``set_hydra_sigma`` so each row carries the continuous
     time axis. ``get_timesteps_sigmas`` is shape-independent (function
     of ``infer_steps`` + ``flow_shift`` only), so ``t`` is canonical
     across buckets.

Reports:

  - Per-(artist, step) ``(t_sampler, sigma_low, e_low, e_high)`` in CSV.
  - Population-aggregated **teacher curve** ``teacher_curve.json``:
    ``[{step, t, n, mu_low, std_low, mu_high, std_high}, ...]`` —
    consumable by ``scripts/distill_turbo.py`` for FEI-trajectory-weighted
    CA (the gap ``teacher_FEI − student_FEI`` weighting CA bands).
  - Per-artist trajectories — high-frequency-dominant styles should
    track lower ``e_low`` throughout, mirroring the training-time
    rankings from ``probe_fei_artist.py``.

Usage::

    uv run python bench/fera_artist/probe_fei_trajectory.py \\
        --k_per_artist 1 --infer_steps 28 --guidance_scale 4.0 --label cfg4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.fei import compute_fei_2band, fei_sigma_low  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fera-trajectory-probe")


_UNTAGGED = "__untagged__"
# Cache filenames look like ``{stem}_{W}x{H}_anima.npz`` (W, H = pixels).
# Mirrors ``probe_fei_artist.py``.
_FNAME_RE = re.compile(r"^(?P<stem>.+)_(?P<w>\d{3,5})x(?P<h>\d{3,5})_anima\.npz$")


def _scan_cache(cache_dir: Path) -> dict[str, tuple[int, int]]:
    """Map ``stem -> (W_px, H_px)`` from the bucketed cache filenames.

    A stem can have multiple cache files if it was re-cached at different
    aspect ratios; we take the first stable-sorted hit (same convention
    as the artist probe).
    """
    out: dict[str, tuple[int, int]] = {}
    for f in sorted(cache_dir.rglob("*_anima.npz")):
        m = _FNAME_RE.match(f.name)
        if m is None:
            continue
        stem = m.group("stem")
        if stem in out:
            continue
        out[stem] = (int(m.group("w")), int(m.group("h")))
    return out


def _build_artist_groups(
    caption_index: Path,
    cache_by_stem: dict[str, tuple[int, int]],
    include_untagged: bool,
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    data = json.loads(caption_index.read_text())
    image_meta = data.get("image_meta", {})
    groups: dict[str, list[str]] = defaultdict(list)
    n_untagged = 0
    n_missing_cache = 0
    for stem, meta in image_meta.items():
        if stem not in cache_by_stem:
            n_missing_cache += 1
            continue
        artists = [a for a in meta.get("artist", []) if a]
        if not artists:
            n_untagged += 1
            if include_untagged:
                groups[_UNTAGGED].append(stem)
            continue
        groups[artists[0]].append(stem)
    log.info(
        f"caption_index: {len(image_meta)} stems, "
        f"{n_missing_cache} missing cache, "
        f"{n_untagged} untagged "
        f"(include_untagged={include_untagged})"
    )
    return dict(groups), image_meta


def _sample_per_artist(
    groups: dict[str, list[str]], k: int, seed: int
) -> list[tuple[str, str]]:
    """Return ``[(artist, stem), ...]`` — K stems per artist, deterministic."""
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for artist in sorted(groups.keys()):
        stems = list(groups[artist])
        rng.shuffle(stems)
        for stem in stems[:k]:
            out.append((artist, stem))
    return out


def _read_caption(image_meta_path: str, image_dataset: Path) -> str:
    """``image_meta[stem]["path"]`` is the .txt sidecar's path under image_dataset."""
    p = image_dataset / image_meta_path
    if not p.exists():
        raise FileNotFoundError(f"caption file missing: {p}")
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    # Collapse newlines that occasionally split tag lists.
    return re.sub(r"\s*\n\s*", " ", text)


# ---- FEI + sigma capture patches --------------------------------------------

# Mutable handle the patched function reads + writes. set_capture_target()
# swaps to a fresh list per generation so each artist's trace is isolated.
_CAPTURE_TARGET: list[dict] | None = None
_CAPTURE_DIV: float = 4.0
_CAPTURE_STEP_COUNTER: dict[int, int] = {}  # id(model) -> step counter
# Per-step ``t`` written by the set_hydra_sigma patch, read by the FEI patch.
# The sampler calls ``set_hydra_sigma(anima, t_expand)`` immediately before
# ``compute_and_set_hydra_fei(anima, latents)`` at generation.py:724-725, so
# the value is fresh when the FEI patch fires.
_LAST_SIGMA_T: float | None = None


def _set_capture(target: list[dict] | None, div: float) -> None:
    global _CAPTURE_TARGET, _CAPTURE_DIV, _LAST_SIGMA_T
    _CAPTURE_TARGET = target
    _CAPTURE_DIV = float(div)
    _LAST_SIGMA_T = None
    _CAPTURE_STEP_COUNTER.clear()


def _install_fei_patch() -> None:
    """Replace ``library.inference.generation.compute_and_set_hydra_fei`` with
    a logger. We patch the *generation* namespace because the function is
    imported by name at module load (``from ... import compute_and_set_hydra_fei``)
    so patching the source module has no effect on the live call site.

    Behaviour:
      - Always computes FEI on the pre-forward latent at our ``_CAPTURE_DIV``.
      - Appends to ``_CAPTURE_TARGET`` (caller-installed per artist).
      - Also calls the original so any attached Hydra router still gets its
        FEI set (no-op for base DiT — div is None there).
    """
    import library.inference.generation as _gen
    import library.inference.adapters as _adapters

    original = _adapters.compute_and_set_hydra_fei

    def patched(model, z):  # type: ignore[no-untyped-def]
        if _CAPTURE_TARGET is not None:
            # `z` is the live (B, C, T, H, W) Anima latent — squeeze the T axis.
            z2d = z.squeeze(2) if z.ndim == 5 else z
            h_lat, w_lat = int(z2d.shape[-2]), int(z2d.shape[-1])
            sigma_low = fei_sigma_low(h_lat, w_lat, _CAPTURE_DIV)
            fei = compute_fei_2band(z2d.detach(), sigma_low)
            step = _CAPTURE_STEP_COUNTER.get(id(model), 0)
            _CAPTURE_STEP_COUNTER[id(model)] = step + 1
            _CAPTURE_TARGET.append(
                {
                    "step": step,
                    "t_sampler": _LAST_SIGMA_T if _LAST_SIGMA_T is not None else float("nan"),
                    "h_lat": h_lat,
                    "w_lat": w_lat,
                    "sigma_low": sigma_low,
                    "e_low": float(fei[0, 0].item()),
                    "e_high": float(fei[0, 1].item()),
                }
            )
        original(model, z)

    _gen.compute_and_set_hydra_fei = patched
    log.info("installed FEI capture patch on library.inference.generation")


def _install_sigma_patch() -> None:
    """Patch ``set_hydra_sigma`` at the *generation* call site (same reason as
    the FEI patch — imported by name at module load) to stash the per-step
    ``t`` into ``_LAST_SIGMA_T``. The FEI patch fires next and reads it.

    ``t_expand`` is a 1-D tensor of length B replicated from the scalar
    ``timesteps[i]``; all entries equal, so ``[0].item()`` is fine.
    """
    import library.inference.generation as _gen
    import library.inference.adapters as _adapters

    original = _adapters.set_hydra_sigma

    def patched(model, timesteps):  # type: ignore[no-untyped-def]
        global _LAST_SIGMA_T
        if _CAPTURE_TARGET is not None:
            try:
                _LAST_SIGMA_T = float(timesteps.reshape(-1)[0].item())
            except Exception:  # pragma: no cover — should never fire
                _LAST_SIGMA_T = None
        original(model, timesteps)

    _gen.set_hydra_sigma = patched
    log.info("installed sigma capture patch on library.inference.generation")


# ---- main -------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--caption_index",
        type=Path,
        default=ROOT / "post_image_dataset" / "captions" / "caption_index.json",
    )
    p.add_argument(
        "--image_dataset",
        type=Path,
        default=ROOT / "image_dataset",
        help="Root holding the .txt caption sidecars referenced by caption_index.",
    )
    p.add_argument(
        "--k_per_artist", type=int, default=1,
        help="Captions drawn per artist (K=1 strict-balanced).",
    )
    p.add_argument("--include_untagged", action="store_true")
    p.add_argument(
        "--max_artists", type=int, default=10,
        help="If >0, cap the number of artists run (useful for smoke).",
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT / "post_image_dataset" / "lora",
        help="Source of per-stem native buckets. Stems with no cache file "
        "are skipped (matches probe_fei_artist.py).",
    )
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fei_sigma_low_div", type=float, default=4.0,
        help="DoG divisor used for the captured FEI (matches live default).",
    )
    p.add_argument(
        "--negative_prompt", type=str,
        default="lowres, bad anatomy, jpeg artifacts, worst quality",
    )
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    p.add_argument(
        "--vae",
        default="models/vae/qwen_image_vae.safetensors",
    )
    p.add_argument(
        "--text_encoder",
        default="models/text_encoders/qwen_3_06b_base.safetensors",
    )
    p.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile (--compile_blocks). On by default — saves "
        "~30%% wall on a 73-artist run after the first-graph build cost.",
    )
    p.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        help="Optional inductor preset (e.g. 'reduce-overhead'). Forwarded "
        "to compile_blocks; None = inductor default.",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if not args.caption_index.exists():
        raise SystemExit(
            f"caption_index not found at {args.caption_index} — run `make caption-index` first"
        )
    if not args.cache_dir.exists():
        raise SystemExit(
            f"cache_dir not found at {args.cache_dir} — run `make preprocess` first"
        )

    cache_by_stem = _scan_cache(args.cache_dir)
    log.info(f"scanned {len(cache_by_stem)} cached stems under {args.cache_dir}")
    if not cache_by_stem:
        raise SystemExit("no cached stems found — nothing to size from")

    groups, image_meta = _build_artist_groups(
        args.caption_index,
        cache_by_stem,
        include_untagged=args.include_untagged,
    )
    picks = _sample_per_artist(groups, k=args.k_per_artist, seed=args.seed)
    if args.max_artists > 0:
        picks = picks[: args.max_artists]
    log.info(
        f"sampled {len(picks)} (artist, stem) pairs across {len(groups)} artists"
    )

    # Resolve captions + native bucket before loading any model so misses fail fast.
    prompts: list[tuple[str, str, str, int, int]] = []  # (artist, stem, caption, W_px, H_px)
    for artist, stem in picks:
        meta = image_meta.get(stem)
        if meta is None:
            log.warning(f"  skip {artist}/{stem}: no image_meta entry")
            continue
        try:
            cap = _read_caption(meta["path"], args.image_dataset)
        except FileNotFoundError as exc:
            log.warning(f"  skip {artist}/{stem}: {exc}")
            continue
        if not cap:
            log.warning(f"  skip {artist}/{stem}: empty caption")
            continue
        bucket = cache_by_stem.get(stem)
        if bucket is None:
            log.warning(f"  skip {artist}/{stem}: cache vanished mid-build")
            continue
        w_px, h_px = bucket
        prompts.append((artist, stem, cap, w_px, h_px))
    log.info(f"{len(prompts)} prompts resolved (after caption filtering)")
    if not prompts:
        raise SystemExit("no usable prompts after filtering")

    # Heavy imports go after the cheap sanity checks.
    from anima_lora import (  # noqa: E402
        GenerationRequest,
        generate,
        get_generation_settings,
    )
    from library.inference.models import load_shared_models  # noqa: E402
    from library.runtime.device import clean_memory_on_device  # noqa: E402

    _install_fei_patch()
    _install_sigma_patch()

    # Build a representative request to seed the parsed-args namespace; we
    # mutate ``prompt`` / ``image_size`` / ``seed`` per stem below so all
    # downstream knobs (dataset config, sampler, etc.) match the embedder
    # default. First prompt's bucket is just a placeholder.
    extra_argv: list[str] = []
    if not args.no_compile:
        extra_argv.append("--compile_blocks")
        if args.compile_inductor_mode:
            extra_argv += ["--compile_inductor_mode", args.compile_inductor_mode]
    _first_w, _first_h = prompts[0][3], prompts[0][4]
    base_req = GenerationRequest(
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        prompt=prompts[0][2],
        negative_prompt=args.negative_prompt,
        save_path="/tmp/_fei_traj_unused.png",  # we never call save_output
        infer_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        image_size=(_first_h, _first_w),  # GenerationRequest takes (H, W)
        seed=args.seed,
        extra_argv=tuple(extra_argv),
    )
    gen_args = base_req.to_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_args.device = device
    gen_settings = get_generation_settings(gen_args)

    # Shared text encoder + on-demand DiT + conds cache. generate() will load
    # the DiT into ``shared["model"]`` on first call and reuse it after.
    shared = load_shared_models(gen_args)
    shared["conds_cache"] = {}

    out_dir = make_run_dir("fera_artist", label=args.label or "trajectory")
    log.info(f"output → {out_dir}")

    rows: list[dict] = []
    bucket_counts: dict[str, int] = defaultdict(int)
    for idx, (artist, stem, caption, w_px, h_px) in enumerate(prompts):
        gen_args.prompt = caption
        gen_args.seed = args.seed  # same seed across artists isolates the prompt effect
        gen_args.image_size = (h_px, w_px)  # native bucket per stem
        capture: list[dict] = []
        _set_capture(capture, args.fei_sigma_low_div)
        try:
            _ = generate(gen_args, gen_settings, shared_models=shared)
        except Exception as exc:
            log.warning(f"  [{idx + 1}/{len(prompts)}] {artist}/{stem} {w_px}x{h_px}: {exc}")
            continue
        finally:
            _set_capture(None, args.fei_sigma_low_div)
        bucket_counts[f"{w_px}x{h_px}"] += 1
        for c in capture:
            rows.append({"artist": artist, "stem": stem, **c})
        log.info(
            f"  [{idx + 1}/{len(prompts)}] {artist}/{stem} @ {w_px}x{h_px}: "
            f"{len(capture)} steps captured"
        )

    # Free GPU before writing artifacts.
    shared.pop("model", None)
    clean_memory_on_device(device)

    csv_path = out_dir / "fei_trajectory.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- per-step population stats ----------------------------------------
    # ``t_sampler`` is schedule-derived (function of infer_steps + flow_shift
    # only, see library/inference/sampling.py::get_timesteps_sigmas), so it's
    # identical across stems/buckets — we just take the first non-NaN we see.
    # ``sigma_low`` (the DoG kernel scale) DOES vary across buckets because
    # σ_low = min(H_lat, W_lat) / fei_sigma_low_div; we record its bucket-
    # weighted mean for reference.
    by_step_low: dict[int, list[float]] = defaultdict(list)
    by_step_high: dict[int, list[float]] = defaultdict(list)
    by_step_sigma: dict[int, list[float]] = defaultdict(list)
    by_step_t: dict[int, float] = {}
    for r in rows:
        s = r["step"]
        by_step_low[s].append(r["e_low"])
        by_step_high[s].append(r["e_high"])
        by_step_sigma[s].append(r["sigma_low"])
        t_val = r.get("t_sampler", float("nan"))
        if t_val == t_val and s not in by_step_t:  # NaN check
            by_step_t[s] = float(t_val)
    steps_sorted = sorted(by_step_low.keys())
    per_step_stats = [
        {
            "step": s,
            "t_sampler": by_step_t.get(s, float("nan")),
            "sigma_low_mean": float(mean(by_step_sigma[s])),
            "n": len(by_step_low[s]),
            "mean_e_low": float(mean(by_step_low[s])),
            "std_e_low": float(pstdev(by_step_low[s])),
            "min_e_low": float(min(by_step_low[s])),
            "max_e_low": float(max(by_step_low[s])),
            "mean_e_high": float(mean(by_step_high[s])),
            "std_e_high": float(pstdev(by_step_high[s])),
        }
        for s in steps_sorted
    ]

    # ---- teacher_curve.json — consumed by distill_turbo for FEI-trajectory
    # weighted CA. Schema is intentionally narrow: step → (t, mu, std) for
    # both bands. Trainer interpolates over `t` (or floors to step) at
    # τ_CA lookup time.
    teacher_curve = {
        "schema_version": 1,
        "infer_steps": args.infer_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "fei_sigma_low_div": args.fei_sigma_low_div,
        "n_prompts": len(prompts),
        "n_artists": len({r["artist"] for r in rows}),
        "bucket_counts": dict(bucket_counts),
        "curve": [
            {
                "step": s["step"],
                "t": s["t_sampler"],
                "n": s["n"],
                "mu_low": s["mean_e_low"],
                "std_low": s["std_e_low"],
                "mu_high": s["mean_e_high"],
                "std_high": s["std_e_high"],
            }
            for s in per_step_stats
        ],
    }
    curve_path = out_dir / "teacher_curve.json"
    curve_path.write_text(json.dumps(teacher_curve, indent=2))
    log.info(f"wrote {curve_path}")

    artifacts: list[str] = [csv_path.name, curve_path.name]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_band, ax_std) = plt.subplots(1, 2, figsize=(13, 5))

        # Per-artist trajectories (faint) + population mean (bold).
        by_artist: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in rows:
            by_artist[r["artist"]].append((r["step"], r["e_low"]))
        for traj in by_artist.values():
            traj.sort()
            xs = [t[0] for t in traj]
            ys = [t[1] for t in traj]
            ax_band.plot(xs, ys, color="gray", alpha=0.15, linewidth=0.7)
        mu = [s["mean_e_low"] for s in per_step_stats]
        sd = [s["std_e_low"] for s in per_step_stats]
        ax_band.plot(steps_sorted, mu, color="C0", linewidth=2, label="population mean")
        ax_band.fill_between(
            steps_sorted,
            [m - s_ for m, s_ in zip(mu, sd)],
            [m + s_ for m, s_ in zip(mu, sd)],
            color="C0", alpha=0.2, label="±1σ",
        )
        ax_band.set_xlabel("denoising step (0 = noise, N−1 = clean)")
        ax_band.set_ylabel("e_low(z_t)")
        ax_band.set_ylim(0, 1)
        ax_band.set_title(
            f"FEI trajectory across {len(by_artist)} artists "
            f"(CFG={args.guidance_scale:g}, {args.infer_steps} steps)"
        )
        ax_band.grid(alpha=0.3)
        ax_band.legend(fontsize=8)

        ax_std.plot(steps_sorted, sd, marker="o", color="C1")
        ax_std.set_xlabel("denoising step")
        ax_std.set_ylabel("std(e_low) across artists")
        ax_std.set_title("Router discriminative signal (higher = better)")
        ax_std.grid(alpha=0.3)

        fig.tight_layout()
        png = out_dir / "fei_trajectory.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        artifacts.append(png.name)
        log.info(f"wrote {png}")
    except Exception as exc:
        log.warning(f"plot failed (continuing): {exc}")

    # ---- console summary ---------------------------------------------------
    print("\n== teacher curve across artists at each step (native buckets) ==")
    print("step | t      | σ_low(μ) |  n  | mean   | std    | min    | max  ")
    print("-" * 72)
    for s in per_step_stats:
        print(
            f"{s['step']:>4} | {s['t_sampler']:.4f} | {s['sigma_low_mean']:>8.3f} | "
            f"{s['n']:>3} | {s['mean_e_low']:.4f} | {s['std_e_low']:.4f} | "
            f"{s['min_e_low']:.4f} | {s['max_e_low']:.4f}"
        )

    # Compare to the matched-bucket training-time probe (probe_fei_artist),
    # which reported std(e_low) ≈ 0.131 at t=0.05 div=4. Printing the
    # nearest-step trajectory equivalent here makes the contrast cheap.
    if per_step_stats:
        peak = max(per_step_stats, key=lambda s: s["std_e_low"])
        print(
            f"\npeak std(e_low) = {peak['std_e_low']:.4f} at step {peak['step']} "
            f"(t={peak['t_sampler']:.3f}, σ_low(μ)={peak['sigma_low_mean']:.2f}); "
            "cf. training-time mixture probe at div=4, t=0.05 ≈ 0.131"
        )

    metrics = {
        "n_artists": len({r["artist"] for r in rows}),
        "k_per_artist": args.k_per_artist,
        "n_prompts": len(prompts),
        "infer_steps": args.infer_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "fei_sigma_low_div": args.fei_sigma_low_div,
        "bucket_counts": dict(bucket_counts),
        "per_step_stats": per_step_stats,
    }
    write_result(
        out_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        label=args.label,
        device=device,
    )
    log.info("done")


if __name__ == "__main__":
    main()
