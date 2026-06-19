"""FlowBender Phase-0 probe — does open-loop EasyControl colorize leave an
uncorrected forward-operator alignment error?

FlowBender (arXiv:2606.20404) trains a closed-loop correction policy over the
model's *own* alignment error: at each step it look-aheads to a clean estimate
x̂₁, re-extracts the condition via the task's forward operator H, and feeds the
deviation H(x̂₁) vs y back as a refinement input. The whole method only has
headroom if the open-loop (vanilla) model actually drifts from the condition —
i.e. H(model output) != y by a margin above the irreducible VAE-roundtrip floor.

This probe measures exactly that on the deployed colorize checkpoint, with the
*real* forward operator H = mangafy (XDoG + screentone), the same operator that
synthesized the training conditions:

    err(t)  = MSE( mangafy(decode(x̂₁_t)),  mangafy(color_target) )
    floor   = MSE( mangafy(decode(encode(color_target))),  mangafy(color_target) )

x̂₁_t is captured per sampling step via a forward hook on the DiT: the Euler
flow estimate x̂₁ = x_t - σ·v (v = velocity prediction, σ = current noise level).
All mangafy calls share a fixed seed so screentone phase matches and the diff
reflects structure (lineart / tone-band), not random dot placement.

Verdict (heuristic, printed): ratio = median(err_final) / median(floor).
  ratio < ~1.5  → LOW headroom; output already mangafy-consistent → FlowBender
                  unlikely to help on colorize. STOP.
  ~1.5–3        → MODERATE; worth a Phase-1 A/B but expect small gains.
  > ~3          → HIGH headroom; real uncorrected alignment error → proceed.

Run:
    uv run python bench/flowbender/probe_alignment_drift.py --num_samples 8

Companion to docs (FlowBender feasibility on i2i). No training; read-only on
the existing anima_colorize_comic checkpoint + staged colorize cond/targets.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))  # repo root (anima_lora/)
# mangafy modules are flat (not a package) — import with their dir on sys.path,
# the same way easycontrol_adapters/colorization/prep.py does.
sys.path.insert(0, str(_ROOT / "easycontrol_adapters/colorization"))

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    load_vae,
)
from bench._common import make_run_dir, write_result  # noqa: E402
from library.inference.models import load_text_encoder  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from mangafy_gpu import mangafy_array_gpu  # noqa: E402
STAGING = _ROOT / "post_image_dataset/easycontrol/colorize/staging"
RESIZED = _ROOT / "post_image_dataset/resized"
_IMG_EXTS = (".png", ".webp", ".jpg", ".jpeg")


# ----------------------------------------------------------------------------- helpers
def _to_uint8_rgb(img_bchw_m11: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] in [-1,1] (B=1) -> uint8 HWC [0,255]."""
    x = img_bchw_m11[0].float().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()


def _load_rgb_uint8(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def _mangafy(rgb_u8: np.ndarray, seed: int, device: torch.device) -> np.ndarray:
    """H = forward operator. Returns float32 [0,1] grayscale-as-RGB."""
    out = mangafy_array_gpu(rgb_u8, seed=seed, device=device)  # uint8 HWC
    return out.astype(np.float32) / 255.0


def _blur(img_f: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img_f
    import cv2

    return cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _luma(rgb_f: np.ndarray) -> np.ndarray:
    return rgb_f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _discover_samples(n: int) -> list[tuple[str, Path, Path]]:
    """Return [(stem, cond_manga_png, color_target_png)] with matching targets."""
    out: list[tuple[str, Path, Path]] = []
    for cond in sorted(STAGING.rglob("*")):
        if cond.suffix.lower() not in _IMG_EXTS:
            continue
        rel_dir = cond.parent.relative_to(STAGING)
        target = None
        for ext in _IMG_EXTS:
            cand = RESIZED / rel_dir / f"{cond.stem}{ext}"
            if cand.exists():
                target = cand
                break
        if target is None:
            continue
        out.append((f"{rel_dir}/{cond.stem}", cond, target))
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=1024, help="square gen resolution")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mangafy_seed", type=int, default=0)
    ap.add_argument("--blur_sigma", type=float, default=1.0)
    ap.add_argument(
        "--ckpt", default=str(_ROOT / "output/ckpt/anima_colorize_comic.safetensors")
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    run_dir = make_run_dir("flowbender", label=args.label or "alignment-drift")
    panels_dir = run_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    samples = _discover_samples(args.num_samples)
    if not samples:
        raise SystemExit(f"no (cond, target) pairs found under {STAGING}")
    print(f"[flowbender] {len(samples)} samples, {args.steps} steps @ {args.size}px")

    ckpts = default_checkpoints()
    # schedule: colorize uses discrete_flow_shift=1.0 (snapshot) -> sigmas = linspace
    _, sigmas = get_timesteps_sigmas(args.steps, shift=1.0, device=torch.device("cpu"))

    # shared models loaded once (VAE + TE not touched by the adapter)
    vae = load_vae(ckpts.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True)
    vae.to(device)

    # build a representative args namespace once (image is per-sample but the
    # parsed knobs are identical) for the text encoder + gen settings
    base_req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt="",  # empty prompt -> auto-colorize
        negative_prompt="",
        save_path=str(run_dir / "_gen.png"),
        infer_steps=args.steps,
        guidance_scale=1.0,  # CFG=1 -> one forward per step, unambiguous capture
        image_size=(args.size, args.size),
        seed=args.seed,
        easycontrol_weight=args.ckpt,
        easycontrol_image=str(samples[0][1]),
    )
    base_args = base_req.to_args()
    base_args.device = device
    base_args.compile = False
    base_args.compile_blocks = False
    base_args.torch_compile = False
    base_args.vae_2d = True
    base_args.sampler = "euler"
    gen_settings = get_generation_settings(base_args)
    text_encoder = load_text_encoder(base_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()

    # forward-hook capture of (x_t, sigma, v) per model call
    caps: list[tuple[torch.Tensor, float, torch.Tensor]] = []

    def hook(_m, inp, out):
        x_t = inp[0].detach().to(torch.float32).cpu()
        sig = float(inp[1].detach().flatten()[0].item())
        v = out.detach().to(torch.float32).cpu()
        caps.append((x_t, sig, v))

    rows: list[dict] = []
    per_sample = []  # (stem, err_final, err_final_blur, floor, floor_blur, luma_mse)

    for si, (stem, cond_png, target_png) in enumerate(samples):
        # reference operator outputs (computed once per sample)
        target_rgb = _load_rgb_uint8(target_png, args.size)
        ref = _mangafy(target_rgb, args.mangafy_seed, device)
        ref_b = _blur(ref, args.blur_sigma)

        # floor: VAE round-trip of the color target, then mangafy
        with torch.no_grad():
            tgt_t = (
                torch.from_numpy(target_rgb).float().div(127.5).sub(1.0)  # [-1,1]
                .permute(2, 0, 1).unsqueeze(0).to(device, torch.bfloat16)
            )
            rt = vae.decode_to_pixels(vae.encode_pixels_to_latents(tgt_t))
            rt_rgb = _to_uint8_rgb(rt)
        floor_manga = _mangafy(rt_rgb, args.mangafy_seed, device)
        floor = _mse(floor_manga, ref)
        floor_b = _mse(_blur(floor_manga, args.blur_sigma), ref_b)

        # fresh DiT per sample (avoids adapter re-apply stacking); generate w/ hook
        req = GenerationRequest(
            dit=ckpts.dit,
            vae=ckpts.vae,
            text_encoder=ckpts.text_encoder,
            prompt="",
            negative_prompt="",
            save_path=str(run_dir / f"_gen_{si}.png"),
            infer_steps=args.steps,
            guidance_scale=1.0,
            image_size=(args.size, args.size),
            seed=args.seed,
            easycontrol_weight=args.ckpt,
            easycontrol_image=str(cond_png),
        )
        a = req.to_args()
        a.device = device
        a.compile = a.compile_blocks = a.torch_compile = False
        a.vae_2d = True
        a.sampler = "euler"
        a.save_path = str(run_dir / f"_gen_{si}.png")

        anima = load_dit_model(a, device, torch.bfloat16)
        shared = {"model": anima, "text_encoder": text_encoder, "vae": vae}
        caps.clear()
        h = anima.register_forward_hook(hook)
        with torch.no_grad():
            generate(a, gen_settings, shared_models=shared)
        h.remove()
        del anima
        torch.cuda.empty_cache()

        # CFG=1 -> expect one forward per step; tolerate a 2x (cond/uncond) layout
        if len(caps) == 2 * args.steps:
            step_caps = caps[0::2]
        elif len(caps) >= args.steps:
            step_caps = caps[: args.steps]
        else:
            raise RuntimeError(
                f"{stem}: captured {len(caps)} forwards, expected >= {args.steps}"
            )

        last_pred_rgb = None
        for i, (x_t, sig, v) in enumerate(step_caps):
            x1 = x_t - sig * v  # Euler clean estimate
            if x1.dim() == 5:
                x1 = x1.squeeze(2)
            with torch.no_grad():
                pred = vae.decode_to_pixels(x1.to(device, torch.bfloat16))
            pred_rgb = _to_uint8_rgb(pred)
            pred_manga = _mangafy(pred_rgb, args.mangafy_seed, device)
            err = _mse(pred_manga, ref)
            err_b = _mse(_blur(pred_manga, args.blur_sigma), ref_b)
            rows.append(
                {
                    "sample": stem,
                    "step": i,
                    "sigma": round(sig, 5),
                    "mangafy_mse": err,
                    "mangafy_mse_blur": err_b,
                }
            )
            if i == len(step_caps) - 1:
                last_pred_rgb = pred_rgb
                last_pred_manga = pred_manga

        # final-step luma consistency (cheap secondary signal)
        luma_mse = _mse(_luma(last_pred_rgb.astype(np.float32) / 255.0),
                        _luma(target_rgb.astype(np.float32) / 255.0))
        err_final = rows[-1]["mangafy_mse"]
        err_final_b = rows[-1]["mangafy_mse_blur"]
        per_sample.append((stem, err_final, err_final_b, floor, floor_b, luma_mse))
        ratio = err_final / floor if floor > 0 else float("inf")
        print(
            f"  [{si}] {stem:40s} err_final={err_final:.5f} floor={floor:.5f} "
            f"ratio={ratio:.2f} luma_mse={luma_mse:.5f}"
        )

        # visual panel: cond | gen | mangafy(gen) | reference mangafy
        try:
            from PIL import Image

            cond_rgb = _load_rgb_uint8(cond_png, args.size)
            strip = np.concatenate(
                [
                    cond_rgb,
                    last_pred_rgb,
                    (last_pred_manga * 255).astype(np.uint8),
                    (ref * 255).astype(np.uint8),
                ],
                axis=1,
            )
            Image.fromarray(strip).save(panels_dir / f"{si:02d}_{stem.replace('/', '_')}.png")
        except Exception as e:  # noqa: BLE001
            print(f"    (panel skipped: {e})")

    # ----------------------------------------------------------------- aggregate
    with (run_dir / "per_step.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    finals = np.array([p[1] for p in per_sample])
    finals_b = np.array([p[2] for p in per_sample])
    floors = np.array([p[3] for p in per_sample])
    floors_b = np.array([p[4] for p in per_sample])
    lumas = np.array([p[5] for p in per_sample])
    med_final, med_floor = float(np.median(finals)), float(np.median(floors))
    ratio = med_final / med_floor if med_floor > 0 else float("inf")
    ratio_b = (
        float(np.median(finals_b)) / float(np.median(floors_b))
        if np.median(floors_b) > 0
        else float("inf")
    )

    if ratio < 1.5:
        verdict = "LOW headroom — output already mangafy-consistent; FlowBender unlikely to help. STOP."
    elif ratio < 3.0:
        verdict = "MODERATE headroom — Phase-1 A/B worthwhile but expect small gains."
    else:
        verdict = "HIGH headroom — real uncorrected alignment error; proceed to Phase 1."

    # per-step curve plot
    artifacts = ["per_step.csv", "panels"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for stem, *_ in per_sample:
            srows = [r for r in rows if r["sample"] == stem]
            ax.plot(
                [r["sigma"] for r in srows],
                [r["mangafy_mse"] for r in srows],
                alpha=0.5,
                lw=1,
            )
        ax.axhline(med_floor, ls="--", color="k", label=f"VAE-roundtrip floor (med={med_floor:.4f})")
        ax.set_xlabel("σ (noise level, high→low along sampling)")
        ax.set_ylabel("mangafy MSE  ‖H(x̂₁) − H(target)‖²")
        ax.invert_xaxis()
        ax.set_title(f"FlowBender Phase-0: alignment drift (ratio={ratio:.2f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "drift_curves.png", dpi=110)
        artifacts.append("drift_curves.png")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    metrics = {
        "n_samples": len(per_sample),
        "steps": args.steps,
        "size": args.size,
        "median_err_final": med_final,
        "median_floor": med_floor,
        "ratio_final_over_floor": ratio,
        "ratio_blurred": ratio_b,
        "median_luma_mse": float(np.median(lumas)),
        "per_sample": [
            {
                "stem": s,
                "err_final": ef,
                "err_final_blur": efb,
                "floor": fl,
                "floor_blur": flb,
                "ratio": (ef / fl if fl > 0 else None),
                "luma_mse": lm,
            }
            for (s, ef, efb, fl, flb, lm) in per_sample
        ],
        "verdict": verdict,
    }
    write_result(run_dir, script=__file__, args=args, metrics=metrics, artifacts=artifacts)

    print("\n" + "=" * 78)
    print(f"median err_final = {med_final:.5f}   median floor = {med_floor:.5f}")
    print(f"ratio (final/floor) = {ratio:.2f}   blurred ratio = {ratio_b:.2f}")
    print(f"median luma MSE = {float(np.median(lumas)):.5f}")
    print(f"VERDICT: {verdict}")
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
