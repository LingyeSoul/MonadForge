#!/usr/bin/env python3
"""Paired holdout-CMMD eval for the ReFT Phase-1' arms.

Same design as bench/memorization/generalize.py (whose helpers this reuses):
prompts drawn from the sincos sample_ratio COMPLEMENT (never trained by any
arm), identical per-item seeds and sizes for every model, 20 steps / CFG 1.0
(the trainer's CMMD convention), PE-Core pooled MMD² against real holdout and
member pools, with a real-vs-real noise floor.

Deltas vs generalize.py:

* ``--adapters`` may mix plain-LoRA checkpoints (loaded via the standard
  inference adapter path) and ``bench.reft.reft_network`` checkpoints
  (dispatched on ``ss_network_module`` metadata; the DiT is loaded bare and
  the bench network is built via ``create_network_from_weights`` +
  ``apply_to`` — the inference loader has no generic dispatch for
  out-of-tree modules).
* Rendering is EAGER for every model (no compile_blocks) so all arms share
  one code path — the register arms grow the seq mid-stack and the paired
  design only needs cross-arm comparability, not speed.
* Saves per-model renders + a contact-sheet montage. The headroom bench's
  arm-A lesson: pixel metrics are blind to framing collapse — every eval
  ships an eyeball artifact.

Usage (from anima_lora/)::

    uv run python bench/reft/eval_cmmd.py \
        --adapters output/ckpt/bench_sincos_half_plain.safetensors \
                   output/ckpt/bench_sincos_half_reft64.safetensors \
                   output/ckpt/bench_sincos_half_reg36.safetensors \
                   output/ckpt/bench_sincos_half_reg36_reft64.safetensors \
        [--with_base] [--num_prompts 24] [--num_refs 96] [--label phase1p]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors import safe_open  # noqa: E402

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
)
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.memorization.eyeball import _pick  # noqa: E402
from bench.memorization.generalize import (  # noqa: E402
    GEN_SEED_BASE,
    Prompt,
    _pil_to_minus1to1,
    encode_pool,
    render_model,
    split_items,
    to_prompts,
)
from bench.reft import reft_network  # noqa: E402
from library.inference import load_shared_models  # noqa: E402
from library.inference.models import load_dit_model  # noqa: E402
from library.inference.output import decode_latent  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.cmmd import cmmd_from_pools  # noqa: E402
from library.vision.encoder import load_pe_encoder  # noqa: E402


def _network_module_of(ckpt: str) -> str:
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        return (f.metadata() or {}).get("ss_network_module", "")


def render_bench_model(
    args,
    label: str,
    adapter: str,
    prompts: list[Prompt],
    device,
    telemetry: dict[str, dict[str, float | None]],
) -> list[torch.Tensor]:
    """Latents for a bench.reft.reft_network checkpoint (manual apply)."""
    ckpt = default_checkpoints()
    base_req = GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt="placeholder",
        save_path="__unused__",
        infer_steps=args.steps,
        guidance_scale=args.cfg,
        seed=0,
    )
    base_args = base_req.to_args()
    base_args.device = device
    gen_settings = get_generation_settings(base_args)
    shared_models = load_shared_models(base_args)
    shared_models["conds_cache"] = {}

    anima = load_dit_model(base_args, device, torch.bfloat16)
    net = reft_network.create_network_from_weights(
        args.multiplier, adapter, None, None, anima
    )
    net.to(device=device, dtype=torch.bfloat16)
    net.apply_to(None, anima, apply_text_encoder=False, apply_unet=True)
    net.eval()
    shared_models["model"] = anima

    latents: list[torch.Tensor] = []
    for i, p in enumerate(prompts):
        pa = dataclasses.replace(
            base_req,
            prompt=p.caption,
            seed=GEN_SEED_BASE + i,  # identical across models — paired design
            image_size=p.size_hw,
        ).to_args()
        pa.device = device
        print(f"[{label}] {i + 1}/{len(prompts)} {p.stem}", flush=True)
        latents.append(generate(pa, gen_settings, shared_models).detach().to("cpu"))

    # Adoption readout from the last forward — distinguishes "no adoption"
    # from "adopted but useless" when reading a null synergy result.
    telemetry[label] = {
        "reg_ratio": net.last_reg_ratio,
        "patch_sink_ratio": net.last_patch_sink_ratio,
    }
    net.remove()
    shared_models.clear()
    del gen_settings, anima, net
    clean_memory_on_device(device)
    return latents


def _to_pil(px: torch.Tensor) -> Image.Image:
    if px.dim() == 4:
        px = px[0]
    arr = ((px.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def save_renders(run_dir: Path, label: str, pixels: list[torch.Tensor], prompts) -> Path:
    out = run_dir / "renders" / label
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for px, p in zip(pixels, prompts):
        img = _to_pil(px)
        img.save(out / f"{p.stem}.png")
        img.thumbnail((256, 256))
        thumbs.append(img)
    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 256, rows * 256), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
    sheet_path = run_dir / f"montage_{label}.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--with_base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--membership_adapter",
        default=None,
        help="Snapshot to replay membership from (default: first adapter).",
    )
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--num_prompts", type=int, default=24)
    ap.add_argument("--num_refs", type=int, default=96)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--pick_seed", type=int, default=0)
    ap.add_argument("--label", default="reft_phase1p")
    args = ap.parse_args()
    args.membership_adapter = args.membership_adapter or args.adapters[0]
    # render_model (generalize.py) reads this; keep every arm eager.
    args.compile_blocks = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = Random(args.pick_seed)

    members, holdouts, target = split_items(args)
    if len(holdouts) < args.num_prompts:
        raise RuntimeError("not enough holdout items for the prompt set")
    prompts = to_prompts(holdouts, args.num_prompts, rng)
    prompt_paths = {p.ref_path for p in prompts}
    ref_holdout_items = [
        it for it in holdouts if Path(it[0].absolute_path) not in prompt_paths
    ]
    ref_hold = _pick(ref_holdout_items, args.num_refs, rng)
    ref_memb = _pick(members, args.num_refs, rng)
    print(
        f"prompts={len(prompts)} ref_holdout={len(ref_hold)} ref_member={len(ref_memb)}",
        flush=True,
    )

    models: list[tuple[str, str | None, str]] = []
    if args.with_base:
        models.append(("base", None, "std"))
    for a in args.adapters:
        kind = (
            "bench"
            if _network_module_of(a) == reft_network.ReFTBenchNetwork.network_module
            else "std"
        )
        models.append((Path(a).stem, a, kind))

    # Phase 1 — DiT sampling per model (paired prompts/seeds; fresh load each).
    latents_by_model: dict[str, list[torch.Tensor]] = {}
    telemetry: dict[str, dict[str, float | None]] = {}
    for label, adapter, kind in models:
        if kind == "bench":
            latents_by_model[label] = render_bench_model(
                args, label, adapter, prompts, device, telemetry
            )
        else:
            latents_by_model[label] = render_model(
                args, label, adapter, prompts, device
            )

    # Phase 2 — one VAE pass for every latent.
    ckpt = default_checkpoints()
    vae = load_vae(
        ckpt.vae, device="cpu", disable_mmap=True, dtype=torch.bfloat16, eval=True
    )
    pixels_by_model = {
        label: [decode_latent(vae, lat, device).to("cpu") for lat in lats]
        for label, lats in latents_by_model.items()
    }
    del vae, latents_by_model
    clean_memory_on_device(device)

    run_dir = make_run_dir("reft", args.label)
    montages = {
        label: save_renders(run_dir, label, pxs, prompts)
        for label, pxs in pixels_by_model.items()
    }

    # Phase 3 — one PE-Core pass for gens + both reference pools.
    bundle = load_pe_encoder(device)
    pools = {
        label: encode_pool(
            bundle, [px.unsqueeze(0) if px.dim() == 3 else px for px in pxs]
        )
        for label, pxs in pixels_by_model.items()
    }
    ref_hold_pool = encode_pool(
        bundle,
        [_pil_to_minus1to1(Image.open(it[0].absolute_path)) for it in ref_hold],
    )
    ref_memb_pool = encode_pool(
        bundle,
        [_pil_to_minus1to1(Image.open(it[0].absolute_path)) for it in ref_memb],
    )
    del bundle
    clean_memory_on_device(device)

    half = len(ref_hold_pool) // 2
    floor = cmmd_from_pools(ref_hold_pool[:half], ref_hold_pool[half:])

    metrics: dict[str, dict[str, float]] = {}
    for label, pool in pools.items():
        metrics[label] = {
            "cmmd_holdout": cmmd_from_pools(ref_hold_pool, pool),
            "cmmd_member": cmmd_from_pools(ref_memb_pool, pool),
        }

    lines = [
        "# ReFT Phase-1' — paired holdout-CMMD",
        "",
        f"- target artist: `{target}` · prompts {len(prompts)} (holdout captions, "
        f"paired seeds) · refs {len(ref_hold)}/{len(ref_memb)} (holdout/member)",
        f"- steps {args.steps} · cfg {args.cfg} · eager · real-vs-real noise floor "
        f"**{floor:.4f}**",
        "",
        "| model | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |",
        "|---|---|---|---|",
    ]
    for label, m in metrics.items():
        lines.append(
            f"| {label} | {m['cmmd_holdout']:.4f} | {m['cmmd_member']:.4f} | "
            f"{m['cmmd_member'] - m['cmmd_holdout']:+.4f} |"
        )
    if telemetry:
        lines += [""] + [
            f"- adoption `{label}`: reg_ratio={t['reg_ratio']} "
            f"patch_sink_ratio={t['patch_sink_ratio']} (sink needs ~14-24×)"
            for label, t in telemetry.items()
        ]
    lines += [
        "",
        "Reading: lower `cmmd_holdout` = closer to the artist's unseen work. "
        "Deltas below the noise floor are not interpretable. Check every "
        "montage for framing collapse before believing any number "
        "(the headroom arm-A lesson).",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={"noise_floor": floor, "adoption": telemetry, **metrics},
        artifacts=[run_dir / "summary.md", *montages.values()],
        extra={"prompts": [p.stem for p in prompts]},
    )
    print(f"done → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
