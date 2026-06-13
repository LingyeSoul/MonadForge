"""Fit a PiD->native-VAE color-match transform on Anima's own latents.

WHY THIS EXISTS
---------------
PiD's distilled pixel decoder color-drifts versus the native Qwen VAE. This is
acknowledged upstream — ``PiD/docs/FLUX2_2kto4k_new_ckpt_compare.md``:

    "the old decoder's final-latent output drifted in color vs. the native VAE;
     _2606 matches it."
    "Early-termination whitening ... earlier timestep -> whiter."

NVIDIA fixed it by *retraining* a corrected checkpoint (``_2606``) — but only for
the **flux2** backbone. The ``qwenimage`` checkpoint Anima rides
(``pid_qwenimage_2kto4k_4step.pth``) has no color-corrected variant, so Anima
inherits the drift. People currently undo it by hand with a ComfyUI chain
(saturation 1.05 / contrast 1.02 / cool tint 0.1 / levels) — empirical, coupled,
un-reproducible.

This bench measures the drift on Anima's real latents and fits a STATIC color
transform candidate(PiD) -> reference(native VAE): the cheap decode-time
equivalent of what NVIDIA baked into ``_2606``.

WHAT IT DOES
------------
For N cached Anima latents it decodes a center crop two ways:
  reference = native Qwen VAE  (WanVAE2d_, the decoder stage1 already validated)
  candidate = PiD 4-step SDE   (the shipping node's exact decode core)
aligns them (PiD is 4x SR -> downsample to VAE resolution), and fits three
transforms, each pixel-storage-free (closed form from accumulated moments):

  1. per-channel linear-RGB affine (gain+bias)         — WB + exposure baseline
  2. linear-RGB 3x3 + bias (the one to BAKE)           — captures cross-channel cast
  3. Oklab per-channel moment match, decoded to the     — the principled version of
     ComfyUI knobs (contrast=L gain, saturation=          the hand-dialed node chain
     chroma gain, tint=(a,b) shift, brightness=L shift)

It reports before/after RMSE with a train/test split (generalization guard) and —
crucially — the PER-IMAGE SPREAD of the offset. A tight spread => a static bake
fixes the drift. A wide spread is the irreducible 4-step SDE variance (the same
mechanism behind the authors' "early-termination whitening") that no static
transform can remove — that number tells you whether calibration is the whole
answer or just most of it.

NOTE: fit at the EXACT step count you decode with (``--steps``). The drift is
timestep-dependent, so a calibration fit at 4 steps will not transfer to 2 or 6.

Run (from anima_lora/):
    uv run python bench/pid/fit_color_calib.py --num_images 24 --steps 4
Output: bench/pid/results/<ts>-colorcalib/  (pid_color_calib.safetensors,
comparison.png, report.txt, result.json).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]  # anima_lora/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling stage1 import

from bench._common import make_run_dir, write_result  # noqa: E402
from stage1_vae_roundtrip import (  # noqa: E402  native Qwen VAE decoder (reuse, no dup)
    _LATENTS_MEAN,
    _LATENTS_STD,
    WanVAE2d_,
)

# Default on-disk locations (verified present in this checkout).
DEFAULT_PID_CKPT = REPO_ROOT.parent / "comfy/models/pid/pid_qwenimage_2kto4k_4step.pth"
DEFAULT_NULL_CAP = REPO_ROOT.parent / "comfy/models/pid/pid_null_caption_gemma.safetensors"
DEFAULT_VAE = REPO_ROOT / "models/vae/QwenImage_VAE_2d.pth"
DEFAULT_NODE_DIR = REPO_ROOT.parent / "comfy/custom_nodes/comfyui-anima-pid"
DEFAULT_LATENT_GLOB = "post_image_dataset/lora/**/*_anima.npz"


def load_pid_core(node_dir: Path):
    """Import the shipping node's ``pid_core`` (and its ``pid_net`` subpackage)
    without executing the package ``__init__`` (which pulls in ComfyUI). We register
    a synthetic package whose ``__path__`` points at the node dir, so pid_core's
    ``from .pid_net import PidNet`` and pid_net's own relative imports resolve."""
    pkg_name = "_anima_pid_vendor"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(node_dir)]
        sys.modules[pkg_name] = pkg
    return importlib.import_module(f"{pkg_name}.pid_core")


# ---- Oklab (Ottosson) — stats + knob decode only; no inverse needed ----
def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb01_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (...,3) in [0,1] sRGB -> Oklab (...,3) = (L, a, b)."""
    r, g, b = srgb_to_linear(rgb.clamp(0, 1)).unbind(-1)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l.clamp(min=0) ** (1 / 3), m.clamp(min=0) ** (1 / 3), s.clamp(min=0) ** (1 / 3)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.stack([L, a, bb], dim=-1)


def load_latent(npz_path: Path) -> torch.Tensor:
    """Return cached Anima latent (16,h,w), already (mu-mean)/std normalized."""
    d = np.load(npz_path)
    key = next(k for k in d.files if k.startswith("latents_"))
    lat = torch.from_numpy(d[key]).float()
    if lat.ndim == 4:
        lat = lat[0]
    return lat  # (16,h,w)


def center_crop(lat: torch.Tensor, crop: int) -> torch.Tensor:
    if crop <= 0:
        return lat  # full latent — real 4x SR (auto-tiled if large)
    _, h, w = lat.shape
    ch, cw = min(crop, h), min(crop, w)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return lat[:, y0 : y0 + ch, x0 : x0 + cw]


def decode_pid(pid_core, net, lat5, args, null_cap, pid_dtype):
    """Full-size PiD decode. Latents larger than --tile_latent go through the tiled
    SR path (fixed tile size -> the compiled block graph is built once and reused
    across every tile and every image); smaller ones run single-pass."""
    gh, gw = lat5.shape[-2], lat5.shape[-1]
    common = dict(steps=args.steps, sigma=0.0, seed=args.seed, dtype=pid_dtype,
                  compile=args.compile, caption_embs=null_cap)
    if max(gh, gw) > args.tile_latent:
        return pid_core.pid_decode_latent_tiled(
            net, lat5, tile=args.tile_latent, overlap=args.tile_overlap, **common), True
    return pid_core.pid_decode_latent(net, lat5, **common), False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_images", type=int, default=24, help="latents to sample (evenly spaced); <=0 = all of the pool")
    ap.add_argument("--one_per_artist", action="store_true",
                    help="pool = one (middle) latent per artist dir, not evenly over all latents (diversity)")
    ap.add_argument("--steps", type=int, default=4, help="PiD SDE steps — FIT AT YOUR DECODE STEP COUNT")
    ap.add_argument("--crop", type=int, default=0, help="center latent crop in latent px; 0 = FULL latent (real 4x SR)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile the PiD net (per-block, fixed tile -> compiles once); --no-compile to disable")
    ap.add_argument("--tile_latent", type=int, default=64,
                    help="latent-grid tile for the SR decode; latents larger than this auto-tile (64 -> 2048px tiles)")
    ap.add_argument("--tile_overlap", type=int, default=16, help="latent-grid tile overlap (feather-blended)")
    ap.add_argument("--max_latent", type=int, default=0, help="skip latents whose grid exceeds this (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0, help="PiD sampling seed (fixed for reproducibility)")
    ap.add_argument("--n_compare", type=int, default=4, help="triplets in comparison.png")
    ap.add_argument("--no_null_caption", action="store_true", help="use zero caption instead of faithful gemma null")
    ap.add_argument("--pid_ckpt", type=Path, default=DEFAULT_PID_CKPT)
    ap.add_argument("--null_caption", type=Path, default=DEFAULT_NULL_CAP)
    ap.add_argument("--vae_pth", type=Path, default=DEFAULT_VAE)
    ap.add_argument("--node_dir", type=Path, default=DEFAULT_NODE_DIR)
    ap.add_argument("--latent_glob", default=DEFAULT_LATENT_GLOB)
    ap.add_argument("--label", default="colorcalib")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    pid_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ---- gather latents (evenly spaced over a sorted glob for content diversity) ----
    all_npz = sorted(REPO_ROOT.glob(args.latent_glob))
    if not all_npz:
        raise SystemExit(f"no latents matched {args.latent_glob} under {REPO_ROOT}")
    if args.one_per_artist:
        by_artist: dict[str, list] = {}
        for p in all_npz:
            by_artist.setdefault(p.parent.name, []).append(p)
        pool = [grp[len(grp) // 2] for _, grp in sorted(by_artist.items())]  # middle latent per artist
        print(f"[data] {len(all_npz)} latents across {len(by_artist)} artists; one (middle) per artist")
    else:
        pool = all_npz
        print(f"[data] {len(all_npz)} latents found")
    limit = args.num_images if args.num_images and args.num_images > 0 else len(pool)
    n = min(limit, len(pool))
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
    npz_paths = [pool[i] for i in idx]
    print(f"[data] sampling {n} latents (evenly spaced over pool)")

    # ---- native Qwen VAE (reference decoder) ----
    print(f"[vae] loading {args.vae_pth}")
    vae = WanVAE2d_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                    attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
    vae.load_state_dict(torch.load(args.vae_pth, map_location="cpu", weights_only=False), strict=False)
    vae = vae.to(dev, torch.float32).eval().requires_grad_(False)
    vae_mean = torch.tensor(_LATENTS_MEAN, device=dev)
    vae_std = torch.tensor(_LATENTS_STD, device=dev)
    vae_scale = [vae_mean, 1.0 / vae_std]

    # ---- PiD decoder (candidate) ----
    pid_core = load_pid_core(args.node_dir)
    print(f"[pid] building net + loading {args.pid_ckpt}")
    net = pid_core.build_pid_net(device=device, dtype=pid_dtype)
    missing, unexpected = pid_core.load_pid_weights(net, str(args.pid_ckpt))
    _, suspect, unexp = pid_core.categorize_load_keys(missing, unexpected)
    if suspect or unexp:
        raise SystemExit(f"PiD ckpt/arch mismatch — suspect_missing={suspect[:5]} unexpected={unexp[:5]}")
    null_cap = None
    if not args.no_null_caption:
        null_cap = pid_core.load_null_caption_embs(str(args.null_caption), dev, pid_dtype)
        print("[pid] using faithful gemma(chi_prompt+'') null caption")
    else:
        print("[pid] using ZERO caption (off-distribution)")

    # ---- accumulators (float64, pixel-storage-free) ----
    # split: even index = train (fit), odd index = test (held-out generalization check)
    def new_acc():
        return {"A": np.zeros((4, 4)), "C": np.zeros((4, 3)), "Drr": np.zeros(3), "n": 0}

    acc = {"train": new_acc(), "test": new_acc(), "all": new_acc()}
    # per-channel scalar sums (for the simple affine + "before" RMSE), over all data
    pc = {k: np.zeros(3) for k in ("sp", "sr", "spp", "srr", "spr")}
    pc["n"] = 0
    # Oklab global moments (knob decode) + per-image offsets (spread diagnostic)
    okl = {f"{w}_{m}": np.zeros(3) for w in ("p", "r") for m in ("s", "ss")}
    okl["n"] = 0
    per_img_off = []  # (dL, da, db) per image — ref minus pid means
    compare = []  # (pid_lowres, ref) for the strip

    processed = 0
    for i, p in enumerate(npz_paths):
        lat = center_crop(load_latent(p), args.crop).to(dev)
        lh, lw = lat.shape[-2:]
        if args.max_latent and max(lh, lw) > args.max_latent:
            print(f"  [{i + 1}/{n}] skip {p.parent.name}/{p.stem}: latent {lh}x{lw} > max_latent {args.max_latent}")
            continue
        lat5 = lat.unsqueeze(0)  # (1,16,h,w)

        with torch.no_grad():
            ref = vae.decode(lat5.float(), vae_scale).clamp(-1, 1)  # (1,3, h*8, w*8) in [-1,1]
            pid, tiled = decode_pid(pid_core, net, lat5.to(pid_dtype), args, null_cap, pid_dtype)
            # (1,3, h*32, w*32) in [-1,1]

        ref01 = (ref + 1.0) * 0.5  # (1,3,Hr,Wr)
        pid01 = (pid.float() + 1.0) * 0.5
        # align: PiD is 4x SR -> downsample to VAE resolution (area = honest color avg)
        pid01 = F.interpolate(pid01, size=ref01.shape[-2:], mode="area").clamp(0, 1)

        # (Npx,3) pixel tables
        P = pid01[0].permute(1, 2, 0).reshape(-1, 3).double()
        R = ref01[0].permute(1, 2, 0).reshape(-1, 3).double()
        Paug = torch.cat([P, torch.ones(P.shape[0], 1, device=dev, dtype=torch.float64)], dim=1)
        A = (Paug.T @ Paug).cpu().numpy()
        C = (Paug.T @ R).cpu().numpy()
        Drr = (R * R).sum(0).cpu().numpy()
        npx = P.shape[0]
        split = "train" if i % 2 == 0 else "test"
        for tgt in (split, "all"):
            acc[tgt]["A"] += A
            acc[tgt]["C"] += C
            acc[tgt]["Drr"] += Drr
            acc[tgt]["n"] += npx

        # per-channel scalar sums
        pc["sp"] += P.sum(0).cpu().numpy()
        pc["sr"] += R.sum(0).cpu().numpy()
        pc["spp"] += (P * P).sum(0).cpu().numpy()
        pc["srr"] += Drr
        pc["spr"] += (P * R).sum(0).cpu().numpy()
        pc["n"] += npx

        # Oklab moments
        ok_p = rgb01_to_oklab(pid01[0].permute(1, 2, 0)).reshape(-1, 3).double()
        ok_r = rgb01_to_oklab(ref01[0].permute(1, 2, 0)).reshape(-1, 3).double()
        okl["p_s"] += ok_p.sum(0).cpu().numpy()
        okl["p_ss"] += (ok_p * ok_p).sum(0).cpu().numpy()
        okl["r_s"] += ok_r.sum(0).cpu().numpy()
        okl["r_ss"] += (ok_r * ok_r).sum(0).cpu().numpy()
        okl["n"] += npx
        per_img_off.append((ok_r.mean(0) - ok_p.mean(0)).cpu().numpy())

        if len(compare) < args.n_compare:
            compare.append((pid01[0].cpu().numpy(), ref01[0].cpu().numpy()))

        processed += 1
        print(f"  [{i + 1}/{n}] {p.parent.name}/{p.stem}  latent {lh}x{lw} -> "
              f"PiD {lh * 32}x{lw * 32}px{' (tiled)' if tiled else ''} | ref {ref01.shape[-2]}x{ref01.shape[-1]}  npx={npx}")
        del lat, lat5, ref, pid, ref01, pid01, P, R, Paug
        if device == "cuda":
            torch.cuda.empty_cache()

    if processed == 0:
        raise SystemExit("no latents processed (all skipped by --max_latent?)")
    n = processed

    # ---- solve transforms ----
    def solve_M(a):  # M_aug (4x3): pred_c = [p,1] . M_aug[:,c]
        return np.linalg.solve(a["A"] + 1e-6 * np.eye(4), a["C"])

    def rmse(a, M):  # RMSE of applying M_aug to split `a`
        sse = 0.0
        for c in range(3):
            mc = M[:, c]
            sse += float(mc @ a["A"] @ mc - 2 * mc @ a["C"][:, c] + a["Drr"][c])
        return float(np.sqrt(max(sse, 0.0) / (a["n"] * 3)))

    M_all = solve_M(acc["all"])
    M_train = solve_M(acc["train"])
    linear_M = M_all[:3, :].T.copy()  # (3,3): out = pid @ linear_M.T + linear_b
    linear_b = M_all[3, :].copy()  # (3,)

    # before RMSE (identity) over all data, from per-channel sums
    before_sse = float((pc["spp"] - 2 * pc["spr"] + pc["srr"]).sum())
    rmse_before = float(np.sqrt(before_sse / (pc["n"] * 3)))
    rmse_after_all = rmse(acc["all"], M_all)
    rmse_train = rmse(acc["train"], M_train)
    rmse_test = rmse(acc["test"], M_train)  # M fit on TRAIN, eval on held-out TEST

    # simple per-channel affine (regression, MSE-optimal per channel)
    nP = pc["n"]
    mp, mr = pc["sp"] / nP, pc["sr"] / nP
    var_p = pc["spp"] / nP - mp * mp
    cov_pr = pc["spr"] / nP - mp * mr
    pc_gain = cov_pr / np.maximum(var_p, 1e-8)
    pc_bias = mr - pc_gain * mp

    # ---- Oklab knobs (the principled ComfyUI-chain equivalent) ----
    nO = okl["n"]
    mp_o = okl["p_s"] / nO
    mr_o = okl["r_s"] / nO
    vp_o = okl["p_ss"] / nO - mp_o * mp_o
    vr_o = okl["r_ss"] / nO - mr_o * mr_o
    L_gain = float(np.sqrt(vr_o[0] / max(vp_o[0], 1e-12)))  # ~ contrast
    L_shift = float(mr_o[0] - L_gain * mp_o[0])  # ~ brightness
    sat = float(np.sqrt((vr_o[1] + vr_o[2]) / max(vp_o[1] + vp_o[2], 1e-12)))  # ~ saturation
    tint_a = float(mr_o[1] - mp_o[1])  # +a redder, -a greener  (ref - pid)
    tint_b = float(mr_o[2] - mp_o[2])  # +b yellower, -b bluer

    # ---- per-image spread (THE diagnostic) ----
    off = np.array(per_img_off)  # (N,3) = ref-pid Oklab mean offset per image
    spread = off.std(0)  # std across images
    mean_off = off.mean(0)

    run_dir = make_run_dir("pid", label=args.label)

    # ---- save the bakeable calib ----
    from safetensors.torch import save_file

    save_file(
        {
            "linear_M": torch.from_numpy(linear_M).float(),  # (3,3)
            "linear_b": torch.from_numpy(linear_b).float(),  # (3,)
            "per_channel_gain": torch.from_numpy(pc_gain).float(),  # (3,)
            "per_channel_bias": torch.from_numpy(pc_bias).float(),  # (3,)
        },
        str(run_dir / "pid_color_calib.safetensors"),
        metadata={
            "apply": "out = (pid_rgb01 @ linear_M.T + linear_b).clamp(0,1)  # PiD->VAE color match",
            "steps": str(args.steps),
            "crop": str(args.crop),
            "null_caption": str(not args.no_null_caption),
        },
    )

    # ---- report ----
    lines = [
        "PiD -> native-VAE color calibration",
        "=" * 52,
        f"images={n}  steps={args.steps}  crop={'full' if args.crop <= 0 else args.crop}  "
        f"compile={args.compile}  tile_latent={args.tile_latent}  "
        f"null_caption={'gemma' if not args.no_null_caption else 'zeros'}  seed={args.seed}",
        "",
        "RMSE (RGB [0,1], lower=closer to native VAE):",
        f"  before (raw PiD)      : {rmse_before:.5f}",
        f"  after 3x3+bias (all)  : {rmse_after_all:.5f}   ({100 * (1 - rmse_after_all / rmse_before):.1f}% drop)",
        f"  3x3 train (fit)       : {rmse_train:.5f}",
        f"  3x3 test  (held-out)  : {rmse_test:.5f}   <- generalization; if >> train, fit set too narrow",
        "",
        "Linear-RGB 3x3 + bias  (THE BAKE: out = pid @ M.T + b):",
        f"  M =\n{np.array2string(linear_M, precision=4, prefix='      ')}",
        f"  b = {np.array2string(linear_b, precision=4)}",
        "",
        "Per-channel affine (simple WB+exposure, regression):",
        f"  gain(R,G,B) = {np.array2string(pc_gain, precision=4)}",
        f"  bias(R,G,B) = {np.array2string(pc_bias, precision=4)}",
        "",
        "Oklab moment-match -> the PRINCIPLED ComfyUI-chain equivalent:",
        f"  contrast (L gain)     = {L_gain:.4f}   (node 'contrast')",
        f"  brightness (L shift)  = {L_shift:+.4f}  (node 'levels' mid)",
        f"  saturation (chroma)   = {sat:.4f}   (node 'saturation')",
        f"  tint a (red<->green)  = {tint_a:+.4f}",
        f"  tint b (yellow<->blue)= {tint_b:+.4f}  ({'pid is WARM -> cool it' if tint_b < 0 else 'pid is COOL -> warm it'})",
        "",
        "PER-IMAGE SPREAD (the key diagnostic):",
        f"  mean offset (ref-pid) Oklab(L,a,b) = {np.array2string(mean_off, precision=4)}",
        f"  std  offset across images          = {np.array2string(spread, precision=4)}",
        "  -> small std  => static bake fixes the drift (systematic).",
        "  -> large std  => irreducible 4-step SDE variance (the authors'",
        "                   'early-termination whitening'); raise --steps / use ODE.",
    ]
    report = "\n".join(lines)
    (run_dir / "report.txt").write_text(report + "\n")
    print("\n" + report)

    # ---- comparison strip (best-effort; never lose the calib over a cosmetic glitch) ----
    artifacts = ["pid_color_calib.safetensors", "report.txt"]
    try:
        M_t = torch.from_numpy(linear_M).float()
        b_t = torch.from_numpy(linear_b).float()
        rows = []
        for pid_lr, ref_img in compare:
            pid_t = torch.from_numpy(pid_lr).permute(1, 2, 0)  # (H,W,3) at ref resolution
            calib = (pid_t @ M_t.T + b_t).clamp(0, 1)
            ref_t = torch.from_numpy(ref_img).permute(1, 2, 0)
            panels = torch.stack([pid_t, calib, ref_t]).permute(0, 3, 1, 2)  # (3,3,H,W)
            s = min(1.0, 512 / max(panels.shape[-2:]))  # cap long side ~512px
            if s < 1.0:
                panels = F.interpolate(panels, scale_factor=s, mode="area")
            row = torch.cat(list(panels.permute(0, 2, 3, 1)), dim=1)  # [PiD|PiD+calib|VAE-ref]
            rows.append(row)
        # rows may differ in width (varying aspect ratios) -> normalize to a common width
        tw = min(r.shape[1] for r in rows)
        rows = [
            F.interpolate(r.permute(2, 0, 1)[None], size=(round(r.shape[0] * tw / r.shape[1]), tw),
                          mode="area")[0].permute(1, 2, 0)
            for r in rows
        ]
        strip = torch.cat(rows, dim=0).clamp(0, 1)
        Image.fromarray((strip.numpy() * 255).astype(np.uint8)).save(run_dir / "comparison.png")
        artifacts.append("comparison.png")
    except Exception as e:  # noqa: BLE001 — strip is a convenience, not the deliverable
        print(f"[warn] comparison strip failed ({e}); calib + report still saved")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        device=device,
        metrics={
            "rmse_before": rmse_before,
            "rmse_after_all": rmse_after_all,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "rmse_drop_pct": 100 * (1 - rmse_after_all / rmse_before),
            "linear_M": linear_M.tolist(),
            "linear_b": linear_b.tolist(),
            "per_channel_gain": pc_gain.tolist(),
            "per_channel_bias": pc_bias.tolist(),
            "oklab_knobs": {
                "contrast_L_gain": L_gain,
                "brightness_L_shift": L_shift,
                "saturation_chroma": sat,
                "tint_a": tint_a,
                "tint_b": tint_b,
            },
            "per_image_offset_mean_Lab": mean_off.tolist(),
            "per_image_offset_std_Lab": spread.tolist(),
        },
        artifacts=artifacts,
    )
    print(f"\n[done] {run_dir}")


if __name__ == "__main__":
    main()
