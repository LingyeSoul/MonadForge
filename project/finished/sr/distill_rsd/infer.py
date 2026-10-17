"""Single-step RSD student inference + eval against the v2 teacher.

The student is a 1-step generator: ẑ0 = G_θ(z_T, z_y, T-1, ε), z_T ~ N(z_y, κ²I),
decoded by the VQGAN. Tiling is handled by chopping the LR latent. Eval mode scores
student (NFE=1) vs the released v2 teacher (NFE=15) vs bicubic on the Phase-0 set.
"""
import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from rsd_models import TEACHER_CKPT, make_eps, predict_x0  # noqa: E402
from utils.util_image import ImageSpliterTh  # noqa: E402  (resshift path added by rsd_models)

REPO = M.REPO
CKPT_DIR = REPO / "output" / "sr" / "rsd"


class FeatheredSpliter(ImageSpliterTh):
    """ImageSpliterTh with raised-cosine feathered overlap blending.

    The upstream gather() is a uniform box average: every covering tile gets weight 1, so
    the blend weight STEPS at the overlap edges (1 tile ↔ 2 tiles). Even with the shared
    noise field, tiles disagree slightly in the overlap (each tile's VQGAN encode + Swin
    sees different context), and the box average concentrates that whole disagreement into
    a visible line exactly at the overlap edges (measured 2026-07-08 on the x2 comfy
    renders: seams sit at the 1→2/2→1 transitions, ~1.5× background even with shared
    noise). Feathering ramps each tile's weight 0→1 over the overlap width from its own
    edge (Hann profile, separable), so any tile disagreement fades in smoothly instead of
    switching on. gather() renormalizes by the accumulated weight, so borders (no
    neighbour → single tile at partial weight) still come out exact; overlap == 0 falls
    back to the box path unchanged."""

    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1):
        super().__init__(im, pch_size, stride, sf=sf, extra_bs=extra_bs)
        self.ramp = (pch_size - stride) * sf   # overlap width at output res = feather span
        self._wcache = {}

    def _tile_weight(self, h, w, dtype, device):
        if self.ramp <= 0:
            return None
        key = (h, w, dtype, device)
        if key not in self._wcache:
            def prof(length):
                d = torch.arange(length, device=device, dtype=torch.float32) + 0.5
                d = torch.minimum(d, length - d).clamp(max=self.ramp) / self.ramp
                return 0.5 - 0.5 * torch.cos(torch.pi * d)
            w2d = (prof(h)[:, None] * prof(w)[None, :]).clamp(min=1e-3)
            self._wcache[key] = w2d.to(dtype)[None, None]
        return self._wcache[key]

    def update(self, pch_res, index_infos):
        wt = self._tile_weight(pch_res.shape[-2], pch_res.shape[-1],
                               pch_res.dtype, pch_res.device)
        if wt is None:
            return super().update(pch_res, index_infos)
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for cur, (h0, h1, w0, w1) in zip(pch_list, index_infos):
            self.im_res[:, :, h0:h1, w0:w1] += cur * wt
            self.pixel_count[:, :, h0:h1, w0:w1] += wt


def latest_ckpt(ckpt_dir: Path) -> Path:
    """Most recently written rsd_student_*.pth (final or numbered)."""
    cands = sorted(ckpt_dir.glob("rsd_student_*.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no rsd_student_*.pth under {ckpt_dir} — train first or pass --ckpt")
    return cands[-1]


def load_lr(path, device, scale):
    """Load LR image, bicubic-upsample x scale to HR size (ResShift convention)."""
    im = Image.open(path).convert("RGB")
    up = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(up, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    return t.to(device), up


def _dc_loss(a, b, k=32):
    """Low-freq (DC + coarse tone) L1 on [-1,1] images — mirrors train.py's dc_loss."""
    return F.l1_loss(F.avg_pool2d(a, k), F.avg_pool2d(b, k)).item()


@torch.no_grad()
def roundtrip_ref(vqgan, gt_t, sf_scale, vdt):
    """decode(z0), z0 = scale_factor * encode(gt) — the roundtrip-consistent color
    reference the training image losses target (train.py:264), NOT raw gt.

    The frozen VQ roundtrip decode(encode(gt)) is not identity: it carries a fixed
    directional color tint on this art data (~[+2.5R, +0.9B] u8), so a latent-faithful
    student's output inherits that tint. Scoring color vs raw HR therefore penalizes the
    correct student and rewards an anti-tint hack — decode(z0) puts the color minimum at
    the latent objective's zero (pred==z0). gt_t is a [-1,1] tensor at output resolution."""
    z0 = vqgan.encode(gt_t.to(vdt)) * sf_scale
    return vqgan.decode(z0, force_not_quantize=True).float().clamp(-1, 1)


@torch.no_grad()
def _sample_tile(cfg, diff, vqgan, student, hr_t, align, device, amp=True,
                 zT_noise=None, eps_noise=None, cond_mode="latent"):
    """One tile: reflect-pad to `align` (Swin needs dims % lq_size), encode -> 1-step
    student -> decode -> crop back. In/out 1x3xHxW in [-1,1].

    Heavy matmuls run under bf16 autocast by default (matches training's --amp, ~2x
    faster + ~half VRAM); the returned tile is cast back to fp32 for seam-averaging.

    The VAE is the VRAM peak (full pixel-res conv stack). When bf16, it runs in NATIVE
    bf16 *outside* autocast: autocast keeps GroupNorm on its fp32 list, which would
    materialize the full-res norm activations in fp32 (the dominant tensors). Native
    bf16 keeps them bf16 too — GroupNorm still reduces in fp32 internally, so it's safe.
    The student stays under autocast (matches training).

    `zT_noise`/`eps_noise` (latent-shaped, chop//f-sized) inject a precomputed noise
    slice instead of drawing fresh per-tile randn — the shared-noise path
    (`student_sr(shared_noise=True)`) uses this so overlapping tiles draw the SAME noise
    field and the overlap-average is seam-free. Both default to None = the original
    independent per-tile draw. Slices are cropped to z_y's actual latent dims, so skinny
    tiles (image dim ≤ chop on one axis, padded to less than chop) work too."""
    h, w = hr_t.shape[2:]
    ph, pw = (-h) % align, (-w) % align
    if ph or pw:
        hr_t = F.pad(hr_t, (0, pw, 0, ph), mode="reflect")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    vdt = next(vqgan.parameters()).dtype   # bf16 when amp (see main), else fp32
    z_y = vqgan.encode(hr_t.to(vdt)) * cfg.diffusion.params.scale_factor
    # Noise slices are chop//f-sized; a skinny tile (image dim ≤ chop on one axis) pads to
    # LESS than chop, so crop the slice to the actual latent dims before injecting.
    hz, wz = z_y.shape[2:]
    zt_n = (torch.randn_like(z_y) if zT_noise is None
            else zT_noise[:, :, :hz, :wz].to(z_y.dtype))
    z_T = z_y + diff.kappa * zt_n
    eps = (make_eps(student, z_y) if eps_noise is None
           else eps_noise[:, :, :hz, :wz].to(z_y.dtype))
    tT = torch.full((z_y.shape[0],), diff.num_timesteps - 1, device=device, dtype=torch.long)
    # `lq` conditioning is NOT the residual base z_y — students distilled after 2026-07-11
    # inherit the released pixel convention (see rsd_models.make_cond_lq). cond_mode comes
    # from the ckpt meta, so pre-fix students keep their latent conditioning and still run.
    c_y = M.make_cond_lq(hr_t, z_y, cond_mode)
    with ctx:
        z0 = predict_x0(diff, student, z_T, c_y, tT, eps)
    img = vqgan.decode(z0.to(vdt), force_not_quantize=True).clamp(-1, 1)
    return img[:, :, :h, :w].float()


@torch.no_grad()
def student_sr(cfg, diff, vqgan, student, lq_img, device, align=256, overlap=None,
               chop=512, tile_batch=1, amp=True, shared_noise=True, noise_seed=None,
               cond_mode="latent"):
    """1-step student SR on an HR-sized (upsampled-LR) tensor 1x3xHxW in [-1,1].

    The student is spatially same-resolution (the x4 lives in the residual-shift, not a
    spatial upscale), so we tile with sf=1. Large images are chopped into overlapping
    `chop`-px tiles and overlap-averaged (ImageSpliterTh) — both for VRAM and because the
    Swin attention is built for the lq_size grid.

    `align` is the Swin pad-alignment (each tile is reflect-padded to a multiple of it);
    `overlap` is the seam overlap that sets the tile stride (`chop - overlap`) and is
    decoupled from `align` so small tiles are possible (default = align).

    `tile_batch` stacks that many tiles into one forward (ImageSpliterTh.extra_bs) — small
    tiles underfill the GPU one-at-a-time, so batching amortizes launch/overhead for a big
    speedup; VRAM scales ~linearly with it, so it's a card-specific cap. Every tile is
    exactly chop×chop (chop is an align multiple), so the batch stacks with no ragged pad.

    `shared_noise` (default on): the student is stochastic — it injects two independent
    noise draws per forward (the 3-ch residual-shift `z_T` noise + the 1-ch `make_eps`
    injection). Drawn per-tile, neighbouring tiles get INDEPENDENT noise fields and the
    overlap-average has to blend that discontinuity, leaving a faint tile-block seam in
    flat regions. Instead we draw ONE full-image latent noise field per source up front
    and slice each tile's noise from it by its latent position (pixel start // f), so
    overlapping tiles read IDENTICAL noise and the average is seam-free. The field is a
    few MB (latent-res, ≪ the VAE decode peak) and drawing it once is marginally cheaper
    than per-tile. `noise_seed` seeds it for reproducibility (None = the prior unseeded
    behaviour, now spatially coherent). Interior tiles align exactly (their pixel starts
    are f-multiples); a last tile whose start isn't an f-multiple is shifted by <f px of
    noise — one boundary, negligible. Only the tiled path uses it (a single un-tiled tile
    has no seam)."""
    if overlap is None:
        overlap = align
    H, W = lq_img.shape[2:]
    if H > chop or W > chop:
        spliter = FeatheredSpliter(lq_img, chop, stride=chop - overlap, sf=1, extra_bs=tile_batch)
        g_zT = g_eps = None
        if shared_noise:
            ch_mult = cfg.autoencoder.params.ddconfig.ch_mult
            f = 2 ** (len(ch_mult) - 1)               # VQ-f4 -> 4
            z_ch = cfg.autoencoder.params.ddconfig.z_channels
            gen = None
            if noise_seed is not None:
                gen = torch.Generator(device=device).manual_seed(int(noise_seed))
            gpad = chop // f                          # margin so any tile start slices in-bounds
            hl, wl = H // f + gpad, W // f + gpad
            g_zT = torch.randn(1, z_ch, hl, wl, device=device, generator=gen)
            g_eps = torch.randn(1, student.noise_channels, hl, wl, device=device, generator=gen)
            lt = chop // f                            # per-tile latent size
        for pch, info in spliter:
            zt_noise = eps_noise = None
            if shared_noise:
                zt_noise = torch.cat([g_zT[:, :, h0 // f:h0 // f + lt, w0 // f:w0 // f + lt]
                                      for (h0, _, w0, _) in info], dim=0)
                eps_noise = torch.cat([g_eps[:, :, h0 // f:h0 // f + lt, w0 // f:w0 // f + lt]
                                       for (h0, _, w0, _) in info], dim=0)
            spliter.update(_sample_tile(cfg, diff, vqgan, student, pch, align, device, amp,
                                        zT_noise=zt_noise, eps_noise=eps_noise,
                                        cond_mode=cond_mode), info)
        img = spliter.gather()
    else:
        img = _sample_tile(cfg, diff, vqgan, student, lq_img, align, device, amp,
                           cond_mode=cond_mode)
    return ((img[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(M.VERSIONS), default="x4",
                    help="scale family: x4 = released teacher schedule; x2 / x4ft = our sr-train "
                         "finetunes. Selects the config (hence sf) and the default ckpt_dir/out_dir. "
                         "Must match the version the student was distilled under.")
    ap.add_argument("--config", default=None, help="override the version's ResShift config yaml")
    ap.add_argument("--cond_lq", choices=["pixel", "latent"], default=None,
                    help="override the `lq` conditioning mode (default: the ckpt's own "
                         "cond_lq meta, or latent for pre-2026-07-11 ckpts that lack it). "
                         "Must match what the student was distilled with.")
    ap.add_argument("--ckpt", default=None,
                    help="rsd_student_*.pth (uses EMA weights); default = most recent in --ckpt_dir")
    ap.add_argument("--ckpt_dir", default=None,
                    help="dir to pick the most recent ckpt from when --ckpt is unset "
                         "(default: output/sr/rsd for x4, output/sr/rsd_<version> otherwise)")
    ap.add_argument("--eval", action="store_true",
                    help="reference eval on the Phase-0 set: MUSIQ (no-ref) plus, when --hr_dir "
                         "matches, DC + LPIPS against the ROUNDTRIP-consistent decode(z0) "
                         "reference (NOT raw HR). The frozen VQ roundtrip carries a fixed color "
                         "tint on this art data, so scoring color vs raw HR penalizes a "
                         "latent-faithful student for the decoder's tint and rewards an anti-tint "
                         "hack — the exact trap the training-loss fix (commit 3119d5d3) removed. "
                         "decode(z0) is the correct color reference; raw-HR DC is kept as a "
                         "diagnostic (dc_hr) so the tint gap stays visible.")
    ap.add_argument("--in_dir", default=str(REPO / "sr" / "data" / "lr_eval"))
    ap.add_argument("--hr_dir", default=str(REPO / "sr" / "data" / "hr_eval"),
                    help="ground-truth HR dir (stem-matched to --in_dir) for the reference DC/LPIPS "
                         "in --eval. Skipped per-image when no stem match exists.")
    ap.add_argument("--out_dir", default=None,
                    help="default: <ckpt_dir>/infer")
    ap.add_argument("--chop", type=int, default=256, help="tile size (px) for large images; must be a multiple of the align stride. 2048 single-tiles the 512->2048 eval (no overlap-redundancy ~2.2x compute saving); lower it (e.g. 1024) if VRAM-bound")
    ap.add_argument("--overlap", type=int, default=64,
                    help="tile seam overlap in px (default 64). Sets the tile stride = chop - "
                         "overlap, which must be > 0. Independent of Swin alignment; only --chop "
                         "must be a multiple of the align stride. The overlap is also the feather "
                         "ramp width (raised-cosine blend weights across it), so 0 = hard-abutted "
                         "box tiles; smaller overlap = fewer redundant tiles but narrower blending.")
    ap.add_argument("--tile_batch", type=int, default=8,
                    help="stack this many chop-tiles into one forward (default 1 = serial). Small "
                         "tiles underfill the GPU one-at-a-time; batching amortizes launch overhead "
                         "for a big speedup. VRAM scales ~linearly, so tune to the card (e.g. "
                         "--chop 256 --tile_batch 8).")
    ap.add_argument("--no_bf16", action="store_true", help="disable bf16 autocast (inference matches training's --amp by default; ~2x slower + 2x VRAM if set)")
    ap.add_argument("--no_shared_noise", action="store_true",
                    help="disable the shared full-image noise field (revert to independent "
                         "per-tile noise). Shared noise (default on) makes overlapping tiles draw "
                         "identical noise so the overlap-average is seam-free — a few MB, quality-"
                         "neutral. Only matters when the image is tiled (> --chop).")
    ap.add_argument("--noise_seed", type=int, default=None,
                    help="seed the shared noise field for reproducible output (default: unseeded).")
    ap.add_argument("--weights", choices=["ema", "student"], default="ema",
                    help="ema = smoothed (best late); student = raw (better for EARLY ckpts, "
                         "EMA still drags the teacher-init there)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(student, dynamic=True) — marks spatial dims dynamic so "
                         "the eval's all-different tile shapes share one graph (avoids a recompile "
                         "per image). Swin window_partition may force graph breaks; first tile pays "
                         "warmup. Worth it for many tiles, a loss for a handful.")
    args = ap.parse_args()
    config_path, _ = M.resolve_version(args.version, args.config)
    default_sub = "rsd" if args.version == "x4" else f"rsd_{args.version}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else REPO / "output" / "sr" / default_sub
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "infer"
    args.out_dir = str(out_dir)
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(ckpt_dir)
    print(f"version={args.version} config={config_path.name} ckpt={ckpt}")
    dev = M.DEVICE
    cfg = M.load_configs(config_path)
    diff = M.build_diffusion(cfg)
    vqgan = M.build_autoencoder(cfg, dev)
    if not args.no_bf16:
        # native bf16 VAE — halves the full-res GroupNorm activations that autocast would
        # otherwise keep in fp32 (the VRAM peak). Codebook is unused (force_not_quantize).
        vqgan = vqgan.to(torch.bfloat16)
    sd = torch.load(ckpt, map_location="cpu")
    # rebuild the noise-injection arch the ckpt was trained with (ckpt_meta from train.py;
    # old ckpts predate it -> default to the "add" reconstruction).
    student = M.build_generator(cfg, str(TEACHER_CKPT), dev, pretrained=False,
                                noise_mode=sd.get("noise_mode", "add"),
                                noise_channels=sd.get("noise_channels"))
    key = args.weights if args.weights in sd else ("ema" if "ema" in sd else None)
    student.load_state_dict(sd[key] if key else sd, strict=True)
    # ckpts written before the 2026-07-11 cond_lq fix carry no meta -> they were distilled
    # under latent conditioning and must keep it.
    cond_mode = args.cond_lq or sd.get("cond_lq", "latent")
    print(f"weights: {key or 'raw'} | noise_mode: {student.noise_mode}"
          f"({student.noise_channels}ch) | cond_lq: {cond_mode}")
    student.eval()
    if args.compile:
        # Bump the recompile ceiling: dynamic=True still specializes on a few discrete
        # guards (Swin window_partition reshapes), so the 24-distinct-shape eval can trip
        # the default limit of 8 and silently fall back to eager. (No backward pass here,
        # so the ContextVar-reversion gotcha that bites training doesn't apply.)
        import torch._dynamo as dynamo
        for attr in ("recompile_limit", "cache_size_limit"):  # renamed across torch versions
            if hasattr(dynamo.config, attr):
                setattr(dynamo.config, attr, 64)
        student = torch.compile(student, dynamic=True)
        print("torch.compile(dynamic=True) enabled on student")
    scale = cfg.diffusion.params.sf
    # Pixel alignment the SwinUNet needs. Tiles are cut at MODEL-INPUT (post-upsample)
    # resolution, so latent = pixels / f_vq and the deepest level runs at latent / 2^(L-1)
    # with a hard window_partition view (no internal pad): each tile dim must be divisible
    # by f_vq·window·2^(L-1) = 256, INDEPENDENT of sf. (The old ×sf formula was correct at
    # ×4 only because sf == f_vq == 4; at ×2 it under-aligned to 128, so an edge tile
    # reflect-padded to a 128-but-not-256 multiple crashed the deepest window_partition.)
    n_levels = len(cfg.model.params.get("channel_mult", [1, 2, 2, 4]))
    f_vq = 2 ** (len(cfg.autoencoder.params.ddconfig.ch_mult) - 1)
    align = f_vq * int(cfg.model.params.window_size) * (2 ** (n_levels - 1))
    if args.chop % align:
        raise SystemExit(f"--chop {args.chop} must be a multiple of {align} (f_vq×window×2^{n_levels-1})")
    overlap = args.overlap if args.overlap is not None else align
    if not 0 <= overlap < args.chop:
        raise SystemExit(f"--overlap {overlap} must be in [0, chop={args.chop}) so the tile "
                         f"stride (chop - overlap) is > 0 (chop == overlap => zero stride)")
    if args.tile_batch < 1:
        raise SystemExit(f"--tile_batch {args.tile_batch} must be >= 1")
    print(f"tiling: chop={args.chop} overlap={overlap} stride={args.chop - overlap} "
          f"align={align} tile_batch={args.tile_batch}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)
    rows = []
    try:
        import pyiqa
        musiq = pyiqa.create_metric("musiq", device=dev)
    except Exception:  # noqa: BLE001
        musiq = None

    # reference color/perceptual metrics (--eval only): scored against decode(z0), not raw HR.
    hr_dir = Path(args.hr_dir) if args.eval else None
    sf_scale = cfg.diffusion.params.scale_factor
    vdt = next(vqgan.parameters()).dtype
    lpips_fn = None
    if args.eval:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(dev).eval()
        except Exception:  # noqa: BLE001
            lpips_fn = None

    lr_paths = sorted(in_dir.glob("*.png"))
    pbar = tqdm(lr_paths, desc="student SR", unit="img")
    for lr_path in pbar:
        lq_t, _ = load_lr(lr_path, dev, scale)
        sr = student_sr(cfg, diff, vqgan, student, lq_t, dev, cond_mode=cond_mode,
                        align=align, overlap=overlap,
                        chop=args.chop, tile_batch=args.tile_batch, amp=not args.no_bf16,
                        shared_noise=not args.no_shared_noise, noise_seed=args.noise_seed)
        Image.fromarray(sr).save(out_dir / lr_path.name)
        row = {"stem": lr_path.stem}
        if musiq is not None:
            t = torch.from_numpy(sr.astype(np.float32) / 255).permute(2, 0, 1)[None]
            row["musiq_student"] = round(float(musiq(t).item()), 3)
            pbar.set_postfix(musiq=row["musiq_student"])
        if hr_dir is not None and (hr_dir / lr_path.name).exists():
            # student output and raw HR (resized to output res) as [-1,1] tensors
            sr_t = (torch.from_numpy(sr.astype(np.float32) / 127.5 - 1)
                    .permute(2, 0, 1)[None].to(dev))
            gt_im = (Image.open(hr_dir / lr_path.name).convert("RGB")
                     .resize((sr.shape[1], sr.shape[0]), Image.BOX))
            gt_t = (torch.from_numpy(np.asarray(gt_im, np.float32) / 127.5 - 1)
                    .permute(2, 0, 1)[None].to(dev))
            with torch.no_grad():
                ref = roundtrip_ref(vqgan, gt_t, sf_scale, vdt)   # decode(z0): correct color ref
                row["dc_z0"] = round(_dc_loss(sr_t, ref), 5)      # primary color metric
                row["dc_hr"] = round(_dc_loss(sr_t, gt_t), 5)     # diagnostic: raw-HR (tint-biased)
                if lpips_fn is not None:
                    row["lpips_z0"] = round(float(lpips_fn(sr_t, ref).item()), 5)
        rows.append(row)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None
    # alloc_retries > 0 means we hit the OOM-edge retry path (free-cache + cudaFree + retry):
    # the tell-tale of a memory-bound regime where a bigger --chop is SLOWER despite fewer
    # FLOPs. reserved >> allocated also signals fragmentation/large-segment churn.
    if dev == "cuda":
        ms = torch.cuda.memory_stats()
        summary_mem = {"peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                       "num_alloc_retries": ms.get("num_alloc_retries", 0),
                       "num_ooms": ms.get("num_ooms", 0)}
    else:
        summary_mem = {}
    def _mean(key):
        vals = [r[key] for r in rows if key in r]
        return round(float(np.mean(vals)), 5) if vals else None

    summary = {"n": len(rows), "ckpt": str(ckpt), "chop": args.chop, "peak_vram_gb": peak_gb,
               **summary_mem,
               "musiq_student_mean": round(float(np.mean([r["musiq_student"] for r in rows
                                                          if "musiq_student" in r])), 3) if musiq else None}
    if args.eval:
        # decode(z0) is the correct color reference; dc_hr (raw-HR) is a tint-biased diagnostic.
        summary.update({"dc_z0_mean": _mean("dc_z0"), "lpips_z0_mean": _mean("lpips_z0"),
                        "dc_hr_mean_diag": _mean("dc_hr"),
                        "n_ref": sum(1 for r in rows if "dc_z0" in r)})
    (out_dir / "infer_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"student 1-step outputs -> {out_dir}")
    if args.eval:
        print("(DC/LPIPS above are vs the roundtrip decode(z0) reference — the training image-loss "
              "target. dc_hr_mean_diag is vs raw HR and is tint-biased: use dc_z0 to rank students. "
              "Teacher/bicubic comparison: run sr-phase0 metrics on the same set.)")


if __name__ == "__main__":
    main()
