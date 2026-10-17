"""ResShift inference — basicsr-free, over the vendored source.

A self-contained reimplementation of ResShift's ``inference_resshift.py`` + the core
of ``sampler.py``, using only the vendored ``sr/resshift`` tree (models / ldm /
util_image) — no basicsr, no datapipe, no external clone. Handles the released realsr
x4 versions (v1/v2 = 15-step, v3 = 4-step) AND our locally-trained art finetunes from
make sr-train (``x2`` / ``x4ft`` / ``x4s4`` — config + ckpt dir in _LOCAL, override the
checkpoint with --ckpt). The scale factor is read from the config (NOT hardcoded), so the
chop/tiling math generalizes. Tiled inference for large images + weight auto-download
(torch.hub) for the released x4 versions.

Output mirrors distill_rsd/infer.py: per-image SR PNGs + infer_summary.json (per-image
MUSIQ for SR and the bicubic baseline, means, peak VRAM) + contact_sheet.png of
[bicubic-up | SR] rows (the train_x2 montage style, minus GT — inference has none).
Local versions default -o to output/sr/<version>/infer; released x4 keeps sr/data/results.

    python sr_infer.py -i <img|dir> -o <out_dir> [--version v3] [--chop_size 512]
    python sr_infer.py -i <img|dir> --version x2 [--ckpt output/sr/x2/resshift_x2_final.pth]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# vendored ResShift source + rsd_models builders
HERE = Path(__file__).resolve().parent          # project/finished/sr/scripts
SR = HERE.parent                                 # project/finished/sr/
REPO = SR.parents[2]                             # repo root (output/ lives there)
sys.path.insert(0, str(SR / "distill_rsd"))
import rsd_models as M  # noqa: E402

from utils import util_common, util_image, util_net  # noqa: E402  (vendored ResShift)
from utils.util_image import ImageSpliterTh  # noqa: E402

# version -> (config filename, checkpoint filename, release URL)
_REL = "https://github.com/zsyOAOA/ResShift/releases/download/v2.0"
_VQGAN = ("autoencoder_vq_f4.pth", f"{_REL}/autoencoder_vq_f4.pth")
_VERSIONS = {
    "v1": ("realsr_swinunet_realesrgan256.yaml", "resshift_realsrx4_s15_v1.pth",
           f"{_REL}/resshift_realsrx4_s15_v1.pth"),
    "v2": ("realsr_swinunet_realesrgan256.yaml", "resshift_realsrx4_s15_v2.pth",
           f"{_REL}/resshift_realsrx4_s15_v2.pth"),
    "v3": ("realsr_swinunet_realesrgan256_journal.yaml", "resshift_realsrx4_s4_v3.pth",
           f"{_REL}/resshift_realsrx4_s4_v3.pth"),
}
# our locally-trained art models (no release URL — checkpoint is produced by make sr-train).
# version -> (config, default ckpt dir); the dirs mirror train_sr/train.py's VERSIONS.
_LOCAL = {
    "x2":   (SR / "configs" / "realsr_x2_art.yaml", REPO / "output" / "sr" / "x2"),
    "x4ft": (SR / "configs" / "realsr_x4_art.yaml", REPO / "output" / "sr" / "x4_art"),
    "x4s4": (SR / "configs" / "realsr_x4_s4_art.yaml", REPO / "output" / "sr" / "x4_s4_art"),
}
# ckpt file prefix per local version (train_sr writes resshift_<train-version>_<step>.pth)
_LOCAL_PREFIX = {"x2": "x2", "x4ft": "x4", "x4s4": "x4s4"}
# `lq` conditioning convention per version (rsd_models.make_cond_lq). The released weights
# want the LR PIXELS at latent resolution; only the x2 line was trained on the VQ latent.
# Passing the latent to a released checkpoint yields neon-green output — that regression
# shipped 2026-06-30..2026-07-11 and is what queued.md misdiagnosed as a bf16 decode bug.
_COND = {"v1": "pixel", "v2": "pixel", "v3": "pixel",
         "x2": "latent", "x4ft": "pixel", "x4s4": "pixel"}


def _latest_local_ckpt(ckpt_dir, prefix):
    """Newest resshift_<prefix>_*.pth in a make-sr-train output dir (prefer *_final)."""
    cands = sorted(Path(ckpt_dir).glob(f"resshift_{prefix}_*.pth"))
    if not cands:
        raise SystemExit(
            f"no resshift_{prefix}_*.pth in {ckpt_dir} — train one first (make sr-train) "
            f"or pass --ckpt <path>.")
    final = [c for c in cands if c.stem.endswith("final")]
    return final[0] if final else cands[-1]


def _musiq_score(metric, im_np):
    """MUSIQ on an HxWx3 uint8 RGB array."""
    t = torch.from_numpy(im_np.astype(np.float32) / 255).permute(2, 0, 1)[None]
    return round(float(metric(t).item()), 3)


def _contact_sheet(entries, path, cell_h=384, pad=4, strip=18):
    """[bicubic-up | SR] rows (train_x2 montage style) with a per-row label strip."""
    bg = (16, 16, 16)
    rows = []
    for lr_up, sr_im, label in entries:
        def _th(im):
            w = max(1, round(im.width * cell_h / im.height))
            return im.resize((w, cell_h), Image.LANCZOS)
        a, b = _th(lr_up), _th(sr_im)
        row = Image.new("RGB", (a.width + b.width + pad, cell_h + strip), bg)
        ImageDraw.Draw(row).text((2, 2), label, fill=(220, 220, 220))
        row.paste(a, (0, strip))
        row.paste(b, (a.width + pad, strip))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows),
                              sum(r.height for r in rows) + pad * (len(rows) - 1)), bg)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + pad
    sheet.save(path)


def _ensure_weight(name, url):
    """Return sr/weights/<name>, downloading from the release if absent."""
    dst = M.WEIGHTS / name
    if not dst.exists():
        M.WEIGHTS.mkdir(parents=True, exist_ok=True)
        print(f"[sr_infer] downloading {name} -> {dst}")
        torch.hub.download_url_to_file(url, str(dst), progress=True)
    return dst


class ResShiftInfer:
    """Released ResShift x4 sampler built from the vendored tree."""

    def __init__(self, version="v3", chop_size=512, chop_stride=-1, chop_bs=1,
                 use_amp=True, seed=12345, device=M.DEVICE, ckpt_path=None, cond_lq=None):
        self.device = device
        self.use_amp = use_amp
        self.version = version
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if version in _LOCAL:
            # locally-trained art model (e.g. x2): config + checkpoint live in-repo,
            # no release URL. Our make-sr-train ckpts are {"ema","model","step"} dicts.
            cfg_path, ckpt_dir = _LOCAL[version]
            cfg = M.load_configs(cfg_path)
            ckpt = Path(ckpt_path) if ckpt_path \
                else _latest_local_ckpt(ckpt_dir, _LOCAL_PREFIX[version])
            print(f"[sr_infer] {version}: cfg={Path(cfg_path).name} ckpt={ckpt.name}")
            sd = torch.load(str(ckpt), map_location=device)
            sd = sd.get("ema") or sd.get("model") or sd.get("state_dict") or sd
        else:
            cfg_name, ckpt_name, ckpt_url = _VERSIONS[version]
            cfg = M.load_configs(M.RESSHIFT / "configs" / cfg_name)
            ckpt = _ensure_weight(ckpt_name, ckpt_url)
            sd = torch.load(str(ckpt), map_location=device)
            sd = sd["state_dict"] if "state_dict" in sd else sd
        self.ckpt_name = str(ckpt)
        cfg.autoencoder.ckpt_path = str(_ensure_weight(*_VQGAN))

        # scale factor is config-driven (x4 released = 4, our art model = 2), so all the
        # chop/tiling math below generalizes instead of assuming x4.
        self.sf = int(cfg.diffusion.params.sf)

        # released ckpts carry DDP/compile prefixes (module. / _orig_mod.) — reload_model
        # reconstructs the target keys, unlike build_teacher's strict load (distill-only).
        model = util_common.instantiate_from_config(cfg.model).to(device)
        util_net.reload_model(model, sd)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.eval()
        self.autoencoder = M.build_autoencoder(cfg, device)
        self.diffusion = M.build_diffusion(cfg)
        self.cond_lq = cfg.model.params.cond_lq
        self.cond_mode = cond_lq or _COND[version]
        # reflect-pad the LR so the encoded latent is divisible by the Swin alignment at
        # every level. latent = y0_px * sf / 4 (VQ-f4); deepest level downsamples by
        # 2^(#levels-1), each needing size % window == 0 -> latent % (window*2^(L-1)) == 0.
        # Solve for the pixel multiple: offset = latent_align * 4 / sf  (x4 -> 64, x2 -> 128).
        # The old hardcoded lq_size=64 only happened to be right for x4 (latent == pixel).
        latent_align = cfg.model.params.window_size * 2 ** (len(cfg.model.params.channel_mult) - 1)
        self.offset = latent_align * 4 // self.sf

        self.chop_arg = chop_size
        self.chop_size = chop_size * (4 // self.sf)
        if chop_stride < 0:
            self.chop_stride = (chop_size - {512: 64, 256: 32, 64: 16}[chop_size]) * (4 // self.sf)
        else:
            self.chop_stride = chop_stride * (4 // self.sf)
        self.chop_bs = chop_bs

    @torch.no_grad()
    def _sample(self, y0):
        """y0: 1xCxHxW in [-1,1] RGB -> SR tensor in [-1,1], reflect-padded to offset.

        We encode the LR to a LATENT (z_y) and run the reverse loop on latents, rather
        than diffusion.p_sample_loop: that one leaves model_kwargs['lq'] to the caller and
        assumes the pixel LR already matches the latent grid — true at x4, false at x2
        (LR 128 vs latent 64). make_cond_lq resamples the LR to the latent grid, so the
        conditioning is scale-agnostic AND stays the LR pixels the released weights expect.
        clip_denoised stays False — latents aren't bounded to [-1,1].
        """
        ori_h, ori_w = y0.shape[2:]
        pad_h = (-ori_h) % self.offset
        pad_w = (-ori_w) % self.offset
        if pad_h or pad_w:
            y0 = F.pad(y0, (0, pad_w, 0, pad_h), mode="reflect")
        z_y = self.diffusion.encode_first_stage(y0, self.autoencoder, up_sample=True)
        z = self.diffusion.prior_sample(z_y)
        # z_y is the residual-shift BASE; the model's `lq` input is a separate conditioning
        # map — the released weights want the LR pixels there, not the latent (make_cond_lq).
        y_up = F.interpolate(y0, scale_factor=self.sf, mode="bicubic", align_corners=False)
        kw = {"lq": M.make_cond_lq(y_up, z_y, self.cond_mode)} if self.cond_lq else None
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((z_y.shape[0],), i, device=z_y.device, dtype=torch.long)
            z = self.diffusion.p_sample(self.model, z, z_y, t, clip_denoised=False,
                                        model_kwargs=kw)["sample"]
        out = self.diffusion.decode_first_stage(z.float(), first_stage_model=self.autoencoder)
        if pad_h or pad_w:
            out = out[:, :, : ori_h * self.sf, : ori_w * self.sf]
        return out.clamp_(-1.0, 1.0)

    @torch.no_grad()
    def _process(self, im_lq):
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp else torch.no_grad()
        if im_lq.shape[2] > self.chop_size or im_lq.shape[3] > self.chop_size:
            spliter = ImageSpliterTh(im_lq, self.chop_size, stride=self.chop_stride,
                                     sf=self.sf, extra_bs=self.chop_bs)
            for pch, info in spliter:
                with ctx:
                    spliter.update(self._sample(pch), info)
            sr = spliter.gather()
        else:
            with ctx:
                sr = self._sample(im_lq)
        return sr * 0.5 + 0.5

    def run(self, in_path, out_path, musiq=True, sheet=True, sheet_max=32):
        """SR every image under in_path into out_path, plus rsd-infer-style extras:
        per-image MUSIQ (SR + bicubic baseline) into infer_summary.json and a
        contact_sheet.png of [bicubic-up | SR] rows (train_x2 montage style)."""
        in_path, out_path = Path(in_path), Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in in_path.rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}) \
            if in_path.is_dir() else [in_path]
        metric = None
        if musiq:
            try:
                import pyiqa
                metric = pyiqa.create_metric("musiq", device=self.device)
            except Exception as e:  # noqa: BLE001
                print(f"[sr_infer] pyiqa/musiq unavailable ({e}) — skipping scores")
        rows, sheet_entries = [], []
        for f in files:
            im = util_image.imread(f, chn="rgb", dtype="float32")          # h x w x c
            t = util_image.img2tensor(im).to(self.device)                  # 1 x c x h x w, [0,1]
            sr = self._process((t - 0.5) / 0.5)
            sr_np = util_image.tensor2img(sr, rgb2bgr=False, min_max=(0.0, 1.0))  # RGB uint8
            Image.fromarray(sr_np).save(out_path / f"{f.stem}.png")
            row = {"stem": f.stem}
            need_lr_up = metric is not None or (sheet and len(sheet_entries) < sheet_max)
            if need_lr_up:
                lr = Image.open(f).convert("RGB")
                lr_up = lr.resize((lr.width * self.sf, lr.height * self.sf), Image.BICUBIC)
            if metric is not None:
                row["musiq_sr"] = _musiq_score(metric, sr_np)
                row["musiq_bicubic"] = _musiq_score(metric, np.asarray(lr_up))
            if sheet and len(sheet_entries) < sheet_max:
                label = f"{f.stem}   bicubic | SR"
                if "musiq_sr" in row:
                    label += f"   musiq {row['musiq_bicubic']} -> {row['musiq_sr']}"
                sheet_entries.append((lr_up, Image.fromarray(sr_np), label))
            rows.append(row)
            print(f"[sr_infer] {f.name} -> {out_path / (f.stem + '.png')}"
                  + (f"  musiq {row['musiq_sr']}" if "musiq_sr" in row else ""))
        if sheet_entries:
            _contact_sheet(sheet_entries, out_path / "contact_sheet.png")
            print(f"[sr_infer] contact sheet ({len(sheet_entries)} rows) -> "
                  f"{out_path / 'contact_sheet.png'}")
        summary = {"n": len(rows), "version": self.version, "ckpt": self.ckpt_name,
                   "chop": self.chop_arg}
        if self.device == "cuda":
            summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if metric is not None and rows:
            for k in ("musiq_sr", "musiq_bicubic"):
                summary[f"{k}_mean"] = round(float(np.mean([r[k] for r in rows])), 3)
        (out_path / "infer_summary.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2))
        print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--in_path", required=True, help="input image or directory")
    ap.add_argument("-o", "--out_path", default=None,
                    help="output dir (default: output/sr/<version>/infer for local models, "
                         "sr/data/results for released x4)")
    ap.add_argument("-v", "--version", default="v3", choices=list(_VERSIONS) + list(_LOCAL),
                    help="released x4: v1/v2/v3 — or our art finetunes x2/x4ft/x4s4 (make sr-train).")
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint (local versions only; default = newest in the "
                         "version's output/sr dir).")
    ap.add_argument("--cond_lq", choices=["pixel", "latent"], default=None,
                    help="override the version's `lq` conditioning convention (default: pixel "
                         "for released/x4 art models, latent for the legacy x2 line).")
    ap.add_argument("--chop_size", type=int, default=512, choices=[512, 256, 64])
    ap.add_argument("--chop_stride", type=int, default=-1)
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no_musiq", action="store_true",
                    help="skip MUSIQ scoring (still writes infer_summary.json)")
    ap.add_argument("--no_sheet", action="store_true", help="skip contact_sheet.png")
    ap.add_argument("--sheet_max", type=int, default=32,
                    help="cap contact-sheet rows (first N inputs)")
    args = ap.parse_args()
    out_path = args.out_path or str(
        REPO / "output" / "sr" / args.version / "infer" if args.version in _LOCAL
        else SR / "data" / "results")
    sampler = ResShiftInfer(version=args.version, chop_size=args.chop_size,
                            chop_stride=args.chop_stride, use_amp=not args.no_amp,
                            seed=args.seed, ckpt_path=args.ckpt, cond_lq=args.cond_lq)
    sampler.run(args.in_path, out_path, musiq=not args.no_musiq,
                sheet=not args.no_sheet, sheet_max=args.sheet_max)
    print(f"[sr_infer] done -> {out_path}")


if __name__ == "__main__":
    main()
