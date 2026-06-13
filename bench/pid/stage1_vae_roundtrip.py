"""Stage 1: prove our cached Anima latents live in PiD's Qwen-Image latent space.

We decode one of our cached ``_anima.npz`` latents with PiD's OWN Qwen-Image VAE
(WanVAE2d_ + checkpoints/QwenImage_VAE_2d.pth) and compare against the original
resized source image. If the reconstruction matches, the latent space +
normalization convention are identical, so PiD's SR decoder will accept our
latents as LQ_latent directly.

Self-contained: the WanVAE2d_ architecture is copied verbatim from
PiD/pid/_src/tokenizers/qwenimage_vae.py minus the imaginaire framework deps
(only torch + einops needed). Run with the repo venv python.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

# Per-channel latent normalization — AutoencoderKLQwenImage config defaults.
# (Byte-for-byte identical to library/models/qwen_vae.py:1037.)
_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
_LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        shape = (dim, 1, 1) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(), nn.Conv2d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return self.proj(x) + identity


class Encoder2d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                 temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = nn.Conv2d(3, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, mode="downsample2d"))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), nn.Conv2d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Decoder2d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                 temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        self.conv1 = nn.Conv2d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
                                    ResidualBlock(dims[0], dims[0], dropout))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, mode="upsample2d"))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), nn.Conv2d(out_dim, 3, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class WanVAE2d_(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                 temperal_downsample=[True, True, False], dropout=0.0, temporal_window=4):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder2d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout)
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder2d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample[::-1], dropout)

    def decode(self, z, scale):
        z = z / scale[1].view(1, self.z_dim, 1, 1) + scale[0].view(1, self.z_dim, 1, 1)
        x = self.conv2(z)
        return self.decoder(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_pth", default="/tmp/pid_ckpts/checkpoints/QwenImage_VAE_2d.pth")
    ap.add_argument("--npz", default="post_image_dataset/lora/kat_(bu-kunn)/11071601_1024x1008_anima.npz")
    ap.add_argument("--src", default="post_image_dataset/resized/kat_(bu-kunn)/11071601.png")
    ap.add_argument("--out_dir", default="bench/pid/results")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float32

    cfg = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
               temperal_downsample=[False, True, True], dropout=0.0)
    model = WanVAE2d_(**cfg)
    ckpt = torch.load(args.vae_pth, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 10):", missing[:10])
    if unexpected:
        print("  unexpected (first 10):", unexpected[:10])
    model = model.to(device, dtype).eval().requires_grad_(False)

    d = np.load(args.npz)
    lat_key = [k for k in d.files if k.startswith("latents_")][0]
    latent = torch.from_numpy(d[lat_key]).to(device, dtype)  # (16, h, w), normalized
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    print(f"[latent] key={lat_key} shape={tuple(latent.shape)} "
          f"min={latent.min():.3f} max={latent.max():.3f} std={latent.std():.3f}")

    mean = torch.tensor(_LATENTS_MEAN, device=device, dtype=dtype)
    std = torch.tensor(_LATENTS_STD, device=device, dtype=dtype)
    scale = [mean, 1.0 / std]

    with torch.no_grad():
        recon = model.decode(latent, scale)  # (1,3,H,W) in [-1,1]
    recon = recon.clamp(-1, 1)[0].float().cpu()
    print(f"[recon] shape={tuple(recon.shape)} min={recon.min():.3f} max={recon.max():.3f}")

    img = ((recon + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "stage1_pidvae_recon.png")
    Image.fromarray(img).save(out_path)
    print(f"[save] {out_path}")

    # Compare against original resized source (metric: PSNR in [-1,1] space, resized to match).
    if os.path.exists(args.src):
        src = Image.open(args.src).convert("RGB").resize((recon.shape[-1], recon.shape[-2]), Image.LANCZOS)
        src_t = torch.from_numpy(np.asarray(src)).float().permute(2, 0, 1) / 127.5 - 1.0
        mse = F.mse_loss(recon, src_t).item()
        psnr = -10.0 * np.log10(mse + 1e-12)
        print(f"[compare vs source] mse={mse:.5f} psnr={psnr:.2f} dB")
        print("  >> PSNR >~20 dB means the latent decodes to the right image (space matches).")


if __name__ == "__main__":
    main()
