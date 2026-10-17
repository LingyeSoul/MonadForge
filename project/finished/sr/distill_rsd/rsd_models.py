"""RSD network builders: teacher / student / fake-critic / discriminator.

All three SR nets are ResShift's UNetModelSwin (174M). The student & fake are made
STOCHASTIC one-step generators by adding a zero-initialized conv on the noise eps
whose output is injected after input_blocks[0] (RSD App. C) — zero-init => identical
to the frozen teacher at step 0. The discriminator taps the fake critic's bottleneck
([B,640,8,8]) per RSD Fig 2 / Eq 12.

Imports the ResShift clone by path; resolves its relative weight paths under RESSHIFT.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

SR = Path(__file__).resolve().parents[1]   # project/finished/sr/
REPO = SR.parents[2]                       # repo root (for output/ + data/ paths)
RESSHIFT = SR / "resshift"                 # vendored ResShift source (no external clone)
WEIGHTS = SR / "weights"                   # model checkpoints (gitignored, see sr/README.md)
if str(RESSHIFT) not in sys.path:
    sys.path.insert(0, str(RESSHIFT))

from utils import util_common  # noqa: E402  (vendored ResShift)
from models.unet import UNetModelSwin  # noqa: E402  (vendored ResShift)

CONFIG = RESSHIFT / "configs" / "realsr_swinunet_realesrgan256.yaml"
TEACHER_CKPT = WEIGHTS / "resshift_realsrx4_s15_v2.pth"

# Finetuned teachers (make sr-train). Their art configs differ from the released x4 one
# only in `sf` / the noise schedule (diffusion args, not weight shapes), so the same
# StochasticUNet arch + warm-start machinery applies unchanged.
#   x2   — our x2 finetune, distilled into the x2 student that ships in the ComfyUI node.
#   x4ft — our x4 finetune (queued.md Stage 1). Same 15-step schedule as the released x4,
#          so the distiller's math is untouched; only the teacher weights change. This is
#          the version to pass for Stage 2 — it removes the need to hand-pass --teacher.
X2_CONFIG = SR / "configs" / "realsr_x2_art.yaml"
X2_TEACHER_CKPT = WEIGHTS / "resshift_x2_final.pth"
X4FT_CONFIG = SR / "configs" / "realsr_x4_art.yaml"
X4FT_TEACHER_CKPT = REPO / "output" / "sr" / "x4_art" / "resshift_x4_final.pth"

# version -> (config, teacher checkpoint, cond_lq mode)
VERSIONS = {
    "x4":   (CONFIG, TEACHER_CKPT, "pixel"),
    "x2":   (X2_CONFIG, X2_TEACHER_CKPT, "latent"),
    "x4ft": (X4FT_CONFIG, X4FT_TEACHER_CKPT, "pixel"),
}

DEVICE = "cuda"


def resolve_version(version="x4", config=None, teacher=None):
    """Map a version tag to (config_path, teacher_ckpt); explicit args override.

    'x4' (default) = released 15-step v2 teacher; 'x2' / 'x4ft' = our sr-train finetunes.
    A 4-step (v3-schedule) x4 finetune has no tag — distill it with
    `--version x4ft --config sr/configs/realsr_x4_s4_art.yaml --teacher <ckpt>`.
    """
    try:
        cfg_path, tch, _ = VERSIONS[version]
    except KeyError:
        raise ValueError(f"unknown version {version!r} (expected one of {list(VERSIONS)})") from None
    return Path(config) if config else cfg_path, Path(teacher) if teacher else tch


def cond_mode_for(version):
    """Default `lq` conditioning mode for a version (see make_cond_lq)."""
    return VERSIONS[version][2]


def make_cond_lq(lq_img, z_y, mode="pixel"):
    """Build the tensor fed to the UNet's `lq` conditioning input.

    ResShift's `lq` input is NOT the residual-shift base — that's `z_y`, the VQ latent of
    the upsampled LR, and it keeps its own (latent) identity in q_sample/prior everywhere.
    `lq` is a separate 3-channel map concatenated onto the latent inside the first conv,
    and the released realsr checkpoints were trained with the **LR PIXELS** there (the
    lq_size == image_size == 64 config resolves feature_extractor to nn.Identity, so what
    goes in is the LR image at latent resolution, values in [-1,1]).

    Feeding the VQ latent instead (mode="latent") is off-manifold for those weights: the
    released teacher's x0 prediction degrades from img-MSE 0.004 to 0.038 (t=0) / 0.51
    (t=7) / 1.50 (t=14) vs GT, and 15-step inference comes out neon green. That bug shipped
    in sr_infer + the distill loop between 2026-06-30 and 2026-07-11 and is what the queued
    "bf16 color bug" actually was (bf16 is innocent — fp32 sampling is equally neon).

    mode="latent" is kept ONLY for the x2 line, whose teacher and 1-step student were both
    trained from scratch under that convention and are therefore self-consistent (it's why
    the x2 finetune log shows "neon -> coherent by ~400 steps": it was re-learning the
    conditioning from the warm-start instead of inheriting it). New finetunes use "pixel"
    so the released prior actually transfers.

    lq_img: the HR-sized bicubic-upsampled LR in [-1,1] (what ArtSRDataset returns as "lq",
            and what encode_first_stage(up_sample=True) encodes into z_y).
    z_y:    the LR latent — only its spatial dims are used here.
    """
    if mode == "latent":
        return z_y
    if mode != "pixel":
        raise ValueError(f"cond_lq mode must be 'pixel' or 'latent', got {mode!r}")
    return torch.nn.functional.interpolate(
        lq_img.float(), size=z_y.shape[-2:], mode="bicubic", align_corners=False,
    ).clamp(-1, 1).to(z_y.dtype)


def predict_x0(diff, model, z_t, c_y, t, eps=None):
    """ResShift x0 prediction (predict_type=xstart): scale input, forward, output IS x0.

    c_y is the `lq` CONDITIONING map — build it with make_cond_lq, do NOT pass the residual
    base z_y directly (that's the latent-lq bug; see make_cond_lq). eps is passed only to
    the stochastic student/fake nets; the teacher is deterministic.
    """
    kw = {"lq": c_y}
    if eps is not None:
        kw["eps"] = eps
    return model(diff._scale_input(z_t, t), t, **kw)


def load_configs(config_path=CONFIG):
    cfg = OmegaConf.load(config_path)
    # autoencoder ckpt is config-relative ("weights/autoencoder_vq_f4.pth"); point it at sr/weights/
    cfg.autoencoder.ckpt_path = str(WEIGHTS / Path(cfg.autoencoder.ckpt_path).name)
    return cfg


def _strip_module(sd):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}


def _extract_unet_sd(raw, prefer="ema"):
    """Pull the bare UNet state_dict out of any of our checkpoint envelopes.

    Handles the released ResShift ckpts ({"state_dict": ...} or a bare state_dict) AND
    our own snapshots: sr-train writes {"ema","model","step"} and the distill loop writes
    {"ema","student","step",...}. Prefer the EMA weights (sample quality) when present.
    """
    if isinstance(raw, dict):
        for key in (prefer, "state_dict", "model", "student"):
            v = raw.get(key)
            if isinstance(v, dict) and v and all(isinstance(k, str) for k in v):
                raw = v
                break
    return _strip_module(raw)


def enable_grad_checkpointing(model):
    """Turn on checkpointing for every Swin BasicLayer (bit-exact, big activation save).

    UNetModelSwin hardcodes use_checkpoint=False on each BasicLayer, so the windowed
    attention at the 64x64 latent grid keeps all activations live for backward. The
    vendored BasicLayer runs checkpoint(use_reentrant=False) — grad-safe regardless of
    where the params get their grad, and composes with torch.compile. Only worth it on
    the trainable nets (student/fake) — frozen no_grad forwards (teacher) build no
    graph anyway.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = True
            n += 1
    return n


class StochasticUNet(UNetModelSwin):
    """UNetModelSwin + a zero-init noise branch -> stochastic one-step generator.

    Two injection modes (both zero-init => bit-identical to the frozen teacher at step 0):
      - "add" (the original reconstruction): a separate zero-init conv `noise_proj`
        maps a 3-channel eps to the post-conv width and is ADDED after input_blocks[0].
      - "concat" (train default; matches the official RSD release, `sampler.py::_add_noise` /
        `trainer.py:1426`): WIDEN input_blocks[0][0] by `noise_channels` (default 1) input
        planes, zero-init on the new slice, and CONCAT eps onto the [latent, lq] stack
        before the first conv. This is the published mechanism; kept behind a flag because
        it changes the first-conv shape (existing "add" checkpoints don't load into it).

    forward(x, timesteps, lq, eps): injects eps per `noise_mode`.
    encode_features(x, lq, t, eps): returns the bottleneck feature map for the GAN head.
    """

    def __init__(self, *args, noise_mode="add", noise_channels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_mode = noise_mode
        # default channel count differs by mode: 3 (add, latent-shaped) vs 1 (concat, official)
        self.noise_channels = noise_channels if noise_channels is not None \
            else (1 if noise_mode == "concat" else 3)
        conv = self.input_blocks[0][0]
        if noise_mode == "add":
            in_lat = conv.out_channels  # post-conv width (160)
            self.noise_proj = nn.Conv2d(self.noise_channels, in_lat, 3, padding=1)
            nn.init.zeros_(self.noise_proj.weight)
            nn.init.zeros_(self.noise_proj.bias)
        elif noise_mode == "concat":
            # widen the first input conv: [in_ch] -> [in_ch + noise_channels], zero everywhere
            # then copy the teacher's original in-planes back; the noise slice stays zero.
            wide = nn.Conv2d(conv.in_channels + self.noise_channels, conv.out_channels,
                             conv.kernel_size, padding=conv.padding)
            with torch.no_grad():
                nn.init.zeros_(wide.weight)
                wide.weight[:, :conv.in_channels].copy_(conv.weight)
                wide.bias.copy_(conv.bias)
            self.input_blocks[0][0] = wide
        else:
            raise ValueError(f"noise_mode must be 'add' or 'concat', got {noise_mode!r}")

    def _embed_and_concat(self, x, timesteps, lq):
        emb = self.time_embed(
            self._timestep_embedding(timesteps)
        ).type(self.dtype)
        if lq is not None:
            assert self.cond_lq
            lq = self.feature_extractor(lq.type(self.dtype))
            x = torch.cat([x, lq], dim=1)
        return x.type(self.dtype), emb

    def _timestep_embedding(self, timesteps):
        from models.unet import timestep_embedding
        return timestep_embedding(timesteps, self.model_channels)

    def _inject_concat(self, h, eps):
        """concat mode: append eps onto the [latent, lq] stack before the first conv."""
        if self.noise_mode == "concat" and eps is not None:
            h = torch.cat([h, eps.type(self.dtype)], dim=1)
        return h

    def _inject_add(self, h, ii, eps):
        """add mode: add noise_proj(eps) after input_blocks[0]."""
        if self.noise_mode == "add" and ii == 0 and eps is not None:
            h = h + self.noise_proj(eps.type(self.dtype))
        return h

    def forward(self, x, timesteps, lq=None, eps=None, mask=None):
        h, emb = self._embed_and_concat(x, timesteps, lq)
        h = self._inject_concat(h, eps)
        hs = []
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            h = self._inject_add(h, ii, eps)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def encode_features(self, x, lq=None, timesteps=None, eps=None):
        """Bottleneck features for the GAN head: input_blocks + middle ResBlock."""
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        h, emb = self._embed_and_concat(x, timesteps, lq)
        h = self._inject_concat(h, eps)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            h = self._inject_add(h, ii, eps)
        # first sub-module of middle_block is the ResBlock -> bottleneck feats
        h = self.middle_block[0](h, emb)
        return h  # [B, 640, 8, 8]


def compile_swin_blocks():
    """Block-compile the vendored SwinUNet (the repo's compile_blocks trick).

    Whole-graph torch.compile on UNetModelSwin is pathological: dynamo traces the full
    174M net once per (net, grad-mode, batch-shape) variant — ~7 giant graphs — and every
    checkpoint() call inside becomes a higher-order op speculate-traced as its own
    subgraph. Measured: tens of minutes of warmup for ~zero steady-state win.

    Compiling the repeated block CLASSES instead keeps the module glue eager, leaves
    checkpoint() OUTSIDE compiled code (backward recompute just re-invokes the compiled
    forward — no higher-order op), and every instance across student/fake/teacher shares
    one compiled artifact per distinct (resolution, width, batch, grad-mode): a handful
    of small graphs. Idempotent; raise the dynamo recompile budget first (the variants
    all live on ONE code object per class).
    """
    from models.swin_transformer import SwinTransformerBlock  # vendored ResShift
    from models.unet import ResBlock
    for cls in (SwinTransformerBlock, ResBlock):
        if not getattr(cls, "_block_compiled", False):
            cls.forward = torch.compile(cls.forward)
            cls._block_compiled = True


def make_eps(model, ref):
    """Injection noise shaped for `model`'s mode: (B, model.noise_channels, H, W), like `ref`.

    add mode wants 3 planes (latent-shaped), concat mode wants 1 (official). Use this for the
    injected eps everywhere the student/fake is called — NOT for the residual-shift diffusion
    noise (that stays latent-shaped, 3-ch, e.g. `z_T = z_y + kappa*randn_like(z_y)`)."""
    return torch.randn(ref.shape[0], model.noise_channels, *ref.shape[2:],
                       device=ref.device, dtype=ref.dtype)


class DiscHead(nn.Module):
    """Small DMD2-style discriminator on the fake-critic bottleneck [B,640,8,8]->[B,1]."""

    def __init__(self, in_ch=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.SiLU(),   # 8->4
            nn.Conv2d(256, 256, 4, 2, 1), nn.GroupNorm(32, 256), nn.SiLU(),     # 4->2
            nn.Conv2d(256, 1, 2, 1, 0),                                          # 2->1
        )

    def forward(self, feats):
        return self.net(feats).flatten(1).mean(1)  # [B]


def _build_unet(cfg, cls):
    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    return cls(**params)


def build_teacher(cfg, ckpt_path, device, dtype=torch.float32):
    model = _build_unet(cfg, UNetModelSwin)
    sd = _extract_unet_sd(torch.load(ckpt_path, map_location="cpu"))
    model.load_state_dict(sd, strict=True)
    model = model.to(device, dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_generator(cfg, teacher_ckpt, device, dtype=torch.float32, grad_ckpt=False,
                    pretrained=True, noise_mode="add", noise_channels=None):
    """Student or fake: teacher weights + zero-init noise branch.

    noise_mode "add" (default) / "concat" (official RSD) — see StochasticUNet.
    grad_ckpt=True enables Swin-attention gradient checkpointing (training only) —
    large activation-memory save, bit-exact. Leave off for inference.
    pretrained=False skips loading the teacher ckpt (random init) — use when the caller
    immediately load_state_dict's a full student ckpt over it (avoids a wasted 174M read).
    """
    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    model = StochasticUNet(noise_mode=noise_mode, noise_channels=noise_channels, **params)
    if pretrained:
        sd = _extract_unet_sd(torch.load(teacher_ckpt, map_location="cpu"))
        if model.noise_mode == "concat":
            # teacher's first conv is narrower than our widened one; graft the teacher planes
            # into the leading slice (noise slice stays zero-init) so load stays strict.
            key = "input_blocks.0.0.weight"
            tw = sd[key]
            w = model.state_dict()[key].clone()   # zeros on the noise slice
            w[:, :tw.shape[1]].copy_(tw)
            sd = {**sd, key: w}
            model.load_state_dict(sd, strict=True)
        else:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # only noise_proj.* should be missing from the ckpt
            assert all("noise_proj" in m for m in missing), f"unexpected missing: {missing}"
            assert not unexpected, f"unexpected keys: {unexpected}"
    if grad_ckpt:
        enable_grad_checkpointing(model)
    return model.to(device, dtype)


def build_autoencoder(cfg, device):
    ae = util_common.get_obj_from_str(cfg.autoencoder.target)(
        **OmegaConf.to_container(cfg.autoencoder.params, resolve=True)
    )
    sd = torch.load(cfg.autoencoder.ckpt_path, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    ae.load_state_dict(_strip_module(sd), strict=False)
    ae = ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def build_diffusion(cfg):
    """ResShift GaussianDiffusion (residual-shift forward). Used for q_sample + x0-pred."""
    return util_common.get_obj_from_str(cfg.diffusion.target)(
        **OmegaConf.to_container(cfg.diffusion.params, resolve=True)
    )
