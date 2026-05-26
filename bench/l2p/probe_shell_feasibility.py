"""L2P Phase 0 — can the FROZEN Anima DiT core be reached from PIXEL space?

Port preconditions for the L2P transfer paradigm (Chen et al.,
*L2P: Unlocking Latent Potential for Pixel Generation*, arXiv:2605.12013) onto
Anima. The paper's whole claim rests on one premise: a latent DiT's frozen
intermediate blocks "function within their native optimization manifold" when
you swap the VAE for large-patch RGB tokenization and re-train only the shallow
input/output shells. The paper shows this empirically on a 3840-dim / 30-block
source LDM. Anima is smaller (2048-dim / 28-block) and we train on a single GPU,
so the open question is **does the premise survive at Anima's scale + budget**,
not "does the concept work."

This is the cheapest test that can falsify the whole approach (the SPD-plan
ethos): freeze the entire DiT, bolt on a fresh RGB patch-embed (input shell) and
a fresh token→pixel decoder (output shell), and overfit a tiny self-generated
image set with the *exact* Anima flow-matching objective. If shells-only can
drive the FM loss down and decode recognizable structure, the frozen core's
priors are reachable from pixel space → pay for Phase 1 (add last-n blocks,
scale data, real DiP detailer with input skips). If it plateaus at noise, the
latent→pixel gap is too wide for cheap transfer here.

Phase-0 simplifications (deliberately stricter / simpler than the full method —
all promoted in Phase 1, see bench/l2p/plan.md):
  * Output shell defaults to a **pure token decoder** (token grid → conv-transpose
    16×), NOT the DiP image-U-Net with noisy-input skip connections. This isolates
    "is there enough signal in the frozen core's *output tokens* to rebuild
    pixels" — a strictly harder bar than a U-Net that can cheat via input skips.
    Pass ``--dip_skip`` to swap in the real DiP detailer (the Phase-1 output shell)
    once the pure-token decoder is confirmed to be the bottleneck.
  * `final_layer` AdaLN modulation is dropped (passthrough); the decoder owns
    all of the token→pixel transform.
  * No last-n block tuning by default (`--train_blocks 0` = shells only). The
    paper's Fig-9b shallow-tuning ablation lives in Phase 1.

FM convention is lifted verbatim from training (train.py / library/runtime/noise):
    noisy = (1-σ)·x0 + σ·ε   ;   target = ε − x0   ;   model sees σ∈[0,1] as `timesteps`.
Here x0 is the RGB image in [-1,1] (not a latent).

Usage::

    python bench/l2p/probe_shell_feasibility.py --dit <path> \
        --image_dir post_image_dataset/resized \
        --te_cache_dir post_image_dataset/lora \
        --num_images 64 --steps 2000 --resolution 1024

Self-generated images (source-Anima samples) are the faithful corpus per the
paper; resized dataset images are an acceptable Phase-0 stand-in (we only need to
know whether the mapping *learns at all*). The README notes the distinction.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Marginal fit at 1024²/bs=1 on 16 GB — defrag the allocator so the last ~tens of
# MB don't fail on fragmentation. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

# bench/ is not an installed package — bootstrap sibling import (see CLAUDE.md).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _anima import add_common_args, build_anima, resolve_dtype  # noqa: E402
from _common import make_run_dir, write_result  # noqa: E402

from library.io.cache import load_cached_text_features  # noqa: E402


# ---------------------------------------------------------------------------
# Shells — fresh-init, trainable. The frozen DiT core sits between them.
# ---------------------------------------------------------------------------


class L2PInputEmbed(nn.Module):
    """RGB large-patch tokenizer replacing ``model.x_embedder`` (the VAE-latent
    patch-embed). Contract matches ``PatchEmbed``: (B, C, T, H, W) → (B,T',H',W',D),
    and it carries ``spatial_patch_size`` so ``prepare_embedded_sequence`` /
    RoPE — which key on the *token grid*, not the patch size — stay transparent.

    With patch=16 on a 1024² image the grid is 64×64 = 4096 tokens, the same
    grid (and RoPE) as Anima's native patch=2 on a 128² VAE latent. Token count
    is preserved across the modality swap — exactly the L2P efficiency trick, and
    it lines up with Anima's constant-token-bucket invariant.
    """

    def __init__(self, in_channels: int, patch: int, out_channels: int):
        super().__init__()
        self.spatial_patch_size = patch
        self.temporal_patch_size = 1
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels * patch * patch, out_channels, bias=False)
        self.dim = in_channels * patch * patch
        std = 1.0 / math.sqrt(self.dim)
        nn.init.trunc_normal_(self.proj.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T=1, H, W) → (B, T'=1, H/p, W/p, D)
        b, c, t, h, w = x.shape
        p = self.spatial_patch_size
        assert h % p == 0 and w % p == 0, f"H,W {(h, w)} not divisible by patch {p}"
        x = x.view(b, c, t, h // p, p, w // p, p)
        # (b, c, t, hg, p, wg, p) → (b, t, hg, wg, c*p*p)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).reshape(b, t, h // p, w // p, c * p * p)
        return self.proj(x)


class _Passthrough(nn.Module):
    """Drop-in for ``model.final_layer``: returns the token grid unchanged,
    swallowing the (t_emb, adaln_lora_B_T_3D=...) modulation args. Phase 0 hands
    the entire token→pixel transform to the decoder (in the unpatchify slot)."""

    def forward(self, x, *args, **kwargs):  # noqa: D401
        return x


class L2PTokenDecoder(nn.Module):
    """Pure token→pixel decoder replacing ``model.unpatchify``. Takes the frozen
    core's output token grid (B,T',H',W',D), drops the singleton temporal axis,
    and conv-transpose-upsamples 16× (=2⁴) back to RGB velocity in pixel space.

    Deliberately *not* the DiP image-U-Net: no skip connections from the noisy
    input, so the only path to a sharp output is through the frozen blocks. A
    strictly harder feasibility bar than the real detailer (Phase 1)."""

    def __init__(self, hidden: int, out_channels: int = 3, base: int = 256):
        super().__init__()
        # token D → base channels at the 1/16 grid, then 4× upsample-by-2 blocks.
        ch = [base, base // 2, base // 4, base // 8, base // 16]  # 256→128→64→32→16
        self.proj_in = nn.Conv2d(hidden, ch[0], 3, padding=1)
        ups = []
        for i in range(4):
            ups += [
                nn.ConvTranspose2d(ch[i], ch[i + 1], 4, stride=2, padding=1),
                nn.GroupNorm(min(8, ch[i + 1]), ch[i + 1]),
                nn.SiLU(),
            ]
        self.ups = nn.Sequential(*ups)
        self.proj_out = nn.Conv2d(ch[-1], out_channels, 3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x_B_T_H_W_D: torch.Tensor) -> torch.Tensor:
        b, t, h, w, d = x_B_T_H_W_D.shape
        assert t == 1, "Phase-0 probe is image-only (T=1)"
        x = x_B_T_H_W_D.squeeze(1).permute(0, 3, 1, 2).contiguous()  # (B, D, H', W')
        x = self.proj_in(x)
        x = self.ups(x)
        return self.proj_out(x)  # (B, 3, H, W)


class L2PDiPDecoder(nn.Module):
    """DiP-style "Detailer Head" (paper §3.2; Table-3 ladder adapted to Anima's
    2048 width) — the Phase-1 output shell, enabled by ``--dip_skip``.

    Unlike ``L2PTokenDecoder``, a U-Net **encoder runs on the noisy input image**
    so high-frequency RGB detail reaches the output through *skip connections*,
    instead of having to be rebuilt from scratch through the frozen latent-trained
    core. The frozen core's output token grid is fused at the **bottleneck**
    (concat → conv), mirroring the paper's ``(512 + 3840) → 512`` — here
    ``(512 + hidden) → 512``. Channels follow the paper: down ``3→64→128→256→512``,
    up ``512→256→128→64→64``, out ``64→3``; 4 symmetric stride-2 stages map full
    res ↔ the 64² token grid (R/16).

    This is the specific element Phase 0 deliberately removed (`plan.md`): it gives
    pixels a path that *bypasses* the frozen core, the suspected lever for the
    shells-only / both-ends-block plateau at the ‖ε−x0‖² noise floor. Caveat: the
    skip lets the head trivially copy the (near-clean) input at low σ, so FM loss
    will overstate the win — read the montage, not just the number.

    The skip input is set per-forward by the caller (`set_skip(noisy_rgb)`) right
    before ``model.forward`` so the head always sees the exact image the DiT saw,
    in both the train and the Euler-sample paths.
    """

    def __init__(self, hidden: int, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        enc_ch = [in_channels, 64, 128, 256, 512]  # paper down-ladder
        self.enc = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(enc_ch[i], enc_ch[i + 1], 4, stride=2, padding=1),
                nn.GroupNorm(min(8, enc_ch[i + 1]), enc_ch[i + 1]),
                nn.SiLU(),
            )
            for i in range(4)  # R → R/16, the token grid
        )
        # Fuse frozen token features (hidden ch @ R/16) with the enc bottleneck.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + hidden, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
        )
        up_ch = [512, 256, 128, 64, 64]  # paper up-ladder
        skip_ch = [256, 128, 64, in_channels]  # symmetric enc skips (R/8…R)
        self.up = nn.ModuleList()
        for i in range(4):
            self.up.append(
                nn.ModuleDict(
                    {
                        "convt": nn.ConvTranspose2d(up_ch[i], up_ch[i + 1], 4, stride=2, padding=1),
                        "fuse": nn.Sequential(
                            nn.Conv2d(up_ch[i + 1] + skip_ch[i], up_ch[i + 1], 3, padding=1),
                            nn.GroupNorm(min(8, up_ch[i + 1]), up_ch[i + 1]),
                            nn.SiLU(),
                        ),
                    }
                )
            )
        self.proj_out = nn.Conv2d(up_ch[-1], out_channels, 3, padding=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self._skip_input: torch.Tensor | None = None

    def set_skip(self, rgb: torch.Tensor) -> None:
        """Stash the noisy RGB (B,3,H,W) the DiT is about to see this forward."""
        self._skip_input = rgb

    def forward(self, x_B_T_H_W_D: torch.Tensor) -> torch.Tensor:
        b, t, h, w, d = x_B_T_H_W_D.shape
        assert t == 1, "Phase-0 probe is image-only (T=1)"
        assert self._skip_input is not None, "call set_skip(noisy_rgb) before forward"
        tok = x_B_T_H_W_D.squeeze(1).permute(0, 3, 1, 2).contiguous()  # (B, D, R/16, R/16)
        e = self._skip_input  # (B, 3, R, R)
        skips: list[torch.Tensor] = []
        for enc in self.enc:
            skips.append(e)  # pre-downsample feature → [in@R, 64@R/2, 128@R/4, 256@R/8]
            e = enc(e)
        x = self.bottleneck(torch.cat([e, tok], dim=1))  # e: 512@R/16
        for i, up in enumerate(self.up):
            x = up["convt"](x)
            x = up["fuse"](torch.cat([x, skips[-(i + 1)]], dim=1))
        return self.proj_out(x)  # (B, 3, R, R)


# ---------------------------------------------------------------------------
# Data — pair RGB images with their cached text embeddings by stem.
# ---------------------------------------------------------------------------


def _discover_pairs(image_dir: Path, te_dir: Path, n: int, seed: int):
    """Find (image_path, te_cache_path) pairs sharing a stem. TE caches are
    ``{stem}_anima_te.safetensors``; images are matched by bare stem."""
    import random

    te_caches = sorted(te_dir.rglob("*_anima_te.safetensors"))
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    img_by_stem: dict[str, Path] = {}
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in exts:
            img_by_stem.setdefault(p.stem, p)

    pairs = []
    for te in te_caches:
        stem = te.name[: -len("_anima_te.safetensors")]
        img = img_by_stem.get(stem)
        if img is not None:
            pairs.append((img, te))
    if not pairs:
        raise SystemExit(
            f"No (image, te-cache) stem matches between {image_dir} and {te_dir}. "
            "Check that resized images and *_anima_te.safetensors share stems."
        )
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def _load_image(path: Path, res: int, device, dtype) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((res, res), Image.BICUBIC)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 255.0
    t = t.view(res, res, 3).permute(2, 0, 1)  # (3, H, W)
    t = t * 2.0 - 1.0  # [-1, 1]
    return t.to(device=device, dtype=dtype)


def _apply_flow_shift(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Reparameterize σ toward higher noise: σ ← shift·σ / (1 + (shift−1)·σ).

    Identity at shift=1. Matches the reference's ``set_timesteps_z_image`` (and
    Anima's own ``"shift"`` timestep_sampling branch, ``noise.py:114``). Applied to
    *both* the training draw and the Euler-sample schedule so they stay aligned."""
    if shift == 1.0:
        return sigma
    return (sigma * shift) / (1.0 + (shift - 1.0) * sigma)


def _shallow_block_indices(num_blocks: int, n: int) -> list[int]:
    """The paper's "shallow tuning" set: the **first n AND last n** DiT blocks
    (both I/O ends), NOT just the last n. Paper §3.3 mod 3: *"we only update the
    initial input projection layer, the first and last n blocks of the DiT, and
    the newly added Detailer Head"*; Fig-9b ablates this shallow set (5 layers
    beats 10 / full). The input-side blocks are arguably the load-bearing ones
    here — the modality swap (RGB patches) happens at the *input*, so leaving the
    first blocks frozen would test the wrong end.

    ``n`` is per-end → 2n blocks, unless the ends meet (2n ≥ num_blocks) in which
    case every block is returned. On Anima's 28 blocks, n=2 ≈ the paper's 5-layer
    shallow default, n=3 brackets it."""
    if 2 * n >= num_blocks:
        return list(range(num_blocks))
    return list(range(n)) + list(range(num_blocks - n, num_blocks))


_SIGMA_BAND_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _sigma_band_stats(sigmas: list[float], losses: list[float]):
    """Mean loss + sample count per σ band.

    The whole point of the breakdown: a DiP input-skip can trivially copy the
    *near-clean* input at low σ, so a low **overall** loss can be carried entirely
    by the low-σ regime while the model learns nothing generative. At high σ the
    input is ≈ pure noise — the skip is useless and the only path to a low loss is
    through the frozen core. Splitting by σ exposes that asymmetry (which the
    single scalar hides). Note even a *working* generator floors the high-σ band at
    ≈Var(x0) — single-step prediction can't recover x0 from pure noise; generation
    is the iterative process. So this is a *cheat detector*, not a quality metric:
    the montage stays the arbiter of generation."""
    n = len(_SIGMA_BAND_EDGES) - 1
    sums = [0.0] * n
    counts = [0] * n
    for s, l in zip(sigmas, losses):
        for i in range(n):
            if s < _SIGMA_BAND_EDGES[i + 1] or i == n - 1:
                sums[i] += l
                counts[i] += 1
                break
    return [(sums[i] / counts[i] if counts[i] else None, counts[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-base-v1.0.safetensors",
        help="Base Anima DiT safetensors (default: Anima 1.0, resolved under "
        "anima_home()). Repo-relative; build_anima resolves it.",
    )
    p.add_argument("--image_dir", type=Path, default=Path("post_image_dataset/resized"))
    p.add_argument("--te_cache_dir", type=Path, default=Path("post_image_dataset/lora"))
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Default 1: backprop to the input shell retains activations through "
        "all 28 frozen blocks, so 1024² needs bs=1 to fit 16 GB.",
    )
    p.add_argument("--patch", type=int, default=16, help="Pixel patch size (1K→64² grid).")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="FM noise-shift applied to the sampled sigma: σ ← shift·σ/(1+(shift−1)·σ) "
        "(paper §3.4 / reference set_timesteps_z_image shift=3). Pixel space "
        "under-corrupts and degenerates to trivial local reconstruction without "
        "it — our base stack is unshifted (timestep_sampling=sigmoid never applies "
        "discrete_flow_shift). 1.0 = no shift (the old probe behaviour).",
    )
    p.add_argument("--lr", type=float, default=5e-5, help="L2P paper AdamW LR.")
    p.add_argument(
        "--train_blocks",
        type=int,
        default=0,
        help="FULL-tune the first N AND last N DiT blocks too (paper §3.3 shallow "
        "tuning at both I/O ends; N per-end; Phase-1 knob; 0 = shells only). "
        "Adds ~2 GB optimizer state → auto-enables gradient checkpointing.",
    )
    p.add_argument(
        "--lora_blocks",
        type=int,
        default=0,
        help="Adapt the first N AND last N DiT blocks (paper §3.3) with a LoRA "
        "instead of full-tuning. N per-end. Tiny optimizer state, activation "
        "graph unchanged from shells-only → fits bs=1 with no checkpointing. The "
        "shippable, in-idiom alternative.",
    )
    p.add_argument("--lora_rank", type=int, default=64, help="LoRA rank for --lora_blocks.")
    p.add_argument(
        "--dip_skip",
        action="store_true",
        help="Use the real DiP detailer (L2PDiPDecoder) instead of the pure token "
        "decoder: a U-Net encoder on the noisy input feeds skip connections so "
        "pixels get a path bypassing the frozen core (paper §3.2). The Phase-1 "
        "output shell + the suspected lever for the shells-only/both-ends plateau. "
        "The full-res head activations still fit at 1024²/bs=1 without "
        "checkpointing (~15.3 GB with --lora_blocks 2).",
    )
    p.add_argument("--sample_every", type=int, default=250)
    p.add_argument("--sample_steps", type=int, default=16, help="Euler steps for viz.")
    add_common_args(p)  # includes --compile_mode (block stack is always compiled)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)

    # Frozen base DiT.
    bundle = build_anima(args, adapter=None, train_mode=False)
    model = bundle.anima
    for prm in model.parameters():
        prm.requires_grad_(False)

    # Swap the shells. prepare_embedded_sequence may concat a padding-mask
    # channel before x_embedder, so size the input embed accordingly + feed an
    # all-ones mask (mirrors inference).
    concat_pad = bool(getattr(model, "concat_padding_mask", False))
    in_ch = 3 + (1 if concat_pad else 0)
    hidden = int(model.model_channels)

    # x_embedder / final_layer are real submodules → assign a Module directly.
    model.x_embedder = L2PInputEmbed(in_ch, args.patch, hidden).to(device, dtype)
    model.final_layer = _Passthrough().to(device, dtype)
    # unpatchify is a *class method*, not a submodule: assigning an nn.Module to it
    # would be diverted into _modules and never retrieved (the class method wins
    # attribute lookup). So register the decoder under a fresh name and shadow
    # unpatchify with a bound method — a plain function lands in instance __dict__
    # and *does* override a non-data-descriptor class attribute.
    import types

    decoder = L2PDiPDecoder(hidden) if args.dip_skip else L2PTokenDecoder(hidden)
    model.l2p_decoder = decoder.to(device, dtype)
    model.unpatchify = types.MethodType(lambda self, x: self.l2p_decoder(x), model)

    trainable = list(model.x_embedder.parameters()) + list(model.l2p_decoder.parameters())

    # --- Block adaptation: full-tune OR LoRA the first-N ∪ last-N blocks (mutually exclusive) ---
    network = None
    if args.train_blocks > 0 and args.lora_blocks > 0:
        raise SystemExit("Pass at most one of --train_blocks / --lora_blocks.")
    if args.train_blocks > 0:
        for i in _shallow_block_indices(model.num_blocks, args.train_blocks):
            for prm in model.blocks[i].parameters():
                prm.requires_grad_(True)
                trainable.append(prm)
    elif args.lora_blocks > 0:
        # Fresh LoRA on the first-N ∪ last-N blocks. The layer-range filter
        # (layer_start/layer_end) is a single CONTIGUOUS window — it can't express
        # the paper's both-ends set — and include_patterns only *rescues excluded*
        # modules (can't restrict). So instead exclude the MIDDLE block indices by
        # regex (fullmatch, appended to the default embedder/final-layer exclude),
        # leaving only the shallow both-ends blocks adapted. apply_to MUST precede
        # compile_blocks (compile-after-apply invariant).
        from networks.lora_anima.factory import create_network

        nb = model.num_blocks
        shallow = set(_shallow_block_indices(nb, args.lora_blocks))
        middle = [i for i in range(nb) if i not in shallow]
        # Trailing `\.` anchors the full index, so `(?:1|12)` can't cross-match
        # (`blocks.12.` never matches the `1` alt). Empty middle → no extra exclude.
        extra_exclude = (
            [r".*blocks\.(?:" + "|".join(map(str, middle)) + r")\..*"] if middle else []
        )
        network = create_network(
            1.0, args.lora_rank, float(args.lora_rank), None, [], model,
            exclude_patterns=extra_exclude,
            # Saves the low-precision LoRA input instead of the fp32 cast — those
            # retained fp32 activations otherwise dominate memory in frozen-DiT +
            # LoRA setups and OOM the marginal 16 GB fit (project_custom_down_autograd).
            use_custom_down_autograd="true",
        )
        network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
        network.to(device, dtype)
        lora_params = [p for p in network.get_trainable_params() if p.requires_grad]
        trainable += lora_params

    n_train = sum(t.numel() for t in trainable)
    adapt = (
        f"lora_blocks={args.lora_blocks}(r{args.lora_rank})"
        if args.lora_blocks
        else f"train_blocks={args.train_blocks}"
    )
    decoder_kind = "DiP-skip U-Net" if args.dip_skip else "pure-token"
    print(
        f"[l2p-p0] trainable params: {n_train / 1e6:.2f}M  "
        f"({adapt}, {decoder_kind} decoder, flow_shift={args.flow_shift})"
    )

    # Compile the block stack. Reaching the input shell backprops through all 28
    # frozen blocks at 4096 tokens — marginally over 16 GB eager. Inductor fusion
    # trims the intermediate allocations enough to fit. compile_blocks() also sets
    # native-flatten (4096 tokens → a single token-count graph here); the swapped
    # shells sit outside the compiled block._forward zone, so the compile-after-
    # apply ordering is satisfied by compiling after the swap.
    model.compile_blocks(mode=args.compile_mode)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    pairs = _discover_pairs(args.image_dir, args.te_cache_dir, args.num_images, args.seed)
    print(f"[l2p-p0] overfitting {len(pairs)} images for {args.steps} steps @ {args.resolution}²")

    # Preload the tiny corpus into VRAM (images + per-sample crossattn_emb).
    imgs, ctxs = [], []
    for img_path, te_path in pairs:
        imgs.append(_load_image(img_path, args.resolution, device, dtype))
        cross, _pooled = load_cached_text_features(str(te_path), variant=0)
        ctxs.append(cross.to(device, dtype))  # (N_text, 1024)

    res = args.resolution
    pad_mask = torch.ones(1, 1, res, res, device=device, dtype=dtype) if concat_pad else None

    def _forward(x0_b, ctx_b, sigma_b):
        """One FM forward in pixel space. x0_b:(B,3,H,W); returns (pred, target)."""
        noise = torch.randn_like(x0_b)
        s = sigma_b.view(-1, 1, 1, 1)
        noisy = (1.0 - s) * x0_b + s * noise
        target = noise - x0_b
        x5 = noisy.unsqueeze(2)  # (B,3,T=1,H,W)
        pm = pad_mask.expand(x0_b.shape[0], -1, -1, -1) if pad_mask is not None else None
        if args.dip_skip:
            model.l2p_decoder.set_skip(noisy)  # (B,3,H,W) — the DiP input-skip
        pred = model.forward(x5, sigma_b, context=ctx_b, padding_mask=pm).squeeze(2)
        return pred, target

    run_dir = make_run_dir("l2p", label=args.label or "shell-feasibility")
    loss_curve: list[float] = []
    sigma_curve: list[float] = []  # per-step σ (post-shift), parallel to loss_curve

    # Unfreezing tail blocks adds grad + optimizer memory on top of the marginal
    # bs=1 fit, so checkpoint the (now-recomputed) block activations. Gating is on
    # self.training, so run train() — safe (frozen core has no dropout/running
    # stats). Shells-only (train_blocks=0) fits eager, so stay in eval() there.
    # Checkpointing composes with compile_blocks: it wraps forward(), compile
    # wrapped _forward() (see compile_blocks docstring).
    # --dip_skip adds the detailer's full-res activations but still fits without
    # checkpointing (`--lora_blocks 2 --dip_skip` lands ~15.3 GB at 1024²/bs=1), so
    # it stays in the eval()/no-ckpt path — checkpointing is reserved for the
    # grad+optimizer cost of --train_blocks.
    if args.train_blocks > 0:
        model.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)
        model.train()
    else:
        model.eval()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(imgs), (args.batch_size,))
        # crossattn lengths vary per sample → pad to the batch max (zero pads act
        # as attention sinks, which is what the pretrained model expects).
        maxlen = max(ctxs[i].shape[0] for i in idx)
        x0_b = torch.stack([imgs[i] for i in idx])
        ctx_b = torch.stack(
            [F.pad(ctxs[i], (0, 0, 0, maxlen - ctxs[i].shape[0])) for i in idx]
        )
        sigma_b = torch.sigmoid(torch.randn(args.batch_size, device=device))
        sigma_b = _apply_flow_shift(sigma_b, args.flow_shift).to(dtype)

        pred, target = _forward(x0_b, ctx_b, sigma_b)
        loss = F.mse_loss(pred.float(), target.float())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        loss_curve.append(loss.item())
        sigma_curve.append(float(sigma_b.float().mean()))  # bin key (bs=1 → exact)
        if step % 50 == 0 or step == 1:
            recent = sum(loss_curve[-50:]) / len(loss_curve[-50:])
            print(f"[l2p-p0] step {step:5d}/{args.steps}  loss {loss.item():.4f}  (ma50 {recent:.4f})")

        if step % args.sample_every == 0 or step == args.steps:
            _sample_montage(
                model, ctxs, pad_mask, res, args.sample_steps, device, dtype, run_dir, step, args.flow_shift
            )

    # ---- verdict: did shells-only actually learn? ----
    base = sum(loss_curve[:50]) / 50
    final = sum(loss_curve[-50:]) / 50
    drop = (base - final) / base if base > 0 else 0.0
    # Pure-noise FM loss floor ≈ E‖ε − x0‖² = Var(ε)+Var(x0) ≈ 1 + Var(x0). A model
    # that learns the mean velocity field drives MSE well below that.
    verdict = "PASS" if drop > 0.30 and final < 0.85 else ("WEAK" if drop > 0.10 else "FAIL")

    # ---- per-σ breakdown: is the low loss carried by the skip-copy regime? ----
    win = max(100, args.steps // 3)
    bands = _sigma_band_stats(sigma_curve[-win:], loss_curve[-win:])
    print("\n[l2p-p0] per-σ loss (final window):")
    for i, (m, c) in enumerate(bands):
        lo, hi = _SIGMA_BAND_EDGES[i], _SIGMA_BAND_EDGES[i + 1]
        cell = f"{m:.3f}" if m is not None else "  -  "
        print(f"           σ[{lo:.1f},{hi:.1f}): {cell}  (n={c})")

    def _band_mean(lo_edge: float, hi_edge: float):
        vals = [
            (m, c)
            for i, (m, c) in enumerate(bands)
            if _SIGMA_BAND_EDGES[i] >= lo_edge and _SIGMA_BAND_EDGES[i + 1] <= hi_edge and m is not None
        ]
        cnt = sum(c for _, c in vals)
        return sum(m * c for m, c in vals) / cnt if cnt else None

    loss_lo = _band_mean(0.0, 0.4)  # skip can copy the near-clean input here
    loss_hi = _band_mean(0.6, 1.0)  # input ≈ noise → only the frozen core can help
    # Skip-cheat signature: low-σ basically solved, high-σ stuck near the floor.
    skip_cheat = bool(
        args.dip_skip
        and loss_lo is not None
        and loss_hi is not None
        and loss_lo < 0.25
        and loss_hi > 3 * loss_lo
        and loss_hi > 0.5
    )

    print(f"\n[l2p-p0] VERDICT: {verdict}  (loss {base:.3f}→{final:.3f}, drop {drop:.0%})")
    if skip_cheat:
        print(
            f"[l2p-p0] ⚠ SKIP-CHEAT SUSPECTED: low-σ {loss_lo:.3f} vs high-σ {loss_hi:.3f} "
            "— the DiP input-skip is copying the near-clean input; the loss PASS is "
            "NOT generative. Trust the montage (from-noise), not this number."
        )
    print("[l2p-p0] PASS → Phase 1 (last-n blocks + DiP detailer + data scaling).")
    print("[l2p-p0] FAIL → latent→pixel gap too wide for cheap transfer at Anima scale.")
    print("[l2p-p0] Always confirm the from-noise montage — single-step loss can't see generation.")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        device=device,
        metrics={
            "verdict": verdict,
            "loss_base_ma50": base,
            "loss_final_ma50": final,
            "loss_drop_frac": drop,
            "trainable_params_M": n_train / 1e6,
            "flow_shift": args.flow_shift,
            "sigma_bands": {
                f"{_SIGMA_BAND_EDGES[i]:.1f}-{_SIGMA_BAND_EDGES[i + 1]:.1f}": bands[i][0]
                for i in range(len(bands))
            },
            "loss_lo_sigma": loss_lo,
            "loss_hi_sigma": loss_hi,
            "skip_cheat_suspected": skip_cheat,
            "n_images": len(pairs),
            "loss_curve": loss_curve,
        },
        artifacts=[p.name for p in sorted(run_dir.glob("sample_*.png"))],
    )
    print(f"[l2p-p0] wrote {run_dir}/result.json")


@torch.no_grad()
def _sample_montage(model, ctxs, pad_mask, res, steps, device, dtype, run_dir, step, flow_shift=1.0):
    """Euler-sample 2 images from noise and save a montage to eyeball coherence.

    The σ schedule carries the same ``flow_shift`` as the training draw — an
    unshifted sampler against a shift-trained model would walk a mismatched
    trajectory and understate quality."""
    from PIL import Image

    n = min(2, len(ctxs))
    ctx = torch.stack([ctxs[i] for i in range(n)])
    x = torch.randn(n, 3, res, res, device=device, dtype=dtype)
    sig = _apply_flow_shift(torch.linspace(1.0, 0.0, steps + 1, device=device), flow_shift)
    pm = pad_mask.expand(n, -1, -1, -1) if pad_mask is not None else None
    dip = hasattr(model.l2p_decoder, "set_skip")
    for i in range(steps):
        s = sig[i].expand(n).to(dtype)
        if dip:
            model.l2p_decoder.set_skip(x)  # current noisy image is the DiP skip
        v = model.forward(x.unsqueeze(2), s, context=ctx, padding_mask=pm).squeeze(2)
        x = x - (sig[i] - sig[i + 1]) * v  # dx = v·dσ, integrate σ:1→0
    img = ((x.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu()
    grid = torch.cat([img[k].permute(1, 2, 0) for k in range(n)], dim=1).numpy()
    Image.fromarray(grid).save(run_dir / f"sample_{step:05d}.png")


if __name__ == "__main__":
    main()
