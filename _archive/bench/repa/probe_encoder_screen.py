#!/usr/bin/env python3
"""REPA encoder screen — which vision encoder is the best REPA *target* for
Anima's frozen DiT, measured with zero training?

THE QUESTION. The shipped REPA term aligns block 8 to **PE-Spatial** patch
tokens (``library/training/repa.py``; default ``_repa_encoder="pe_spatial"``).
PE is CLIP-lineage (vision-language contrastive), and the REPA literature
(Yu et al., arXiv:2410.06940) finds *self-supervised* targets (DINO) beat
language-contrastive ones because the generative scaffold wants visual-structural
organization, not caption semantics. Three challengers, spanning the axis that
matters for a captioned-art domain — how much language vs. pure-visual structure
the target carries:

  * **pe_spatial**  — incumbent. Language-contrastive, spatial-tuned. (vendored)
  * **dinov2-large**— pure SSL, dense-optimized.            (facebook/dinov2-large)
  * **dinov3-vitl16**— pure SSL, *Gram-anchored* dense maps. (facebook/dinov3-…)
  * **tipsv2-l14**  — hybrid: image-text contrastive *plus* DINO-style spatial
                       self-distillation. The midpoint that tells us whether PE
                       loses because "language targets are bad" (DINO wins, TIPS
                       lands near PE) or "PE specifically is weak" (TIPS wins).
                       (google/tipsv2-l14)

THE RULER (identical to ``bench/repa/probe_layer_sigma_cka.py``). Centered linear
CKA between pooled DiT block tokens and the encoder patch tokens — the
no-training analog of the relational Gram loss the training term optimizes
(``relational_gram_loss``), its centering = the ``spatial_norm`` DC removal. The
**headline is the floor-subtracted content gap** ``useful_gap`` = (matched −
same-grid-mismatched) − (its σ→1 floor): strips the shared-spatial-layout
confound (via the mismatch) AND the caption-reconstruction confound (via the
σ→1 floor), leaving the alignment REPA can actually *inject*. Higher useful_gap
in the early **representation regime** (≤``--repr_regime_max``, the
REPA-defensible zone least redundant with the flow-matching loss) ⇒ the better
target.

FAIR COMPARISON. Every encoder sees the **same** image: each cached latent is
VAE-decoded to RGB once, then fed to all encoders at their own native norm,
square-resized so every encoder yields a **32×32** patch grid (last ``gh*gw``
tokens — patch tokens always follow the CLS/register prefix, so the slice is
model-agnostic). One grid for all ⇒ the DiT side pools once per (layer, σ) and
is compared against each encoder's tokens. VAE-decode (not the real image) is
deliberate: it's the manifold the model actually denoises on, and the
reconstruction error is uniform across encoders so it can't bias the *ranking*.

This is a **Phase-0 pre-screen**, not the verdict. It says how each encoder will
behave (anchor vs. inject regime, best align-layer) and ranks the no-training
alignment headroom. The decision is a short LoRA A/B (encoder the only variable,
judged by CMMD) on the winner(s) — which needs registering the encoder in
``library/vision/encoders.py`` + a caching pass, deliberately deferred until the
screen says it's worth it.

Run from anima_lora/ (downloads DINOv2/v3/TIPS on first use; DINOv3 is
manually-gated — request access at the model page if it skips)::

    uv run python bench/repa/probe_encoder_screen.py --num_samples 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima, resolve_dtype  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.repa.probe_layer_sigma_cka import (  # noqa: E402
    _all_block_features,
    linear_cka,
)
from bench.soft_tokens_contrastive.reward_premise_probe import (  # noqa: E402
    EmbCache,
    discover_pairs,
)
from library.io.cache import load_cached_latents  # noqa: E402
from library.training.repa import pool_dit_tokens_to_grid  # noqa: E402

log = logging.getLogger("bench.repa.encoder_screen")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_DATA = "post_image_dataset/lora"
DEFAULT_SIGMAS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
SHIPPED_LAYER = 8
GRID = 32  # every encoder is resized to yield a GRID×GRID patch grid

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

# name → (repo / loader spec). patch sets the square input res = GRID*patch so
# the encoder emits exactly GRID×GRID = 1024 patch tokens.
ENCODER_SPECS = {
    "pe_spatial": {"kind": "pe", "patch": 16},
    "dinov2-large": {"kind": "hf", "repo": "facebook/dinov2-large", "patch": 14},
    "dinov3-vitl16": {
        "kind": "hf",
        "repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "patch": 16,
    },
    "tipsv2-l14": {"kind": "tips", "repo": "google/tipsv2-l14", "patch": 14},
    # The "open door": Qwen3-VL vision tower → MERGED tokens, in Anima's own
    # Qwen language family. Tests whether language-alignment (why PE beats DINO)
    # extends to the conditioning's own space.
    "qwen3vl-embed-2b": {"kind": "qwenvl", "repo": "Qwen/Qwen3-VL-Embedding-2B"},
}
DEFAULT_ENCODERS = list(ENCODER_SPECS)


# ─────────────────────────────────────────────────────────────────────────────
# Dense-token extractors. Each returns (B, GRID*GRID, D) fp32 patch tokens from
# an RGB batch in [0, 1]. The "last gh*gw tokens" rule is model-agnostic: CLS +
# register tokens always prefix the patch tokens, so the tail is the grid.
# ─────────────────────────────────────────────────────────────────────────────
class _HFDense:
    def __init__(self, name, repo, patch, device, dtype, trust_remote_code=False):
        from transformers import AutoModel

        self.name = name
        self.patch = patch
        self.res = GRID * patch
        self.device = device
        self.dtype = dtype
        self.model = (
            AutoModel.from_pretrained(repo, trust_remote_code=trust_remote_code)
            .to(device, dtype)
            .eval()
        )
        self.model.requires_grad_(False)
        mean, std = _resolve_norm(repo, trust_remote_code)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.d = None

    @torch.no_grad()
    def encode(self, rgb01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            rgb01, size=(self.res, self.res), mode="bicubic", align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        x = (x - self.mean.to(rgb01)) / self.std.to(rgb01)
        out = self.model(pixel_values=x.to(self.device, self.dtype))
        h = getattr(out, "last_hidden_state", None)
        if h is None:
            h = out[0] if isinstance(out, (tuple, list)) else out
        n = GRID * GRID
        if h.dim() != 3 or h.shape[1] < n:
            raise RuntimeError(
                f"{self.name}: got {tuple(h.shape)} tokens, need ≥{n} patch tokens "
                f"at res {self.res} (patch {self.patch}). Wrong patch size?"
            )
        tok = h[:, -n:, :].float()  # (B, 1024, D) — patch tokens (tail)
        self.d = int(tok.shape[-1])
        return tok


class _TIPSDense:
    """TIPSv2 (hybrid image-text + DINO-style spatial). Custom modeling: its
    ``encode_image`` takes **[0, 1]** pixels (normalizes internally) and returns
    a ``TIPSv2ImageOutput`` whose ``patch_tokens`` are already patch-only (CLS /
    register tokens are separate fields) — so no tail-slice, no ImageNet norm."""

    def __init__(self, name, repo, patch, device, dtype):
        from transformers import AutoModel

        self.name = name
        self.patch = patch
        self.res = GRID * patch  # 448 @ patch14 → 32×32 patch tokens
        self.device = device
        self.dtype = dtype
        self.model = (
            AutoModel.from_pretrained(repo, trust_remote_code=True)
            .to(device, dtype)
            .eval()
        )
        self.model.requires_grad_(False)
        self.d = None

    @torch.no_grad()
    def encode(self, rgb01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            rgb01, size=(self.res, self.res), mode="bicubic", align_corners=False,
            antialias=True,
        ).clamp(0, 1)  # TIPS wants [0, 1], normalizes internally
        out = self.model.encode_image(x.to(self.device, self.dtype))
        tok = out.patch_tokens.float()  # (B, N, D), patch-only
        n = GRID * GRID
        if tok.shape[1] != n:
            raise RuntimeError(
                f"{self.name}: {tok.shape[1]} patch tokens at res {self.res}, "
                f"need {n} (32×32). Wrong patch size?"
            )
        self.d = int(tok.shape[-1])
        return tok


class _QwenVLDense:
    """Qwen3-VL vision tower → **merged** tokens (out_hidden_size), the
    "vision in Qwen language space" representation fed to the LLM — the only
    candidate that shares geometry with Anima's Qwen text conditioning.

    Uses the Qwen processor's ``grid_thw`` plumbing: a 1024² square → 64×64
    patches → (spatial_merge_size=2) → 32×32 merged tokens. The merger output
    order is row-major spatial; if it weren't, the same-grid mismatch control
    would collapse useful_gap → ~0, which is detectable, not silent.
    """

    def __init__(self, name, repo, device, dtype):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = name
        self.device = device
        self.dtype = dtype
        self.proc = AutoProcessor.from_pretrained(repo)
        model = (
            AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype)
            .to(device)
            .eval()
        )
        model.requires_grad_(False)
        self._model = model  # keep alive
        self.visual = getattr(model, "visual", None) or model.model.visual
        vc = self.visual.config
        self.merge = int(getattr(vc, "spatial_merge_size", 2))
        self.patch = int(getattr(vc, "patch_size", 16))
        self.res = GRID * self.merge * self.patch  # 32*2*16 = 1024
        self.d = None

    @torch.no_grad()
    def encode(self, rgb01: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.functional import to_pil_image

        x = F.interpolate(
            rgb01, size=(self.res, self.res), mode="bicubic", align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        inp = self.proc.image_processor(images=to_pil_image(x[0].cpu()), return_tensors="pt")
        pv = inp["pixel_values"].to(self.device, self.dtype)
        thw = inp["image_grid_thw"].to(self.device)
        out = self.visual(pv, grid_thw=thw)
        # pooler_output = merger(hidden_states): the post-spatial-merge sequence
        # (1024 tokens at out_hidden_size = LLM dim) — the language-space target.
        # last_hidden_state is the pre-merge 4096-patch grid (raw visual). Despite
        # the name, pooler_output here is the full merged sequence, not a vector.
        emb = getattr(out, "pooler_output", None)
        if emb is None:
            emb = getattr(out, "last_hidden_state", None)
        if emb is None:
            emb = out[0] if isinstance(out, (tuple, list)) else out
        if emb.dim() == 3:
            emb = emb[0]  # (N, D)
        n = GRID * GRID
        if emb.shape[0] != n:
            raise RuntimeError(
                f"{self.name}: {emb.shape[0]} merged tokens (grid_thw={thw.tolist()}), "
                f"need {n} (32×32). spatial_merge={self.merge}, patch={self.patch}."
            )
        self.d = int(emb.shape[-1])
        return emb.unsqueeze(0).float()  # (1, N, D)


class _PEDense:
    """PE-Spatial via the vendored registry loader (no HF transformers path)."""

    def __init__(self, name, device, dtype):
        from library.vision.encoder import load_pe_encoder

        self.name = name
        self.patch = 16
        self.res = GRID * 16  # 512 — PE-Spatial-B16-512's native input
        self.device = device
        self.dtype = dtype
        self.bundle = load_pe_encoder(device, name="pe_spatial", dtype=dtype)
        self.d = None

    @torch.no_grad()
    def encode(self, rgb01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            rgb01, size=(self.res, self.res), mode="bicubic", align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        x = (x - 0.5) / 0.5  # PE Normalize(0.5, 0.5): [0,1] → [-1,1]
        out = self.bundle.encoder(x.to(self.device, self.dtype))
        h = out.last_hidden_state  # (B, 1+1024, D), CLS at 0
        tok = h[:, -GRID * GRID :, :].float()
        self.d = int(tok.shape[-1])
        return tok


def _resolve_norm(repo: str, trust_remote_code: bool):
    """Pull (mean, std) from the model's image processor; ImageNet fallback."""
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(
            repo, trust_remote_code=trust_remote_code
        )
        mean = getattr(proc, "image_mean", None)
        std = getattr(proc, "image_std", None)
        if mean and std and len(mean) == 3:
            return tuple(mean), tuple(std)
    except Exception as e:  # noqa: BLE001
        log.info(f"  ({repo}: no image processor [{type(e).__name__}], ImageNet norm)")
    return IMAGENET_MEAN, IMAGENET_STD


def build_encoder(name: str, device, dtype):
    spec = ENCODER_SPECS[name]
    if spec["kind"] == "pe":
        return _PEDense(name, device, dtype)
    if spec["kind"] == "tips":
        return _TIPSDense(name, spec["repo"], spec["patch"], device, dtype)
    if spec["kind"] == "qwenvl":
        return _QwenVLDense(name, spec["repo"], device, dtype)
    return _HFDense(
        name, spec["repo"], spec["patch"], device, dtype,
        trust_remote_code=spec.get("trust_remote_code", False),
    )


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=True, text_encoder=False)
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--num_seeds", type=int, default=1)
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all")
    ap.add_argument(
        "--encoders", nargs="+", default=DEFAULT_ENCODERS,
        help=f"subset of {DEFAULT_ENCODERS}",
    )
    ap.add_argument("--band_lo", type=float, default=0.45)
    ap.add_argument("--band_hi", type=float, default=0.90)
    ap.add_argument("--repr_regime_max", type=int, default=11)
    add_common_args(ap)
    args = ap.parse_args()

    device = torch.device(
        args.device if isinstance(args.device, str) else "cuda"
    )
    dtype = resolve_dtype(args.dtype)
    sigma_grid = sorted(float(s) for s in args.sigmas)
    band = [s for s in sigma_grid if args.band_lo <= s <= args.band_hi] or sigma_grid
    n_seeds = max(1, args.num_seeds)

    pairs = discover_pairs(args.data_dir)
    embs = EmbCache(pairs)
    rng = np.random.default_rng(args.seed)
    # Need a latent + a caption embedding per sample.
    pool = sorted(s for s in pairs)
    rng.shuffle(pool)

    # ── Phase A: VAE-decode latents → RGB[0,1], keep emb + latent on CPU ───────
    from library.models.qwen_vae import load_vae

    vae = load_vae(args.vae, device=device, dtype=torch.bfloat16, eval=True, vae_2d=True)
    samples: list[dict] = []
    for stem in pool:
        if len(samples) >= args.num_samples:
            break
        emb = embs.get(stem)
        if emb is None:
            continue
        npz_path, _te = pairs[stem]
        lat, _res, _oh, _ow = load_cached_latents(npz_path)  # (C,H,W) fp32
        with torch.no_grad():
            rgb = vae.decode_to_pixels(lat.unsqueeze(0).to(device, torch.bfloat16))
            rgb01 = ((rgb.float() + 1.0) / 2.0).clamp(0, 1).cpu()  # (1,3,H,W)
        samples.append(
            {
                "stem": stem,
                "x0": lat.unsqueeze(0).unsqueeze(2),  # (1,C,1,H,W) CPU
                "emb": emb.unsqueeze(0),  # (1,S,D) CPU
                "hw": (int(lat.shape[-2]), int(lat.shape[-1])),
                "rgb01": rgb01,
            }
        )
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not samples:
        raise SystemExit("no samples with both a latent and a caption embedding")
    n_img = len(samples)
    # Same-grid mismatched partner = next image (all share the 32×32 grid).
    partner = [(i + 1) % n_img for i in range(n_img)]
    log.info(f"decoded {n_img} images · encoders={args.encoders} · σ={sigma_grid}")

    # ── Phase B: per-encoder dense tokens (one encoder resident at a time) ─────
    # enc_tok[name] = list over images of (1024, D) fp16 CPU tensors.
    enc_tok: dict[str, list[torch.Tensor]] = {}
    enc_dim: dict[str, int] = {}
    for name in args.encoders:
        if name not in ENCODER_SPECS:
            log.warning(f"  skip unknown encoder '{name}'")
            continue
        try:
            enc = build_encoder(name, device, dtype)
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"  ✗ {name}: load failed [{type(e).__name__}: {str(e)[:140]}] — "
                "skipped (DINOv3 is gated; request access or drop it)."
            )
            continue
        toks: list[torch.Tensor] = []
        try:
            for s in samples:
                t = enc.encode(s["rgb01"].to(device))  # (1,1024,D)
                toks.append(t[0].half().cpu())
        except Exception as e:  # noqa: BLE001
            log.warning(f"  ✗ {name}: encode failed [{type(e).__name__}: {str(e)[:140]}]")
            del enc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        enc_tok[name] = toks
        enc_dim[name] = enc.d
        log.info(f"  ✓ {name}: D={enc.d}, res={enc.res}, {n_img} images")
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    encoders = [n for n in args.encoders if n in enc_tok]
    if not encoders:
        raise SystemExit("no encoder loaded — nothing to score")

    # ── Phase C: DiT forwards (shared across encoders) → CKA per (enc,layer,σ) ─
    bundle = build_anima(args, adapter=None, train_mode=False)
    anima = bundle.anima
    patch = int(anima.patch_spatial)
    n_blocks = len(anima.blocks)
    layers = sorted(args.layers) if args.layers else list(range(n_blocks))
    if any(not (0 <= ell < n_blocks) for ell in layers):
        raise SystemExit(f"--layers out of range (DiT has {n_blocks} blocks)")
    layer_set = set(layers)

    # acc[name][layer][σ] = (matched_list, mismatch_list) per image (seed-avg).
    cka_m = {e: {ell: {s: [] for s in sigma_grid} for ell in layers} for e in encoders}
    cka_x = {e: {ell: {s: [] for s in sigma_grid} for ell in layers} for e in encoders}

    for ai, s_rec in enumerate(samples):
        H, W = s_rec["hw"]
        x0 = s_rec["x0"].to(device, dtype)
        x0_f = x0.float()
        emb_b = s_rec["emb"].to(device, dtype)
        pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
        pe_now = {e: enc_tok[e][ai].to(device).float().unsqueeze(0) for e in encoders}
        pe_x = {
            e: enc_tok[e][partner[ai]].to(device).float().unsqueeze(0) for e in encoders
        }
        for s in sigma_grid:
            acc_m = {e: {ell: [] for ell in layers} for e in encoders}
            acc_x = {e: {ell: [] for ell in layers} for e in encoders}
            for sj in range(n_seeds):
                g = torch.Generator(device=device).manual_seed(args.seed + ai * 1000 + sj)
                eps = torch.randn(x0.shape, generator=g, device=device, dtype=dtype)
                noisy = ((1.0 - s) * x0_f + s * eps.float()).to(dtype)
                t_b = torch.full((1,), float(s), device=device, dtype=dtype)
                feats = _all_block_features(anima, noisy, t_b, emb_b, pad, layer_set)
                for ell in layers:
                    dit_tok = pool_dit_tokens_to_grid(feats[ell], (H, W), patch, GRID, GRID)
                    d0 = dit_tok[0]
                    for e in encoders:
                        acc_m[e][ell].append(linear_cka(d0, pe_now[e][0]))
                        acc_x[e][ell].append(linear_cka(d0, pe_x[e][0]))
            for e in encoders:
                for ell in layers:
                    cka_m[e][ell][s].append(float(np.mean(acc_m[e][ell])))
                    cka_x[e][ell][s].append(float(np.mean(acc_x[e][ell])))
        if (ai + 1) % 10 == 0 or ai + 1 == n_img:
            log.info(f"  [{ai + 1}/{n_img}] scored")
        if device.type == "cuda" and (ai + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # ── aggregate → per-encoder useful_gap(layer×σ) + cross-encoder ranking ────
    band_cols = [sigma_grid.index(s) for s in band]
    repr_rows = [i for i, ell in enumerate(layers) if ell <= args.repr_regime_max]
    ship_row = int(np.argmin([abs(ell - SHIPPED_LAYER) for ell in layers]))

    per_enc: dict[str, dict] = {}
    for e in encoders:
        m = np.array([[float(np.mean(cka_m[e][ell][s])) for s in sigma_grid] for ell in layers])
        x = np.array([[float(np.mean(cka_x[e][ell][s])) for s in sigma_grid] for ell in layers])
        gap = m - x
        floor = gap[:, -1:].copy()  # σ→1 caption-reconstruction floor
        useful = np.clip(gap - floor, 0.0, None)
        band_useful = useful[:, band_cols].mean(axis=1)
        band_matched = m[:, band_cols].mean(axis=1)
        l_star = int(layers[int(np.argmax(band_useful))])
        if repr_rows:
            rr = repr_rows[int(np.argmax(band_useful[repr_rows]))]
            l_star_repr = int(layers[rr])
            repr_score = float(band_useful[rr])
        else:
            l_star_repr, repr_score = SHIPPED_LAYER, float("nan")
        ridge = [int(layers[int(np.argmax(useful[:, j]))]) for j in range(len(sigma_grid))]
        ceiling = useful.max(axis=0)
        sigma_weight = ceiling / max(float(ceiling.max()), 1e-12)
        per_enc[e] = {
            "useful": useful,
            "matched": m,
            "band_useful": band_useful,
            "band_matched": band_matched,
            "l_star": l_star,
            "best_band_useful": float(band_useful.max()),
            "l_star_repr": l_star_repr,
            "repr_band_useful": repr_score,
            "shipped_band_useful": float(band_useful[ship_row]),
            "shipped_band_matched": float(band_matched[ship_row]),
            "ridge": ridge,
            "sigma_weight": [float(w) for w in sigma_weight],
            "d_enc": enc_dim[e],
        }

    # Rank by the REPA-defensible signal: content gap in the representation regime.
    ranking = sorted(encoders, key=lambda e: per_enc[e]["repr_band_useful"], reverse=True)
    winner = ranking[0]
    base = "pe_spatial" if "pe_spatial" in per_enc else None
    base_repr = per_enc[base]["repr_band_useful"] if base else float("nan")

    # ── artifacts ──────────────────────────────────────────────────────────────
    run_dir = make_run_dir("repa", label=args.label or "encoder-screen")
    np.savez(
        run_dir / "heatmaps.npz",
        layers=np.array(layers),
        sigmas=np.array(sigma_grid),
        encoders=np.array(encoders),
        **{f"useful_{e}": per_enc[e]["useful"] for e in encoders},
        **{f"matched_{e}": per_enc[e]["matched"] for e in encoders},
    )

    M = ["# REPA encoder screen — which target maximizes injectable alignment\n"]
    M.append(
        f"- images: **{n_img}** (VAE-decoded, shared across encoders) · "
        f"{n_seeds} noise draw(s) · base DiT, no adapter\n"
        f"- ruler: centered linear CKA; **headline = useful_gap** "
        f"(matched − same-grid mismatched, σ→1 floor subtracted) on the "
        f"σ∈[{args.band_lo},{args.band_hi}] band\n"
        f"- ranked by **representation-regime** useful_gap (best layer "
        f"≤{args.repr_regime_max} — the REPA-defensible zone, least redundant "
        f"with the flow loss)\n"
        f"- shipped: **pe_spatial** @ layer {SHIPPED_LAYER}\n"
    )
    skipped = [e for e in args.encoders if e in ENCODER_SPECS and e not in per_enc]
    if skipped:
        M.append(f"- ⚠️ skipped (load/encode failed): {', '.join(skipped)}\n")

    M.append("\n## Ranking (representation-regime content gap)\n")
    M.append(
        "| rank | encoder | D | l\\*(≤" + str(args.repr_regime_max) + ") | "
        "repr useful_gap | best l\\* | best useful_gap | @layer8 useful_gap | "
        "@layer8 matched-CKA |"
    )
    M.append("|---|---|---|---|---|---|---|---|---|")
    for r, e in enumerate(ranking, 1):
        p = per_enc[e]
        flag = " ⭐" if e == winner else (" (shipped)" if e == base else "")
        M.append(
            f"| {r}{flag} | {e} | {p['d_enc']} | {p['l_star_repr']} | "
            f"{p['repr_band_useful']:.4f} | {p['l_star']} | "
            f"{p['best_band_useful']:.4f} | {p['shipped_band_useful']:.4f} | "
            f"{p['shipped_band_matched']:.4f} |"
        )

    if base and winner != base and not np.isnan(base_repr):
        lift = per_enc[winner]["repr_band_useful"] - base_repr
        rel = lift / base_repr * 100 if base_repr > 1e-9 else float("inf")
        M.append(
            f"\n**Verdict:** `{winner}` leads the incumbent `pe_spatial` on "
            f"representation-regime injectable alignment by **+{lift:.4f}** "
            f"({rel:+.0f}%). This is a *no-training proxy* — promote to a LoRA "
            f"A/B (encoder-only variable, CMMD-judged) before committing the "
            f"re-cache.\n"
        )
    elif base and winner == base:
        M.append(
            "\n**Verdict:** the incumbent `pe_spatial` already leads on the "
            "no-training proxy — no encoder swap justified by this screen. The "
            "DINO/TIPS prior didn't transfer to Anima's manifold.\n"
        )

    M.append("\n## Per-encoder σ-weight (useful_gap ceiling, peak-norm) + ridge\n")
    M.append("| encoder | " + " | ".join(f"σ={s:g}" for s in sigma_grid) + " |")
    M.append("|---|" + "---|" * len(sigma_grid))
    for e in ranking:
        sw = " | ".join(f"{w:.2f}" for w in per_enc[e]["sigma_weight"])
        M.append(f"| {e} (σ-wt) | {sw} |")
        rd = " | ".join(str(v) for v in per_enc[e]["ridge"])
        M.append(f"| {e} (l\\*(σ)) | {rd} |")

    M.append("\n## Reading it\n")
    M.append(
        "- **useful_gap** is the part of the alignment REPA can *inject*: matched "
        "CKA minus generic-layout (mismatch) minus caption-reconstruction (σ→1 "
        "floor). The encoder with the most of it in the early layers gives REPA "
        "the most non-redundant signal to anchor to.\n"
        "- A high @layer8 matched-CKA but low useful_gap = the DiT already tracks "
        "that encoder's *layout* (little to inject; pure anchoring regime).\n"
        "- This screen ranks *headroom*, not final quality — a low-headroom "
        "incumbent can still win the training A/B via anchoring. Confirm with the "
        "A/B before re-caching.\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    metrics = {
        "n_images": n_img,
        "num_seeds": n_seeds,
        "encoders_scored": encoders,
        "encoders_skipped": skipped,
        "layers": layers,
        "sigma_grid": sigma_grid,
        "band": band,
        "repr_regime_max": args.repr_regime_max,
        "shipped_layer": SHIPPED_LAYER,
        "winner": winner,
        "ranking": ranking,
        "per_encoder": {
            e: {
                "d_enc": per_enc[e]["d_enc"],
                "l_star": per_enc[e]["l_star"],
                "best_band_useful": per_enc[e]["best_band_useful"],
                "l_star_repr": per_enc[e]["l_star_repr"],
                "repr_band_useful": per_enc[e]["repr_band_useful"],
                "shipped_band_useful": per_enc[e]["shipped_band_useful"],
                "shipped_band_matched": per_enc[e]["shipped_band_matched"],
                "ridge": per_enc[e]["ridge"],
                "sigma_weight": per_enc[e]["sigma_weight"],
            }
            for e in encoders
        },
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["summary.md", "heatmaps.npz"],
        device=device,
    )

    log.info("\n" + "=" * 72)
    log.info(f"  REPA encoder screen → {run_dir}")
    for r, e in enumerate(ranking, 1):
        p = per_enc[e]
        tag = " ⭐" if e == winner else (" (shipped)" if e == base else "")
        log.info(
            f"  {r}. {e}{tag}: repr useful_gap {p['repr_band_useful']:.4f} "
            f"@l{p['l_star_repr']} | best {p['best_band_useful']:.4f} @l{p['l_star']} "
            f"| layer8 {p['shipped_band_useful']:.4f}"
        )
    if skipped:
        log.info(f"  skipped: {skipped}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
