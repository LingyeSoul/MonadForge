#!/usr/bin/env python3
"""OSCAR Phase-0 probe — can we skip PE (and the per-step VAE decode)?

arXiv 2510.09060v2 ("Letting Trajectories Spread"). OSCAR pushes a *set* of
same-prompt trajectories apart by descending a log-det volume energy
``E(Z) = −½ log det(I + τ ZZᵀ)`` on endpoint features ``Z = φ(x̂0)``, with the
push projected orthogonal to the base velocity v_θ. The faithful φ is PE-Core,
which forces ``VAE.decode(x̂0) → RGB → PE`` every step (and a VJP back through
both). The free alternative is to run the volume energy on the **bare latent**
x̂0 — no decode, no tower.

This probe asks the load-bearing question before we wire anything: can the
*free* bare-latent push substitute for the decode+PE push, in the σ-window where
x0 is forming (≲0.45)? Cosine between the two gradients is a weak read on its
own — they live in different spaces (147k-dim raw latent vs a pooled D-vector),
so disagreement is expected even when both are valid. The decisive, space-
agnostic test is a **finite-difference transfer**: step the latent set along
each push and measure the change in PE-space volume energy E_pe. If the latent
push lowers E_pe nearly as much as the PE push does, the decode is redundant.

Each push is also validated *in its own space* before the transfer is trusted:
the latent push must point off the latent centroid (lat_cen_cos), and the PE
energy gradient must point off the PE-feature centroid (feat_push_cos). The PE
*pullback* (the VJP through decode+PE) is validated end-to-end by ΔE_pe<0.

IMPORTANT — precision: the PE path (VAE decode + PE ViT + the VJP back through
both) runs in **float32**. A bf16 backward through that depth collapses the
pullback direction to quantization noise (every cosine ≈ 0); that was the
original bug. `_setup` recasts both models to fp32 for this reason.

No engine edits. Self-contained batched euler rollout (bare DiT) captures
(x_t, v, σ) per step; at a handful of probe σ's it recomputes the volume
gradient in {latent, PE} via autograd and runs the transfer test.

Run from repo root (anima_lora/):
    python bench/oscar/probe_feature_space.py --label base-1024
    python bench/oscar/probe_feature_space.py --cfg 4.0 --size 1248 832 --label cfg4-portrait
    # decode-grad runs fp32 with VAE gradient checkpointing on; if it still
    # OOMs at 1024², drop --size and/or --batch:
    python bench/oscar/probe_feature_space.py --size 768 768 --batch 4

Read (forming band, σ≤0.45):
    First the gates (each push valid in its own space, pullback delivers):
      feat_push_cos < 0.10 → COLLAPSED: endpoints too similar to spread in PE.
      lat_cen_cos   < 0.10 → DEGENERATE latent gradient.
      ΔE_pe(pe) ≥ 0        → PULLBACK-BROKEN: PE VJP not delivering (precision?).
    Then the decision — transfer = ΔE_pe(latent) / ΔE_pe(PE):
      ≥0.50 → GO: free latent push recovers ≥half the diversity gain → skip decode.
      0.20–0.50 → PARTIAL: PE likely load-bearing for full semantic spread.
      <0.20 → NO: latent push buys ~no PE diversity → decode+PE is the price.
    frac_par_lat high (≳0.7) → ⊥v safeguard nukes the latent push → unusable.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

# fp32 decode-grad is memory-heavy; reduce allocator fragmentation before torch
# initializes its CUDA caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from bench._common import make_run_dir, write_result

log = logging.getLogger("bench.oscar.probe_feature_space")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DIT = os.environ.get("ANIMA_DIT", "models/diffusion_models/anima-base-v1.0.safetensors")
VAE = os.environ.get("ANIMA_VAE", "models/vae/qwen_image_vae.safetensors")
TEXT_ENCODER = os.environ.get(
    "ANIMA_TEXT_ENCODER", "models/text_encoders/qwen_3_06b_base.safetensors"
)

# Varied content so the read reflects the model, not one prompt's manifold.
DEFAULT_PROMPTS = [
    "a sprawling cyberpunk city at night, neon signs, rain, wide shot",
    "a close-up portrait of an old fisherman, weathered skin, sharp texture",
    "a calm watercolor landscape, rolling hills, soft gradients, minimal detail",
]

# σ-forming boundary from project_sigma_signal_resolves_by_045.
FORMING_SIGMA = 0.45
EPS = 1e-12


# --------------------------------------------------------------------------- #
# OSCAR set-volume energy (shared by both feature spaces).
# --------------------------------------------------------------------------- #
def set_volume_energy(Z: torch.Tensor, tau: float) -> torch.Tensor:
    """E(Z) = −½ log det(I_m + τ ZZᵀ) over per-sample rows of Z ([m, D]).

    OSCAR Eq. 3 verbatim: the ``+I`` term is the trace stabilizer that *obviates
    manual centering*, and the raw features are used (no per-row normalization —
    that would destroy the relative magnitudes carrying the set geometry). We
    apply only a single **global** scale (divide by the mean row-norm) so τ
    behaves comparably across the latent and PE feature spaces. Lower E ⇒ larger
    spread; the diversity push is −∇E.
    """
    Z = Z.float()
    Z = Z / Z.norm(dim=1).mean().clamp_min(EPS)  # global scale only
    m = Z.shape[0]
    G = Z @ Z.t()
    eye = torch.eye(m, device=Z.device, dtype=Z.dtype)
    return -0.5 * torch.logdet(eye + tau * G)


def _perp(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Component of g orthogonal to v (OSCAR's fidelity safeguard)."""
    denom = (v * v).sum().clamp_min(EPS)
    return g - ((g * v).sum() / denom) * v


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item())


# --------------------------------------------------------------------------- #
# Model setup (bare DiT + prompt embeds + GPU VAE + PE tower).
# --------------------------------------------------------------------------- #
def _setup(args, device):
    import inference as inference_mod
    from library.inference import sampling as inference_utils
    from library.inference.models import load_dit_model
    from library.inference.text import (
        MAX_CROSSATTN_TOKENS,
        ensure_text_strategies,
        prepare_text_inputs,
    )
    from library.models import qwen_vae
    from library.vision import load_pe_encoder

    H, W = args.size
    infer_argv = [
        "--dit", args.dit, "--text_encoder", args.text_encoder, "--vae", args.vae,
        "--vae_chunk_size", "64", "--vae_disable_cache", "--attn_mode", "flash",
        "--prompt", args.prompts[0], "--negative_prompt", "",
        "--image_size", str(H), str(W), "--infer_steps", str(args.steps),
        "--flow_shift", str(args.flow_shift), "--guidance_scale", str(args.cfg),
        "--seed", "0", "--device", str(device), "--save_path", "output/tests",
    ]
    _saved = sys.argv
    try:
        sys.argv = ["inference.py", *infer_argv]
        iargs = inference_mod.parse_args()
    finally:
        sys.argv = _saved
    iargs.lora_weight = None
    iargs.sampler = "euler"

    ensure_text_strategies(args.text_encoder, MAX_CROSSATTN_TOKENS)
    log.info("Loading DiT (no LoRA)%s ...", " + block-compile" if args.compile else " (eager)")
    anima = load_dit_model(iargs, device, torch.bfloat16)
    if args.compile:
        # No adapter here, so compile-after-apply reduces to compile-after-load.
        anima.compile_blocks(mode=args.compile_mode)

    embeds = []
    neg_embed = None
    for prompt in args.prompts:
        iargs.prompt = prompt
        ctx, ctx_null = prepare_text_inputs(iargs, device, anima, shared_models=None)
        embeds.append((prompt, ctx["embed"][0].to(device, torch.bfloat16)))
        if neg_embed is None:
            neg_embed = ctx_null["embed"][0].to(device, torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # VAE + PE on-device in **float32**. This is load-bearing: the PE push is a
    # VJP back through the whole decoder + ViT, and a bf16 backward through that
    # depth turns the gradient *direction* into quantization noise (cos with
    # anything ≈ 0). The latent push is already fp32; running PE in bf16 made the
    # two spaces incomparable. The loaders hardcode bf16, so we recast.
    vae = qwen_vae.load_vae(args.vae, device=device, spatial_chunk_size=64)
    vae.to(torch.float32).eval()
    # fp32 decode-grad at full res is memory-heavy; checkpoint the decoder
    # up-blocks (also disables the causal-conv cache, required for correct
    # checkpoint recompute — equivalent for our single-frame decode).
    vae.enable_gradient_checkpointing()
    pe = load_pe_encoder(device, dtype=torch.float32)
    pe.encoder.inner.to(torch.float32)  # cast the vendored PE tower (loader hardcodes bf16)
    pe.dtype = torch.float32  # encode_pe_from_imageminus1to1 casts inputs to this

    _, sigmas = inference_utils.get_timesteps_sigmas(args.steps, args.flow_shift, device)
    return anima, embeds, neg_embed, vae, pe, sigmas.to(device)


def _rollout(anima, embed, neg_embed, sigmas, cfg, m, h_lat, w_lat, device, base_seed, probe_steps):
    """Batched euler rollout of m seeds for one prompt. At each probe step we
    stash (σ, x̂0, v) on CPU (x̂0 = x_t − σ·v); everything else is discarded so
    the DiT can be freed before the memory-heavy decode-grad phase."""
    try:
        from library.inference.adapters import set_hydra_sigma
    except Exception:  # noqa: BLE001
        set_hydra_sigma = lambda *_a, **_k: None  # noqa: E731 — bare DiT has no hydra

    gen = torch.Generator(device="cpu")
    lats = []
    for j in range(m):
        gen.manual_seed(base_seed + j)
        lats.append(
            torch.randn((1, anima.LATENT_CHANNELS, 1, h_lat, w_lat), generator=gen).to(
                device, torch.bfloat16
            )
        )
    x = torch.cat(lats, dim=0)  # [m,16,1,h,w]

    emb = embed.expand(m, -1, -1)
    nemb = neg_embed.expand(m, -1, -1)
    pad = torch.zeros(m, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    do_cfg = cfg != 1.0
    n = len(sigmas) - 1
    want = set(probe_steps)

    probes = {}  # step -> (sigma, x0_cpu, v_cpu)
    with torch.no_grad():
        for i in range(n):
            sigma = float(sigmas[i])
            t = x.new_full((m,), sigma)
            set_hydra_sigma(anima, t)
            v_c = anima(x, t, emb, padding_mask=pad)
            if do_cfg:
                v_u = anima(x, t, nemb, padding_mask=pad)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = v_c
            if i in want:
                x0 = x.float() - sigma * v.float()  # x̂0 = x_t − σ·v
                probes[i] = (sigma, x0.to("cpu", torch.bfloat16), v.to("cpu", torch.bfloat16))
            dt = float(sigmas[i + 1]) - sigma
            x = (x.float() + v.float() * dt).to(torch.bfloat16)
    return [(probes[i][0], probes[i][1], probes[i][2]) for i in sorted(probes)]


# --------------------------------------------------------------------------- #
# Gradient comparison at one probe step.
# --------------------------------------------------------------------------- #
def _latent_push(x0: torch.Tensor, tau: float) -> torch.Tensor:
    """−∇_{x̂0} E over the full bare latent (φ = identity; no decode).

    This is the honest Particle-Guidance-style latent push: the energy runs on
    the flattened latent itself, so the gradient lives in *true* latent space.
    (A random projection was tried and rejected — it confines the gradient to a
    random low-dim subspace, making any full-space cosine ≈ √(D/N), an artifact.)
    The gradient is small in per-element norm (spread over ~150k dims) but its
    *direction* is what we compare, and cosine is scale-invariant."""
    m = x0.shape[0]
    leaf = x0.detach().float().requires_grad_(True)
    E = set_volume_energy(leaf.reshape(m, -1), tau)
    (g,) = torch.autograd.grad(E, leaf)
    return -g  # push direction


def _pe_feature(x0_i, vae, pe):
    """Pooled+L2-normalized PE feature of one decoded endpoint. x0_i: [1,16,1,h,w]."""
    from library.training.cmmd import pool_and_normalize
    from library.vision import encode_pe_from_imageminus1to1

    rgb = vae.decode_to_pixels(x0_i.to(torch.float32))  # [1,3,1,H,W] for 5D input
    if rgb.dim() == 5:  # drop the temporal axis → [1,3,H,W] for the PE tower
        rgb = rgb.squeeze(2)
    feats = encode_pe_from_imageminus1to1(pe, rgb, same_bucket=True)[0]  # [T,D]
    return pool_and_normalize(feats)  # [D]


def _pe_features(x0, vae, pe):
    """Stack of pooled+L2 PE features for the batch — [m, D], no grad."""
    with torch.no_grad():
        return torch.stack(
            [_pe_feature(x0[i : i + 1], vae, pe) for i in range(x0.shape[0])], dim=0
        ).float()


def _pe_energy(x0, vae, pe, tau):
    """PE-space set-volume energy E_pe(φ(decode(x̂0))), no grad. Lower ⇒ more
    semantic spread. The finite-difference probe steps the latent along a push
    and asks whether *this* number drops — the space-agnostic test of whether a
    push actually buys PE-space diversity."""
    return float(set_volume_energy(_pe_features(x0, vae, pe), tau).item())


def _pe_push(x0, vae, pe, tau):
    """−∇_{x̂0} E over PE features of the decoded endpoints, pulled back to latent
    via a **per-sample VJP** (one decode backprop at a time — OSCAR's own scheme,
    and the only way this fits in memory). Pass A builds Z and the analytic
    dE/dZ; Pass B back-props each sample's decode→PE with that upstream grad.

    Returns ``(push, Z, dEdZ)``: ``Z`` (the feature rows) and ``dEdZ`` (the
    in-feature-space energy gradient) let the caller validate the push *in PE
    space* — cos(−dEdZ_i, Z_i−Z̄) — before trusting the latent-space pullback.

    Everything runs in fp32: the leaf, the decode+PE graph (models recast to
    fp32 in ``_setup``), and the upstream grad. A bf16 backward through this
    depth was the bug — it collapsed the pullback direction to noise."""
    m = x0.shape[0]
    with torch.no_grad():  # Pass A: features (no graph)
        Z = _pe_features(x0, vae, pe)
    Zl = Z.detach().float().requires_grad_(True)
    E = set_volume_energy(Zl, tau)
    (dEdZ,) = torch.autograd.grad(E, Zl)  # [m,D] — cross-sample coupling

    g = torch.zeros_like(x0, dtype=torch.float32)
    for i in range(m):  # Pass B: per-sample VJP
        leaf = x0[i : i + 1].detach().to(torch.float32).requires_grad_(True)
        z = _pe_feature(leaf, vae, pe)  # [D], same boundary as Z's rows
        try:
            (gi,) = torch.autograd.grad(z, leaf, grad_outputs=dEdZ[i].to(z.dtype))
        except RuntimeError as e:  # encoder/decoder severed the graph
            raise RuntimeError(
                "PE-path gradient did not flow to x̂0 — the VAE decode or PE "
                "encoder likely runs under no_grad. Both must be differentiable. "
                f"(original: {e})"
            ) from e
        g[i] = gi.float()
        if x0.device.type == "cuda":
            torch.cuda.empty_cache()
    return -g, Z.detach(), dEdZ.detach()


def _fd_delta_pe(x0, push, vae, pe, tau, eps_frac, E0):
    """Finite-difference ΔE_pe along a (per-sample) push, applied set-wide.

    Step each sample by ``eps_frac · ‖x̂0_i‖`` along its unit push direction —
    OSCAR moves the whole set at once, so the energy change must be measured on
    the joint step, not one sample. Returns ``E_pe(x̂0 + step) − E0``; **negative
    ⇒ the push genuinely increased PE-space diversity** (lower volume energy)."""
    n = push.flatten(1).norm(dim=1).clamp_min(EPS)
    scale = eps_frac * x0.float().flatten(1).norm(dim=1)  # per-sample [m]
    step = push.float() * (scale / n).view(-1, *([1] * (push.dim() - 1)))
    return _pe_energy(x0.float() + step, vae, pe, tau) - E0


def _compare_step(x0, v, vae, pe, tau, eps_frac):
    """Per-step comparison of the latent push vs the PE push at one σ.

    Three families of read:

    * **Direction** — cos(g_lat, g_pe) raw and after the ⊥v projection (the push
      OSCAR actually applies), plus the fraction of each push killed by ⊥v.
    * **Validity, each in its own space** — ``lat_cen_cos`` = cos(g_lat, latent
      centroid offset) checks the latent push spreads in latent space;
      ``feat_push_cos`` = cos(−dEdZ_i, Z_i−Z̄) checks the PE *energy gradient* is
      a real spread direction in PE space (independent of the noisy pullback).
      (We no longer gate on the PE push vs the *latent* centroid — those live in
      different spaces, so disagreement there is the signal, not a failure.)
    * **Transfer (the actionable read)** — finite-difference ΔE_pe when we step
      the latent along g_lat vs along g_pe. ``dE_pe < 0`` confirms the PE push
      works end-to-end; ``dE_lat / dE_pe`` is how much PE-space diversity the
      *free* latent push buys relative to the decode+PE push."""
    g_lat = _latent_push(x0, tau)
    g_pe, Z, dEdZ = _pe_push(x0, vae, pe, tau)
    m = x0.shape[0]
    xc = x0.float() - x0.float().mean(dim=0, keepdim=True)  # latent centroid offsets
    Zc = Z.float() - Z.float().mean(dim=0, keepdim=True)     # PE-feature centroid offsets
    rows = []
    for i in range(m):
        gl, gp = g_lat[i].float(), g_pe[i].float()
        vi = v[i].float()
        gl_p, gp_p = _perp(gl, vi), _perp(gp, vi)
        nl, npe = gl.norm().clamp_min(EPS), gp.norm().clamp_min(EPS)
        rows.append(
            dict(
                cos=_cos(gl, gp),
                cos_perp=_cos(gl_p, gp_p),
                frac_par_lat=float((gl.norm().pow(2) - gl_p.norm().pow(2)).clamp_min(0).sqrt() / nl),
                frac_par_pe=float((gp.norm().pow(2) - gp_p.norm().pow(2)).clamp_min(0).sqrt() / npe),
                mag_ratio=float(nl / npe),
                lat_cen_cos=_cos(gl, xc[i]),
                feat_push_cos=_cos(-dEdZ[i].float(), Zc[i]),  # PE-space validity
            )
        )
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    # Finite-difference transfer test (the decision metric). Reuses Z for E0.
    E0 = float(set_volume_energy(Z.float(), tau).item())
    out["dE_pe_from_pe"] = _fd_delta_pe(x0, g_pe, vae, pe, tau, eps_frac, E0)
    out["dE_pe_from_lat"] = _fd_delta_pe(x0, g_lat, vae, pe, tau, eps_frac, E0)
    # transfer = how much of the PE push's diversity gain the free latent push
    # recovers. Defined only when the PE push itself lowers E_pe (dE_pe_from_pe<0).
    out["transfer"] = (
        float(out["dE_pe_from_lat"] / out["dE_pe_from_pe"])
        if out["dE_pe_from_pe"] < -EPS
        else float("nan")
    )

    if x0.device.type == "cuda":
        torch.cuda.empty_cache()
    # Sanity diagnostics: how spread are the endpoints, and are the pushes
    # nonzero? rel_spread≈0 ⇒ collapsed set (cosines moot).
    out["rel_spread"] = float(xc.flatten(1).norm(dim=1).mean() / x0.float().flatten(1).norm(dim=1).mean().clamp_min(EPS))
    out["gnorm_lat"] = float(g_lat.flatten(1).norm(dim=1).mean())
    out["gnorm_pe"] = float(g_pe.flatten(1).norm(dim=1).mean())
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", default=DIT)
    ap.add_argument("--vae", default=VAE)
    ap.add_argument("--text_encoder", default=TEXT_ENCODER)
    ap.add_argument("--prompts", nargs="+", default=None, help="Prompts (default: 2 varied built-ins).")
    ap.add_argument("--n_prompts", type=int, default=2, help="How many built-in prompts to use if --prompts unset.")
    ap.add_argument("--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    ap.add_argument("--batch", type=int, default=6, help="Set size m (seeds per prompt).")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument(
        "--probe_sigmas", type=float, nargs="+",
        default=[0.85, 0.65, 0.45, 0.30, 0.15],
        help="Target σ's to probe; nearest rollout step is used.",
    )
    ap.add_argument("--tau", type=float, default=1.0, help="Volume-energy regularizer τ.")
    ap.add_argument("--eps_frac", type=float, default=0.05,
                    help="Finite-difference step as a fraction of ‖x̂0‖ for the ΔE_pe transfer test.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="Block-compile the DiT (compile_blocks). Off by default — see note.")
    ap.add_argument("--compile_mode", type=str, default=None, help="inductor mode, e.g. reduce-overhead.")
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    if args.prompts is None:
        args.prompts = DEFAULT_PROMPTS[: max(1, args.n_prompts)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = args.size
    h_lat, w_lat = H // 8, W // 8

    anima, embeds, neg_embed, vae, pe, sigmas = _setup(args, device)
    sig_np = sigmas.float().cpu().numpy()
    n = len(sig_np) - 1
    # map each target σ to the nearest rollout step.
    step_sigmas = sig_np[:n]
    probe_steps = sorted({int(np.argmin(np.abs(step_sigmas - s))) for s in args.probe_sigmas})
    log.info("Probe steps %s → σ %s", probe_steps, [round(float(step_sigmas[i]), 3) for i in probe_steps])

    # Roll out every prompt first (DiT resident), stashing probe-step (x̂0, v) on
    # CPU; then free the DiT so the decode-grad phase has the GPU to itself.
    captured: list[tuple] = []  # (prompt, step, sigma, x0_cpu, v_cpu)
    for p_idx, (prompt, embed) in enumerate(embeds):
        log.info("\n=== rollout prompt %d/%d: %s", p_idx + 1, len(embeds), prompt[:60])
        probes = _rollout(
            anima, embed, neg_embed, sigmas, args.cfg, args.batch,
            h_lat, w_lat, device, args.seed + 1000 * p_idx, probe_steps,
        )
        for step, (sigma, x0c, vc) in zip(probe_steps, probes):
            captured.append((prompt, step, sigma, x0c, vc))

    n_prompts = len(embeds)
    del anima, embeds, neg_embed
    # torch.compile/dynamo caches hold references to the compiled DiT, so a bare
    # `del` won't return its GPU memory — reset dynamo + gc before reclaiming.
    import torch._dynamo as _dynamo
    _dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info("\nDiT freed — running feature-space comparisons (per-sample VJP) ...")

    per_step: list[dict] = []
    for prompt, step, sigma, x0c, vc in captured:
        x0 = x0c.to(device)
        v = vc.to(device)
        stats = _compare_step(x0, v, vae, pe, args.tau, args.eps_frac)
        stats.update(prompt=prompt, step=step, sigma=float(sigma))
        per_step.append(stats)
        log.info(
            "  σ=%.3f  rel_spread=%.3f  cos⊥v=%+.3f  lat_cen=%+.2f  feat_push=%+.2f  "
            "ΔE_pe(pe/lat)=%+.2e/%+.2e  transfer=%+.2f",
            sigma, stats["rel_spread"], stats["cos_perp"], stats["lat_cen_cos"],
            stats["feat_push_cos"], stats["dE_pe_from_pe"], stats["dE_pe_from_lat"],
            stats["transfer"],
        )

    # band aggregation: forming (σ≤0.45) vs early (σ>0.45).
    def _band(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    forming = [r for r in per_step if r["sigma"] <= FORMING_SIGMA]
    early = [r for r in per_step if r["sigma"] > FORMING_SIGMA]
    cos_perp_form = _band(forming, "cos_perp")
    cos_perp_early = _band(early, "cos_perp")
    cos_form = _band(forming, "cos")
    frac_par_lat_form = _band(forming, "frac_par_lat")
    frac_par_pe_form = _band(forming, "frac_par_pe")
    lat_cen_form = _band(forming, "lat_cen_cos")
    feat_push_form = _band(forming, "feat_push_cos")
    dE_pe_form = _band(forming, "dE_pe_from_pe")
    dE_lat_form = _band(forming, "dE_pe_from_lat")
    # Aggregate transfer from the *summed* ΔE in the band (robust to per-step
    # ΔE_pe≈0 blowups that wreck a mean-of-ratios).
    transfer_form = float(dE_lat_form / dE_pe_form) if dE_pe_form < -EPS else float("nan")

    # Validity gates — each push checked in *its own* space:
    #   latent push must spread in latent space (points off the latent centroid);
    #   the PE energy gradient must be a real spread direction in PE space.
    # The PE *pullback* is then validated end-to-end by dE_pe_form < 0 (the push
    # actually lowers PE-space volume energy). Only then is `transfer` meaningful.
    lat_valid = lat_cen_form >= 0.10
    pe_energy_valid = feat_push_form >= 0.10
    pe_pullback_valid = dE_pe_form < -EPS
    safeguard_kills_latent = frac_par_lat_form >= 0.70

    if np.isnan(cos_perp_form):
        verdict = "INCONCLUSIVE — no probe σ landed in the forming band (σ≤0.45)"
    elif not pe_energy_valid:
        verdict = (f"COLLAPSED — PE features don't spread (feat_push_cos={feat_push_form:+.2f}); "
                   f"endpoints too similar to drive set-volume in PE space. Check rel_spread / endpoint estimator.")
    elif not lat_valid:
        verdict = (f"DEGENERATE — latent push has ~0 centroid alignment "
                   f"(lat_cen={lat_cen_form:+.2f}); latent energy gradient is invalid.")
    elif not pe_pullback_valid:
        verdict = (f"PULLBACK-BROKEN — PE energy is valid (feat_push={feat_push_form:+.2f}) but stepping "
                   f"along the pulled-back push does NOT lower E_pe (ΔE_pe={dE_pe_form:+.2e}); the decode/PE "
                   f"VJP isn't delivering (precision? grad path?). Fix before reading transfer.")
    elif transfer_form >= 0.50:
        verdict = (f"GO — the free latent push recovers {transfer_form:.0%} of the PE push's diversity gain "
                   f"→ skip PE + per-step decode")
    elif transfer_form >= 0.20:
        verdict = (f"PARTIAL — latent push recovers only {transfer_form:.0%} of the PE diversity gain; "
                   f"PE likely load-bearing for full semantic spread")
    else:
        verdict = (f"NO — latent push buys ~none of the PE diversity gain (transfer={transfer_form:+.2f}) "
                   f"→ decode+PE is the price of semantic diversity")
    if safeguard_kills_latent and pe_energy_valid and lat_valid:
        verdict += " | WARN: latent push is mostly ∥v_θ — ⊥v safeguard nukes it"

    run_dir = make_run_dir("oscar", label=args.label)
    # CSV of per-(prompt,step) rows.
    csv = run_dir / "feature_space_alignment.csv"
    cols = ["prompt", "step", "sigma", "cos", "cos_perp", "frac_par_lat", "frac_par_pe",
            "lat_cen_cos", "feat_push_cos", "dE_pe_from_pe", "dE_pe_from_lat", "transfer", "rel_spread"]
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in per_step:
            f.write(",".join(str(r[c]).replace(",", " ") for c in cols) + "\n")
    artifacts = ["feature_space_alignment.csv"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # aggregate over prompts per σ for the curve.
        sig_sorted = sorted({r["sigma"] for r in per_step}, reverse=True)
        def _curve(key):
            return [float(np.mean([r[key] for r in per_step if r["sigma"] == s])) for s in sig_sorted]

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(sig_sorted, _curve("cos"), "o-", label="cos(raw)")
        ax[0].plot(sig_sorted, _curve("cos_perp"), "s-", label="cos(⊥v)")
        ax[0].plot(sig_sorted, _curve("lat_cen_cos"), "^-", c="tab:green", label="latent validity")
        ax[0].plot(sig_sorted, _curve("feat_push_cos"), "v-", c="tab:red", label="PE validity")
        ax[0].axvline(FORMING_SIGMA, ls="--", c="gray", alpha=0.6)
        ax[0].axhline(0.0, ls="-", c="black", alpha=0.3)
        ax[0].set(title="push direction agreement + per-space validity",
                  xlabel="σ (→0 = done)", ylabel="cosine", ylim=(-1, 1))
        ax[0].invert_xaxis()
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        transfer = np.clip(_curve("transfer"), -0.5, 1.5)
        ax[1].plot(sig_sorted, transfer, "D-", c="tab:purple", label="ΔE_lat / ΔE_pe")
        ax[1].axvline(FORMING_SIGMA, ls="--", c="gray", alpha=0.6)
        ax[1].axhline(0.5, ls=":", c="green", alpha=0.6)  # GO threshold
        ax[1].axhline(0.0, ls="-", c="black", alpha=0.3)
        ax[1].set(title="transfer: free latent push's share of PE diversity gain",
                  xlabel="σ (→0 = done)", ylabel="transfer (1 = matches PE push)", ylim=(-0.5, 1.5))
        ax[1].invert_xaxis()
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "alignment.png", dpi=110)
        artifacts.append("alignment.png")
    except Exception as e:  # noqa: BLE001
        log.info("  (plot skipped: %s)", e)

    metrics = {
        "n_prompts": n_prompts,
        "batch_m": args.batch,
        "size_hw": [H, W],
        "cfg": args.cfg,
        "tau": args.tau,
        "probe_sigmas": [round(float(step_sigmas[i]), 4) for i in probe_steps],
        "eps_frac": args.eps_frac,
        "transfer_forming": transfer_form,   # headline: free latent push's share of PE diversity gain
        "dE_pe_from_pe_forming": dE_pe_form,  # PE push's own ΔE_pe (<0 ⇒ pullback works)
        "dE_pe_from_lat_forming": dE_lat_form,
        "feat_push_cos_forming": feat_push_form,   # PE-space validity of the energy gradient
        "lat_centroid_cos_forming": lat_cen_form,  # latent-space validity
        "cos_perp_forming": cos_perp_form,   # secondary: ⊥v direction agreement, σ≤0.45
        "cos_perp_early": cos_perp_early,
        "cos_raw_forming": cos_form,
        "frac_par_lat_forming": frac_par_lat_form,
        "frac_par_pe_forming": frac_par_pe_form,
        "verdict": verdict,
    }
    write_result(run_dir, script=__file__, args=args, metrics=metrics,
                 label=args.label, artifacts=artifacts, device=device)

    log.info("\n=== OSCAR feature-space probe ===")
    log.info("  transfer forming(σ≤0.45): %+.2f  (ΔE_pe lat/pe = %+.2e / %+.2e)",
             transfer_form, dE_lat_form, dE_pe_form)
    log.info("  validity forming  latent_cen=%+.2f  PE feat_push=%+.2f  (both >0.10 ⇒ valid)",
             lat_cen_form, feat_push_form)
    log.info("  cos⊥v forming: %+.3f  early: %+.3f   frac∥v(lat/pe): %.2f / %.2f",
             cos_perp_form, cos_perp_early, frac_par_lat_form, frac_par_pe_form)
    log.info("  → %s", run_dir)
    log.info("  Phase-0 read: %s", verdict)


if __name__ == "__main__":
    main()
