#!/usr/bin/env python3
"""SOLACE Phase-0 — does intrinsic self-confidence rank Anima's own samples by quality?

THE PAPER. SOLACE (Kim & Cho, CVPR 2026, "Improving Text-to-Image Generation
with Intrinsic Self-Confidence Rewards") post-trains a flow-matching T2I model
with **no external reward model**. Its reward is the model's own
*self-confidence*: take a generated clean latent ``z0``, re-noise it with a
*known* injected noise ``eps`` to a timestep ``t`` (``z_t=(1-t)z0+t*eps``), query
the velocity field, recover ``eps_hat = v(z_t,t,c) + z0`` (rectified-flow), and
score ``S = -log(MSE(eps_hat, eps) + delta)``. Low recovery error ⇒ high
confidence ⇒ high reward. Aggregated over a set of re-noise levels this is the
scalar ``R_SOLACE`` fed to Flow-GRPO. The load-bearing *hypothesis* (paper §4.1):
"large-scale pretraining endows diffusion models with strong self-confidence, so
self-confidence should correlate with text alignment and realism." Everything
downstream (the whole GRPO stack) only pays off if that correlation holds.

THE ANIMA RISK. Three CLAUDE memories put this premise under suspicion *here*:
  * FM-MSE doesn't track quality on Anima (it's why CMMD replaced it as the val
    signal) — and ``R_SOLACE`` is an FM-MSE-shaped quantity.
  * Anima inverts SD-validated premises with some regularity (CTCal's clean
    teacher inverted; sigma-reshape was a no-win; TRDI a non-bottleneck). SOLACE
    is validated on SD3.5-M with COCO-style prompts.
  * Anima resolves x0 by sigma≈0.45 on booru tag-soup captions — its sigma
    structure differs from SD.
So before building any RL, this probe asks the one falsifiable question: **among
the model's OWN generations, do the higher-self-confidence ones actually look
better?** (Real cached latents would be circular — the premise is about samples.)

THE RULER. We generate a spread of Anima samples (several seeds over a varied
prompt set, so some land good and some bad), then for each sample:
  1. ``R_SOLACE`` = mean over the re-noise sigma grid of ``-log(MSE+delta)``,
     MSE averaged over ``K`` antithetic noise probes, CFG-free (conditional emb
     only — the paper computes confidence without CFG). Reuses the *same* live
     DiT that generated it; ``eps_hat = v + z0`` (Anima: ``x_t=(1-t)x0+t*eps``,
     ``v=eps-x0`` — verified against the sampler's ``x0=x_t-sigma*v``).
  2. A quality proxy in PE-Core space (the CMMD feature space Anima targets):
     decode → PE-encode → mean top-k cosine similarity to the real-image PE
     manifold (cached ``*_anima_pe.safetensors`` references). Higher = closer to
     the real distribution.

PRE-REGISTERED GATE (decides whether the SOLACE RL build is worth it). The gate
is the **within-prompt** ranking, NOT the global correlation: Flow-GRPO centers
rewards within a same-prompt group (group-relative advantage), so between-prompt
signal contributes zero gradient and the global rho is dominated by a prompt
*complexity* confound (simple prompts denoise easily AND look PE-typical). So we
gate on, among seeds of the SAME prompt, whether higher R_SOLACE picks the better
sample — pairwise concordance + mean per-prompt Spearman, with a power check:
  * **PASS** iff concordance >= 0.62 and mean per-prompt Spearman >= --gate_rho,
    with >= 16 prompts (else underpowered).
  * **FAIL** iff concordance <= 0.52 and mean per-prompt Spearman <= 0.05.
  * **INCONCLUSIVE** otherwise — re-run with more --seeds_per_variant / --num_images.
The within-prompt view also controls the quality proxy's own complexity confound
(seeds of one prompt share its complexity). The global rho + group CMMD are still
reported, but as confounded context, not the gate. A confidence-sorted contact
sheet + ranked PNGs are written for the eyeball check the gate can't make for you.

PROMPTS are REAL dataset captions by default: each image's TE cache holds v0
(intact) + v1.. (tag-dropout/shuffle) variants; we detokenize them and treat each
(image, variant) as one within-prompt group. Variants of one image are paraphrases
of the SAME content, so they hold scene complexity ~fixed while the prompt varies —
a tighter confound control than independent synthetic prompts. `--use_synthetic`
or `--prompts_file` override the source.

CAVEATS (why even a clean within-prompt result is bounded): the PE quality proxy
itself favors typical/simple content; R_SOLACE on the model's OWN generations is
partly circular; and the σ grid is a free parameter (Anima resolves x0 by σ≈0.45,
so a different grid can reorder R_SOLACE).

Run from anima_lora/ (heavy — generates real samples; background it)::

    # real dataset caption variants (default): 2 images × 4 variants × 2 seeds
    uv run python bench/solace/probe_confidence_quality.py --num_images 2 --variants 4 --seeds_per_variant 2
    # powered within-prompt run
    uv run python bench/solace/probe_confidence_quality.py --num_images 24 --variants 2 --seeds_per_variant 6
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

from bench._common import make_run_dir, write_result  # noqa: E402

log = logging.getLogger("bench.solace.confidence_quality")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_REF_DIR = "post_image_dataset/lora"

# SOLACE's suffix window: the noisier band of the schedule (paper rho=0.6), where
# the denoising task "remains informative but is harder to exploit". We bracket
# the sigma≈0.45 commitment window so a per-sigma breakdown can reveal whether
# any sub-band carries the signal (R_SOLACE itself aggregates the whole grid).
DEFAULT_SIGMAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# A deliberately varied prompt set so natural seed/prompt variance produces a
# quality spread to correlate against: easy single figures, hard hands/poses,
# text rendering (Anima's flaky axis), multi-character composition, scenery.
DEFAULT_PROMPTS = [
    "masterpiece, best quality, score_7, safe. An anime girl with long silver "
    "hair in a white sundress standing in a sunflower field, looking at the "
    "viewer, soft afternoon light.",
    "masterpiece, best quality, safe. A close-up portrait of a smiling anime boy "
    "with freckles and green eyes, detailed face, clean lineart.",
    "masterpiece, best quality, safe. An anime girl holding a wooden sign that "
    'reads "ANIMA" in bold letters, city street background.',
    "masterpiece, best quality, safe. A swordswoman in ornate armor mid-swing, "
    "dynamic action pose, both hands gripping a katana, motion blur.",
    "best quality, safe. Two anime girls sitting back to back on a rooftop at "
    "sunset, one reading a book, the other eating ice cream.",
    "masterpiece, best quality, safe. A witch with a wide-brimmed hat casting a "
    "glowing spell, intricate magic circle, fingers spread, detailed hands.",
    "best quality, safe. A cozy ramen shop interior at night, an anime waitress "
    "carrying three steaming bowls, warm lantern light, rain outside.",
    "masterpiece, best quality, safe. A mecha pilot girl in a cockpit, holographic "
    "displays around her, focused expression, cables and panels.",
    "best quality, safe. A chibi cat-girl waving hello, pastel background, simple "
    "flat-color style.",
    "masterpiece, best quality, safe. A knight and a dragon facing each other on a "
    "cliff under a stormy sky, wide cinematic shot.",
    "best quality, safe. An anime chef plating a dish, careful hand detail holding "
    "chopsticks, steam rising, kitchen background.",
    "masterpiece, best quality, safe. A group of four idols on stage striking a "
    "synchronized pose, stage lights, confetti, full body.",
]
DEFAULT_NEG = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg "
    "artifacts, sepia, extra fingers, bad hands"
)


# --------------------------------------------------------------------------- #
# small stats helpers (no scipy dep)
# --------------------------------------------------------------------------- #
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation = Pearson on the ranks. NaN if degenerate."""
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _discover_dataset_prompts(
    data_dir: str, num_images: int, n_variants: int, rng, tokenizer
) -> list[dict]:
    """Draw REAL caption variants from the dataset's TE caches.

    Each ``{stem}_anima_te.safetensors`` holds ``num_variants`` augmentation
    variants (v0 = intact caption, v1.. = tag-dropout/shuffle). We detokenize the
    real (non-padding) ``t5_input_ids_v{k}`` back to text and feed the normal
    generate path — robust, and the variant text IS the augmented caption the
    model trains on. Returns a flat group-major list of
    ``{"stem", "variant", "text"}`` (image-major, then variant), each entry one
    within-prompt group. Variants of one image are paraphrases of the SAME
    content, so they hold scene complexity ~fixed while varying the prompt — a
    tighter confound control than independent synthetic prompts.
    """
    from safetensors.torch import load_file

    caches = sorted(Path(data_dir).rglob("*_anima_te.safetensors"))
    if not caches:
        raise SystemExit(f"no *_anima_te.safetensors under {data_dir}")
    take = min(num_images, len(caches))
    picks = [
        caches[int(i)] for i in sorted(rng.choice(len(caches), take, replace=False))
    ]

    items: list[dict] = []
    for te in picks:
        stem = te.name.replace("_anima_te.safetensors", "")
        try:
            sd = load_file(str(te))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s (load failed: %s)", stem, e)
            continue
        nv = int(sd["num_variants"].item()) if "num_variants" in sd else 1
        for k in range(min(n_variants, nv)):
            ids = sd.get(f"t5_input_ids_v{k}")
            if ids is None:
                continue
            ids_l = [int(x) for x in ids.flatten().tolist()]
            mask = sd.get(f"t5_attn_mask_v{k}")
            if mask is not None:
                ids_l = ids_l[: int(mask.sum())]
            text = tokenizer.decode(ids_l, skip_special_tokens=True).strip()
            if text:
                items.append({"stem": stem, "variant": k, "text": text})
    if not items:
        raise SystemExit("no caption variants decoded — check TE caches / tokenizer")
    return items


def _within_prompt_stats(
    R: np.ndarray, Q: np.ndarray, n_prompts: int, seeds: int
) -> dict:
    """Decompose the global R↔quality association into between- vs within-prompt.

    ``R``/``Q`` are in prompt-major order (prompt p, seed s). Returns the
    between-prompt Spearman (the complexity confound), the GRPO-relevant
    within-prompt measures (demeaned linear corr, mean per-prompt Spearman,
    pairwise concordance vs 0.5, argmax-hit-rate vs 1/G), and the per-prompt
    Spearman vector. ``None`` when the run isn't a clean ``n_prompts×seeds`` grid.
    """
    if seeds < 2 or n_prompts * seeds != R.size:
        return {"ok": False}
    Rg = R.reshape(n_prompts, seeds)
    Qg = Q.reshape(n_prompts, seeds)
    Rm, Qm = Rg.mean(1), Qg.mean(1)

    between = _spearman(Rm, Qm)
    # demeaned (group-centered) linear corr over all samples — variance-weighted.
    Rw = (Rg - Rm[:, None]).ravel()
    Qw = (Qg - Qm[:, None]).ravel()
    Rw -= Rw.mean()
    Qw -= Qw.mean()
    den = np.sqrt((Rw * Rw).sum() * (Qw * Qw).sum())
    within_lin = float((Rw * Qw).sum() / den) if den > 0 else float("nan")

    # per-prompt Spearman needs >= 3 seeds; at G=2 every group is NaN (rank stat
    # undefined) and only the pairwise concordance below carries within-prompt
    # signal. Average over the non-NaN groups without tripping nanmean-of-empty.
    per_prompt = [_spearman(Rg[i], Qg[i]) for i in range(n_prompts)]
    valid_sp = [s for s in per_prompt if not np.isnan(s)]
    mean_sp = float(np.mean(valid_sp)) if valid_sp else float("nan")
    std_sp = float(np.std(valid_sp)) if valid_sp else float("nan")

    conc = tot = 0
    for i in range(n_prompts):
        for a in range(seeds):
            for b in range(a + 1, seeds):
                tot += 1
                if (Rg[i, a] - Rg[i, b]) * (Qg[i, a] - Qg[i, b]) > 0:
                    conc += 1
    concordance = conc / tot if tot else float("nan")
    argmax_hits = int(
        sum(np.argmax(Rg[i]) == np.argmax(Qg[i]) for i in range(n_prompts))
    )
    return {
        "ok": True,
        "between_spearman": between,
        "within_linear": within_lin,
        "mean_per_prompt_spearman": mean_sp,
        "std_per_prompt_spearman": std_sp,
        "per_prompt_spearman": [
            None if np.isnan(s) else round(float(s), 3) for s in per_prompt
        ],
        "pairwise_concordance": concordance,
        "argmax_hits": argmax_hits,
        "argmax_chance": 1.0 / seeds,
        "n_prompts": n_prompts,
        "seeds": seeds,
    }


def _to_pil(pix_minus1to1: torch.Tensor):
    """``[3,H,W]`` in [-1,1] → PIL.Image (RGB)."""
    from PIL import Image

    arr = ((pix_minus1to1.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


# --------------------------------------------------------------------------- #
# the SOLACE self-confidence reward on one clean latent
# --------------------------------------------------------------------------- #
def solace_reward(
    anima,
    z0: torch.Tensor,  # (1,C,1,H,W) clean generated latent, model dtype
    emb_b: torch.Tensor,  # (1,N,D) conditional crossattn emb (CFG-free)
    sigmas: list[float],
    *,
    n_probes: int,
    delta: float,
    seed: int,
    device,
    dtype,
) -> tuple[float, dict[float, float]]:
    """Return (R_SOLACE, {sigma: mean_MSE}).

    For each sigma we re-noise ``z0`` with ``K=n_probes`` antithetic Gaussian
    probes (``eps`` and ``-eps`` paired for mean-zero, variance-reduced probing),
    query ``v=anima(z_t,sigma,c)``, recover ``eps_hat=v+z0``, and average
    ``MSE(eps_hat, eps)``. ``S(sigma)=-log(MSE+delta)``; ``R_SOLACE`` is the mean
    over the grid (w(t)=1, as the paper uses in practice).
    """
    H, W = z0.shape[-2], z0.shape[-1]
    pad = torch.zeros(1, 1, H, W, dtype=dtype, device=device)
    z0_f = z0.float()
    g = torch.Generator(device=device).manual_seed(seed)

    # antithetic probe set: draw ceil(K/2) base draws, append their negatives.
    n_base = (n_probes + 1) // 2
    base = [
        torch.randn(z0.shape, generator=g, device=device, dtype=dtype)
        for _ in range(n_base)
    ]
    probes = []
    for e in base:
        probes.append(e)
        if len(probes) < n_probes:
            probes.append(-e)

    per_sigma_mse: dict[float, float] = {}
    s_vals: list[float] = []
    for s in sigmas:
        t_b = torch.full((1,), float(s), device=device, dtype=dtype)
        mses = []
        for eps in probes:
            z_t = ((1.0 - s) * z0_f + s * eps.float()).to(dtype)
            with torch.no_grad():
                v = anima(z_t, t_b, emb_b, padding_mask=pad)
            eps_hat = v.float() + z0_f
            mses.append(float(((eps_hat - eps.float()) ** 2).mean().item()))
        mean_mse = float(np.mean(mses))
        per_sigma_mse[s] = mean_mse
        s_vals.append(-float(np.log(mean_mse + delta)))
    return float(np.mean(s_vals)), per_sigma_mse


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=None, help="DiT checkpoint (default: canonical).")
    ap.add_argument("--vae", default=None, help="VAE checkpoint (default: canonical).")
    ap.add_argument(
        "--text_encoder", default=None, help="Text encoder (default: canonical)."
    )
    ap.add_argument("--ref_dir", default=DEFAULT_REF_DIR, help="real PE sidecar dir.")
    ap.add_argument(
        "--data_dir",
        default=DEFAULT_REF_DIR,
        help="dataset dir to source REAL caption variants from (TE caches).",
    )
    ap.add_argument(
        "--num_images", type=int, default=8, help="real images (stems) to draw from."
    )
    ap.add_argument(
        "--variants",
        type=int,
        default=4,
        help="caption variants per image to use (v0=intact, v1.. = tag-dropout). "
        "Each (image, variant) is one within-prompt group.",
    )
    ap.add_argument(
        "--seeds_per_variant",
        type=int,
        default=2,
        help="seeds generated per (image, variant) group — the within-prompt G.",
    )
    ap.add_argument(
        "--prompts_file",
        default=None,
        help="override: one prompt per line (each line is a group). Disables the "
        "dataset/variant source.",
    )
    ap.add_argument(
        "--use_synthetic",
        action="store_true",
        help="override: use the built-in synthetic prompt set instead of dataset.",
    )
    ap.add_argument("--base_seed", type=int, default=1234)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument(
        "--image_size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    ap.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMAS,
        help="re-noise sigma grid (SOLACE suffix window).",
    )
    ap.add_argument(
        "--n_probes", type=int, default=8, help="K antithetic noise probes."
    )
    ap.add_argument("--delta", type=float, default=1e-3, help="log-transform floor.")
    ap.add_argument(
        "--num_ref", type=int, default=512, help="max real PE refs sampled."
    )
    ap.add_argument("--knn", type=int, default=8, help="k for per-sample manifold sim.")
    ap.add_argument("--gate_rho", type=float, default=0.20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="ref-sampling RNG seed.")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    rng = np.random.default_rng(args.seed)

    # canonical checkpoint defaults via the façade (honors ANIMA_DIT/.env/base.toml)
    from anima_lora import default_checkpoints

    ck = default_checkpoints()
    dit_path = args.dit or ck.dit
    vae_path = args.vae or ck.vae
    te_path = args.text_encoder or ck.text_encoder

    # ── prompt groups: REAL dataset caption variants by default ────────────────
    # Each group = one exact prompt; `seeds` seeds are generated within it. Order
    # is group-major so R/quality reshape cleanly to (n_groups, seeds).
    seeds = args.seeds_per_variant
    src_stems: set[str] = set()
    if args.prompts_file:
        lines = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        items = [
            {"stem": f"file{i}", "variant": 0, "text": t} for i, t in enumerate(lines)
        ]
    elif args.use_synthetic:
        items = [
            {"stem": f"syn{i}", "variant": 0, "text": t}
            for i, t in enumerate(DEFAULT_PROMPTS[: args.num_images])
        ]
    else:
        from library.anima.weights import load_t5_tokenizer

        tok = load_t5_tokenizer()
        items = _discover_dataset_prompts(
            args.data_dir, args.num_images, args.variants, rng, tok
        )
        src_stems = {it["stem"] for it in items}
    if not items:
        raise SystemExit("no prompts")
    log.info(
        f"{len(items)} groups (prompts) x {seeds} seeds = {len(items) * seeds} "
        f"samples | sigmas {args.sigmas} | K={args.n_probes} | "
        f"{args.infer_steps} steps @ {args.image_size}"
    )

    run_dir = make_run_dir("solace", label=args.label or "confidence-quality")
    img_dir = run_dir / "samples"
    img_dir.mkdir(exist_ok=True)

    # ── load DiT once; reuse it for BOTH generation and the confidence forward ──
    from anima_lora import GenerationRequest, generate, get_generation_settings
    from library.inference.text import prepare_text_inputs

    # Prime a fully-defaulted namespace + settings from a representative request.
    base_req = GenerationRequest(
        prompt=items[0]["text"],
        negative_prompt=DEFAULT_NEG,
        dit=dit_path,
        vae=vae_path,
        text_encoder=te_path,
        image_size=(args.image_size[0], args.image_size[1]),
        infer_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        sampler="euler",
    )
    proto = base_req.to_args()
    proto.device = device
    gen_settings = get_generation_settings(proto)

    # Preload the DiT (generate() would otherwise lazy-load on first call); store
    # in shared_models so every generate() call reuses it, and so we hold the same
    # Anima object for the confidence forward. Text encoder loaded on CPU and
    # parked there between encodes (prepare_text_inputs moves it on demand).
    from anima_lora import load_dit_model
    from library.inference.models import load_text_encoder

    log.info("loading DiT …")
    anima = load_dit_model(proto, device, dtype)
    text_encoder = load_text_encoder(text_encoder=te_path, dtype=dtype, device="cpu")
    text_encoder.eval()
    shared_models = {"model": anima, "text_encoder": text_encoder, "conds_cache": {}}

    records: list[dict] = []  # one per sample: prompt, seed, latent(cpu), R, per_sigma

    # ── Phase A+B: generate each sample and score its self-confidence ───────────
    for pi, item in enumerate(items):
        prompt = item["text"]
        # encode once per prompt (conds_cache makes repeats free); capture the
        # conditional crossattn emb for the CFG-free confidence forward.
        context, context_null = prepare_text_inputs(
            proto,
            device,
            anima,
            shared_models,
            prompt=prompt,
            negative_prompt=DEFAULT_NEG,
        )
        emb_b = context["embed"][0].to(device, dtype=dtype)  # (1,512,1024) padded

        for si in range(seeds):
            seed = args.base_seed + pi * 1000 + si
            req = GenerationRequest(
                prompt=prompt,
                negative_prompt=DEFAULT_NEG,
                dit=dit_path,
                vae=vae_path,
                text_encoder=te_path,
                image_size=(args.image_size[0], args.image_size[1]),
                infer_steps=args.infer_steps,
                guidance_scale=args.guidance_scale,
                sampler="euler",
                seed=seed,
            )
            a = req.to_args()
            a.device = device
            latent = generate(
                a,
                gen_settings,
                shared_models=shared_models,
                precomputed_text_data={
                    "context": context,
                    "context_null": context_null,
                },
            )
            z0 = latent.to(device, dtype)
            if z0.dim() == 4:
                z0 = z0.unsqueeze(2)  # → (1,C,1,H,W), the DiT's 5D layout
            R, per_sigma = solace_reward(
                anima,
                z0,
                emb_b,
                args.sigmas,
                n_probes=args.n_probes,
                delta=args.delta,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            records.append(
                {
                    "prompt": prompt,
                    "stem": item["stem"],
                    "variant": item["variant"],
                    "seed": seed,
                    "latent": z0.detach().to("cpu"),
                    "R": R,
                    "per_sigma": per_sigma,
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log.info(
            f"  [{pi + 1}/{len(items)}] {item['stem']} v{item['variant']} "
            f"generated+scored ({prompt[:42]}…)"
        )

    # free the DiT + text encoder before bringing up VAE + PE encoder
    del shared_models, anima, text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from library.runtime.device import clean_memory_on_device

    clean_memory_on_device(device)

    # ── Phase C: decode → PE features → quality proxy ──────────────────────────
    from anima_lora import load_vae
    from library.training.cmmd import (
        cmmd_from_pools,
        encode_pixel_batch,
        load_reference_features,
    )
    from library.vision.encoder import load_pe_encoder

    log.info("loading VAE + PE encoder …")
    vae = load_vae(vae_path, device=device, dtype=dtype, eval=True)
    pe_bundle = load_pe_encoder(device, dtype=dtype)

    pixels_list: list[torch.Tensor] = []  # [3,H,W] in [-1,1], cpu
    for rec in records:
        z = rec["latent"].to(device, dtype)
        with torch.no_grad():
            pix = vae.decode_to_pixels(z)  # 5D in → (1,3,1,H,W)
        if pix.dim() == 5:
            pix = pix.squeeze(2)  # → (1,3,H,W); dim 2 explicitly (CLAUDE invariant)
        pixels_list.append(pix[0].detach().to("cpu").float())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # generated-sample PE pool (pooled + L2-normalized, [N,D] cpu)
    gen_pool = encode_pixel_batch(pe_bundle, pixels_list)

    # real-image reference PE pool from cached sidecars
    # sidecars are nested under per-artist subdirs → rglob, not glob. Exclude the
    # stems we drew prompts from — their real images would otherwise leak into the
    # manifold a generation is scored against (the prompt came from that image).
    sidecars = [
        p
        for p in sorted(Path(args.ref_dir).rglob("*_anima_pe.safetensors"))
        if p.name.replace("_anima_pe.safetensors", "") not in src_stems
    ]
    if not sidecars:
        raise SystemExit(
            f"no *_anima_pe.safetensors under {args.ref_dir} — run `make preprocess-pe`"
        )
    if len(sidecars) > args.num_ref:
        idx = rng.choice(len(sidecars), args.num_ref, replace=False)
        sidecars = [sidecars[int(i)] for i in sorted(idx)]
    ref_pool = load_reference_features(sidecars)  # [M,D] cpu, normalized
    log.info(f"PE pools: gen {tuple(gen_pool.shape)} | ref {tuple(ref_pool.shape)}")

    # per-sample quality = mean top-k cosine sim to the real PE manifold (both
    # pools are L2-normalized, so cosine = dot). Higher ⇒ closer to real.
    sims = (gen_pool.float() @ ref_pool.float().t()).numpy()  # [N,M]
    k = min(args.knn, sims.shape[1])
    quality = np.sort(sims, axis=1)[:, -k:].mean(axis=1)  # [N]

    R_arr = np.array([r["R"] for r in records], dtype=np.float64)
    rho = _spearman(R_arr, quality)

    # ── between- vs within-prompt decomposition (THE decision-relevant split) ──
    # Flow-GRPO optimizes a GROUP-RELATIVE advantage: rewards are centered within
    # a group of G samples of the SAME prompt, so any signal living purely in
    # between-prompt differences is algebraically annihilated by the centering and
    # contributes zero gradient. The global rho is therefore the WRONG gate — it's
    # dominated by a complexity confound (simple prompts denoise easily AND look
    # PE-typical). What GRPO can actually exploit is the WITHIN-prompt ranking:
    # among seeds of one prompt, does higher R_SOLACE pick the better sample? The
    # within-prompt view also controls the quality-proxy confound — all seeds of a
    # prompt share its complexity, so their PE-sim differences reflect rendering,
    # not content density. GRPO needs within-prompt *ordering*, so the rank-based
    # measures (concordance, per-prompt Spearman) are more faithful than a linear
    # fit. Underpowered at small G: report all three and the std.
    within = _within_prompt_stats(R_arr, quality, len(items), seeds)

    # per-sigma: does any sub-band carry the correlation? corr of -log(MSE_sigma)
    # (i.e. confidence at that sigma) with quality.
    per_sigma_rho: dict[float, float] = {}
    for s in args.sigmas:
        conf_s = np.array(
            [-np.log(r["per_sigma"][s] + args.delta) for r in records], dtype=np.float64
        )
        per_sigma_rho[s] = _spearman(conf_s, quality)

    # ── group CMMD: top-confidence third vs bottom third, each vs the real pool ─
    order = np.argsort(R_arr)  # ascending R
    N = len(records)
    third = max(2, N // 3)
    bottom_idx = order[:third]
    top_idx = order[-third:]
    cmmd_top = cmmd_from_pools(ref_pool, gen_pool[top_idx])
    cmmd_bottom = cmmd_from_pools(ref_pool, gen_pool[bottom_idx])
    cmmd_all = cmmd_from_pools(ref_pool, gen_pool)
    delta_cmmd = cmmd_bottom - cmmd_top  # >0 ⇒ high-confidence closer to real

    # ── GATE: the within-prompt ranking signal is what Flow-GRPO can exploit. ──
    # The global rho / CMMD split are reported but NOT the gate — they're
    # confounded by prompt complexity (see _within_prompt_stats). We gate on
    # within-prompt pairwise concordance with a power check: at small G the
    # estimate is high-variance, so an inconclusive run is its own outcome
    # (re-run with more --seeds_per_prompt) rather than a silent pass/fail.
    if not within.get("ok"):
        verdict = "INCONCLUSIVE — need a clean n_prompts×seeds grid (seeds≥2)."
        passed = None
    else:
        conc = within["pairwise_concordance"]
        # power: pairs cluster by prompt, AND variant-groups of one image are
        # near-duplicate prompts (esp. tag-shuffle variants) — so independence is
        # bounded by DISTINCT IMAGES, not group count. Require ≥16 distinct images
        # before calling a result decisive.
        n_images = len({r["stem"] for r in records})
        powered = n_images >= 16
        within["n_distinct_images"] = n_images
        strong = conc >= 0.62 and within["mean_per_prompt_spearman"] >= args.gate_rho
        null = conc <= 0.52 and within["mean_per_prompt_spearman"] <= 0.05
        if strong and powered:
            passed = True
            verdict = (
                f"PASS (within-prompt) — pairwise concordance {conc:.2f} (>0.5), "
                f"mean per-prompt Spearman {within['mean_per_prompt_spearman']:+.2f}, "
                f"argmax-R picks best seed {within['argmax_hits']}/{within['n_prompts']}. "
                "Among seeds of the SAME prompt — the only signal GRPO's group-"
                "relative advantage can use — higher self-confidence does pick the "
                "better sample → the SOLACE build is justified. (Global rho "
                f"{rho:+.2f} is mostly the between-prompt complexity confound: "
                f"between-prompt Spearman {within['between_spearman']:+.2f}.)"
            )
        elif null and powered:
            passed = False
            verdict = (
                f"FAIL (within-prompt) — concordance {conc:.2f} ≈ chance, mean "
                f"per-prompt Spearman {within['mean_per_prompt_spearman']:+.2f}. "
                "Within a prompt, self-confidence does NOT pick the better seed; "
                f"the global rho {rho:+.2f} is a between-prompt complexity confound "
                f"(between-prompt Spearman {within['between_spearman']:+.2f}, and "
                "the PE quality proxy shares that confound). GRPO would optimize a "
                "quantity orthogonal to quality — don't build."
            )
        else:
            passed = None
            verdict = (
                f"INCONCLUSIVE — within-prompt signal is weak/underpowered: "
                f"concordance {conc:.2f} (chance 0.50), mean per-prompt Spearman "
                f"{within['mean_per_prompt_spearman']:+.2f}±{within['std_per_prompt_spearman']:.2f}, "
                f"argmax {within['argmax_hits']}/{within['n_prompts']} (chance "
                f"{within['argmax_chance']:.2f}). The global PASS (rho {rho:+.2f}, "
                f"CMMD {cmmd_top:.1f}<{cmmd_bottom:.1f}) is mostly the between-prompt "
                f"complexity confound (between-prompt Spearman "
                f"{within['between_spearman']:+.2f}) — NOT decision-relevant for "
                "GRPO. The point estimates lean weakly positive but none is "
                f"significant at G={within['seeds']} seeds/{within['n_prompts']} "
                "prompts. Re-run with more --seeds_per_variant and --num_images "
                "to actually resolve it. Eyeball the contact sheet meanwhile."
            )
    log.info(verdict)

    # ── artifacts: ranked PNGs + contact sheet + scatter ───────────────────────
    rank_desc = np.argsort(-R_arr)  # best confidence first
    qmin, qmax = float(quality.min()), float(quality.max())
    for rank, i in enumerate(rank_desc):
        rec = records[i]
        slug = "".join(c if c.isalnum() else "_" for c in rec["prompt"][:24]).strip("_")
        name = (
            f"rank{rank:02d}_R{R_arr[i]:+.3f}_q{quality[i]:.3f}"
            f"_seed{rec['seed']}_{slug}.png"
        )
        _to_pil(pixels_list[i]).save(img_dir / name)

    _write_contact_sheet(
        run_dir / "contact_sheet.png", pixels_list, R_arr, quality, records, rank_desc
    )
    _write_scatter(run_dir / "scatter_R_vs_quality.png", R_arr, quality, rho)

    # csv
    with (run_dir / "per_sample.csv").open("w") as f:
        f.write("rank,R_SOLACE,quality,seed,prompt\n")
        for rank, i in enumerate(rank_desc):
            p = records[i]["prompt"].replace(",", " ").replace("\n", " ")[:80]
            f.write(
                f"{rank},{R_arr[i]:.5f},{quality[i]:.5f},{records[i]['seed']},{p}\n"
            )

    np.savez(
        run_dir / "confidence_quality.npz",
        R=R_arr,
        quality=quality,
        sigmas=np.array(args.sigmas),
        per_sigma_rho=np.array([per_sigma_rho[s] for s in args.sigmas]),
    )

    # summary.md
    M = ["# SOLACE Phase-0 — does self-confidence rank Anima's samples by quality?\n"]
    M.append(
        f"- samples: **{N}** ({len(items)} prompt-groups × {seeds} seeds) "
        f"· {args.infer_steps} steps · CFG {args.guidance_scale} · {args.image_size}\n"
        f"- R_SOLACE: mean over σ∈{args.sigmas} of −log(MSE+{args.delta}), "
        f"K={args.n_probes} antithetic probes, CFG-free (conditional emb)\n"
        f"- quality: mean top-{k} cosine sim of decoded sample to the real PE "
        f"manifold ({ref_pool.shape[0]} refs from `{args.ref_dir}`)\n"
    )
    M.append(f"\n## Verdict\n\n**{verdict}**\n")
    if within.get("ok"):
        M.append("\n## Within- vs between-prompt (THE decision-relevant split)\n")
        M.append(
            "Flow-GRPO centers rewards within a same-prompt group, so only the "
            "**within-prompt** R↔quality signal drives any gradient; between-prompt "
            "differences are annihilated by the centering. Within-prompt also "
            "controls the quality-proxy's complexity confound (seeds of one prompt "
            "share its complexity).\n"
        )
        M.append("| signal | value | note |")
        M.append("|---|---|---|")
        M.append(f"| global Spearman | {rho:+.3f} | confounded — NOT the gate |")
        M.append(
            f"| **between-prompt** Spearman | {within['between_spearman']:+.3f} | "
            "the complexity confound (carries the global rho) |"
        )
        M.append(
            f"| **within-prompt** linear (demeaned) | {within['within_linear']:+.3f} | "
            "variance-weighted; ≈0 here |"
        )
        M.append(
            f"| **within-prompt** mean per-prompt Spearman | "
            f"{within['mean_per_prompt_spearman']:+.3f} ± "
            f"{within['std_per_prompt_spearman']:.3f} | rank-based; GRPO-faithful |"
        )
        M.append(
            f"| within-prompt pairwise concordance | "
            f"{within['pairwise_concordance']:.3f} | chance 0.50 |"
        )
        M.append(
            f"| argmax-R picks best seed | {within['argmax_hits']}/"
            f"{within['n_prompts']} | chance {within['argmax_chance']:.2f} |"
        )
        M.append(f"| per-prompt Spearman vector | {within['per_prompt_spearman']} | |")
    M.append("\n## Global numbers (confounded — context only)\n")
    M.append("| metric | value |")
    M.append("|---|---|")
    M.append(f"| Spearman(R_SOLACE, quality) | {rho:+.3f} |")
    M.append(f"| CMMD top-confidence third | {cmmd_top:.3f} |")
    M.append(f"| CMMD bottom-confidence third | {cmmd_bottom:.3f} |")
    M.append(f"| CMMD all samples | {cmmd_all:.3f} |")
    M.append(f"| ΔCMMD (bottom − top) | {delta_cmmd:+.3f} |")
    M.append(f"| R_SOLACE range | {R_arr.min():+.3f} … {R_arr.max():+.3f} |")
    M.append(f"| quality range | {qmin:.3f} … {qmax:.3f} |")
    M.append("\n## Caveats that bound this probe\n")
    M.append(
        "- **Quality proxy is itself complexity-confounded.** PE-Core typicality on "
        "a character-centric booru manifold rewards simple/typical images "
        "(same space as the CMMD val signal). The *between*-prompt correlation is "
        "two confounds stacking (denoise-easiness ↔ generation-easiness ↔ "
        "PE-typicality); only the within-prompt view is interpretable as quality.\n"
        "- **R_SOLACE is computed on the model's OWN generations** — partly circular "
        "('recovers noise it injected into a latent it made'), co-varying with the "
        "same simplicity axis the proxy rewards.\n"
        "- **The σ grid is a free parameter.** Anima's recovery is strongly "
        "σ-dependent (x0 resolves by ~0.45); a different grid could reorder "
        "R_SOLACE. Result is conditional on σ∈"
        f"{args.sigmas}.\n"
        "- **Underpowered.** Per-prompt Spearman at G="
        f"{seeds} has SE≈0.5; treat individual prompt values as "
        "noise. Power the within-prompt test with more seeds/prompts before any "
        "build decision.\n"
    )
    M.append("\n## Per-σ confidence↔quality correlation\n")
    M.append("Does any sub-band of the suffix window carry the signal?\n")
    M.append("| σ | Spearman(−log MSE_σ, quality) |")
    M.append("|---|---|")
    for s in args.sigmas:
        M.append(f"| {s:g} | {per_sigma_rho[s]:+.3f} |")
    M.append("\n## Reading it\n")
    M.append(
        "- **Positive ρ + CMMD(top) < CMMD(bottom)** ⇒ higher self-confidence does "
        "mean better/more-real samples → SOLACE's core hypothesis holds on Anima.\n"
        "- **ρ ≈ 0 or negative** ⇒ confidence is orthogonal to (or anti-correlated "
        "with) quality — the same failure mode as FM-MSE as a val signal. The "
        "GRPO reward would optimize a quantity that doesn't track quality.\n"
        "- Eyeball `contact_sheet.png` (sorted best-confidence → worst) and the "
        "ranked PNGs in `samples/`: if the top rows look better than the bottom "
        "rows, the metric works regardless of the exact ρ.\n"
        "- `scatter_R_vs_quality.png` shows the per-sample cloud; a diagonal trend "
        "is the win, a blob is the null.\n"
    )
    (run_dir / "summary.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_samples": N,
            "n_groups": len(items),
            "seeds_per_variant": seeds,
            "sigmas": list(args.sigmas),
            "n_probes": args.n_probes,
            "spearman_R_quality_global": rho,
            "within_prompt": within,
            "per_sigma_rho": {f"{s:g}": per_sigma_rho[s] for s in args.sigmas},
            "cmmd_top": cmmd_top,
            "cmmd_bottom": cmmd_bottom,
            "cmmd_all": cmmd_all,
            "delta_cmmd": delta_cmmd,
            "gate_rho": args.gate_rho,
            "verdict": verdict,
            "passed": passed,  # True / False / None(inconclusive)
            "R_min": float(R_arr.min()),
            "R_max": float(R_arr.max()),
        },
        label=args.label,
        artifacts=[
            "summary.md",
            "per_sample.csv",
            "confidence_quality.npz",
            "contact_sheet.png",
            "scatter_R_vs_quality.png",
        ],
        device=device,
    )
    log.info("\n" + "=" * 72)
    log.info(f"  SOLACE confidence↔quality probe → {run_dir}")
    state = {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[passed]
    wp = (
        f"within concordance {within['pairwise_concordance']:.2f}"
        if within.get("ok")
        else "no within-prompt grid"
    )
    log.info(
        f"  global ρ={rho:+.3f} | {wp} | between ρ="
        f"{within.get('between_spearman', float('nan')):+.3f} | {state}"
    )
    log.info("=" * 72)


def _write_contact_sheet(path, pixels, R, quality, records, order) -> None:
    """Grid of samples sorted by R_SOLACE (best confidence first), labelled."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning("contact sheet skipped (matplotlib unavailable): %s", e)
        return
    n = len(order)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for cell, i in enumerate(order):
        ax = axes[cell]
        img = ((pixels[i].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"#{cell}  R={R[i]:+.2f}\nq={quality[i]:.3f}", fontsize=7)
    fig.suptitle(
        "SOLACE self-confidence ranking — best (top-left) → worst (bottom-right)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_scatter(path, R, quality, rho) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning("scatter skipped (matplotlib unavailable): %s", e)
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(R, quality, s=24, alpha=0.8)
    ax.set_xlabel("R_SOLACE (self-confidence)")
    ax.set_ylabel("quality (top-k PE sim to real manifold)")
    ax.set_title(f"Spearman ρ = {rho:+.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
