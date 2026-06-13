"""Text-Jacobian alignment probe for the mod-guidance ``pooled_text_proj`` head.

Measures the sensitivity that mod-guidance steering actually rides on: how
faithfully the distilled modulation MLP reproduces the *teacher's local
response to a text change*. The inference-time steering

    emb_at_block = base_emb + w * (proj(pool(p+)) - proj(pool(p-)))

is a first-order text perturbation in modulation space, so the quantity that
predicts whether steering behaves like real text conditioning is the directional
derivative of the student's noise prediction w.r.t. text, compared to the
teacher's. This is exactly what a GAD (geometry-aware distillation) term would
optimize — so this probe is the Step-0 "is there a deficiency to repair"
measurement, and the same script scores any future GAD-trained head.

The two forwards mirror ``scripts/distill_mod/distill.py`` exactly:
  - teacher: real crossattn, ``skip_pooled_text_proj=True``     (text via cross-attn)
  - student: T5("") crossattn + ``pooled_text_override``        (text via modulation MLP)

For a held-out (latent, σ, noise) we perturb the text from sample A toward
sample B by a factor ``h`` (h=1.0 = full prompt swap; small h = local Jacobian)
and compare output deltas:

    ΔT = v_teacher(cross_A + h·(cross_B − cross_A)) − v_teacher(cross_A)
    ΔS = v_student(pool_A  + h·(pool_B  − pool_A))  − v_student(pool_A)

Reported per σ:
  - cos(ΔS, ΔT)         direction match (1.0 = student reproduces teacher's text response)
  - ‖ΔS‖ / ‖ΔT‖         magnitude match (cross-pathway, so ≈1 is ideal but not guaranteed)
  - ‖ΔT‖                sanity: the text change must actually move the teacher

A cos well below 1 on held-out text pairs (especially at high σ, where the doc
says modulation dominates) is the deficiency GAD would target. If cos is already
≈1, GAD has nothing to fix here.

Usage::

    python -m bench.mod_guidance.text_jacobian \
        --pooled_text_proj output/ckpt/pooled_text_proj.safetensors \
        --n_pairs 96 --sigmas 0.1 0.4 0.7 0.9 --h 1.0
"""

from __future__ import annotations

import argparse
import csv
import logging
import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from bench._anima import DEFAULT_DIT
from bench._common import make_run_dir, write_result
from library.anima import weights as anima_utils
from library.anima.models import Anima
from library.datasets.cache import CachedDataset
from library.runtime.harness import compile_dit_blocks
from library.training.forward import from_dit_5d, make_padding_mask, to_dit_5d
from library.inference.uncond import (
    default_uncond_path,
    load_uncond_crossattn,
    uncond_for_batch,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _dc_ac(x4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Split a velocity delta ``(1, C, H, W)`` into DC (per-channel spatial mean)
    and AC (the residual), DAVE-style.

    Returns ``(dc_vec, ac_vec, dc_field_energy, ac_energy)``:
      * ``dc_vec`` — ``(C,)`` the per-channel mean; cosines on it are scale-free
        so the H·W field-vs-vector factor is irrelevant for direction.
      * ``ac_vec`` — flattened ``h − μ`` residual.
      * energies use the **field** convention (DC energy ×H·W) so they're additive:
        ``‖x‖² = dc_field_energy + ac_energy`` (DC field ⊥ AC by construction).

    The shift term of AdaLN injects a spatially-uniform per-channel constant ⇒
    pure DC; the scale/gate gains rescale the AC that's already present. So
    comparing dS's vs dT's DC/AC split is exactly the "what subspace can the mod
    head reach, and how much of the teacher response lives there" question.
    """
    HW = x4.shape[-2] * x4.shape[-1]
    dc = x4.mean(dim=(2, 3))  # (1, C)
    ac = x4 - dc[:, :, None, None]  # (1, C, H, W)
    dc_energy = float((dc * dc).sum()) * HW
    ac_energy = float((ac * ac).sum())
    return dc.flatten(), ac.flatten(), dc_energy, ac_energy


def parse_args():
    p = argparse.ArgumentParser(
        description="Text-Jacobian alignment probe for mod-guidance"
    )
    p.add_argument(
        "--pooled_text_proj",
        required=True,
        help="Trained pooled_text_proj.safetensors to probe.",
    )
    p.add_argument(
        "--data_dir",
        default="post_image_dataset/lora",
        help="Cached latents + TE sidecars (same as distill --data_dir).",
    )
    p.add_argument(
        "--synth_data_dir",
        default=None,
        help="Teacher-synthetic latent dir (distill Phase 2). MUST match the "
        "head's training distribution — a synth-trained head probed on real "
        "latents is off-distribution (err_a inflated, cos collapses).",
    )
    p.add_argument(
        "--uncond_te_path",
        default=None,
        help='T5("") sidecar; defaults to the canonical distill-prep path.',
    )
    p.add_argument("--dit_path", default=DEFAULT_DIT)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.1, 0.4, 0.7, 0.9],
        help="Fixed noise levels to probe (text response is σ-dependent).",
    )
    p.add_argument(
        "--n_pairs",
        type=int,
        default=96,
        help="Number of (latent, textA, textB) trials per σ.",
    )
    p.add_argument(
        "--h",
        type=float,
        default=1.0,
        help="Text perturbation scale. 1.0 = full A→B prompt swap; "
        "small (e.g. 0.1) = local Jacobian (GAD-faithful) variant.",
    )
    p.add_argument(
        "--validation_split",
        type=float,
        default=0.05,
        help="Held-out fraction; probe runs on split='val' so it never "
        "touches data the head trained on.",
    )
    p.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Must match the distill run's --validation_seed for a true holdout.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap val samples loaded into memory (per bucket).",
    )
    p.add_argument("--seed", type=int, default=0, help="Trial-sampling / noise seed.")
    p.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=True,
        help="torch.compile each Block._forward (native-shape; one graph "
        "per token count). On by default — amortises the 384 forwards.",
    )
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--label", default=None)
    return p.parse_args()


def load_model(args, device, dtype) -> Anima:
    model: Anima = anima_utils.load_anima_model(
        device,
        args.dit_path,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    # pooled_text_proj is not in the base checkpoint — materialize then load trained weights.
    model.pooled_text_proj.to_empty(device="cpu")
    state = load_file(args.pooled_text_proj)
    # σ-FiLM weights (if present) ride under a 'sigma_film.' prefix.
    film_state = {
        k[len("sigma_film.") :]: v
        for k, v in state.items()
        if k.startswith("sigma_film.")
    }
    model.pooled_text_proj.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("sigma_film.")}
    )
    model.pooled_text_proj.to(device=device, dtype=torch.float32)
    if film_state:
        model.pooled_text_sigma_film.to_empty(device="cpu")
        model.pooled_text_sigma_film.load_state_dict(film_state)
        model.pooled_text_sigma_film.to(device=device, dtype=torch.float32)
        model.enable_pooled_text_sigma_film = True
        print("σ-FiLM weights present → probing the timestep-conditioned mod head")
    # The mod-guidance steering buffers are non-persistent zeros created on CPU
    # (distill.py moves them via place_dit_for_training, which we skip). They stay
    # zero here — the probe does its own text perturbation, not inference steering —
    # but must sit on-device or _run_blocks' unconditional `t_emb + schedule*delta`
    # arithmetic hits a cross-device add.
    model._mod_guidance_delta = model._mod_guidance_delta.to(device)
    model._mod_guidance_schedule = model._mod_guidance_schedule.to(device)
    model._mod_guidance_final_w = model._mod_guidance_final_w.to(device)
    # Arm the student path; the teacher call still passes skip_pooled_text_proj=True.
    model.enable_pooled_text_modulation = True
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    uncond_te_path = args.uncond_te_path or str(default_uncond_path())
    uncond_te_1 = load_uncond_crossattn(uncond_te_path, device, dtype)

    logger.info("Loading DiT + pooled_text_proj...")
    model = load_model(args, device, dtype)
    # COMPILE LAST — proj weights are already loaded, no adapter monkey-patches.
    compile_dit_blocks(model, enabled=args.compile)

    # --- Collect held-out samples into memory (val split is small) ---
    val = CachedDataset(
        args.data_dir,
        batch_size=1,
        split="val",
        validation_split=args.validation_split,
        validation_seed=args.validation_seed,
        synth_data_dir=args.synth_data_dir,
    )
    n = len(val)
    if args.max_samples is not None:
        n = min(n, args.max_samples)
    if n < 2:
        raise SystemExit(
            f"Need >=2 val samples to form text pairs, got {n}. "
            f"Raise --validation_split or --max_samples."
        )
    latents, crossattns, pooleds = [], [], []
    for k in range(n):
        _idx, lat, cross, pooled = val[k]
        latents.append(lat)  # (16, H, W)
        crossattns.append(cross)  # (seq, 1024) — max-padded, uniform seq
        pooleds.append(pooled)  # (1024,)
    logger.info(f"Loaded {n} held-out samples.")

    def fwd(noisy, sigma, crossattn_emb, *, skip_pooled, pooled_override=None):
        """One DiT noise prediction; mirrors distill.py teacher/student calls."""
        B = noisy.shape[0]
        pad = make_padding_mask(noisy, dtype)
        ts = torch.full((B,), float(sigma), device=device, dtype=dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            out = model.forward_mini_train_dit(
                noisy,
                ts,
                crossattn_emb,
                padding_mask=pad,
                skip_pooled_text_proj=skip_pooled,
                pooled_text_override=pooled_override,
            )
        return out.float()

    # --- Probe loop ---
    gen = torch.Generator().manual_seed(args.seed)
    rows = []  # (sigma, cos, ratio, dT_norm)
    eps = 1e-8
    for sigma in args.sigmas:
        for trial in range(args.n_pairs):
            i = int(torch.randint(0, n, (1,), generator=gen).item())
            j = int(torch.randint(0, n, (1,), generator=gen).item())
            if j == i:
                j = (j + 1) % n

            lat = latents[i].to(device, dtype=dtype).unsqueeze(0)  # (1,16,H,W)
            cross_a = crossattns[i].to(device, dtype=dtype).unsqueeze(0)  # (1,seq,1024)
            cross_b = crossattns[j].to(device, dtype=dtype).unsqueeze(0)
            pool_a = pooleds[i].to(device, dtype=dtype).unsqueeze(0)  # (1,1024)
            pool_b = pooleds[j].to(device, dtype=dtype).unsqueeze(0)

            # Fixed (latent, noise, σ); vary only text. Deterministic per trial.
            noise = torch.randn(lat.shape, generator=gen).to(device, dtype=dtype)
            noisy = to_dit_5d((1.0 - sigma) * lat + sigma * noise)  # (1,16,1,H,W)

            # Perturb text from A toward B by h (h=1 → full swap to B).
            cross_p = cross_a + args.h * (cross_b - cross_a)
            pool_p = pool_a + args.h * (pool_b - pool_a)

            # Teacher (text via cross-attn, proj skipped). Drop dim-2 (T=1)
            # to (1,16,H,W) so the DC/AC split reads the spatial grid.
            t_a = from_dit_5d(fwd(noisy, sigma, cross_a, skip_pooled=True))
            t_p = from_dit_5d(fwd(noisy, sigma, cross_p, skip_pooled=True))
            dT4 = t_p - t_a  # (1,16,H,W)
            dT = dT4.flatten()

            # Student (text via modulation MLP; crossattn pinned at uncond).
            uncond = uncond_for_batch(uncond_te_1, cross_a)
            s_a = from_dit_5d(
                fwd(noisy, sigma, uncond, skip_pooled=False, pooled_override=pool_a)
            )
            s_p = from_dit_5d(
                fwd(noisy, sigma, uncond, skip_pooled=False, pooled_override=pool_p)
            )
            dS4 = s_p - s_a
            dS = dS4.flatten()

            dT_norm = dT.norm().item()
            if dT_norm < eps:
                continue  # texts too similar — no teacher signal to align against
            cos = F.cosine_similarity(dS, dT, dim=0).item()
            ratio = dS.norm().item() / (dT_norm + eps)

            # --- DAVE DC/AC decomposition of the two response deltas ---
            dT_dc, dT_ac, dT_dce, dT_ace = _dc_ac(dT4)
            dS_dc, dS_ac, dS_dce, dS_ace = _dc_ac(dS4)
            cos_dc = F.cosine_similarity(
                dS_dc, dT_dc, dim=0
            ).item()  # within-DC aim (σ-FiLM owns this)
            cos_ac = F.cosine_similarity(
                dS_ac, dT_ac, dim=0
            ).item()  # gain-rescaling reach
            dT_tot = dT_dce + dT_ace
            dS_tot = dS_dce + dS_ace
            dT_ac_frac = dT_ace / (dT_tot + eps)  # teacher response: AC share
            dS_ac_frac = dS_ace / (dS_tot + eps)  # student response: AC share
            # Hard ceiling for ANY pure-DC head: best full-cos = ‖dT_DC‖/‖dT‖.
            cos_ceiling = math.sqrt(max(0.0, 1.0 - dT_ac_frac))
            # Achievable full-cos if the DC aim were perfected (cos_dc→1) at the
            # CURRENT energy split — i.e. the headroom σ-FiLM/more training can buy.
            dc_aligned_full = (
                math.sqrt(max(dS_dce, 0.0) * max(dT_dce, 0.0))
                + cos_ac * math.sqrt(max(dS_ace, 0.0) * max(dT_ace, 0.0))
            ) / (math.sqrt(dS_tot * dT_tot) + eps)
            # Held-out pointwise distillation residual at A: this is exactly what
            # MSE distillation minimizes. If err_a >~ ‖ΔT‖ (delta_snr < 1) then the
            # text-swap signal is buried under the head's own pointwise error and
            # cos≈0 is uninformative noise; if err_a ≪ ‖ΔT‖ then a low cos is a
            # genuine text-derivative misalignment (the GAD-relevant case).
            err_a = (s_a.flatten() - t_a.flatten()).norm().item()
            rel_err_a = err_a / (
                t_a.flatten().norm().item() + eps
            )  # held-out distill residual, normalized
            rows.append(
                (
                    float(sigma),
                    cos,
                    ratio,
                    dT_norm,
                    err_a,
                    rel_err_a,
                    cos_dc,
                    cos_ac,
                    dT_ac_frac,
                    dS_ac_frac,
                    cos_ceiling,
                    dc_aligned_full,
                )
            )

    if not rows:
        raise SystemExit(
            "No valid trials (all text deltas below eps). Check the dataset."
        )

    # --- Aggregate per σ + overall ---
    def _col(rs_, sig, idx):
        return [r[idx] for r in rs_ if r[0] == sig]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    per_sigma = {}
    for sigma in args.sigmas:
        cs = _col(rows, sigma, 1)
        if not cs:
            continue
        rs = _col(rows, sigma, 2)
        dn = _col(rows, sigma, 3)
        ea = _col(rows, sigma, 4)
        re_ = _col(rows, sigma, 5)
        cdc = _col(rows, sigma, 6)
        cac = _col(rows, sigma, 7)
        dtacf = _col(rows, sigma, 8)
        dsacf = _col(rows, sigma, 9)
        ceil = _col(rows, sigma, 10)
        dcaf = _col(rows, sigma, 11)
        mean = sum(cs) / len(cs)
        std = math.sqrt(sum((c - mean) ** 2 for c in cs) / len(cs))
        dn_mean = _mean(dn)
        ea_mean = _mean(ea)
        per_sigma[f"{sigma:.2f}"] = {
            "cos_mean": mean,
            "cos_std": std,
            "ratio_mean": _mean(rs),
            "dT_norm_mean": dn_mean,
            "err_a_mean": ea_mean,
            "rel_err_a_mean": _mean(re_),  # ‖s_a−t_a‖/‖t_a‖ — comparable to val MSE
            "delta_snr": dn_mean / (ea_mean + eps),  # ‖ΔT‖ vs head pointwise error
            # --- DAVE DC/AC decomposition ---
            "cos_dc_mean": _mean(
                cdc
            ),  # within-DC aim — the factor σ-FiLM/training owns
            "cos_ac_mean": _mean(cac),  # within-AC aim — limited to gain-rescaling
            "dT_ac_frac_mean": _mean(
                dtacf
            ),  # teacher response: AC energy share (ceiling driver)
            "dS_ac_frac_mean": _mean(
                dsacf
            ),  # student response: AC energy share (gain reach)
            "cos_ceiling_mean": _mean(
                ceil
            ),  # √(dT DC frac): best full-cos for ANY pure-DC head
            "dc_aligned_full_mean": _mean(
                dcaf
            ),  # full-cos if DC aim perfected at current split
            "n": len(cs),
        }
        logger.info(
            f"σ={sigma:.2f}  cos={mean:.4f}±{std:.4f}  ratio={_mean(rs):.3f}  "
            f"ΔSNR={dn_mean / (ea_mean + eps):.2f}  ||  "
            f"cos_dc={_mean(cdc):.4f}  cos_ac={_mean(cac):.4f}  "
            f"dT_ac={_mean(dtacf):.3f}  dS_ac={_mean(dsacf):.3f}  "
            f"ceiling={_mean(ceil):.3f}  dc_aligned={_mean(dcaf):.3f}  (n={len(cs)})"
        )

    all_cos = [r[1] for r in rows]
    overall = {
        "cos_mean": sum(all_cos) / len(all_cos),
        "ratio_mean": sum(r[2] for r in rows) / len(rows),
        "n": len(rows),
    }
    logger.info(f"OVERALL cos_mean={overall['cos_mean']:.4f}  n={overall['n']}")

    # --- Write envelope + per-trial CSV ---
    run_dir = make_run_dir("mod_guidance", label=args.label or "text-jacobian")
    csv_path = run_dir / "per_trial.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sigma",
                "cos",
                "ratio",
                "dT_norm",
                "err_a",
                "rel_err_a",
                "cos_dc",
                "cos_ac",
                "dT_ac_frac",
                "dS_ac_frac",
                "cos_ceiling",
                "dc_aligned_full",
            ]
        )
        w.writerows(rows)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"per_sigma": per_sigma, "overall": overall},
        label=args.label or "text-jacobian",
        artifacts=["per_trial.csv"],
        device=device,
    )
    logger.info(f"Wrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
