"""Full-spectrum SVD seeding for chimera — headroom probe.

Motivated by "Sparse Spectral LoRA: Routed Experts for Medical VLMs" (Nejati
Manzari et al., CVPR'26 / MedQwen). They seed each MoE expert from a
**non-overlapping SVD segment distributed across the WHOLE singular spectrum**
(expert ``j`` starts at singular index ``(j-1)·min(m,n)/N``), arguing that
intermediate / minor singular bands carry task-dependent signal. Our chimera
instead seeds every pool from the **top** ``(K_c+K_f)·r`` left- and top ``2r``
right-singular vectors (``chimera.py`` SVD partition). With the frozen-Cayley
path the ``r×r`` rotation is a no-op on an ``r``-column span, so
``colspace(ΔW)`` is **caged** to those top directions for the whole run.

Is there ΔW energy outside the top slice that full-spectrum seeding would
capture and top-slice cannot? Two parts, both on the REAL Anima DiT weights:

  * Part A — mechanism check (always). The frozen-Cayley adapter's reachable
    ΔW set is exactly ``{X : col(X) ⊆ span(P_bases), row(X) ⊆ span(Q_bases)}``,
    so best-achievable capture of a band target is the analytic projection
    ``‖Pᵀ T Q‖²/‖T‖²`` — no fit, no bf16/gate confounds. Mirror both seedings'
    band index sets and confirm the cage: top-slice captures an easy top band
    but ~0 of a deep band; spectrum (bands relocated across the spectrum)
    captures the deep band. Proves the lever does what we claim.

  * Part B — the decisive real signal (auto, needs DiT + a freely-trained LoRA).
    A free LoRA's ΔW is NOT caged, so where its energy lands in the base
    weight's singular spectrum tells us where unconstrained training *wants* to
    put mass. For every matched Linear we SVD the base ``W`` and ask the only
    decision-relevant question with the SAME analytic cage Part A uses: of the
    real ΔW, what fraction would each seeding actually *reconstruct*?

      cap_seed = ‖P_seedᵀ ΔW Q_seed‖² / ‖ΔW‖²

    Three cages — ``top`` (chimera today), ``spectrum`` (MedQwen relocation),
    and a sampled ``random`` index set of the same widths (the null). The
    verdict is **relative**: full-spectrum has headroom only if it both beats
    the top-slice (``cap_spectrum − cap_top``) *and* beats the random cage —
    otherwise the energy is just diffuse and any equal-width band set catches
    the same amount (⇒ needs **rank**, not relocated bands).

    Why the old "fraction beyond the top slice" marginal was not enough: it is
    one-sided and rewards *any* tail energy, including energy spread uniformly
    across the tail that no narrow strided band can capture. We keep it as
    context but add a **band-concentration enrichment** per axis: of the
    beyond-top energy, the fraction landing inside the strided spectrum bands
    divided by the fraction of the tail those bands cover. ``enrichment ≫ 1`` ⇒
    band-concentrated, full-spectrum-capturable; ``≈ 1`` ⇒ diffuse, needs rank.

Caveat (Part B): the free LoRA was trained on its own task; it is the best
offline proxy for "where free training puts ΔW", not chimera's exact target.
Control it — point ``--lora`` at a converged, full-data, **plain** (no
ortho / ortho_init) LoRA over the same Linears; an ``use_ortho_init`` ckpt is
top-32-seeded and structurally invalid here. Runs end-to-end on GPU when
available (the per-layer SVD + projection matmuls are the expensive part);
falls back to CPU.
"""

from __future__ import annotations

from pathlib import Path
import torch
from safetensors import safe_open

from bench._common import make_run_dir, write_result

NAME = "full_spectrum"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R, K_C, K_F = 32, 6, 2  # match the shipped chimera-0616 pools
N_LEFT = (K_C + K_F) * R  # top-slice left (U) boundary  = 256
N_RIGHT = 2 * R  # top-slice right (V) boundary = 64


def add_args(p):
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-base-v1.0.safetensors",
        help="Base DiT safetensors (repo-relative ok).",
    )
    p.add_argument(
        "--lora",
        default="output/ckpt/anima_channel.safetensors",
        help="A FREELY-trained LoRA ckpt (uncaged ΔW) for Part B. Skip if absent.",
    )
    p.add_argument(
        "--layer",
        default="net.blocks.0.mlp.layer1.weight",
        help="DiT weight key for the Part-A mechanism check.",
    )
    p.add_argument("--max-layers", type=int, default=64, help="Part B cap.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)


def _resolve(path: str) -> Path:
    from library.env import resolve_under_home

    return resolve_under_home(path)


def _svd(W: torch.Tensor):
    """Full SVD on DEVICE (8192×2048 DiT Linears are slow on CPU)."""
    return torch.linalg.svd(W.to(DEVICE).float(), full_matrices=False)


# ---------------------------------------------------------------------------
# Part A — mechanism: does spectrum seeding reach bands top-slice cannot?
# ---------------------------------------------------------------------------


def _seed_bands(U, V):
    """Left (P) and right (Q) singular-column index sets for both seedings,
    mirroring ``chimera.py`` exactly.

      * top      — ``chimera.__init__``: P = top ``(K_c+K_f)·r`` of U,
                    Q = top ``2r`` of V.
      * spectrum — MedQwen relocation: P = ``K_c+K_f`` non-overlapping r-wide
                    bands strided across U; Q = 2 r-wide bands at the two ends
                    of V (content A at 0, freq A at k/2).

    Returns ``(P_idx, Q_idx)`` per seeding plus the spectrum band starts (so the
    deep target can be placed ON a spectrum band — its raison d'être).
    """
    k_out, k_in = U.shape[1], V.shape[1]
    nL = K_C + K_F
    segL = k_out // nL
    segR = k_in // 2
    p_bands = [s for j in range(nL) for s in [j * segL]]
    top = {
        "P": torch.arange(N_LEFT),
        "Q": torch.arange(N_RIGHT),
    }
    spectrum = {
        "P": torch.cat([torch.arange(s, s + R) for s in p_bands]),
        "Q": torch.cat([torch.arange(0, R), torch.arange(segR, segR + R)]),
    }
    return top, spectrum, p_bands, segR


def _capture(U, V, P_idx, Q_idx, left_idx, right_idx):
    """Exact best-achievable capture of a bilinear target whose left support is
    ``U[:, left_idx]`` and right support is ``V[:, right_idx]``, by a caged
    adapter with ``col ⊆ span(U[:,P_idx])`` and ``row ⊆ span(V[:,Q_idx])``.

    For orthonormal column subsets the reachable best-approx of T is
    ``P Pᵀ T Q Qᵀ`` and captured fraction = ``‖Pᵀ T Q‖² / ‖T‖²`` — no fit, no
    bf16, no gate confounds. Target core ``G`` is random full-rank in the band.
    """
    torch.manual_seed(0)
    g = torch.randn(len(left_idx), len(right_idx), device=U.device)
    Ul, Vr = U[:, left_idx], V[:, right_idx]
    T = Ul @ g @ Vr.T  # (out, in)
    P, Q = U[:, P_idx], V[:, Q_idx]
    proj = P.T @ T @ Q  # (p, q)
    return float(proj.pow(2).sum() / T.pow(2).sum().clamp_min(1e-30))


def _part_a(args, W):
    U, S, Vh = _svd(W)
    V = Vh.T
    out = {"layer": args.layer, "shape": list(W.shape), "k": int(S.shape[0])}
    energy = (S**2).cumsum(0) / (S**2).sum()
    out["spectrum_decay"] = {
        "energy_in_top_left_slice": float(energy[min(N_LEFT, len(S)) - 1]),
        "energy_in_top_right_slice": float(energy[min(N_RIGHT, len(S)) - 1]),
        "s_top_over_s_mid": float(S[0] / S[len(S) // 2].clamp_min(1e-9)),
        "s_top_over_s_tail": float(S[0] / S[-1].clamp_min(1e-9)),
    }
    top, spectrum, p_bands, segR = _seed_bands(U, V)

    # Two targets:
    #   easy  — band 0 on both sides; BOTH seedings hold it (sanity control).
    #   deep  — left support on a DEEP spectrum P-band, right support on the
    #           spectrum freq-A band (k/2). Capturable by spectrum, not by
    #           top-slice (its bands are all in the top 2r/256).
    deep_left = next(s for s in reversed(p_bands) if s + R <= U.shape[1])
    targets = {
        "easy_band": (torch.arange(0, R), torch.arange(0, R)),
        "deep_band": (
            torch.arange(deep_left, deep_left + R),
            torch.arange(segR, segR + R),
        ),
    }
    results = {}
    for seeding, bands in (("top", top), ("spectrum", spectrum)):
        for tname, (li, ri) in targets.items():
            cap = _capture(
                U,
                V,
                bands["P"].to(U.device),
                bands["Q"].to(U.device),
                li.to(U.device),
                ri.to(U.device),
            )
            results[f"{seeding}/{tname}"] = cap
            print(f"  {seeding:8s} {tname:10s} frac_captured={cap:.3f}")
    out["captured"] = results
    out["deep_left_singular_index"] = int(deep_left)
    # Mechanism claim: top-slice can't reach the deep band; spectrum can.
    out["top_deep_captured"] = results["top/deep_band"]
    out["spectrum_deep_captured"] = results["spectrum/deep_band"]
    return out


# ---------------------------------------------------------------------------
# Part B — where does free, uncaged ΔW land in W's singular spectrum?
# ---------------------------------------------------------------------------


def _cage_capture(dW, U, V, P_idx, Q_idx):
    """Fraction of the REAL ΔW a frozen-Cayley cage (``col ⊆ span(U[:,P]))``,
    ``row ⊆ span(V[:,Q])``) can reconstruct: ``‖PᵀΔW Q‖²/‖ΔW‖²``. Same analytic
    projection as Part A's ``_capture`` but with the trained ΔW as the target —
    no fit, no gate confounds."""
    P, Q = U[:, P_idx], V[:, Q_idx]
    num = (P.T @ dW @ Q).pow(2).sum()
    return float(num / dW.pow(2).sum().clamp_min(1e-30))


def _rand_cage_capture(dW, U, V, n_p, n_q, seed, trials=4):
    """Null: mean capture of a random index set of the same widths. If the
    spectrum cage can't beat this, ΔW is diffuse and relocating bands buys
    nothing over any equal-width set (the energy needs rank, not a new basis)."""
    k = U.shape[1]
    g = torch.Generator(device=U.device).manual_seed(seed)
    vals = [
        _cage_capture(
            dW,
            U,
            V,
            torch.randperm(k, generator=g, device=U.device)[:n_p],
            torch.randperm(k, generator=g, device=U.device)[:n_q],
        )
        for _ in range(trials)
    ]
    return float(torch.tensor(vals).mean())


def _band_enrichment(energy, band_idx, top_boundary, k):
    """Of the beyond-top energy on one axis, how concentrated is it in the
    strided spectrum bands vs spread uniformly across the tail?

    ``enrichment = (in-band fraction of beyond-top energy) / (fraction of the
    tail the bands cover)``. ``≫ 1`` ⇒ band-concentrated (full-spectrum can
    grab it); ``≈ 1`` ⇒ diffuse (only rank helps)."""
    top = min(top_boundary, k)
    tail_n = k - top
    if tail_n <= 0:
        return None
    beyond = energy[top:].sum().clamp_min(1e-30)
    band_in_tail = band_idx[band_idx >= top]
    in_band = (
        energy[band_in_tail].sum() if band_in_tail.numel() else energy.new_zeros(())
    )
    coverage = band_in_tail.numel() / tail_n  # uniform-null in-band fraction
    in_band_frac = float(in_band / beyond)
    return {
        "in_band_frac": in_band_frac,
        "band_coverage": coverage,
        "enrichment": (in_band_frac / coverage) if coverage > 0 else None,
    }


def _lora_to_dit_key(prefix: str) -> str | None:
    """``lora_unet_blocks_0_mlp_layer1`` → ``net.blocks.0.mlp.layer1.weight``.

    Only the cleanly-mapping (non-fused) Linears; fused q/k/v on the lora side
    are already split to match the DiT, so the underscore→dot inversion holds
    for the leaf names we care about (mlp + the split attn projections)."""
    if not prefix.startswith("lora_unet_"):
        return None
    return f"net.{prefix[len('lora_unet_') :].replace('_', '.')}.weight"


def _part_b(args):
    dit_path = _resolve(args.dit)
    lora_path = _resolve(args.lora)
    if not dit_path.exists() or not lora_path.exists():
        return {"skipped": f"missing dit={dit_path.exists()} lora={lora_path.exists()}"}

    lf = safe_open(str(lora_path), "pt")
    lkeys = set(lf.keys())
    prefixes = sorted(
        k[: -len(".lora_down.weight")] for k in lkeys if k.endswith(".lora_down.weight")
    )
    df = safe_open(str(dit_path), "pt")
    dit_keys = set(df.keys())

    per_layer = []
    agg = {
        k: []
        for k in (
            "cap_top",
            "cap_spectrum",
            "cap_random",
            "headroom",
            "lb",
            "rb",
            "enr_left",
            "enr_right",
        )
    }
    n = 0
    for prefix in prefixes:
        if n >= args.max_layers:
            break
        dkey = _lora_to_dit_key(prefix)
        if dkey is None or dkey not in dit_keys:
            continue
        down = lf.get_tensor(f"{prefix}.lora_down.weight").float().to(DEVICE)  # (r, in)
        up = lf.get_tensor(f"{prefix}.lora_up.weight").float().to(DEVICE)  # (out, r)
        alpha = (
            float(lf.get_tensor(f"{prefix}.alpha"))
            if f"{prefix}.alpha" in lkeys
            else float(down.shape[0])
        )
        dW = (up @ down) * (alpha / down.shape[0])  # (out, in)
        W = df.get_tensor(dkey).float().to(DEVICE)
        if W.shape != dW.shape:
            continue
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.T
        k = S.shape[0]
        left = (U.T @ dW).pow(2).sum(dim=1)  # (k,) energy per left-singular idx
        right = (dW @ V).pow(2).sum(dim=0)  # (k,) energy per right-singular idx
        lt, rt = left.sum().clamp_min(1e-30), right.sum().clamp_min(1e-30)
        idx = torch.arange(k, device=DEVICE, dtype=torch.float32)
        lb = float(left[min(N_LEFT, k) :].sum() / lt)
        rb = float(right[min(N_RIGHT, k) :].sum() / rt)

        # Sharpened, decision-relevant signal: two-sided cage capture of the
        # real ΔW under each seeding, vs a same-width random null.
        top, spectrum, _, _ = _seed_bands(U, V)
        p_top, q_top = top["P"].to(DEVICE), top["Q"].to(DEVICE)
        p_spec, q_spec = spectrum["P"].to(DEVICE), spectrum["Q"].to(DEVICE)
        cap_top = _cage_capture(dW, U, V, p_top, q_top)
        cap_spec = _cage_capture(dW, U, V, p_spec, q_spec)
        cap_rand = _rand_cage_capture(
            dW, U, V, p_spec.numel(), q_spec.numel(), args.seed + n
        )
        enr_l = _band_enrichment(left, spectrum["P"].to(DEVICE), N_LEFT, k)
        enr_r = _band_enrichment(right, spectrum["Q"].to(DEVICE), N_RIGHT, k)

        per_layer.append(
            {
                "key": dkey,
                "left_energy_beyond_top_slice": lb,
                "right_energy_beyond_top_slice": rb,
                "left_centroid_frac": float((left * idx).sum() / lt / k),
                "right_centroid_frac": float((right * idx).sum() / rt / k),
                "cap_top": cap_top,
                "cap_spectrum": cap_spec,
                "cap_random": cap_rand,
                "headroom_spectrum_over_top": cap_spec - cap_top,
                "left_band_enrichment": enr_l,
                "right_band_enrichment": enr_r,
            }
        )
        agg["cap_top"].append(cap_top)
        agg["cap_spectrum"].append(cap_spec)
        agg["cap_random"].append(cap_rand)
        agg["headroom"].append(cap_spec - cap_top)
        agg["lb"].append(lb)
        agg["rb"].append(rb)
        if enr_l and enr_l["enrichment"] is not None:
            agg["enr_left"].append(enr_l["enrichment"])
        if enr_r and enr_r["enrichment"] is not None:
            agg["enr_right"].append(enr_r["enrichment"])
        n += 1

    def _med(xs):
        return float(torch.tensor(xs).median()) if xs else None

    return {
        "lora": lora_path.name,
        "n_layers": len(per_layer),
        "N_left_boundary": N_LEFT,
        "N_right_boundary": N_RIGHT,
        # legacy one-sided marginals (context only — see docstring)
        "median_left_energy_beyond_top_slice": _med(agg["lb"]),
        "median_right_energy_beyond_top_slice": _med(agg["rb"]),
        # sharpened decision metric: cage capture of the real ΔW
        "median_cap_top": _med(agg["cap_top"]),
        "median_cap_spectrum": _med(agg["cap_spectrum"]),
        "median_cap_random": _med(agg["cap_random"]),
        "median_headroom_spectrum_over_top": _med(agg["headroom"]),
        # band-concentration enrichment (≫1 band-concentrated, ≈1 diffuse)
        "median_left_band_enrichment": _med(agg["enr_left"]),
        "median_right_band_enrichment": _med(agg["enr_right"]),
        "per_layer": per_layer,
    }


def run(args):
    torch.manual_seed(args.seed)
    print(f"[device={DEVICE}]")
    print("Part A — structural reach (top-slice vs full-spectrum seeding):")
    W = safe_open(str(_resolve(args.dit)), "pt").get_tensor(args.layer).float()
    a = _part_a(args, W)
    print(
        f"  deep-band captured: top-slice={a['top_deep_captured']:.2f} "
        f"spectrum={a['spectrum_deep_captured']:.2f} "
        "(top≈0 & spectrum≈1 ⇒ cage confirmed + spectrum reaches it)"
    )

    print("\nPart B — where free, uncaged ΔW lands in W's spectrum:")
    b = _part_b(args)
    if "skipped" in b:
        print(f"  SKIPPED: {b['skipped']}")
    else:
        ct, cs, cr = (
            b["median_cap_top"],
            b["median_cap_spectrum"],
            b["median_cap_random"],
        )
        head = b["median_headroom_spectrum_over_top"]
        enr = max(
            b["median_left_band_enrichment"] or 0.0,
            b["median_right_band_enrichment"] or 0.0,
        )
        print(f"  proxy={b['lora']}  layers={b['n_layers']}")
        print(
            f"  cage capture of real ΔW: top={ct:.4f}  spectrum={cs:.4f}  "
            f"random-null={cr:.4f}  (headroom spectrum−top={head:+.4f})"
        )
        print(
            f"  band enrichment (≫1 concentrated / ≈1 diffuse): "
            f"left={b['median_left_band_enrichment']} "
            f"right={b['median_right_band_enrichment']}"
        )
        print(
            f"  [context] one-sided energy beyond top-slice: "
            f"left={b['median_left_energy_beyond_top_slice']:.3f} "
            f"right={b['median_right_energy_beyond_top_slice']:.3f}"
        )
        # Relative verdict — full-spectrum earns the A/B only if it reconstructs
        # meaningfully more of the real ΔW than BOTH the top-slice AND a random
        # equal-width cage, and the tail energy is band-concentrated (not diffuse).
        beats_top = head > 0.02
        beats_null = (cs - cr) > 0.02
        concentrated = enr > 1.3
        if beats_top and beats_null and concentrated:
            verdict = "HEADROOM — full-spectrum beats top-slice & random null and ΔW is band-concentrated ⇒ ship the training A/B"
        elif not concentrated and not beats_null:
            verdict = "DIFFUSE — spectrum ≈ random null & no band concentration ⇒ needs RANK, not relocated bands; do NOT ship full-spectrum"
        else:
            verdict = "TOP-SUFFICIENT/PROXY-WEAK — top-slice already captures ΔW (or signal too weak to gate); do NOT ship on this proxy"
        print("verdict:", verdict)

    metrics = {"part_a_reach": a, "part_b_free_lora_localization": b}
    label = f"{NAME}-{args.label}" if args.label else NAME
    run_dir = make_run_dir("chimera", label=label)
    write_result(
        run_dir, script=__file__, args=vars(args), metrics=metrics, device=DEVICE
    )
    print(f"\nwrote {run_dir / 'result.json'}")
