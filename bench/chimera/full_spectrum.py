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
    put mass. For every matched Linear: SVD the base ``W``, project ``ΔW`` onto
    ``W``'s left/right singular directions, report the fraction of ΔW energy
    whose singular index sits **beyond the top slice**. High ⇒ full-spectrum
    has headroom ⇒ run the training A/B. Near zero ⇒ top-slice already captures
    it ⇒ don't bother.

Caveat (Part B): the free LoRA was trained on its own task; it is the best
offline proxy for "where free training puts ΔW", not chimera's exact target.
Runs end-to-end on GPU when available (the per-layer SVD + projection matmuls
are the expensive part); falls back to CPU.
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
        "deep_band": (torch.arange(deep_left, deep_left + R), torch.arange(segR, segR + R)),
    }
    results = {}
    for seeding, bands in (("top", top), ("spectrum", spectrum)):
        for tname, (li, ri) in targets.items():
            cap = _capture(U, V, bands["P"].to(U.device), bands["Q"].to(U.device), li.to(U.device), ri.to(U.device))
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


def _lora_to_dit_key(prefix: str) -> str | None:
    """``lora_unet_blocks_0_mlp_layer1`` → ``net.blocks.0.mlp.layer1.weight``.

    Only the cleanly-mapping (non-fused) Linears; fused q/k/v on the lora side
    are already split to match the DiT, so the underscore→dot inversion holds
    for the leaf names we care about (mlp + the split attn projections)."""
    if not prefix.startswith("lora_unet_"):
        return None
    return f"net.{prefix[len('lora_unet_'):].replace('_', '.')}.weight"


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

    per_layer, mlp_left, all_left, all_right = [], [], [], []
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
        left = (U.T @ dW).pow(2).sum(dim=1)  # (k,) energy per left-singular idx
        right = (dW @ V).pow(2).sum(dim=0)  # (k,) energy per right-singular idx
        lt, rt = left.sum().clamp_min(1e-30), right.sum().clamp_min(1e-30)
        k = S.shape[0]
        idx = torch.arange(k, device=DEVICE, dtype=torch.float32)
        lb = float(left[min(N_LEFT, k) :].sum() / lt)
        rb = float(right[min(N_RIGHT, k) :].sum() / rt)
        per_layer.append(
            {
                "key": dkey,
                "left_energy_beyond_top_slice": lb,
                "right_energy_beyond_top_slice": rb,
                "left_centroid_frac": float((left * idx).sum() / lt / k),
                "right_centroid_frac": float((right * idx).sum() / rt / k),
            }
        )
        all_left.append(lb)
        all_right.append(rb)
        if "mlp" in dkey:
            mlp_left.append(lb)
        n += 1

    def _med(xs):
        return float(torch.tensor(xs).median()) if xs else None

    return {
        "lora": lora_path.name,
        "n_layers": len(per_layer),
        "N_left_boundary": N_LEFT,
        "N_right_boundary": N_RIGHT,
        "median_left_energy_beyond_top_slice": _med(all_left),
        "median_right_energy_beyond_top_slice": _med(all_right),
        "median_mlp_left_energy_beyond_top_slice": _med(mlp_left),
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
        print(
            f"  layers={b['n_layers']}  median ΔW energy beyond top-slice: "
            f"left={b['median_left_energy_beyond_top_slice']:.3f} "
            f"right={b['median_right_energy_beyond_top_slice']:.3f} "
            f"(mlp left={b['median_mlp_left_energy_beyond_top_slice']})"
        )
        hi = max(
            b["median_left_energy_beyond_top_slice"] or 0.0,
            b["median_right_energy_beyond_top_slice"] or 0.0,
        )
        print(
            "verdict:",
            "HEADROOM — ship the training A/B"
            if hi > 0.3
            else "top-slice already captures ΔW — full-spectrum unlikely to help",
        )

    metrics = {"part_a_reach": a, "part_b_free_lora_localization": b}
    label = f"{NAME}-{args.label}" if args.label else NAME
    run_dir = make_run_dir("chimera", label=label)
    write_result(run_dir, script=__file__, args=vars(args), metrics=metrics, device=DEVICE)
    print(f"\nwrote {run_dir / 'result.json'}")
