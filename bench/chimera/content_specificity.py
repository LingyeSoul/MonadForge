"""Does chimera's shared-A content half learn a content-specific representation?

The chimera content half is one **shared** down-projection ``A_c`` + ``K_c``
B-heads, gated by the network-level ContentRouter on pooled ``crossattn_emb``.
Under the shipped frozen-Cayley path the ``r×r`` rotation is a no-op on an
``r``-column span, so ``A_c x`` is a FIXED (content-agnostic) projection of the
layer input. The ONLY route to content specificity is therefore:

    Δy_c = Σ_c (π_c[c] − 1/K_c) · B_c[c] (A_c x)

— the gate ``π_c`` (a function of the prompt) selecting a per-prompt mixture of
**distinct** B-heads. So "does shared A really learn content-specific
representation" reduces to three measurable, necessary conditions:

  1. The ContentRouter is **responsive** — π_c actually moves across prompts
     (if it's ~constant, the centered gate makes Δy_c ≈ 0: a dead content half).
  2. π_c **discriminates real content** — semantically different prompts get
     different expert mixtures, above a label-shuffled null.
  3. The B-heads are **diverse** — if all K_c heads point the same way, routing
     is moot regardless of how well π_c moves.

All three are necessary; content-specificity is real only if all hold. Pure
read of the trained checkpoint against REAL training captions — loads the
ContentRouter + B-head stacks from ``*_chimera.safetensors`` and the already-
cached ``crossattn_emb`` from the TE caches (no DiT, no text encoder, CPU-only).
Artist directory is used as a free, weak content/style label.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

from bench._common import make_run_dir, write_result
from networks.lora_anima.routers import ContentRouter

NAME = "content_specificity"
TE_GLOB = "*_anima_te.safetensors"


def add_args(p):
    p.add_argument(
        "--ckpt",
        default="output/ckpt/anima_chimera-0616_chimera.safetensors",
        help="Trained *_chimera.safetensors (carries content_router.* + lora_ups_c).",
    )
    p.add_argument(
        "--te-dir",
        default="post_image_dataset/lora",
        help="Dir of cached *_anima_te.safetensors (real training captions).",
    )
    p.add_argument("--n-prompts", type=int, default=400)
    p.add_argument("--min-per-artist", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)


def _resolve(path: str) -> Path:
    from library.env import resolve_under_home

    return resolve_under_home(path)


def _load_content_router(ckpt):
    keys = set(ckpt.keys())
    if "content_router.net.0.weight" not in keys:
        raise SystemExit(
            "checkpoint has no content_router.* — not a content-routed chimera."
        )
    w0 = ckpt.get_tensor("content_router.net.0.weight")  # (hidden, in)
    w2 = ckpt.get_tensor("content_router.net.2.weight")  # (K_c, hidden)
    in_dim, hidden, k_c = w0.shape[1], w0.shape[0], w2.shape[0]
    md = ckpt.metadata() or {}
    # Content tau is not separately stamped; the ContentRouter default (1.0) is
    # what training used. LN-in presence IS stamped.
    ln = str(md.get("ss_chimera_content_router_layer_norm", "true")).lower() == "true"
    router = ContentRouter(in_dim, k_c, hidden_dim=hidden, tau=1.0, apply_layer_norm=ln)
    router.load_state_dict(
        {
            "net.0.weight": w0,
            "net.0.bias": ckpt.get_tensor("content_router.net.0.bias"),
            "net.2.weight": w2,
            "net.2.bias": ckpt.get_tensor("content_router.net.2.bias"),
        },
        strict=False,
    )
    router.eval()
    return router, k_c


def _bhead_diversity(ckpt):
    """Mean / max cross-expert cosine of the content B-heads per Linear.

    Each head is (out, r), flattened. Low cosine ⇒ experts occupy distinct
    output directions ⇒ routing between them is meaningful.
    """
    keys = list(ckpt.keys())
    prefixes = sorted(
        {
            k[: k.index(".lora_ups_c.")]
            for k in keys
            if ".lora_ups_c." in k and k.endswith(".weight")
        }
    )
    per_mean, per_max = [], []
    for prefix in prefixes:
        idxs = sorted(
            int(k.split(".lora_ups_c.")[1].split(".")[0])
            for k in keys
            if k.startswith(f"{prefix}.lora_ups_c.") and k.endswith(".weight")
        )
        heads = [
            ckpt.get_tensor(f"{prefix}.lora_ups_c.{i}.weight").float().flatten()
            for i in idxs
        ]
        if len(heads) < 2:
            continue
        H = torch.stack(heads)
        H = H / H.norm(dim=1, keepdim=True).clamp_min(1e-9)
        g = (H @ H.T).abs()
        off = g[~torch.eye(len(heads), dtype=torch.bool)]
        per_mean.append(off.mean().item())
        per_max.append(off.max().item())
    t = torch.tensor(per_mean) if per_mean else torch.tensor([float("nan")])
    return {
        "n_layers": len(per_mean),
        "median_cross_expert_cos": float(t.median()),
        "p90_cross_expert_cos": float(t.quantile(0.9)) if per_mean else None,
        "max_cross_expert_cos": float(max(per_max)) if per_max else None,
    }


def _sample_prompts(te_dir: Path, n: int, min_per_artist: int, seed: int):
    files = list(te_dir.rglob(TE_GLOB))
    if not files:
        raise SystemExit(f"no {TE_GLOB} under {te_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    by_artist: dict[str, list[Path]] = {}
    for f in files:
        by_artist.setdefault(f.parent.name, []).append(f)
    artists = [a for a, fs in by_artist.items() if len(fs) >= min_per_artist]
    rng.shuffle(artists)
    chosen: list[tuple[Path, str]] = []
    for a in artists:
        for f in by_artist[a]:
            chosen.append((f, a))
            if len(chosen) >= n:
                break
        if len(chosen) >= n:
            break
    embs, labels = [], []
    for f, a in chosen:
        with safe_open(str(f), "pt") as t:
            tkeys = set(t.keys())
            if "v0_intact" in tkeys and float(t.get_tensor("v0_intact")) < 0.5:
                continue
            if "crossattn_emb_v0" not in tkeys:
                continue
            embs.append(t.get_tensor("crossattn_emb_v0").float())  # (L, D)
            labels.append(a)
    return torch.stack(embs), labels  # (N, L, D)


def _entropy(p):
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def _nmi(labels, clusters):
    """Normalized MI between artist labels and argmax-expert clusters. Chance is
    handled by the shuffle null the caller z-scores against."""
    la = {v: i for i, v in enumerate(sorted(set(labels)))}
    ca = {v: i for i, v in enumerate(sorted(set(clusters)))}
    a = [la[x] for x in labels]
    c = [ca[x] for x in clusters]
    n = len(a)
    cont = torch.zeros(len(la), len(ca))
    for i, j in zip(a, c):
        cont[i, j] += 1
    pi = cont.sum(1) / n
    pj = cont.sum(0) / n
    pij = cont / n
    mask = pij > 0
    ratio = pij[mask] / (pi.unsqueeze(1) * pj.unsqueeze(0))[mask]
    mi = float((pij[mask] * ratio.log()).sum())
    denom = math.sqrt(max(_entropy(pi), 1e-12) * max(_entropy(pj), 1e-12))
    return mi / denom if denom > 0 else 0.0


def run(args):
    torch.manual_seed(args.seed)
    ckpt_path = _resolve(args.ckpt)
    ckpt = safe_open(str(ckpt_path), "pt")
    router, k_c = _load_content_router(ckpt)
    bhead = _bhead_diversity(ckpt)

    embs, labels = _sample_prompts(
        _resolve(args.te_dir), args.n_prompts, args.min_per_artist, args.seed
    )
    with torch.no_grad():
        pi = router(embs)  # (N, K_c) — pools + LN + MLP + softmax, exactly as training
    N = pi.shape[0]

    # C1: responsiveness — does π_c move across prompts?
    mean_gate = pi.mean(0)
    responsiveness = float(pi.std(0).mean())
    gate_entropy = _entropy(mean_gate) / math.log(k_c)

    # C2: real-content discrimination vs shuffle null
    argmax_expert = pi.argmax(1).tolist()
    nmi = _nmi(labels, argmax_expert)
    rng = random.Random(args.seed)
    null = []
    for _ in range(50):
        shuf = labels[:]
        rng.shuffle(shuf)
        null.append(_nmi(shuf, argmax_expert))
    null_t = torch.tensor(null)
    nmi_z = (nmi - float(null_t.mean())) / max(float(null_t.std()), 1e-9)

    # Utilization / dead experts
    usage = Counter(argmax_expert)
    util = [usage.get(i, 0) / N for i in range(k_c)]
    dead = sum(1 for u in util if u == 0.0)

    metrics = {
        "ckpt": ckpt_path.name,
        "n_prompts": N,
        "n_artists": len(set(labels)),
        "K_c": k_c,
        "condition1_responsiveness": {
            "mean_per_expert_gate_std": responsiveness,
            "gate_usage_entropy_norm": gate_entropy,
            "mean_gate": mean_gate.tolist(),
            "max_minus_min_mean_gate": float(mean_gate.max() - mean_gate.min()),
        },
        "condition2_discrimination": {
            "nmi_artist_vs_expert": nmi,
            "nmi_shuffle_mean": float(null_t.mean()),
            "nmi_z_vs_shuffle": nmi_z,
        },
        "condition3_bhead_diversity": bhead,
        "utilization": {"per_expert_argmax_frac": util, "dead_experts": dead},
    }

    # Discrimination (does the gate track content, z vs shuffle) and amplitude
    # (how hard it drives the content half, per-prompt std) are DISTINCT axes —
    # a gate can be tightly content-locked yet barely move (weak-but-real half).
    disc_ok = nmi_z > 3.0  # gate direction tracks content above chance
    strong = responsiveness > 0.02  # gate amplitude is non-trivial
    div_ok = bhead["median_cross_expert_cos"] < 0.5  # experts genuinely distinct
    effective_experts = k_c - dead
    if not disc_ok:
        verdict = "NOT content-specific — gate does not track content (dead/noise half)"
    elif not div_ok:
        verdict = "NOT content-specific — B-heads collapsed (routing is moot)"
    elif strong:
        verdict = f"content-specific — discriminative + high-amplitude (eff K_c={effective_experts})"
    else:
        verdict = (
            "content-LOCKED but LOW-AMPLITUDE — gate tracks content "
            f"(z={nmi_z:.0f}) yet barely moves (std={responsiveness:.4f}); "
            f"effective K_c={effective_experts}/{k_c} (the content half is real "
            "but weak / over-provisioned)"
        )
    metrics["verdict"] = verdict
    metrics["effective_experts"] = effective_experts

    print(f"K_c={k_c}  prompts={N}  artists={len(set(labels))}")
    print(
        f"  C1 responsiveness: per-expert gate std={responsiveness:.4f}  "
        f"usage_entropy={gate_entropy:.3f}  dead_experts={dead}"
    )
    print(
        f"  C2 discrimination: NMI={nmi:.4f}  shuffle={float(null_t.mean()):.4f}  "
        f"z={nmi_z:.2f}"
    )
    print(
        f"  C3 B-head diversity: median cross-expert cos="
        f"{bhead['median_cross_expert_cos']:.3f}  (p90={bhead['p90_cross_expert_cos']})"
    )
    print(f"  VERDICT: {verdict}")

    label = f"{NAME}-{args.label}" if args.label else NAME
    run_dir = make_run_dir("chimera", label=label)
    write_result(run_dir, script=__file__, args=vars(args), metrics=metrics, device="cpu")
    print(f"wrote {run_dir / 'result.json'}")
