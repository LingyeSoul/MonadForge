#!/usr/bin/env python
"""Router-target leverage: is routing a given Linear (e.g. cross_attn.output_proj)
worth it, vs. leaving it as a plain LoRA?

Analytic, forward-only. No image generation, no CMMD, no FM val loss (which does
not track quality on Anima — see ``project_fm_val_loss_uninformative``).

The exact null model
--------------------
Within one expert pool the experts share a single down-projection ``A`` (HydraLoRA /
chimera per-pool). So the pool's contribution at a module is

    y(x) = ( Σ_k g_k(x) · B_k ) · A · x   =   B_eff(g(x)) · A · x

A K-expert routed pool is *exactly* a single rank-r LoRA whose B-matrix is the
gate-weighted sum of the per-expert B-heads. Freeze the gate to its dataset mean
``ḡ`` and you get **literally a plain LoRA**: ``B_eff(ḡ)·A·x``. That is the null
hypothesis "this module is not worth routing" — it would behave identically to
plain LoRA.

So "is module M worth being a router_target" reduces to: does the per-sample gate
deviation produce output the mean-gate adapter cannot?

    ρ_M  =  E_x ‖ ΔB(x)·A·x ‖²  /  E_x ‖ B_eff(ḡ)·A·x ‖² ,   ΔB(x) = Σ_k (g_k(x)−ḡ_k) B_k

  ρ ≈ 0  → routing at M is numerically indistinguishable from plain LoRA → drop M
           from router_targets (keep the rank, lose the router).
  ρ large → the gate genuinely re-steers M's output per sample → routing earns it.

Cheap to measure exactly. Because

    ‖Σ_k c_k B_k a‖²  =  Σ_{k,l} c_k c_l · tr(B_kᵀ B_l · a aᵀ)  =  Σ_{k,l} c_k c_l ⟨T_kl, S⟩

with the per-expert-pair Gram ``T_kl = B_kᵀ B_l`` (r×r, static) and the per-forward
rank-space input Gram ``S = aᵀa`` (r×r, where ``a = A·x`` flattened over batch+tokens),
the whole estimator needs only ``(gate, S)`` per forward — both O(r²). No big
activations are stored.

When ρ is low, two sub-diagnostics attribute *why* (both are necessary conditions):
  * gate uninformative  → ``gate_norm_entropy`` high but ``gate_drift`` ≈ 0 (gate
    barely moves across samples ⇒ ΔB ≈ 0 ⇒ exactly plain LoRA);
  * experts collinear   → ``expert_subspace_overlap`` ≈ 1 (the B_k span the same
    subspace ⇒ routing among clones).

The verdict for a single module is *relative*: rank ρ across every router_target
and compare cross_attn.output_proj against mlp.layer1/2.

Coverage
--------
Handles the loaded inference forms:
  * ``HydraLoRAModule`` — shared-A Hydra / FeRA-shared / σ / FEI / crossattn_emb
    routing, both ``route_per_layer=True`` (per-Linear router) and ``False``
    (network-level GlobalRouter; the broadcast gate returns through
    ``_compute_gate`` either way). If ``num_experts_content>0`` (chimera-runtime
    form) the gate is split into content/freq pools for reporting.
  * ``ChimeraHydraInferenceModule`` — dual free-form pools (content + freq),
    captured via a forward_pre_hook.

NOT covered: the Cayley *training-form* ``ChimeraHydraLoRAModule`` (run on a saved
checkpoint, which always loads as the inference form), and ``independent_A``
FeRA's ``StackedExpertsLoRAModule`` (per-expert A breaks the single-S trick — the
estimator would need per-expert input Grams; extend if needed).

Usage
-----
    python bench/router_targets/analyze_router_target_leverage.py \\
        --lora_weight output/ckpt/anima-chimera-XXXX.safetensors \\
        --dataset_dir post_image_dataset/lora \\
        --num_samples 32
"""

import argparse
import glob
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench._common import make_run_dir, write_result
from library.anima import weights as anima_utils
from library.log import setup_logging
from library.training.router_conditioning import apply_router_conditioning
from networks import lora_anima
from networks.lora_modules import ChimeraHydraInferenceModule, HydraLoRAModule

setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_weight", required=True, help="Trained routed checkpoint")
    p.add_argument(
        "--dit", default="models/diffusion_models/anima-base-v1.0.safetensors"
    )
    p.add_argument(
        "--dataset_dir",
        default="post_image_dataset/lora",
        help="Dir with cached <stem>_*_anima.npz + <stem>_anima_te.safetensors",
    )
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument(
        "--sigmas",
        default="0.05,0.15,0.3,0.45,0.6,0.75,0.9",
        help="Comma-separated flow-matching sigmas to forward at.",
    )
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--print_top_n", type=int, default=8)
    p.add_argument(
        "--worth_it_threshold",
        type=float,
        default=0.05,
        help="ρ below this ⇒ module's routing is ~plain-LoRA-equivalent (rule of thumb)",
    )
    return p.parse_args()


# --------- I/O helpers (from bench/hydralora/analyze_router_sigma_correlation.py) ---


def find_sample_stems(dataset_dir, n, seed):
    # Caches may sit flat in dataset_dir or one level deep in per-artist
    # subdirs (post_image_dataset/lora/<artist>/...), so glob recursively.
    te_files = sorted(
        glob.glob(
            os.path.join(dataset_dir, "**", "*_anima_te.safetensors"), recursive=True
        )
    )
    if not te_files:
        raise FileNotFoundError(f"no *_anima_te.safetensors found in {dataset_dir}")
    stems = []
    for te_path in te_files:
        base = os.path.basename(te_path).replace("_anima_te.safetensors", "")
        # Pair the npz from the TE file's own directory.
        npz = glob.glob(
            os.path.join(os.path.dirname(te_path), f"{base}_*_anima.npz")
        )
        if npz:
            stems.append((base, npz[0], te_path))
    if not stems:
        raise FileNotFoundError(f"no paired latent/TE samples in {dataset_dir}")
    rng = np.random.default_rng(seed)
    if len(stems) > n:
        idx = rng.choice(len(stems), size=n, replace=False)
        stems = [stems[i] for i in sorted(idx.tolist())]
    return stems


def load_latent_npz(npz_path):
    z = np.load(npz_path)
    key = next(
        (k for k in z.files if k.startswith("latents_") and "flip" not in k), None
    )
    if key is None:
        raise KeyError(f"no latents_* key in {npz_path}")
    return torch.from_numpy(z[key]).float()


def load_cached_te(te_path):
    sd = load_file(te_path)
    key = "crossattn_emb_v0" if "crossattn_emb_v0" in sd else "crossattn_emb"
    if key not in sd:
        raise KeyError(f"no crossattn_emb* key in {te_path}: {list(sd.keys())[:5]}")
    emb = sd[key].float()
    if emb.ndim == 2:
        emb = emb.unsqueeze(0)
    return emb


def classify_module(lora_name: str) -> str:
    p = lora_name.lower().replace("_", ".")
    if "self.attn.qkv" in p:
        return "self_attn.qkv"
    if "self.attn.output.proj" in p or "self.attn.out.proj" in p:
        return "self_attn.out"
    if "cross.attn.q.proj" in p:
        return "cross_attn.q"
    if (
        "cross.attn.kv.proj" in p
        or "cross.attn.k.proj" in p
        or "cross.attn.v.proj" in p
    ):
        return "cross_attn.kv"
    if "cross.attn.output.proj" in p or "cross.attn.out.proj" in p:
        return "cross_attn.out"
    if (
        "mlp.layer1" in p
        or "mlp.fc1" in p
        or "mlp.gate.proj" in p
        or "mlp.up.proj" in p
    ):
        return "mlp.layer1"
    if "mlp.layer2" in p or "mlp.fc2" in p or "mlp.down.proj" in p:
        return "mlp.layer2"
    return "other"


def block_depth(lora_name: str) -> int:
    parts = lora_name.split("_")
    try:
        return int(parts[parts.index("blocks") + 1])
    except (ValueError, IndexError):
        return -1


# --------- expert-static quantities -------------------------------------------


def pairwise_T(B: torch.Tensor) -> torch.Tensor:
    """T[k,l] = B_kᵀ B_l, shape (K, K, r, r), fp32. B is (K, out, r)."""
    Bf = B.detach().float()
    return torch.einsum("koi,loj->klij", Bf, Bf)


def subspace_overlap(B: torch.Tensor) -> float:
    """Mean off-diagonal column-space overlap of the B-heads.

    Orthonormalize each B_k's columns (QR) → Q_k (out, r). overlap_kl =
    ‖Q_kᵀ Q_l‖_F² / r ∈ [0,1]; 1 = identical subspace, 0 = orthogonal. Returns
    the mean over k≠l (0.0 for a single expert).
    """
    Bf = B.detach().float()
    K = Bf.shape[0]
    if K < 2:
        return 0.0
    Qs = []
    for k in range(K):
        q, _ = torch.linalg.qr(Bf[k])  # (out, r)
        Qs.append(q)
    r = Bf.shape[-1]
    vals = []
    for k in range(K):
        for j in range(k + 1, K):
            m = Qs[k].transpose(0, 1) @ Qs[j]
            vals.append(float((m.pow(2).sum() / r).clamp(0, 1)))
    return float(np.mean(vals)) if vals else 0.0


# --------- gate balance (marginal entropy etc.) -------------------------------


def gate_balance(gates: np.ndarray) -> dict:
    """gates: (N, K). Marginal-usage entropy + per-sample gate drift."""
    N, K = gates.shape
    eps = 1e-12
    mean_gate = gates.mean(axis=0).astype(np.float64)
    p = np.clip(mean_gate / (mean_gate.sum() + eps), eps, 1.0)
    H = -np.sum(p * np.log2(p))
    norm_H = float(H / (np.log2(K) if K > 1 else 1.0))
    # per-sample drift from the dataset-mean gate (L2), normalized by ‖ḡ‖.
    drift = np.linalg.norm(gates - mean_gate[None, :], axis=1)
    gate_drift = float(drift.mean() / (np.linalg.norm(mean_gate) + eps))
    dead = int((mean_gate < 0.5 / K).sum())
    return {
        "num_experts": int(K),
        "gate_norm_entropy": norm_H,
        "gate_drift": gate_drift,
        "dead_experts": dead,
        "mean_gate": mean_gate.tolist(),
    }


def leverage_ratio(gates: np.ndarray, S_stack: np.ndarray, T: np.ndarray) -> float:
    """ρ = mean_i ‖Σ(g_i−ḡ) B‖² / mean_i ‖Σ ḡ B‖² over forwards i.

    gates: (N, K) realized gates.  S_stack: (N, r, r) per-forward input Grams.
    T: (K, K, r, r) static B_kᵀB_l.  Uses ⟨T_kl, S_i⟩ = tr(B_kᵀB_l a aᵀ).
    """
    gbar = gates.mean(axis=0)  # (K,)
    Tt = torch.from_numpy(T) if not torch.is_tensor(T) else T
    num = 0.0
    den = 0.0
    for i in range(gates.shape[0]):
        S = (
            torch.from_numpy(S_stack[i])
            if not torch.is_tensor(S_stack[i])
            else S_stack[i]
        )
        G = torch.einsum("klij,ij->kl", Tt, S).numpy()  # (K, K)
        dg = gates[i] - gbar
        num += float(dg @ G @ dg)
        den += float(gbar @ G @ gbar)
    return float(num / den) if den > 1e-30 else 0.0


# --------- capture ------------------------------------------------------------


class Capture:
    """Per (module, pool) accumulator of (gate row, input Gram S)."""

    def __init__(self):
        self.gates = defaultdict(list)  # key -> list[(K,)]
        self.S = defaultdict(list)  # key -> list[(r,r)]

    def add(self, key, gate_rows: torch.Tensor, S: torch.Tensor):
        # gate_rows (B, K); S is the shared per-forward input Gram (r, r).
        g = gate_rows.detach().float().cpu().numpy()
        Snp = S.detach().float().cpu().numpy()
        for b in range(g.shape[0]):
            self.gates[key].append(g[b])
            self.S[key].append(Snp)


def _input_gram(lx: torch.Tensor) -> torch.Tensor:
    """S = aᵀa over flattened (batch*tokens), where a is the rank-R signal lx."""
    a = lx.detach().float()
    a2 = a.reshape(-1, a.shape[-1])  # (N, r)
    return a2.transpose(0, 1) @ a2  # (r, r)


def _find_org_linears(unet, modules):
    """Map id(adapter module) → its wrapped DiT Linear.

    ``LoRAModule.apply_to`` rebinds ``org_module.forward = self.forward`` (an
    instance attribute on the Linear) and deletes the back-reference, so a
    ``register_forward_pre_hook`` on the adapter module never fires — the DiT
    calls the *Linear's* ``__call__``, which dispatches straight to the bound
    ``forward`` without ever invoking the adapter module's own ``__call__``.
    Recover each Linear by matching the bound method stashed in its ``__dict__``.
    """
    want = {id(m): m for m in modules}
    out = {}
    for sub in unet.modules():
        fwd = sub.__dict__.get("forward")
        owner = getattr(fwd, "__self__", None)
        if owner is not None and id(owner) in want:
            out[id(owner)] = sub
    return out


def install_hooks(network, cap: Capture, unet=None):
    """Wrap Hydra ``_compute_gate`` and pre-hook chimera-inference forwards.

    Returns (pools, restore) where pools maps a pool-key →
    {"module", "pool", "B" (K,out,r), "T", "overlap", "name", "group", "block"}.
    """
    pools = {}
    restorers = []

    hydra = [m for m in network.modules() if isinstance(m, HydraLoRAModule)]
    chim = [m for m in network.modules() if isinstance(m, ChimeraHydraInferenceModule)]
    # Chimera modules inject via the org Linear's rebound ``forward`` (see
    # _find_org_linears), so the capture pre-hook must sit on the Linear.
    org_linears = _find_org_linears(unet, chim) if (chim and unet is not None) else {}
    if not hydra and not chim:
        raise RuntimeError(
            "no HydraLoRAModule / ChimeraHydraInferenceModule found — checkpoint is "
            "not a routed (Hydra/chimera) adapter, or chimera loaded as its training "
            "(Cayley) form. Run on a saved routed checkpoint."
        )
    logger.info(f"hooking {len(hydra)} Hydra + {len(chim)} chimera-inference modules")

    def register_pool(key, module, pool, B):
        pools[key] = {
            "module": module.lora_name,
            "pool": pool,
            "B": B.detach().float().cpu(),
            "T": pairwise_T(B).cpu(),
            "overlap": subspace_overlap(B),
            "name": module.lora_name,
            "group": classify_module(module.lora_name),
            "block": block_depth(module.lora_name),
        }

    # ---- Hydra: wrap _compute_gate(lx) — lx is the rank signal, gate the return.
    for m in hydra:
        K_c = int(getattr(m, "num_experts_content", 0) or 0)
        Bw = m.lora_up_weight  # (E, out, r)
        if K_c > 0:  # chimera-runtime form: split content/freq for reporting
            register_pool(f"{m.lora_name}::content", m, "content", Bw[:K_c])
            register_pool(f"{m.lora_name}::freq", m, "freq", Bw[K_c:])
        else:
            register_pool(f"{m.lora_name}::all", m, "all", Bw)

        orig = m._compute_gate

        def make(mod, orig_fn, k_c):
            name = mod.lora_name

            def wrapped(lx):
                gate = orig_fn(lx)  # (B, E)
                S = _input_gram(lx)
                if k_c > 0:
                    cap.add(f"{name}::content", gate[:, :k_c], S)
                    cap.add(f"{name}::freq", gate[:, k_c:], S)
                else:
                    cap.add(f"{name}::all", gate, S)
                return gate

            return wrapped

        m._compute_gate = make(m, orig, K_c)
        restorers.append((m, "_compute_gate", orig))

    # ---- Chimera inference: pre-hook captures x → lx_c / lx_f, gates per pool.
    for m in chim:
        register_pool(f"{m.lora_name}::content", m, "content", m.lora_up_c_weight)
        register_pool(f"{m.lora_name}::freq", m, "freq", m.lora_up_f_weight)

        # ``mod`` is the chimera adapter module (carries the routing buffers /
        # down-projections); the hook fires on its org Linear, whose first hook
        # arg is the Linear, not ``mod`` — so reference ``mod`` by closure.
        def make_pre(mod):
            name = mod.lora_name

            def pre_hook(_linear, inputs):
                x = inputs[0]
                x_lora = mod._rebalance(x).float()
                lx_c = torch.nn.functional.linear(
                    x_lora, mod.lora_down_c.weight.float()
                )
                lx_f = torch.nn.functional.linear(
                    x_lora, mod.lora_down_f.weight.float()
                )
                if mod.use_global_content_router:
                    pi_c = mod._content_routing_weights
                    if pi_c.dim() == 1:
                        pi_c = pi_c.unsqueeze(0)
                    if pi_c.shape[0] == 1 and lx_c.shape[0] > 1:
                        pi_c = pi_c.expand(lx_c.shape[0], -1)
                    pi_c = pi_c.float()
                else:
                    pi_c = mod._compute_content_gate(lx_c).float()  # (B, K_c)
                pi_f = mod._freq_routing_weights
                if pi_f.dim() == 1:
                    pi_f = pi_f.unsqueeze(0)
                pi_f = pi_f.float().expand(pi_c.shape[0], -1)
                cap.add(f"{name}::content", pi_c, _input_gram(lx_c))
                cap.add(f"{name}::freq", pi_f, _input_gram(lx_f))

            return pre_hook

        org_linear = org_linears.get(id(m))
        if org_linear is None:
            logger.warning(
                f"no org Linear found for chimera module {m.lora_name} — "
                "skipping (its routing leverage will be absent from the report)"
            )
            continue
        h = org_linear.register_forward_pre_hook(make_pre(m))
        restorers.append((h, "_handle", None))

    def restore():
        for obj, attr, orig in restorers:
            if attr == "_handle":
                obj.remove()
            else:
                setattr(obj, attr, orig)

    return pools, restore


# --------- driving routing exactly like training ------------------------------


def drive_routing(network, timesteps, noisy_5d, emb):
    apply_router_conditioning(
        network=network,
        noisy_model_input=noisy_5d,
        timesteps=timesteps,
        is_train=False,
        warmup_step=0,
        max_train_steps=1,
        crossattn_emb=emb,
    )
    # Network-level crossattn_emb GlobalRouter (fired separately in train.py).
    if getattr(network, "use_crossattn_router", False) and hasattr(
        network, "set_crossattn_routing"
    ):
        network.set_crossattn_routing(emb)
    # Chimera FreqRouter needs FEI even when use_fei_router is False (router_source
    # != "fei" on chimera). apply_router_conditioning only sets FEI under the flag.
    if getattr(network, "freq_router", None) is not None and not getattr(
        network, "use_fei_router", False
    ):
        from library.runtime.fei import compute_fei_2band, fei_sigma_low

        z = noisy_5d.squeeze(2) if noisy_5d.dim() == 5 else noisy_5d
        div = float(getattr(network.cfg, "fei_sigma_low_div", 8.0))
        fei = compute_fei_2band(
            z, fei_sigma_low(int(z.shape[-2]), int(z.shape[-1]), div)
        )
        network.set_fei(fei)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    stems = find_sample_stems(args.dataset_dir, args.num_samples, args.seed)
    sigma_list = [float(s) for s in args.sigmas.split(",") if s.strip()]
    logger.info(f"{len(stems)} samples × {len(sigma_list)} sigmas")

    logger.info(f"loading DiT from {args.dit}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.eval().requires_grad_(False)
    anima.to(device)

    logger.info(f"loading adapter from {args.lora_weight}")
    # Load via `file=` (not a pre-loaded `weights_sd=`) so the loader reads the
    # safetensors metadata: the three-axis routing stamps (ss_use_moe_style /
    # ss_route_per_layer / ss_router_source) live in `__metadata__`, which
    # `load_file()` drops — passing a pre-loaded dict makes MoE checkpoints
    # fail the three-axis stamp check in LoRANetworkCfg.from_weights.
    network, weights_sd = lora_anima.create_network_from_weights(
        multiplier=1.0,
        file=args.lora_weight,
        ae=None,
        text_encoders=[],
        unet=anima,
        weights_sd=None,
        for_inference=False,
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.missing_keys:
        logger.warning(f"missing keys: {len(info.missing_keys)}")
    if info.unexpected_keys:
        logger.warning(f"unexpected keys: {len(info.unexpected_keys)}")
    network.to(device, dtype=torch.bfloat16)
    network.eval()
    for p in network.parameters():
        p.requires_grad_(False)

    cfg = network.cfg
    axes = {
        "use_moe_style": getattr(cfg, "use_moe_style", None),
        "route_per_layer": getattr(cfg, "route_per_layer", None),
        "router_source": getattr(cfg, "router_source", None),
        "content_router_source": getattr(cfg, "content_router_source", None),
    }
    logger.info(f"routing axes: {axes}")

    cap = Capture()
    pools, restore = install_hooks(network, cap, anima)

    logger.info("running forward passes")
    n_forward = 0
    with torch.no_grad():
        for stem, npz_path, te_path in stems:
            lat = load_latent_npz(npz_path).to(device).unsqueeze(0).float()
            emb = load_cached_te(te_path).to(device, dtype=torch.bfloat16)
            h_lat, w_lat = lat.shape[-2], lat.shape[-1]
            pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
            for sigma in sigma_list:
                noise = torch.randn_like(lat)
                sv = torch.tensor(sigma, device=device).view(1, 1, 1, 1)
                noisy = ((1.0 - sv) * lat + sv * noise).to(torch.bfloat16).unsqueeze(2)
                t = torch.tensor([sigma], device=device, dtype=torch.bfloat16)
                drive_routing(network, t, noisy, emb)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _ = anima(noisy, t, emb, padding_mask=pad)
                n_forward += 1
            logger.info(f"  {stem}")
    restore()
    if hasattr(network, "clear_step_caches"):
        network.clear_step_caches()
    logger.info(f"captured {n_forward} forwards")

    # --------- per-pool metrics ---------
    per_pool = {}
    for key, meta in pools.items():
        if not cap.gates[key]:
            continue
        gates = np.stack(cap.gates[key], axis=0)  # (N, K)
        S_stack = np.stack(cap.S[key], axis=0)  # (N, r, r)
        rho = leverage_ratio(gates, S_stack, meta["T"])
        bal = gate_balance(gates)
        per_pool[key] = {
            "name": meta["name"],
            "group": meta["group"],
            "block": meta["block"],
            "pool": meta["pool"],
            "rho": rho,
            "rms_frac": float(np.sqrt(max(rho, 0.0))),
            "expert_subspace_overlap": meta["overlap"],
            **bal,
        }

    # --------- report ---------
    print("\n" + "=" * 80)
    print(f"Router-target leverage  —  {os.path.basename(args.lora_weight)}")
    print(f"  axes: {axes}")
    print(f"  pools: {len(per_pool)}   forwards: {n_forward}")
    print("=" * 80)
    print(
        "\nρ = routing-induced output variance / mean-adapter output magnitude².\n"
        "  ρ≈0 ⇒ this module routes ≈ a plain LoRA (drop from router_targets).\n"
        "  rms_frac = √ρ ≈ fractional RMS change routing adds to the adapter output."
    )

    # group × pool aggregation — the relative read the verdict hangs on.
    grp = defaultdict(list)
    for m in per_pool.values():
        grp[(m["group"], m["pool"])].append(m)

    print("\nPer (module-group, pool) — median ρ ranks where routing earns its keep:")
    rows = []
    for (g, pool), ms in grp.items():
        rho_arr = np.array([x["rho"] for x in ms])
        ov = np.array([x["expert_subspace_overlap"] for x in ms])
        dr = np.array([x["gate_drift"] for x in ms])
        rows.append(
            (
                g,
                pool,
                len(ms),
                float(np.median(rho_arr)),
                float(rho_arr.mean()),
                float(np.median(ov)),
                float(np.median(dr)),
            )
        )
    rows.sort(key=lambda r: -r[3])
    print(
        f"  {'group':<16s} {'pool':<8s} {'n':>3s} {'medρ':>8s} {'meanρ':>8s} "
        f"{'ovlp':>6s} {'drift':>6s}  verdict"
    )
    for g, pool, n, medr, meanr, ov, dr in rows:
        if medr < args.worth_it_threshold:
            why = (
                "collinear"
                if ov > 0.8
                else ("gate-flat" if dr < 0.05 else "low-leverage")
            )
            v = f"NOT worth ({why})"
        else:
            v = "worth routing"
        print(
            f"  {g:<16s} {pool:<8s} {n:>3d} {medr:>8.4f} {meanr:>8.4f} "
            f"{ov:>6.3f} {dr:>6.3f}  {v}"
        )

    # spotlight: cross_attn.out vs mlp.layer1/2 (the live router_targets question)
    print("\nrouter_targets focus — cross_attn.out vs mlp.layer[12]:")
    focus = defaultdict(dict)
    for (g, pool), ms in grp.items():
        if g in ("cross_attn.out", "mlp.layer1", "mlp.layer2"):
            focus[pool][g] = float(np.median([x["rho"] for x in ms]))
    for pool, gmap in focus.items():
        ca = gmap.get("cross_attn.out")
        mlp = [v for k, v in gmap.items() if k.startswith("mlp")]
        if ca is not None and mlp:
            mlp_med = float(np.median(mlp))
            rel = ca / mlp_med if mlp_med > 1e-9 else float("inf")
            print(
                f"  [{pool}] cross_attn.out medρ={ca:.4f}  mlp medρ={mlp_med:.4f}  "
                f"ratio={rel:.2f}×  →  "
                f"{'cross_attn.out competitive — keep it' if rel >= 0.5 else 'cross_attn.out lags — consider dropping from router_targets'}"
            )

    print("\nTop modules by ρ (routing matters most):")
    for m in sorted(per_pool.values(), key=lambda x: -x["rho"])[: args.print_top_n]:
        blk = f"blk{m['block']:02d}" if m["block"] >= 0 else "----"
        print(
            f"  {blk} [{m['pool']:<7s}] ρ={m['rho']:.4f} ovlp={m['expert_subspace_overlap']:.3f} "
            f"drift={m['gate_drift']:.3f} H={m['gate_norm_entropy']:.3f}  {m['name']}"
        )
    print("\nBottom modules by ρ (routing ≈ plain LoRA):")
    for m in sorted(per_pool.values(), key=lambda x: x["rho"])[: args.print_top_n]:
        blk = f"blk{m['block']:02d}" if m["block"] >= 0 else "----"
        print(
            f"  {blk} [{m['pool']:<7s}] ρ={m['rho']:.4f} ovlp={m['expert_subspace_overlap']:.3f} "
            f"drift={m['gate_drift']:.3f} H={m['gate_norm_entropy']:.3f}  {m['name']}"
        )

    all_rho = np.array([m["rho"] for m in per_pool.values()])
    metrics = {
        "lora_weight": args.lora_weight,
        "routing_axes": axes,
        "sigmas": sigma_list,
        "num_samples": len(stems),
        "n_forward": n_forward,
        "worth_it_threshold": args.worth_it_threshold,
        "overall": {
            "median_rho": float(np.median(all_rho)),
            "mean_rho": float(all_rho.mean()),
            "frac_below_threshold": float((all_rho < args.worth_it_threshold).mean()),
        },
        "per_group_pool": [
            {
                "group": g,
                "pool": pool,
                "n": n,
                "median_rho": medr,
                "mean_rho": meanr,
                "median_overlap": ov,
                "median_drift": dr,
            }
            for g, pool, n, medr, meanr, ov, dr in rows
        ],
        "per_pool": per_pool,
    }
    label = args.label or os.path.splitext(os.path.basename(args.lora_weight))[0]
    out_dir = make_run_dir("router_targets", label=label)
    result_path = write_result(
        out_dir, script=__file__, args=args, label=label, metrics=metrics, device=device
    )
    logger.info(f"result → {result_path}")


if __name__ == "__main__":
    main()
