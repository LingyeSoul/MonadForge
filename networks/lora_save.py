"""Save-pipeline handlers for LoRA / Ortho / Hydra / DoRA checkpoints.

Extracted from ``networks.lora_anima.LoRANetwork.save_weights``. The save
flow is expressed as a pipeline of key-triggered conversion steps plus a
variant-agnostic ``q/k/v`` defuse, capped by a variant-specific write.

Ordering is fixed and must not change:

    1. ortho_hydra_to_hydra   (keys ending ``.S_p`` with dim 3)
    2. ortho_to_lora          (keys ending ``.S_p`` with dim 2)
    3. legacy_ortho_to_lora   (keys ending ``.base_lambda``)
    4. variant dispatch:
         - save_variant == "hydra_moe" or "ortho_hydra_to_hydra":
               expand per-expert ups, hydra q/k/v split, write ``*_moe.safetensors``
         - otherwise:
               rename DoRA magnitude, standard q/k/v defuse, write ``*.safetensors``

Step 1 must run before step 2 because both key off ``.S_p`` — the 3-D
case (OrthoHydraLoRA) would otherwise be mis-reduced by the 2-D handler.
Step 3 handles legacy checkpoints from the deprecated sig-type OrthoLoRA
(see ``lora_deprecated.OrthoLoRAModule``); current training never emits
those keys.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch

from library.log import setup_logging
from networks.lora_anima.attn_fuse import ATTN_FUSE_SPECS, AttnFuseSpec
from networks.lora_modules import OrthoHydraLoRAExpModule, OrthoLoRAExpModule

setup_logging()
logger = logging.getLogger(__name__)


def _match_fused_spec(prefix: str) -> Optional[AttnFuseSpec]:
    """Return the AttnFuseSpec whose ``fused_frag`` ends ``prefix``, else None.

    Replaces the old ``_FUSED_SPLIT`` dict lookup. Iterates the small
    shared spec tuple — single source of truth with ``loading.py``.
    """
    for spec in ATTN_FUSE_SPECS:
        if prefix.endswith(spec.fused_frag):
            return spec
    return None


# ---------------------------------------------------------------------------
# Step 1: OrthoHydraLoRAExp → HydraLoRA (Cayley params → lora_down/up_weight)
# ---------------------------------------------------------------------------


def _convert_ortho_hydra_to_hydra(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    """Mutates state_dict in place.

    Converts ``.S_p`` / ``.S_q`` / ``.P_bases`` (or legacy 2-D ``.P_basis``) /
    ``.Q_basis`` / ``.lambda_layer`` keys (OrthoHydraLoRAExp) to shared
    ``.lora_down.weight`` + per-expert stacked ``.lora_up_weight`` (HydraLoRA
    runtime form).
    """
    prefixes = set()
    for key in list(state_dict.keys()):
        if not (key.endswith(".S_p") and state_dict[key].dim() == 3):
            continue
        prefix = key[: -len(".S_p")]
        # Discriminator vs StackedExperts ortho: OrthoHydra's S_q is 2-D
        # ``(r, r)`` (shared across experts), StackedExperts ortho's S_q is
        # 3-D ``(E, r, r)`` (per-expert). The 3-D case is handled by
        # ``_convert_ortho_stacked_experts`` which runs before this step.
        S_q_key = f"{prefix}.S_q"
        if S_q_key not in state_dict:
            continue
        if state_dict[S_q_key].dim() != 2:
            continue
        prefixes.add(prefix)

    for prefix in prefixes:
        S_p = state_dict[f"{prefix}.S_p"]  # (E, r, r)
        S_q = state_dict[f"{prefix}.S_q"]  # (r, r)
        # Per-expert disjoint bases (new) or legacy shared basis (old ckpts).
        P_bases = state_dict.get(f"{prefix}.P_bases")
        if P_bases is None:
            P_bases = state_dict[f"{prefix}.P_basis"]  # (out, r) legacy
        Q_basis = state_dict[f"{prefix}.Q_basis"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]  # (1, r)
        alpha = state_dict.get(f"{prefix}.alpha")
        save_dtype = dtype if dtype is not None else P_bases.dtype

        R_q = OrthoHydraLoRAExpModule._cayley(S_q.float())  # (r, r)
        Q_eff = R_q @ Q_basis.float()  # (r, in)

        R_p = OrthoHydraLoRAExpModule._cayley(S_p.float())  # (E, r, r)
        if P_bases.dim() == 3:
            # (E, out, r) @ (E, r, r) = (E, out, r)
            P_eff = P_bases.float() @ R_p
        else:
            # legacy shared (out, r): broadcast over experts
            P_eff = P_bases.float().unsqueeze(0) @ R_p  # (E, out, r)

        # sqrt-split lambda so ΔW = P @ diag(λ) @ Q is preserved bit-exactly
        lam_1d = lam.squeeze(0).float()
        lam_abs = lam_1d.abs()
        lam_sign = lam_1d.sign()
        lam_sqrt = lam_abs.sqrt()

        lora_down = (
            (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
        )
        lora_up_weight = (
            (P_eff * (lam_sqrt * lam_sign).unsqueeze(0).unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        for suffix in ("S_p", "S_q", "lambda_layer", "P_basis", "P_bases", "Q_basis"):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        state_dict[f"{prefix}.lora_up_weight"] = lora_up_weight
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


# ---------------------------------------------------------------------------
# Step 2: OrthoLoRA → standard LoRA (Cayley params → lora_down/up)
# ---------------------------------------------------------------------------


def _convert_ortho_to_lora(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".S_p"):
            prefixes.add(key[: -len(".S_p")])

    for prefix in prefixes:
        S_p = state_dict[f"{prefix}.S_p"]
        S_q = state_dict[f"{prefix}.S_q"]
        P_basis = state_dict[f"{prefix}.P_basis"]
        Q_basis = state_dict[f"{prefix}.Q_basis"]
        lam = state_dict[f"{prefix}.lambda_layer"]  # (1, r)
        alpha = state_dict.get(f"{prefix}.alpha")
        save_dtype = dtype if dtype is not None else P_basis.dtype

        R_p = OrthoLoRAExpModule._cayley(S_p.float())
        R_q = OrthoLoRAExpModule._cayley(S_q.float())
        P_eff = P_basis.float() @ R_p  # (out, r)
        Q_eff = R_q @ Q_basis.float()  # (r, in)

        # ΔW = P_eff @ diag(λ) @ Q_eff is already exactly rank r — factor directly.
        lam_abs = lam.squeeze(0).float().abs()
        lam_sign = lam.squeeze(0).float().sign()
        lam_sqrt = lam_abs.sqrt()
        lora_up = (
            (P_eff * (lam_sqrt * lam_sign).unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
        )

        for suffix in ("S_p", "S_q", "lambda_layer", "P_basis", "Q_basis"):
            state_dict.pop(f"{prefix}.{suffix}", None)
        # inv_scale stays — shared buffer, not an ortho-exp-only key.

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


# ---------------------------------------------------------------------------
# Step 3: Legacy OrthoLoRA (sig-type) → standard LoRA (SVD of ΔW in 2r-dim subspace)
# ---------------------------------------------------------------------------


def _convert_legacy_ortho_to_lora(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".base_lambda"):
            prefixes.add(key[: -len(".base_lambda")])

    for prefix in prefixes:
        P = state_dict[f"{prefix}.p_layer.weight"]  # (out, r)
        Q = state_dict[f"{prefix}.q_layer.weight"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]
        P_base = state_dict[f"{prefix}.base_p_weight"]
        Q_base = state_dict[f"{prefix}.base_q_weight"]
        lam_base = state_dict[f"{prefix}.base_lambda"]
        alpha = state_dict.get(f"{prefix}.alpha")
        rank = Q.shape[0]

        # ΔW = P·diag(λ)·Q − P_base·diag(λ_base)·Q_base is rank ≤ 2r. SVD
        # works in the small 2r-dim column/row space instead of on the full
        # (out × in) matrix: ΔW = [P|P_base] @ M @ [Q; Q_base], then SVD of M.
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dtype = dtype if dtype is not None else P.dtype

        P_cat = torch.cat([P, P_base], dim=1).float().to(svd_device)  # (out, 2r)
        Q_cat = torch.cat([Q, Q_base], dim=0).float().to(svd_device)  # (2r, in)
        lam_diag = torch.diag(lam.squeeze(0).float().to(svd_device))
        lam_base_diag = torch.diag(lam_base.squeeze(0).float().to(svd_device))

        M = torch.zeros(2 * rank, 2 * rank, device=svd_device)
        M[:rank, :rank] = lam_diag
        M[rank:, rank:] = -lam_base_diag

        Qp, Rp = torch.linalg.qr(P_cat)
        Qq, Rq = torch.linalg.qr(Q_cat.T)

        core = Rp @ M @ Rq.T
        Uc, Sc, Vhc = torch.linalg.svd(core)

        lora_up = (
            (Qp @ Uc[:, :rank] * Sc[:rank].sqrt().unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Sc[:rank].sqrt().unsqueeze(1) * Vhc[:rank, :] @ Qq.T)
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        for suffix in (
            "p_layer.weight",
            "q_layer.weight",
            "lambda_layer",
            "base_p_weight",
            "base_q_weight",
            "base_lambda",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


# ---------------------------------------------------------------------------
# Step 4: q/k/v defuse — standard LoRA variant
# ---------------------------------------------------------------------------


def _rename_dora_and_defuse_standard(
    state_dict: Dict[str, torch.Tensor],
) -> None:
    # DoRA: rename magnitude → dora_scale for ComfyUI; drop internal buffer.
    for key in list(state_dict.keys()):
        if key.endswith(".magnitude"):
            new_key = key.replace(".magnitude", ".dora_scale")
            state_dict[new_key] = state_dict.pop(key)
        elif key.endswith("._org_weight_norm"):
            del state_dict[key]

    # Split fused qkv_proj / kv_proj into per-component weights.
    fused_groups: List[tuple] = []
    for key in list(state_dict.keys()):
        if not key.endswith(".lora_down.weight"):
            continue
        prefix = key.removesuffix(".lora_down.weight")
        spec = _match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        suffixes = spec.component_letters
        n = len(suffixes)
        down = state_dict.pop(f"{prefix}.lora_down.weight")
        up = state_dict.pop(f"{prefix}.lora_up.weight")
        alpha = state_dict.pop(f"{prefix}.alpha", None)
        dora_scale = state_dict.pop(f"{prefix}.dora_scale", None)

        up_chunks = up.chunk(n, dim=0)
        dora_chunks = (
            dora_scale.chunk(n, dim=0) if dora_scale is not None else [None] * n
        )

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for letter, up_chunk, dora_chunk in zip(suffixes, up_chunks, dora_chunks):
            new_prefix = base_prefix + spec.component_frag(letter)
            state_dict[f"{new_prefix}.lora_down.weight"] = down.clone()
            state_dict[f"{new_prefix}.lora_up.weight"] = up_chunk
            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            if dora_chunk is not None:
                state_dict[f"{new_prefix}.dora_scale"] = dora_chunk


# ---------------------------------------------------------------------------
# Step 4 (hydra variant): expand per-expert ups, q/k/v split per-expert
# ---------------------------------------------------------------------------


def _build_hydra_moe_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Build the _moe.safetensors payload from a state_dict whose hydra keys
    are in the training-runtime form (stacked ``.lora_up_weight``).

    Expands ``.lora_up_weight`` of shape (E, out, r) back into per-expert
    ``.lora_ups.N.weight`` keys, then splits fused qkv/kv attention prefixes
    per-expert so the ComfyUI HydraLoRA custom node sees separate q/k/v
    component names.
    """
    hydra_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        v = v.detach().clone().to("cpu")
        if k.endswith(".lora_up_weight"):
            prefix = k.removesuffix(".lora_up_weight")
            for i in range(v.size(0)):
                hydra_sd[f"{prefix}.lora_ups.{i}.weight"] = v[i]
        else:
            hydra_sd[k] = v

    # Split fused attention prefixes per-expert. lora_down / alpha / router /
    # inv_scale are shared across q/k/v (same layer input, same routing
    # decision), so clone them into each split component.
    hydra_fused_groups: List[tuple] = []
    for key in list(hydra_sd.keys()):
        if not key.endswith(".lora_down.weight"):
            continue
        prefix = key.removesuffix(".lora_down.weight")
        spec = _match_fused_spec(prefix)
        if spec is not None:
            hydra_fused_groups.append((prefix, spec))

    for prefix, spec in hydra_fused_groups:
        suffixes = spec.component_letters
        n = len(suffixes)
        down = hydra_sd.pop(f"{prefix}.lora_down.weight")
        alpha = hydra_sd.pop(f"{prefix}.alpha", None)
        router_w = hydra_sd.pop(f"{prefix}.router.weight", None)
        router_b = hydra_sd.pop(f"{prefix}.router.bias", None)
        inv_scale = hydra_sd.pop(f"{prefix}.inv_scale", None)
        # σ-conditional router MLP (optional). Shared across q/k/v for the
        # same reason as router.weight/bias — routing is driven by the same
        # pooled layer input.
        sigma_mlp_keys = [
            k
            for k in list(hydra_sd.keys())
            if k.startswith(f"{prefix}.sigma_mlp.")
        ]
        sigma_mlp_state = {k: hydra_sd.pop(k) for k in sigma_mlp_keys}

        ups_keys = sorted(
            (
                k
                for k in list(hydra_sd.keys())
                if k.startswith(f"{prefix}.lora_ups.") and k.endswith(".weight")
            ),
            key=lambda k: int(
                k.removeprefix(f"{prefix}.lora_ups.").removesuffix(".weight")
            ),
        )
        ups = [hydra_sd.pop(k) for k in ups_keys]
        ups_chunked = [u.chunk(n, dim=0) for u in ups]

        # Plain-LoRA leg (present when hydra_router_layers excluded this
        # module — the fused qkv then carries the standard ``.lora_up.weight``
        # and optional DoRA ``.dora_scale`` instead of the hydra stack).
        # Split these per-component so q/k/v keys are consistent with the
        # already-split ``.lora_down.weight`` above.
        plain_up = hydra_sd.pop(f"{prefix}.lora_up.weight", None)
        plain_up_chunks = (
            plain_up.chunk(n, dim=0) if plain_up is not None else None
        )
        dora_scale = hydra_sd.pop(f"{prefix}.dora_scale", None)
        dora_chunks = (
            dora_scale.chunk(n, dim=0) if dora_scale is not None else None
        )

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for ci, letter in enumerate(suffixes):
            new_prefix = base_prefix + spec.component_frag(letter)
            hydra_sd[f"{new_prefix}.lora_down.weight"] = down.clone()
            for ei, u_chunks in enumerate(ups_chunked):
                hydra_sd[f"{new_prefix}.lora_ups.{ei}.weight"] = (
                    u_chunks[ci].contiguous().clone()
                )
            if plain_up_chunks is not None:
                hydra_sd[f"{new_prefix}.lora_up.weight"] = (
                    plain_up_chunks[ci].contiguous().clone()
                )
            if dora_chunks is not None:
                hydra_sd[f"{new_prefix}.dora_scale"] = dora_chunks[ci].clone()
            if alpha is not None:
                hydra_sd[f"{new_prefix}.alpha"] = alpha.clone()
            if router_w is not None:
                hydra_sd[f"{new_prefix}.router.weight"] = router_w.clone()
            if router_b is not None:
                hydra_sd[f"{new_prefix}.router.bias"] = router_b.clone()
            if inv_scale is not None:
                hydra_sd[f"{new_prefix}.inv_scale"] = inv_scale.clone()
            for k, v in sigma_mlp_state.items():
                subkey = k.removeprefix(f"{prefix}.")
                hydra_sd[f"{new_prefix}.{subkey}"] = v.clone()

    if dtype is not None:
        hydra_sd = {k: v.to(dtype) for k, v in hydra_sd.items()}
    return hydra_sd


# ---------------------------------------------------------------------------
# Step 1b: ortho StackedExperts → free StackedExperts (per-expert lora_down/up)
# ---------------------------------------------------------------------------


def _convert_ortho_stacked_experts(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    """Mutate state_dict in place.

    Ortho ``StackedExpertsLoRAModule`` saves with per-expert ``S_p``,
    ``S_q``, ``lambda_layer`` plus shared ``P_basis``, ``Q_basis`` — the
    Cayley-rotated SVD layout. Convert to the free per-expert
    ``lora_down_weight (E, r, in)`` + ``lora_up_weight (E, out, r)``
    layout so the on-disk file matches the free StackedExperts shape and
    can be loaded by either mode (the runtime ortho path is reachable
    only via the cfg — checkpoints distill at save).

    Discriminator from OrthoHydra: ``S_q`` here is 3-D ``(E, r, r)``,
    versus OrthoHydra's 2-D ``S_q (r, r)``. So we detect by checking
    both ``.S_p`` and ``.S_q`` are 3-D for the same prefix.
    """
    prefixes = set()
    for key in list(state_dict.keys()):
        if not key.endswith(".S_q"):
            continue
        if state_dict[key].dim() != 3:
            continue
        prefix = key[: -len(".S_q")]
        if state_dict.get(f"{prefix}.S_p") is None or state_dict[f"{prefix}.S_p"].dim() != 3:
            continue
        prefixes.add(prefix)

    for prefix in prefixes:
        S_p = state_dict[f"{prefix}.S_p"]  # (E, r, r)
        S_q = state_dict[f"{prefix}.S_q"]  # (E, r, r)
        P_basis = state_dict[f"{prefix}.P_basis"]  # (out, r)
        Q_basis = state_dict[f"{prefix}.Q_basis"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]  # (E, r)
        alpha = state_dict.get(f"{prefix}.alpha")
        save_dtype = dtype if dtype is not None else P_basis.dtype

        # Batched Cayley over S_p and S_q for every expert. Same parameter-free
        # transform as ``StackedExpertsLoRAModule._cayley_rotations`` but in
        # fp32 here for save-time stability.
        E, r, _ = S_p.shape
        skew = torch.cat([S_q.float(), S_p.float()], dim=0)  # (2E, r, r)
        A = skew - skew.transpose(-2, -1)
        eye = torch.eye(r, dtype=torch.float32, device=skew.device)
        R = torch.linalg.solve(eye + A, eye - A)  # (2E, r, r)
        R_q = R[:E]  # (E, r, r)
        R_p = R[E:]  # (E, r, r)

        # Per-expert effective bases.
        # Q_eff[e]  = R_q[e] @ Q_basis  → (r, in)
        # P_eff[e]  = P_basis @ R_p[e]  → (out, r)
        Q_eff = torch.einsum("erj,ji->eri", R_q, Q_basis.float())
        P_eff = torch.einsum("oj,ejr->eor", P_basis.float(), R_p)

        # Sqrt-split λ between sides so ΔW = P_eff @ diag(λ) @ Q_eff is
        # preserved bit-exactly under the (down, up) factorization.
        lam_abs = lam.float().abs()
        lam_sign = lam.float().sign()
        lam_sqrt = lam_abs.sqrt()  # (E, r)

        # lora_down_weight: (E, r, in) — absorb |sqrt(λ)| into Q_eff's rank axis.
        lora_down_weight = (
            Q_eff * lam_sqrt.unsqueeze(-1)
        ).to(save_dtype).cpu().contiguous()
        # lora_up_weight: (E, out, r) — absorb sign*sqrt(λ) into P_eff's rank axis.
        lora_up_weight = (
            P_eff * (lam_sqrt * lam_sign).unsqueeze(1)
        ).to(save_dtype).cpu().contiguous()

        for suffix in ("S_p", "S_q", "lambda_layer", "P_basis", "Q_basis", "_eye_r"):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_down_weight"] = lora_down_weight
        state_dict[f"{prefix}.lora_up_weight"] = lora_up_weight
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


# ---------------------------------------------------------------------------
# Step 4 (stacked-experts variant): per-expert ups + downs, q/k/v split
# ---------------------------------------------------------------------------


def _build_stacked_experts_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Build the StackedExperts ``_moe.safetensors`` payload.

    Independent-A variant of :func:`_build_hydra_moe_state_dict`: BOTH
    ``lora_up_weight (E, out, r)`` AND ``lora_down_weight (E, r, in)`` are
    expanded per-expert (``.lora_ups.{i}.weight`` / ``.lora_downs.{i}.weight``),
    then fused attention prefixes are split per-expert per-component so q/k/v
    keys land on separate components.

    No router/sigma_mlp/inv_scale handling here — the GlobalRouter lives at
    network top-level (``global_router.*``), not per-Linear.
    """
    sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        v = v.detach().clone().to("cpu")
        if k.endswith(".lora_up_weight"):
            prefix = k.removesuffix(".lora_up_weight")
            for i in range(v.size(0)):
                sd[f"{prefix}.lora_ups.{i}.weight"] = v[i]
        elif k.endswith(".lora_down_weight"):
            prefix = k.removesuffix(".lora_down_weight")
            for i in range(v.size(0)):
                sd[f"{prefix}.lora_downs.{i}.weight"] = v[i]
        else:
            sd[k] = v

    # Per-expert q/k/v split for fused attention prefixes.
    fused_groups: List[tuple] = []
    for key in list(sd.keys()):
        if not key.endswith(".lora_downs.0.weight"):
            continue
        prefix = key.removesuffix(".lora_downs.0.weight")
        spec = _match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        suffixes = spec.component_letters
        n = len(suffixes)
        alpha = sd.pop(f"{prefix}.alpha", None)

        # Collect per-expert ups + downs.
        ups_keys = sorted(
            (k for k in list(sd.keys()) if k.startswith(f"{prefix}.lora_ups.") and k.endswith(".weight")),
            key=lambda k: int(k.removeprefix(f"{prefix}.lora_ups.").removesuffix(".weight")),
        )
        downs_keys = sorted(
            (k for k in list(sd.keys()) if k.startswith(f"{prefix}.lora_downs.") and k.endswith(".weight")),
            key=lambda k: int(k.removeprefix(f"{prefix}.lora_downs.").removesuffix(".weight")),
        )
        ups = [sd.pop(k) for k in ups_keys]
        downs = [sd.pop(k) for k in downs_keys]
        # Per-expert chunk-of-out_dim across q/k/v components.
        ups_chunked = [u.chunk(n, dim=0) for u in ups]

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for ci, letter in enumerate(suffixes):
            new_prefix = base_prefix + spec.component_frag(letter)
            for ei, u_chunks in enumerate(ups_chunked):
                sd[f"{new_prefix}.lora_ups.{ei}.weight"] = (
                    u_chunks[ci].contiguous().clone()
                )
            # Downs are shared across q/k/v inputs (the fused Linear sees one
            # input vector), so clone each expert's down into every component.
            for ei, d in enumerate(downs):
                sd[f"{new_prefix}.lora_downs.{ei}.weight"] = d.clone()
            if alpha is not None:
                sd[f"{new_prefix}.alpha"] = alpha.clone()

    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
) -> None:
    """Run the full save pipeline: conversion chain + variant write.

    Mutates ``state_dict`` in place.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Steps 1–3: key-triggered conversions. Ordering is load-bearing:
    #   * ortho_stacked_experts runs first (3-D S_q only).
    #   * ortho_hydra_to_hydra (2-D S_q + 3-D S_p) then.
    #   * ortho_to_lora (2-D S_p + 2-D S_q) last among the S_p paths.
    # The S_q dimensionality is the discriminator.
    _convert_ortho_stacked_experts(state_dict, dtype)
    _convert_ortho_hydra_to_hydra(state_dict, dtype)
    _convert_ortho_to_lora(state_dict, dtype)
    _convert_legacy_ortho_to_lora(state_dict, dtype)

    # Variant dispatch. ``stacked_experts_global_fei`` writes the
    # independent-A per-expert (lora_downs.{i}, lora_ups.{i}) layout;
    # ``hydra_moe`` / ``ortho_hydra_to_hydra`` write the shared-A Hydra
    # layout (single lora_down, lora_ups.{i}). Auto-fallback on any
    # ``.lora_up_weight`` key for backward-compat with paths that don't
    # plumb ``save_variant`` through.
    is_stacked_experts_variant = save_variant == "stacked_experts_global_fei"
    is_hydra_variant = (
        save_variant in ("hydra_moe", "ortho_hydra_to_hydra")
        or any(k.endswith(".lora_up_weight") for k in state_dict.keys())
    ) and not is_stacked_experts_variant

    if is_stacked_experts_variant:
        # StackedExperts: discriminator is per-expert lora_down_weight
        # (3-D). Loader sniffs ``.lora_downs.{i}.weight`` to disambiguate
        # from Hydra's shared down at load time.
        se_file = os.path.splitext(file)[0] + "_moe.safetensors"
        se_sd = _build_stacked_experts_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(se_sd, se_file, metadata or {})
        logger.info(f"StackedExperts full format saved to {se_file}")
        return

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = _build_hydra_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    # Standard (lora / ortho / dora) path.
    _rename_dora_and_defuse_standard(state_dict)

    if dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key].detach().clone().to("cpu").to(dtype)
            state_dict[key] = v

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(
            state_dict, metadata
        )
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)
