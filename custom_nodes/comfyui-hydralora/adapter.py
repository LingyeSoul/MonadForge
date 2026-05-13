"""LoRA / HydraLoRA / ReFT loading and application for the Anima adapter node.

A single safetensors file may contain any combination of the three; each is
auto-detected from key patterns. Plain LoRA goes through ComfyUI's
weight-patch path. HydraLoRA and ReFT are applied as per-Linear / per-block
``forward_hook``s swapped in via ``ModelPatcher.add_object_patch``
(overriding ``forward`` strands block weights on CPU under ComfyUI's
cast-weights path). Hydra hooks reproduce the trained
``HydraLoRAModule.forward`` exactly — per-sample router gate, per-expert
``lora_up`` blend — so style separation actually fires at inference time
instead of being averaged out by a uniform bake.
"""

import json
import logging
import math
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Cache: path -> parsed adapter bundle ({"lora": dict|None, "hydra": dict|None, "reft": dict|None}).
_adapter_cache: Dict[str, dict] = {}

_REFT_KEY_RE = re.compile(r"^reft_unet_blocks_(\d+)\.(.+)$")


# ---------------------------------------------------------------------------
# FEI (Frequency-Energy Index) — FeRA-style content-aware routing key.
#
# 2-band Laplacian decomposition of the current latent z_t on the simplex.
# Mirrors ``library/runtime/fei.py`` from the training tree so the node stays
# standalone (no anima_lora source on the runtime path).
# ---------------------------------------------------------------------------

_GAUSS_CACHE: Dict[tuple, torch.Tensor] = {}


def _gaussian_kernel_1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (round(sigma, 4), device, dtype)
    cached = _GAUSS_CACHE.get(key)
    if cached is not None:
        return cached
    half = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    _GAUSS_CACHE[key] = k
    return k


def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian along (H, W) with reflect padding. ``x`` is ``[B, C, H, W]``."""
    if sigma <= 0:
        return x
    k1 = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    K = k1.numel()
    pad = K // 2
    C = x.shape[1]
    kw = k1.view(1, 1, 1, K).expand(C, 1, 1, K).contiguous()
    kh = k1.view(1, 1, K, 1).expand(C, 1, K, 1).contiguous()
    x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, kw, groups=C)
    x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="reflect")
    x = torch.nn.functional.conv2d(x, kh, groups=C)
    return x


def _compute_fei_2band(z: torch.Tensor, sigma_low: float) -> torch.Tensor:
    """Return ``[B, 2]`` simplex ``(e_low, e_high)`` on a 4D latent.

    Promoted to fp32 internally — the squared norm + simplex divide can
    underflow at small ``e_low`` in bf16. Negligible cost on
    ``H_lat·W_lat ≈ 4096`` patch grids vs the DiT forward.
    """
    z = z.float()
    lp = _gaussian_blur_2d(z, sigma_low)
    e_low = lp.pow(2).flatten(1).sum(-1)
    e_high = (z - lp).pow(2).flatten(1).sum(-1)
    energies = torch.stack([e_low, e_high], dim=-1)
    return energies / energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _fei_sigma_low(h_lat: int, w_lat: int, fei_sigma_low_div: float) -> float:
    """σ_low scaled to ``min(H_lat, W_lat)`` so the band semantic is bucket-invariant."""
    return float(min(h_lat, w_lat)) / float(fei_sigma_low_div)


def _parse_reft(weights_sd: Dict[str, torch.Tensor]) -> Optional[Dict[int, dict]]:
    """Group ReFT keys by block index. Returns None if no ReFT keys present."""
    by_idx: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, value in weights_sd.items():
        m = _REFT_KEY_RE.match(key)
        if m is None:
            continue
        idx = int(m.group(1))
        by_idx.setdefault(idx, {})[m.group(2)] = value

    if not by_idx:
        return None

    reft: Dict[int, dict] = {}
    for idx, d in by_idx.items():
        if "rotate_layer.weight" not in d or "learned_source.weight" not in d:
            logger.warning(
                f"ReFT block {idx} is missing rotate_layer.weight or "
                f"learned_source.weight -- skipping."
            )
            continue
        rotate = d["rotate_layer.weight"]
        reft_dim = rotate.size(0)
        alpha_t = d.get("alpha")
        if alpha_t is None:
            alpha = float(reft_dim)
        else:
            alpha = float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
        reft[idx] = {
            "rotate": rotate,                              # (reft_dim, x_dim)
            "source_w": d["learned_source.weight"],        # (reft_dim, x_dim)
            "source_b": d["learned_source.bias"],          # (reft_dim,)
            "scale": alpha / reft_dim,
        }
    return reft or None


def _parse_hydra(weights_sd: Dict[str, torch.Tensor]) -> Optional[dict]:
    """Group Hydra multi-head keys. Returns None if no per-expert ups found.

    Captures router (``router.weight`` / ``router.bias``). σ-conditional
    routing is driven by the router input directly — sinusoidal(σ) is
    concatenated onto the pooled rank-R vector, so the router weight is
    ``Linear(rank + sigma_feature_dim, E)``; σ dim is recovered downstream
    from ``router_w.shape[1] - lora_down.shape[0]``. The legacy additive
    ``sigma_mlp.*`` bias path was removed on the training side (see
    ``docs/methods/hydra-lora.md`` §Fixes); no current checkpoint writes
    those keys, so we ignore them.

    Only prefixes with per-expert ``lora_ups`` are returned; prefixes that
    are plain LoRA (``lora_up.weight`` singular) are left to the
    ``_extract_lora_sd`` path.
    """
    modules: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if key.startswith("reft_"):
            continue
        parts = key.split(".")
        prefix = parts[0]
        rest = ".".join(parts[1:])
        mod = modules.setdefault(prefix, {})
        if rest == "lora_down.weight":
            mod["lora_down"] = value
        elif rest.startswith("lora_ups.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            mod.setdefault("lora_ups", {})[idx] = value
        elif rest == "alpha":
            mod["alpha"] = value
        elif rest == "inv_scale":
            mod["inv_scale"] = value
        elif rest == "router.weight":
            mod["router_w"] = value
        elif rest == "router.bias":
            mod["router_b"] = value

    # Keep only real hydra modules (have per-expert ups). Prefixes that only
    # carry plain LoRA keys (lora_up.weight, singular) flow through the
    # standard LoRA path; including them here would just produce noisy
    # "missing lora_ups" skip warnings.
    hydra_only: Dict[str, dict] = {
        prefix: mod for prefix, mod in modules.items() if "lora_ups" in mod
    }
    if not hydra_only:
        return None
    num_experts = max(max(m["lora_ups"].keys()) + 1 for m in hydra_only.values())
    return {"num_experts": num_experts, "modules": hydra_only}


def _extract_lora_sd(
    weights_sd: Dict[str, torch.Tensor],
) -> Optional[Dict[str, torch.Tensor]]:
    """Pull standard LoRA keys (lora_down/lora_up/alpha/dora_scale).
    Returns None if no lora_up.weight keys are present.

    Hydra-prefix keys are excluded *entirely* — not just the per-expert
    ``.lora_ups.*`` ones. A hydra module's ``lora_down.weight`` / ``router.*``
    / ``alpha`` are orphans from ComfyUI's ``load_lora`` perspective and
    would surface as "lora key not loaded" warnings if passed through; they
    belong to the hydra path.
    """
    hydra_prefixes = {
        key.rsplit(".lora_ups.", 1)[0]
        for key in weights_sd
        if ".lora_ups." in key
    }

    out: Dict[str, torch.Tensor] = {}
    has_up = False
    for key, value in weights_sd.items():
        if key.startswith("reft_"):
            continue
        if key.endswith(".lora_up_weight"):
            continue  # Hydra stacked-ups runtime form (shouldn't appear post-save)
        prefix = key.split(".", 1)[0]
        if prefix in hydra_prefixes:
            continue  # Hydra module — handled via _parse_hydra
        out[key] = value
        if key.endswith(".lora_up.weight"):
            has_up = True
    return out if has_up else None


def load_adapter(file_path: str) -> dict:
    """Parse an Anima adapter file once, cache by path."""
    if file_path in _adapter_cache:
        return _adapter_cache[file_path]

    from safetensors import safe_open
    from safetensors.torch import load_file

    weights_sd = load_file(file_path)
    with safe_open(file_path, framework="pt") as f:
        file_metadata = dict(f.metadata() or {})

    if any(k.startswith("_hydra_router") for k in weights_sd.keys()):
        raise ValueError(
            f"{file_path} uses the deprecated global HydraLoRA router format. "
            "Retrain with the current codebase."
        )

    # Plan2 stacked_experts_global_fei (FeRA cell of the three-axis routing
    # matrix) saves with ``global_router.net.*`` + per-expert split
    # ``.lora_downs.{i}.weight`` / ``.lora_ups.{i}.weight`` — incompatible
    # with the shared-A Hydra parser below. Detect early and point users at
    # AnimaFeraLoader instead of letting them hit the unhelpful "missing
    # lora_down/lora_ups" skip path.
    is_stacked_experts = (
        file_metadata.get("ss_network_spec") == "stacked_experts_global_fei"
        or (
            any(k.startswith("global_router.net.") for k in weights_sd)
            and any(
                ".lora_downs." in k and k.endswith(".weight")
                for k in weights_sd
            )
        )
    )
    if is_stacked_experts:
        raise ValueError(
            f"{file_path} is a plan2 stacked_experts_global_fei (FeRA) "
            "checkpoint — load it with AnimaFeraLoader, not AnimaAdapterLoader. "
            "Independent-A stacked experts + a network-level GlobalRouter "
            "use a different application path than HydraLoRA's per-Linear "
            "shared-A router."
        )

    hydra = _parse_hydra(weights_sd)
    if hydra is not None:
        # Hard σ-band partition is non-persistent (training-side `_expert_band`
        # buffer is registered with persistent=False), so it has to come back
        # from metadata. Without this, soft routing operates over all E experts
        # at inference and the partition trained into the router weights is
        # silently ignored.
        band_on = (
            str(file_metadata.get("ss_specialize_experts_by_sigma_buckets", "")).lower()
            == "true"
        )
        num_buckets = (
            int(file_metadata["ss_num_sigma_buckets"])
            if band_on and "ss_num_sigma_buckets" in file_metadata
            else 0
        )
        if band_on and hydra["num_experts"] % max(num_buckets, 1) != 0:
            logger.warning(
                f"{file_path}: σ-band metadata declares num_sigma_buckets="
                f"{num_buckets} but num_experts={hydra['num_experts']} is not "
                "divisible -- disabling band partition."
            )
            band_on = False
            num_buckets = 0
        # Optional custom σ-bucket boundaries: length B+1 list of edges from
        # 0.0 to 1.0. When absent, defaults to uniform linspace(0, 1, B+1) —
        # matches training's behaviour with the `sigma_bucket_boundaries`
        # kwarg unset.
        boundaries: Optional[List[float]] = None
        if band_on and "ss_sigma_bucket_boundaries" in file_metadata:
            try:
                parsed = json.loads(file_metadata["ss_sigma_bucket_boundaries"])
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    f"{file_path}: ss_sigma_bucket_boundaries is malformed "
                    f"({exc}) — falling back to uniform σ-bucket edges."
                )
                parsed = None
            if (
                isinstance(parsed, list)
                and len(parsed) == num_buckets + 1
            ):
                boundaries = [float(v) for v in parsed]
            elif parsed is not None:
                logger.warning(
                    f"{file_path}: ss_sigma_bucket_boundaries length "
                    f"{len(parsed) if isinstance(parsed, list) else 'N/A'} "
                    f"!= num_sigma_buckets+1={num_buckets + 1} — falling back "
                    "to uniform σ-bucket edges."
                )
        hydra["sigma_band_partition"] = band_on
        hydra["num_sigma_buckets"] = num_buckets
        hydra["sigma_bucket_boundaries"] = boundaries

        # FeRA-style FEI router (content-aware routing). Trained when
        # `use_fei_router=true`; the router's input dim then has
        # `fei_feature_dim` columns past the σ-feature slice, fed per-step
        # 2-band Laplacian energies of the current latent. Without these
        # metadata flags we cannot distinguish FEI columns from σ-feature
        # columns by shape alone — they'd be misinterpreted as sinusoidal(σ)
        # columns and the gate would route on the wrong signal.
        fei_on = (
            str(file_metadata.get("ss_use_fei_router", "")).lower() == "true"
        )
        try:
            fei_feature_dim = (
                int(file_metadata["ss_fei_feature_dim"])
                if fei_on and "ss_fei_feature_dim" in file_metadata
                else 0
            )
        except (TypeError, ValueError):
            logger.warning(
                f"{file_path}: ss_fei_feature_dim is malformed "
                f"({file_metadata.get('ss_fei_feature_dim')!r}) — disabling FEI router."
            )
            fei_on = False
            fei_feature_dim = 0
        try:
            fei_sigma_low_div = (
                float(file_metadata["ss_fei_sigma_low_div"])
                if fei_on and "ss_fei_sigma_low_div" in file_metadata
                else 8.0  # training-side default (configs/methods/fera.toml)
            )
        except (TypeError, ValueError):
            logger.warning(
                f"{file_path}: ss_fei_sigma_low_div is malformed "
                f"({file_metadata.get('ss_fei_sigma_low_div')!r}) — using default 8.0."
            )
            fei_sigma_low_div = 8.0
        hydra["use_fei_router"] = fei_on and fei_feature_dim > 0
        hydra["fei_feature_dim"] = fei_feature_dim if hydra["use_fei_router"] else 0
        hydra["fei_sigma_low_div"] = fei_sigma_low_div

    bundle = {
        "path": file_path,
        "lora": _extract_lora_sd(weights_sd),
        "hydra": hydra,
        "reft": _parse_reft(weights_sd),
    }
    _adapter_cache[file_path] = bundle

    summary = []
    if bundle["lora"] is not None:
        summary.append(
            f"{sum(1 for k in bundle['lora'] if k.endswith('.lora_up.weight'))} LoRA modules"
        )
    if bundle["hydra"] is not None:
        routing = []
        if bundle["hydra"].get("use_fei_router"):
            routing.append(f"FEI={bundle['hydra']['fei_feature_dim']}d")
        if bundle["hydra"].get("sigma_band_partition"):
            routing.append(f"σ-band={bundle['hydra']['num_sigma_buckets']}")
        routing_str = f", {', '.join(routing)}" if routing else ""
        summary.append(
            f"Hydra({bundle['hydra']['num_experts']} experts, "
            f"{len(bundle['hydra']['modules'])} modules{routing_str})"
        )
    if bundle["reft"] is not None:
        summary.append(f"ReFT({len(bundle['reft'])} blocks)")
    logger.info(
        f"Loaded Anima adapter: {', '.join(summary) or 'empty'} from {file_path}"
    )
    return bundle


def _sigma_sinusoidal_features(
    sigma: torch.Tensor, sigma_feature_dim: int
) -> torch.Tensor:
    """Sinusoidal σ features matching ``networks/lora_modules.py`` verbatim.

    Trained σ-conditional bias is sensitive to this functional form, so any
    drift between training and inference shows up directly as wrong gates.

    ``sigma`` is expected to already be fp32 — ``sigma_pre_hook`` normalizes
    it once per step in eager Python so this hot-path call doesn't introduce
    a redundant ``.float()`` (and a per-Linear inductor DeviceCopy warning).
    """
    t = sigma.flatten()
    half_dim = sigma_feature_dim // 2
    exponent = (
        -math.log(10000)
        * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        / max(half_dim, 1)
    )
    freqs = torch.exp(exponent)
    angles = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _make_hydra_hook(params: dict, strength: float, sigma_state: dict):
    """Forward hook reproducing ``HydraLoRAModule.forward`` per Linear.

    Lazy-moves loaded tensors to the input's device on first call (saved
    dtype is preserved; bottleneck matmuls upcast to fp32 to match the CLI
    precision policy — see ``LoRAModule.forward`` rationale). ``sigma_state``
    is shared across all hydra hooks for this checkpoint; the
    diffusion-model pre-hook writes ``sigma_state["sigma"]`` (and, when a
    FEI router is attached, ``sigma_state["fei"]``) once per denoising
    step, and each hook reads them to build the per-sample router input.
    The router-input concat order matches ``HydraLoRAModule._compute_gate``:
    ``[pooled, sinusoidal(σ), FEI]`` — any slice may be empty.

    When ``sigma_band_partition`` is on, expert logits outside each sample's
    σ band are masked to ``-inf`` before softmax — mirrors
    ``networks/lora_modules/hydra.py::_apply_sigma_band_mask``. The expert→
    band lookup uses interleaved layout (``e mod N``) and the bucket lookup
    uses ``torch.bucketize`` against the optional custom edge list, matching
    the training-side ``_register_sigma_band_partition``.
    """
    state = {
        "lora_down": params["lora_down"],
        "lora_ups": params["lora_ups"],          # (E, out, rank)
        "router_w": params["router_w"],          # (E, rank + sigma_dim + fei_dim)
        "router_b": params["router_b"],          # (E,)
        "inv_scale": params.get("inv_scale"),    # (in_dim,) or None
        "scale": params["scale"],
        "sigma_feature_dim": int(params.get("sigma_feature_dim", 0)),
        "fei_feature_dim": int(params.get("fei_feature_dim", 0)),
        "sigma_band_partition": bool(params.get("sigma_band_partition", False)),
        "num_sigma_buckets": int(params.get("num_sigma_buckets", 0)),
        "expert_band": params.get("expert_band"),  # (E,) long, or None
        "sigma_edges": params.get("sigma_edges"),  # (B-1,) fp32, or None
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        # Hot path runs router math + bottleneck matmuls in fp32 (CLI precision
        # policy). Cast once on device migration instead of per-call .float()
        # — otherwise inductor traces every cast as a DeviceCopy in the input
        # program and emits a warning per adapted Linear per compile.
        for k in ("lora_down", "lora_ups", "router_w", "router_b", "inv_scale"):
            if state[k] is not None:
                state[k] = state[k].to(device=x.device, dtype=torch.float32)
        if state["expert_band"] is not None:
            # long bucket-id lookup; no dtype cast.
            state["expert_band"] = state["expert_band"].to(device=x.device)
        if state["sigma_edges"] is not None:
            state["sigma_edges"] = state["sigma_edges"].to(
                device=x.device, dtype=torch.float32
            )
        state["device"] = x.device

    def hydra_hook(module, inputs, output):
        x = inputs[0]
        _ensure_on_device(x)

        x_lora = x.float()
        if state["inv_scale"] is not None:
            x_lora = x_lora * state["inv_scale"]

        # down projection (B, *, rank), fp32 — feeds both the router and the
        # gate-weighted bmm downstream.
        lx = torch.nn.functional.linear(x_lora, state["lora_down"])

        # router gate — RMS pool the rank-R signal over the sequence dim, then
        # optionally concat sinusoidal(σ) before the router linear. Mirrors
        # HydraLoRAModule._compute_gate after the σ-input rewiring (see
        # docs/methods/hydra-lora.md §Fixes): previously σ entered as an
        # additive sigma_mlp bias on the logits; the router now reads it as
        # input, so the σ-feature columns of router.weight train on the same
        # chain rule as the content columns.
        B = lx.shape[0]
        if lx.dim() >= 3:
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        if state["sigma_feature_dim"] > 0 and sigma_state.get("sigma") is not None:
            # σ enters via comfy's timesteps on the model device, and the
            # sinusoidal builder returns fp32 — same device/dtype as pooled,
            # so no trailing .to() is needed (avoids inductor DeviceCopy).
            sigma_feat = _sigma_sinusoidal_features(
                sigma_state["sigma"], state["sigma_feature_dim"]
            )
            # Broadcast σ features across the batch when σ is shape (1,) but
            # pooled is CFG-doubled.
            if sigma_feat.shape[0] == 1 and pooled.shape[0] != 1:
                sigma_feat = sigma_feat.expand(pooled.shape[0], -1)
            pooled = torch.cat([pooled, sigma_feat], dim=-1)
        elif state["sigma_feature_dim"] > 0:
            # σ-conditional router but no σ captured yet (shouldn't happen
            # under the wrapper, but stay safe): zero-pad to keep shape.
            pooled = torch.nn.functional.pad(
                pooled, (0, state["sigma_feature_dim"])
            )
        if state["fei_feature_dim"] > 0 and sigma_state.get("fei") is not None:
            # FEI is already fp32 from _compute_fei_2band and lives on the
            # latent's device — same as pooled after _ensure_on_device.
            # Slice in case the router's fei_feature_dim is smaller than the
            # available simplex width (unused today; 2-band == 2-d here).
            fei_feat = sigma_state["fei"][:, : state["fei_feature_dim"]]
            if fei_feat.shape[0] == 1 and pooled.shape[0] != 1:
                fei_feat = fei_feat.expand(pooled.shape[0], -1)
            pooled = torch.cat([pooled, fei_feat], dim=-1)
        elif state["fei_feature_dim"] > 0:
            # FEI router but pre-hook didn't fire (defensive): keep shape.
            pooled = torch.nn.functional.pad(
                pooled, (0, state["fei_feature_dim"])
            )
        logits = torch.nn.functional.linear(
            pooled, state["router_w"], state["router_b"]
        )
        if state["sigma_band_partition"] and sigma_state.get("sigma") is not None:
            # sigma is already fp32 (normalized in sigma_pre_hook).
            sigma = sigma_state["sigma"].flatten()
            num_buckets = state["num_sigma_buckets"]
            bucket_ids = torch.bucketize(sigma, state["sigma_edges"]).clamp(
                0, num_buckets - 1
            )
            if bucket_ids.shape[0] == 1 and logits.shape[0] > 1:
                bucket_ids = bucket_ids.expand(logits.shape[0])
            in_band = bucket_ids[:, None] == state["expert_band"][None, :]
            logits = logits.masked_fill(~in_band, float("-inf"))
        gate = torch.softmax(logits, dim=-1)

        # gate-weighted combined ups (B, out, rank)
        combined = torch.einsum("be,eor->bor", gate, state["lora_ups"])

        # apply via batched matmul
        orig_shape = lx.shape
        lx_3d = lx.reshape(B, -1, orig_shape[-1])
        delta = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(
            *orig_shape[:-1], -1
        )
        return output + (delta * (state["scale"] * strength)).to(output.dtype)

    return hydra_hook


def _make_router_pre_hook(
    router_state: dict,
    fei_enabled: bool,
    fei_sigma_low_div: float,
):
    """Forward pre-hook that records the per-step routing inputs.

    Always writes ``router_state["sigma"]`` from ``args[1]`` (timesteps).
    When ``fei_enabled`` is true, also computes the per-sample 2-band
    Laplacian FEI from ``args[0]`` (the latent ``x``) and writes
    ``router_state["fei"]`` of shape ``(B, 2)``. Each hydra hook reads
    whichever it needs during gate computation.

    For Anima, ``args[0]`` is the 5D ``(B, C, T, H, W)`` latent passed to
    the cosmos backbone; the T=1 dim is squeezed before the 2D blur. FEI
    compute is one separable Gaussian on a ``H_lat·W_lat ≈ 4096`` grid —
    negligible vs the DiT forward.

    Why a pre-hook rather than overriding ``diffusion_model.forward``:
    replacing ``forward`` via ``add_object_patch`` strands sub-Linears
    (e.g. cosmos ``x_embedder.proj``) on CPU under ComfyUI's lowvram-aware
    load path — exactly the failure mode that retired the old
    ``block.forward`` override in favor of ``_forward_hooks``. A pre-hook
    leaves ``forward`` untouched and torch.compile traces cleanly through it
    (with the dynamo-disable guard below for safety on the dict stores and
    the FEI conv2d).
    """

    @torch._dynamo.disable
    def router_pre_hook(module, args):
        if len(args) >= 2 and args[1] is not None:
            # Normalize to fp32 once per denoising step in eager Python; the
            # hydra hooks downstream then read it without re-casting (each
            # cast inside the compiled graph would log a DeviceCopy warning
            # per adapted Linear). `.detach()` so autograd never sees it,
            # `.float()` is a no-op when already fp32 (comfy's typical case).
            router_state["sigma"] = args[1].detach().float()
        if fei_enabled and len(args) >= 1 and args[0] is not None:
            x = args[0].detach()
            # Anima/cosmos passes a 5D (B, C, T, H, W) latent; collapse T=1
            # so the 2D Laplacian sees (B, C, H, W). Already-4D latents
            # (other backbones) pass through unchanged.
            if x.dim() == 5:
                x = x.squeeze(2)
            h_lat = int(x.shape[-2])
            w_lat = int(x.shape[-1])
            sigma_low = _fei_sigma_low(h_lat, w_lat, fei_sigma_low_div)
            router_state["fei"] = _compute_fei_2band(x, sigma_low)

    return router_pre_hook


def _resolve_module(model, dotted_path: str):
    """Walk attribute / index path under ``model.model``."""
    obj = model.model
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _apply_hydra_live_to_model(
    model, hydra_data: dict, strength: float
) -> int:
    """Install live-routing forward hooks on each Hydra-adapted Linear.

    Replaces the previous uniform-bake fallback. Per-Linear hooks reproduce
    the trained ``HydraLoRAModule.forward`` (per-sample router from layer
    input, per-expert ``lora_up`` blend) so the multi-head specialization
    fires at inference. σ-conditional router bias is captured via a forward
    pre-hook on ``diffusion_model`` that records ``timesteps`` into shared
    state read by each hook.

    Returns number of hooks installed.
    """
    import comfy.lora

    if strength == 0:
        return 0

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})

    # Per-checkpoint shared routing state. The pre-hook writes "sigma" every
    # step (and "fei" when FeRA-style routing is on); every per-Linear hook
    # reads from this dict.
    sigma_state: dict = {}

    # FEI router metadata. Populated by `load_adapter` from `ss_use_fei_router`
    # / `ss_fei_feature_dim` / `ss_fei_sigma_low_div`. Without metadata, FEI
    # stays off — and the per-module sigma_feature_dim calc below collapses
    # to the original (rank → σ) split, so non-FEI checkpoints behave exactly
    # as before.
    fei_on = bool(hydra_data.get("use_fei_router", False))
    fei_feature_dim = int(hydra_data.get("fei_feature_dim", 0))
    fei_sigma_low_div = float(hydra_data.get("fei_sigma_low_div", 8.0))

    # Reconstruct the σ-band → expert lookup once per checkpoint. Identical to
    # training's `_register_sigma_band_partition`: interleaved layout
    # (expert e → band ``e mod N``) plus optional custom σ edges from
    # ss_sigma_bucket_boundaries (defaults to uniform ``linspace(0, 1, N+1)``).
    # Shared across all per-Linear hooks since the partition is a property of
    # the checkpoint, not the layer.
    band_partition_on = bool(hydra_data.get("sigma_band_partition", False))
    num_sigma_buckets = int(hydra_data.get("num_sigma_buckets", 0))
    expert_band: Optional[torch.Tensor] = None
    sigma_edges: Optional[torch.Tensor] = None
    if band_partition_on and num_sigma_buckets > 1:
        E = int(hydra_data["num_experts"])
        expert_band = torch.arange(E, dtype=torch.long) % num_sigma_buckets
        boundaries = hydra_data.get("sigma_bucket_boundaries")
        if boundaries is None:
            edges_full = torch.linspace(0.0, 1.0, num_sigma_buckets + 1)
        else:
            edges_full = torch.tensor(boundaries, dtype=torch.float32)
        sigma_edges = edges_full[1:-1].contiguous()

    # Install a forward pre-hook on diffusion_model to record σ (and FEI on
    # FeRA-style checkpoints). Patch _forward_pre_hooks (an OrderedDict) via
    # add_object_patch so it's reverted on ModelPatcher.unpatch_model.
    # Composes with any prior diffusion_model.forward object_patch (postfix
    # wraps forward; the pre-hook fires before that wrapper sees args).
    diffusion_model = model.get_model_object("diffusion_model")
    router_pre_hook = _make_router_pre_hook(
        sigma_state, fei_on, fei_sigma_low_div
    )
    new_pre_hooks = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre_hooks[id(router_pre_hook)] = router_pre_hook
    model.add_object_patch(
        "diffusion_model._forward_pre_hooks", new_pre_hooks
    )

    patched = 0
    skipped: list[str] = []
    for prefix, mod in hydra_data["modules"].items():
        if "lora_down" not in mod or "lora_ups" not in mod:
            skipped.append(f"{prefix}: missing lora_down/lora_ups")
            continue
        if "router_w" not in mod or "router_b" not in mod:
            skipped.append(f"{prefix}: missing router")
            continue

        comfy_sd_key = key_map.get(prefix)
        if comfy_sd_key is None:
            skipped.append(f"{prefix}: not in ComfyUI key_map")
            continue
        module_path = (
            comfy_sd_key[: -len(".weight")]
            if comfy_sd_key.endswith(".weight")
            else comfy_sd_key
        )

        try:
            linear = _resolve_module(model, module_path)
        except (AttributeError, IndexError, ValueError) as e:
            skipped.append(f"{prefix}: resolve {module_path} failed ({e})")
            continue

        ups_dict = mod["lora_ups"]
        ups_stacked = torch.stack(
            [ups_dict[i] for i in sorted(ups_dict.keys())], dim=0
        )
        rank = mod["lora_down"].shape[0]

        # Router input layout matches HydraLoRAModule._compute_gate's concat
        # order: [pooled rank-R, sinusoidal(σ), FEI]. FEI dim comes from
        # safetensors metadata (uniform across modules in shipped configs);
        # σ dim is whatever's left after stripping rank + fei.
        router_in = mod["router_w"].shape[1]
        sigma_feature_dim = router_in - rank - fei_feature_dim
        if sigma_feature_dim < 0:
            skipped.append(
                f"{prefix}: router input {router_in} < rank {rank} + "
                f"fei_feature_dim {fei_feature_dim} "
                f"(shape {tuple(mod['router_w'].shape)}) -- checkpoint malformed"
            )
            continue
        alpha_t = mod.get("alpha")
        alpha = (
            float(alpha_t.item() if hasattr(alpha_t, "item") else alpha_t)
            if alpha_t is not None
            else float(rank)
        )

        params = {
            "lora_down": mod["lora_down"],
            "lora_ups": ups_stacked,
            "router_w": mod["router_w"],
            "router_b": mod["router_b"],
            "inv_scale": mod.get("inv_scale"),
            "scale": alpha / rank,
            "sigma_feature_dim": sigma_feature_dim,
            "fei_feature_dim": fei_feature_dim,
            "sigma_band_partition": expert_band is not None,
            "num_sigma_buckets": num_sigma_buckets,
            "expert_band": expert_band,
            "sigma_edges": sigma_edges,
        }

        hook = _make_hydra_hook(params, strength, sigma_state)
        new_hooks = OrderedDict(linear._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(f"{module_path}._forward_hooks", new_hooks)
        patched += 1

    if skipped:
        logger.warning(
            f"Hydra live-routing skipped {len(skipped)} prefix(es); "
            f"first few: {skipped[:5]}"
        )
    # Decide what's actually being routed on by checking the router-input
    # split, not the raw shape. With FEI metadata in play, "router_in > rank"
    # alone no longer implies σ-conditional.
    has_sigma = any(
        "router_w" in m
        and "lora_down" in m
        and m["router_w"].shape[1] - m["lora_down"].shape[0] - fei_feature_dim > 0
        for m in hydra_data["modules"].values()
    )
    band_msg = (
        f", σ-band={num_sigma_buckets} buckets"
        if expert_band is not None
        else ""
    )
    fei_msg = (
        f", FEI={fei_feature_dim}d (σ_low_div={fei_sigma_low_div:g})"
        if fei_on
        else ""
    )
    logger.info(
        f"Hydra live-routing installed {patched} hooks "
        f"(strength={strength}, σ-conditional={'yes' if has_sigma else 'no'}"
        f"{fei_msg}{band_msg})"
    )
    return patched


def _apply_lora_sd_to_model(model, lora_sd: Dict[str, torch.Tensor], strength: float):
    """Apply a standard LoRA state_dict via ComfyUI's weight patching."""
    import comfy.lora
    import comfy.lora_convert

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    model.add_patches(loaded, strength)


def _make_reft_hook(params: dict, strength: float):
    """Forward hook that adds ReFT's residual edit to the block output.

    Hook params (rotate / source_w / source_b) are moved/cast to the input's
    device/dtype on first call and cached for subsequent calls.
    """
    scale = params["scale"]
    state = {
        "rotate": params["rotate"],
        "source_w": params["source_w"],
        "source_b": params["source_b"],
        "ready": False,
    }

    def reft_hook(module, inputs, output):
        h = output
        if (
            not state["ready"]
            or state["rotate"].device != h.device
            or state["rotate"].dtype != h.dtype
        ):
            state["rotate"] = state["rotate"].to(device=h.device, dtype=h.dtype)
            state["source_w"] = state["source_w"].to(device=h.device, dtype=h.dtype)
            state["source_b"] = state["source_b"].to(device=h.device, dtype=h.dtype)
            state["ready"] = True
        delta = torch.nn.functional.linear(h, state["source_w"], state["source_b"])
        edit = torch.nn.functional.linear(delta, state["rotate"].T)
        return h + edit * (strength * scale)

    return reft_hook


def _apply_reft_to_model(model, reft_blocks: Dict[int, dict], strength: float) -> int:
    """Install per-block ReFT edits as ComfyUI object patches.

    Uses a ``forward_hook`` per block (swapped in by replacing the block's
    ``_forward_hooks`` OrderedDict via ``add_object_patch``) instead of
    overriding ``block.forward``. Replacing ``forward`` interferes with
    ComfyUI's weight-loading path — the block's Linears were ending up with
    ``comfy_cast_weights=False`` and their weights left on CPU, producing a
    device mismatch when the block ran. A forward hook leaves ``forward``
    (and ComfyUI's view of it) untouched, and torch.compile traces cleanly
    through it.

    Returns the number of blocks actually patched. The original
    ``_forward_hooks`` dict is restored on ``ModelPatcher.unpatch_model``.
    """
    diffusion = model.get_model_object("diffusion_model")
    if not hasattr(diffusion, "blocks"):
        raise ValueError(
            "ReFT adapter requires a DiT with `.blocks` ModuleList "
            f"(got {type(diffusion).__name__})."
        )
    num_blocks = len(diffusion.blocks)

    patched = 0
    for idx, params in reft_blocks.items():
        if idx < 0 or idx >= num_blocks:
            logger.warning(
                f"ReFT block index {idx} out of range [0, {num_blocks}); skipping"
            )
            continue
        block = diffusion.blocks[idx]
        hook = _make_reft_hook(params, strength)
        new_hooks = OrderedDict(block._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(
            f"diffusion_model.blocks.{idx}._forward_hooks", new_hooks
        )
        patched += 1
    return patched


def apply_adapter(
    model, file_path: str, strength_lora: float, strength_reft: float
) -> bool:
    """Apply LoRA / Hydra / ReFT components from ``file_path`` to ``model``
    in place. ``model`` must already be a clone. Returns True if anything
    was applied.
    """
    bundle = load_adapter(file_path)
    applied_any = False

    if bundle["hydra"] is not None:
        n = _apply_hydra_live_to_model(model, bundle["hydra"], strength_lora)
        if n > 0:
            applied_any = True
    if bundle["lora"] is not None:
        # Plain LoRA — apply directly. Hydra + plain-LoRA coexist in the same
        # file when ``router_targets`` is a subset regex (e.g. mlp only):
        # hydra handles mlp prefixes, plain LoRA handles cross_attn / self_attn.
        # ``_parse_hydra`` filters itself to hydra-only prefixes and
        # ``_extract_lora_sd`` skips ``.lora_ups.*`` keys, so the two paths
        # target disjoint modules — no double-patching.
        _apply_lora_sd_to_model(model, bundle["lora"], strength_lora)
        applied_any = True

    if bundle["reft"] is not None:
        n = _apply_reft_to_model(model, bundle["reft"], strength_reft)
        if n > 0:
            applied_any = True

    if not applied_any:
        logger.warning(
            f"Anima adapter at {file_path} contained no recognizable "
            "LoRA, Hydra, or ReFT keys."
        )
    return applied_any
