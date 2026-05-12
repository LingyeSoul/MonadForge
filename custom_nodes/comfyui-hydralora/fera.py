"""FeRA (Frequency-Energy constrained Routing) loading for Anima.

Author-faithful port of Yin et al., arXiv:2511.17979 — the
``networks.methods.fera`` training-side module. Distinct from the
FEI-router-on-Hydra path that ``adapter.py`` handles (those load via
``AnimaAdapterLoader``):

  * One **global router** consumes the latent's spectral energy and
    emits a single ``(B, num_experts)`` gate that every adapted Linear
    reuses for this step. Hydra routes per-Linear from its own input.
  * Each Linear carries **independent** stacked low-rank experts —
    ``lora_down: (E, r, in)`` / ``lora_up: (E, out, r)`` — no shared-A
    pooling.
  * Save format is incompatible with vanilla LoRA loaders. Stacked
    Parameters appear as ``lora_unet_*.lora_down`` / ``lora_unet_*.lora_up``
    *without* the trailing ``.weight`` that plain LoRA / Hydra use (their
    ``lora_down`` is an ``nn.Linear`` whose weight is a child tensor;
    FeRA's is a flat Parameter).
  * Mutually exclusive with HydraLoRA-moe at the inference layer —
    ``library/inference/models.py`` refuses to load both. Same rule here.

Application strategy mirrors the training-side semantics:

  1. One ``forward_pre_hook`` on ``diffusion_model._forward_pre_hooks``
     computes the per-step Frequency-Energy Indicator from ``args[0]``
     (the latent) and runs the router, writing ``(B, num_experts)`` gates
     into shared state.
  2. One ``forward_hook`` per adapted Linear adds the gated stacked-expert
     correction to that Linear's output.

Both use ``ModelPatcher.add_object_patch`` on ``_forward_hooks`` /
``_forward_pre_hooks`` rather than overriding ``forward``. Overriding
``forward`` strands sub-Linears on CPU under ComfyUI's cast-weights path
(see CLAUDE.md). A hook leaves ``forward`` untouched and is properly
reverted on ``unpatch_model``.
"""

import logging
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F

# Reuse the kernel cache + separable 2D blur from the Hydra loader so
# both paths share one cache. Adapter's ``_compute_fei_2band`` is hard-
# coded to 2 bands with ``[e_low, e_high]`` ordering — FeRA defaults to 3
# bands with ``[high, ..., low]`` ordering, so the band code is local.
from .adapter import _gaussian_blur_2d

logger = logging.getLogger(__name__)

# Cache: path -> parsed bundle. Reuses adapter.py's pattern.
_fera_cache: Dict[str, dict] = {}


def _compute_fei_nband(
    z: torch.Tensor, sigma_low: float, num_bands: int
) -> torch.Tensor:
    """Return ``(B, num_bands)`` simplex energies, ordered ``[high, ..., low]``.

    Bit-identical to ``networks/methods/fera.py::FrequencyEnergyIndicator``:
    bands are differences of adjacent pyramid levels (high-freq first),
    followed by the coarsest LP as the residual low-band; σ-scales double
    outward from ``σ_low``. Promoted to fp32 internally — squared norms
    underflow at small energies in bf16.

    The router weights were trained against this exact ordering, so any
    permutation here would corrupt the gate at inference.
    """
    z = z.float()
    sigmas: List[float] = [sigma_low * (2.0**k) for k in range(num_bands - 1)]
    pyramid = [z]
    for s in sigmas:
        pyramid.append(_gaussian_blur_2d(pyramid[-1], s))
    bands = [pyramid[k] - pyramid[k + 1] for k in range(num_bands - 1)]
    bands.append(pyramid[-1])
    energies = torch.stack(
        [b.pow(2).flatten(1).sum(-1) for b in bands], dim=-1
    )
    return energies / energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _looks_like_fera(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """Cheap key sniff for a FeRA checkpoint.

    The save format pairs ``router.net.*`` MLP keys with stacked
    Parameter keys ``lora_unet_*.lora_down`` / ``lora_unet_*.lora_up``
    that do *not* end in ``.weight``. Plain LoRA / Hydra use
    ``.lora_down.weight`` (the Linear's weight tensor is a child), so the
    no-``.weight`` suffix on a stacked Parameter is the disambiguator.
    """
    has_router = any(k.startswith("router.net.") for k in weights_sd)
    has_stacked = any(
        k.startswith("lora_unet_")
        and (k.endswith(".lora_down") or k.endswith(".lora_up"))
        for k in weights_sd
    )
    return has_router and has_stacked


def load_fera(file_path: str) -> dict:
    """Parse a FeRA checkpoint once, cache by path.

    Returns a bundle with the router MLP weights, per-Linear stacked
    expert weights keyed by ``lora_unet_<dotted>`` prefix, and a config
    dict of (rank, alpha, num_experts, num_bands, router_tau,
    router_hidden, fei_sigma_low_div, scale). Layer prefixes are matched
    to live DiT modules at apply time via
    ``comfy.lora.model_lora_keys_unet``.
    """
    if file_path in _fera_cache:
        return _fera_cache[file_path]

    from safetensors import safe_open
    from safetensors.torch import load_file

    weights_sd = load_file(file_path)
    with safe_open(file_path, framework="pt") as f:
        meta = dict(f.metadata() or {})

    declared = meta.get("ss_network_module") == "networks.methods.fera"
    if not declared and not _looks_like_fera(weights_sd):
        raise ValueError(
            f"{file_path} doesn't look like a FeRA checkpoint "
            "(no router.net.* + lora_unet_*.lora_down / .lora_up keys). "
            "For LoRA / HydraLoRA / ReFT files, use AnimaAdapterLoader."
        )

    # ─── Router params ─────────────────────────────────────────────────
    # SoftFrequencyRouter.net = Linear → ReLU → Linear, so the saved keys
    # are router.net.0.{weight,bias} (hidden) + router.net.2.{weight,bias}
    # (output). The ReLU at index 1 has no parameters.
    required = (
        "router.net.0.weight",
        "router.net.0.bias",
        "router.net.2.weight",
        "router.net.2.bias",
    )
    missing = [k for k in required if k not in weights_sd]
    if missing:
        raise ValueError(
            f"{file_path}: router keys missing ({missing}) — checkpoint "
            "may be from a non-faithful FeRA variant."
        )
    router = {
        "w1": weights_sd["router.net.0.weight"],
        "b1": weights_sd["router.net.0.bias"],
        "w2": weights_sd["router.net.2.weight"],
        "b2": weights_sd["router.net.2.bias"],
    }

    # ─── Per-Linear stacked experts ────────────────────────────────────
    layers: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if not key.startswith("lora_unet_"):
            continue
        if key.endswith(".lora_down"):
            prefix = key[: -len(".lora_down")]
            layers.setdefault(prefix, {})["lora_down"] = value  # (E, r, in)
        elif key.endswith(".lora_up"):
            prefix = key[: -len(".lora_up")]
            layers.setdefault(prefix, {})["lora_up"] = value  # (E, out, r)

    incomplete = [
        p for p, d in layers.items() if "lora_down" not in d or "lora_up" not in d
    ]
    if incomplete:
        logger.warning(
            f"FeRA: {len(incomplete)} prefix(es) missing lora_down or "
            f"lora_up; skipping (first few: {incomplete[:5]})"
        )
        for p in incomplete:
            del layers[p]
    if not layers:
        raise ValueError(
            f"{file_path}: parsed router but no usable per-Linear experts."
        )

    # ─── Hyperparams (metadata first, infer from shapes as fallback) ───
    sample = next(iter(layers.values()))
    sample_down = sample["lora_down"]  # (E, r, in)
    E_shape, r_shape = int(sample_down.shape[0]), int(sample_down.shape[1])
    router_hidden, router_in = int(router["w1"].shape[0]), int(router["w1"].shape[1])

    def _meta_int(key: str, fallback: int) -> int:
        v = meta.get(f"ss_{key}")
        try:
            return int(v) if v is not None else fallback
        except (TypeError, ValueError):
            return fallback

    def _meta_float(key: str, fallback: float) -> float:
        v = meta.get(f"ss_{key}")
        try:
            return float(v) if v is not None else fallback
        except (TypeError, ValueError):
            return fallback

    cfg = {
        "rank": _meta_int("fera_rank", r_shape),
        "alpha": _meta_float("fera_alpha", float(r_shape)),
        "num_experts": _meta_int("fera_num_experts", E_shape),
        "num_bands": _meta_int("fera_num_bands", router_in),
        "router_tau": _meta_float("fera_router_tau", 0.7),
        "router_hidden": _meta_int("fera_router_hidden", router_hidden),
        "fei_sigma_low_div": _meta_float("fei_sigma_low_div", 8.0),
    }
    # Shape wins over metadata on conflict — the tensors are authoritative.
    if cfg["num_experts"] != E_shape:
        logger.warning(
            f"FeRA: ss_fera_num_experts={cfg['num_experts']} disagrees with "
            f"stacked-expert axis ({E_shape}); using shape."
        )
        cfg["num_experts"] = E_shape
    if cfg["num_bands"] != router_in:
        logger.warning(
            f"FeRA: ss_fera_num_bands={cfg['num_bands']} disagrees with "
            f"router input dim ({router_in}); using shape."
        )
        cfg["num_bands"] = router_in
    if cfg["rank"] != r_shape:
        logger.warning(
            f"FeRA: ss_fera_rank={cfg['rank']} disagrees with stacked rank "
            f"axis ({r_shape}); using shape."
        )
        cfg["rank"] = r_shape
    cfg["scale"] = cfg["alpha"] / cfg["rank"]

    bundle = {
        "path": file_path,
        "router": router,
        "layers": layers,
        "cfg": cfg,
    }
    _fera_cache[file_path] = bundle

    logger.info(
        f"Loaded FeRA: {len(layers)} adapted Linears, "
        f"{cfg['num_experts']} experts × rank {cfg['rank']}, "
        f"router({cfg['num_bands']} bands → {cfg['router_hidden']} → "
        f"{cfg['num_experts']}, τ={cfg['router_tau']:.2f}), "
        f"σ_low_div={cfg['fei_sigma_low_div']:g} from {file_path}"
    )
    return bundle


def _make_fera_pre_hook(router: dict, cfg: dict, fera_state: dict):
    """Forward pre-hook that runs the global router on the current latent.

    Writes ``fera_state["gates"]`` of shape ``(B, num_experts)`` once per
    ``diffusion_model`` forward. Per-Linear hooks read from there. Router
    weights migrate to the latent's device + fp32 on first call and stay
    there for the rest of the session.

    The ``@torch._dynamo.disable`` guard mirrors
    ``adapter.py::_make_router_pre_hook`` — the dict store + FEI conv2d
    shouldn't get inlined into the compiled DiT graph or every per-Linear
    cast logs a DeviceCopy warning per step.
    """
    state = {
        "w1": router["w1"],
        "b1": router["b1"],
        "w2": router["w2"],
        "b2": router["b2"],
        "device": None,
    }
    tau = float(cfg["router_tau"])
    num_bands = int(cfg["num_bands"])
    fei_sigma_low_div = float(cfg["fei_sigma_low_div"])

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        for k in ("w1", "b1", "w2", "b2"):
            state[k] = state[k].to(device=x.device, dtype=torch.float32)
        state["device"] = x.device

    @torch._dynamo.disable
    def fera_pre_hook(module, args):
        if len(args) < 1 or args[0] is None:
            return
        x = args[0].detach()
        _ensure_on_device(x)
        # Anima/cosmos passes a 5D (B, C, T, H, W) latent; collapse T=1
        # so the 2D Laplacian sees (B, C, H, W). Other backbones pass 4D
        # through unchanged.
        if x.dim() == 5:
            x = x.squeeze(2)
        h_lat, w_lat = int(x.shape[-2]), int(x.shape[-1])
        sigma_low = float(min(h_lat, w_lat)) / fei_sigma_low_div
        fei = _compute_fei_nband(x, sigma_low, num_bands)
        hidden = F.relu(F.linear(fei, state["w1"], state["b1"]))
        logits = F.linear(hidden, state["w2"], state["b2"])
        fera_state["gates"] = F.softmax(logits / tau, dim=-1)

    return fera_pre_hook


def _make_fera_clear_hook(fera_state: dict):
    """Forward (post) hook that drops gates at the end of each
    ``diffusion_model.forward``.

    Defense-in-depth against stale gates leaking into code paths that
    run adapted Linears *outside* the diffusion forward — e.g. ComfyUI's
    ``BaseModel.extra_conds`` calls ``diffusion_model.preprocess_text_embeds``
    before sampling starts, and on a re-run the previous sample's
    last-step gates (shape ``(B_latent, E)`` with B_latent doubled by CFG)
    would otherwise broadcast through any FeRA hook fired in that
    pre-sample path. The LLM adapter is excluded from the key map for
    exactly this reason, but this clear keeps the invariant
    independent of which modules end up adapted.
    """

    @torch._dynamo.disable
    def fera_clear_hook(module, inputs, output):
        fera_state["gates"] = None
        return output

    return fera_clear_hook


def _make_fera_hook(
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scale: float,
    strength: float,
    fera_state: dict,
):
    """Per-Linear forward hook that adds ``Σ_k w_k · U_k @ D_k @ x``.

    Mirrors ``FeRALinear.forward`` in ``networks/methods/fera.py``: one
    batched ``einsum`` for the down projection over E experts, multiply
    by the per-batch gates, one batched ``einsum`` for the up projection.
    Weight tensors migrate to the input's device + fp32 on first call;
    subsequent calls skip the migration. Per-Linear hot path stays in
    fp32 to match the CLI's precision policy (also matches
    ``_make_hydra_hook``).

    Returns the original output untouched when ``strength=0`` or
    ``gates`` hasn't been written yet (defensive — the pre-hook should
    have fired in the same forward).
    """
    state = {
        "lora_down": lora_down,  # (E, r, in)
        "lora_up": lora_up,      # (E, out, r)
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        state["lora_down"] = state["lora_down"].to(
            device=x.device, dtype=torch.float32
        )
        state["lora_up"] = state["lora_up"].to(
            device=x.device, dtype=torch.float32
        )
        state["device"] = x.device

    def fera_hook(module, inputs, output):
        if strength == 0.0:
            return output
        gates = fera_state.get("gates")
        if gates is None:
            return output

        x = inputs[0]
        _ensure_on_device(x)
        x_c = x.float()

        # (..., in) × (E, r, in)ᵀ → (..., E, r). Stacked-einsum saves
        # one (..., E, r) activation instead of E × (..., D_out); not as
        # impactful at inference (no backward) but the layout matches
        # the training code so semantics are bit-identical.
        lx = torch.einsum("...i,eri->...er", x_c, state["lora_down"])

        # Broadcast gates (B, E) across any mid dims (e.g. token T).
        B, E = gates.shape
        n_mid = lx.ndim - 3  # dims between batch and (E, r)
        view_shape = (B,) + (1,) * n_mid + (E, 1)
        lx = lx * gates.view(view_shape).to(torch.float32)

        # (..., E, r) × (E, out, r)ᵀ → (..., out)
        delta = torch.einsum("...er,eor->...o", lx, state["lora_up"])
        return output + (delta * (scale * strength)).to(output.dtype)

    return fera_hook


def _resolve_module(model, dotted_path: str):
    """Walk attribute / index path under ``model.model``. Same idiom as
    ``adapter.py::_resolve_module``."""
    obj = model.model
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _build_fera_key_map(model) -> Dict[str, str]:
    """Inverse-map FeRA's ``lora_unet_<...>`` prefixes to live module paths
    under ``model.model``.

    Built from a direct walk of ``diffusion_model.named_modules()`` rather
    than ``comfy.lora.model_lora_keys_unet`` because ComfyUI's helper
    doesn't emit keys for fused projections (cosmos ``qkv_proj`` /
    ``kv_proj``) — it was designed around the split q/k/v convention from
    older UNet checkpoints. FeRA's ``_scan_targets`` walks
    ``named_modules()`` directly on the training side and matches the
    fused names through its target regex, so a checkpoint trained on the
    full default target set (attn + MLP) has 2 prefixes per block that
    aren't in ComfyUI's map.

    The FeRA training-side prefix convention is

        lora_name = "lora_unet_" + dotted_path.replace(".", "_")

    so this map is the literal inverse, built once at apply time. No
    ambiguity in the inverse direction — each live Linear produces
    exactly one ``lora_name``.

    ``llm_adapter.*`` Linears are intentionally excluded. The LLM adapter
    runs *outside* ``diffusion_model.forward`` at ComfyUI inference time —
    ``model_base.Anima.extra_conds`` calls ``preprocess_text_embeds``
    directly before the sampling loop, so the router pre-hook (which is
    on ``diffusion_model._forward_pre_hooks``) has not fired yet and
    ``fera_state["gates"]`` is either ``None`` (first sample) or *stale*
    from the previous sample's last step. Stale gates from a CFG-doubled
    latent (B=2) broadcast the per-Linear delta to B=2 in a B=1 text
    forward, blowing up ``q_proj``'s output to twice the expected size
    and crashing ``view([B, T, n_heads, head_dim])`` in the next line of
    ``Attention.forward``. Training-side FeRA contributions on
    ``llm_adapter.*_q_proj`` therefore can't be replayed at ComfyUI
    inference — drop them on the floor here. The DiT (cosmos backbone)
    FeRA contributions still apply correctly because those Linears run
    inside ``diffusion_model.forward`` *after* the router pre-hook fires.
    """
    import torch.nn as nn

    diffusion = model.get_model_object("diffusion_model")
    out: Dict[str, str] = {}
    for name, child in diffusion.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Strip torch.compile wrapper if any — matches
        # FeRANetwork._scan_targets so the lora_name keys agree.
        clean = name.replace("_orig_mod.", "")
        if not clean:
            continue
        if clean.startswith("llm_adapter.") or clean == "llm_adapter":
            continue
        lora_name = "lora_unet_" + clean.replace(".", "_")
        out[lora_name] = f"diffusion_model.{clean}"
    return out


def apply_fera(model, file_path: str, strength: float) -> bool:
    """Apply a FeRA adapter to ``model`` in place. ``model`` must already
    be a clone. Returns True if at least one hook was installed.
    """
    if strength == 0:
        logger.info("FeRA: strength=0 — installing no hooks.")
        return False

    bundle = load_fera(file_path)
    layers = bundle["layers"]
    cfg = bundle["cfg"]

    # Per-checkpoint shared state. Pre-hook writes "gates", every
    # per-Linear hook reads from this dict by closure capture.
    fera_state: dict = {}

    # Install the model-level pre-hook (router runs once per forward).
    # Patch _forward_pre_hooks (an OrderedDict) via add_object_patch so
    # it's reverted on ModelPatcher.unpatch_model. Composes with any
    # prior diffusion_model.forward object_patch (postfix wraps forward;
    # the pre-hook fires before that wrapper sees args).
    diffusion_model = model.get_model_object("diffusion_model")
    pre_hook = _make_fera_pre_hook(bundle["router"], cfg, fera_state)
    new_pre_hooks = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre_hooks[id(pre_hook)] = pre_hook
    model.add_object_patch(
        "diffusion_model._forward_pre_hooks", new_pre_hooks
    )

    # Post-hook clears gates at the end of every diffusion forward so
    # subsequent samples don't read stale CFG-batched gates from the
    # previous run's last step (see _make_fera_clear_hook).
    clear_hook = _make_fera_clear_hook(fera_state)
    new_dm_post_hooks = OrderedDict(diffusion_model._forward_hooks)
    new_dm_post_hooks[id(clear_hook)] = clear_hook
    model.add_object_patch(
        "diffusion_model._forward_hooks", new_dm_post_hooks
    )

    # Direct walk of diffusion_model — covers fused qkv/kv projections
    # that ComfyUI's model_lora_keys_unet doesn't enumerate. See
    # ``_build_fera_key_map`` for why.
    key_map = _build_fera_key_map(model)
    scale = float(cfg["scale"])

    patched = 0
    skipped: list[str] = []
    for prefix, layer in layers.items():
        module_path = key_map.get(prefix)
        if module_path is None:
            skipped.append(f"{prefix}: no matching Linear under diffusion_model")
            continue
        try:
            linear = _resolve_module(model, module_path)
        except (AttributeError, IndexError, ValueError) as e:
            skipped.append(f"{prefix}: resolve {module_path} failed ({e})")
            continue
        hook = _make_fera_hook(
            layer["lora_down"], layer["lora_up"], scale, strength, fera_state
        )
        new_hooks = OrderedDict(linear._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(f"{module_path}._forward_hooks", new_hooks)
        patched += 1

    if skipped:
        logger.warning(
            f"FeRA: skipped {len(skipped)} prefix(es); "
            f"first few: {skipped[:5]}"
        )
    logger.info(
        f"FeRA: installed router pre-hook + {patched} per-Linear hooks "
        f"(strength={strength}, {cfg['num_experts']} experts × rank "
        f"{cfg['rank']}, {cfg['num_bands']} bands, "
        f"σ_low_div={cfg['fei_sigma_low_div']:g})"
    )
    return patched > 0
