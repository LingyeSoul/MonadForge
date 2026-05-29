"""ChimeraHydra (dual-pool additive routing) for the Anima adapter node.

Two HydraLoRAs glued at the residual: a **content** pool routed by a per-
Linear rank-R router (or, under ``content_router_source="crossattn"``, a
network-level ContentRouter that pools the post-LLM-adapter context) and
a **frequency** pool routed by the network-level FreqRouter MLP fed
``concat(FEI, sinusoidal(σ))``. Pool outputs are summed. Two on-disk
formats exist:

* **Legacy single-A** (pre-c4851b6): shared ``lora_down`` + per-expert
  ``lora_ups.{i}`` over E = K_c + K_f experts. Goes through
  ``_parse_hydra`` (in ``adapter.py``) for the shape parse, then
  ``_attach_single_a_chimera_metadata`` here decorates the hydra bundle
  with the FreqRouter / ContentRouter state and K_c / K_f. Hook factory
  is ``_make_chimera_hook``.
* **Dual-A** (post-c4851b6): per-pool ``lora_down_{c,f}.weight`` + per-
  pool stacked ups ``lora_ups_{c,f}.{i}.weight``, content-only K_c-narrow
  router. Goes through ``_parse_chimera_dual_a`` here; the loader's
  follow-up validation lives in ``_finalize_dual_a_chimera``. Hook
  factory is ``_make_chimera_dual_a_hook``, apply is
  ``_apply_chimera_dual_a_to_model``.

Both formats are mutually exclusive on a single prefix and use the same
FreqRouter pre-hook (``_make_chimera_pre_hook``) and, when present, the
same global ContentRouter hook on ``diffusion_model.llm_adapter``
(``_make_content_router_llm_adapter_hook``). See
``networks/lora_modules/chimera.py`` for the training-side module these
hooks reproduce, and ``docs/proposal/chimera_hydra.md`` for the
proposal.
"""

import logging
from collections import OrderedDict
from typing import Dict, Optional

import torch

from .adapter import (
    _T5_PAD_LEN,
    _resolve_module,
    compute_fei_2band,
    fei_sigma_low,
    fei_temperature,
    sigma_sinusoidal_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_freq_router_mode(file_metadata: Dict[str, str]) -> tuple[str, float]:
    """Resolve the freq-pool routing mode + FEI temperature from metadata.

    ``ss_chimera_freq_router_mode`` ∈ {"learned", "fei"}; absent ⇒ "learned"
    (every pre-2026-05-27 chimera checkpoint carries a FreqRouter MLP).
    ``ss_chimera_freq_router_tau`` is the power-temperature for the hardwired
    FEI gate (``normalize(FEI ** (1/τ))``); only meaningful under "fei".
    """
    mode = str(
        file_metadata.get("ss_chimera_freq_router_mode", "learned")
    ).strip().lower() or "learned"
    if mode not in ("learned", "fei"):
        raise ValueError(
            f"ss_chimera_freq_router_mode={mode!r}: expected 'learned' or 'fei'."
        )
    try:
        tau = float(file_metadata.get("ss_chimera_freq_router_tau", 1.0))
    except (TypeError, ValueError):
        tau = 1.0
    return mode, tau


def _parse_chimera_content_router(
    weights_sd: Dict[str, torch.Tensor],
    file_metadata: Dict[str, str],
    file_path: str,
    K_c: int,
) -> Optional[dict]:
    """Pick up the network-level ContentRouter state when chimera was trained
    with ``content_router_source="crossattn"``.

    Returns None when the per-Linear router is in use (default), otherwise a
    dict with the MLP weights, parameterless-LN flag, expected input dim, and
    K_c. Same shape contract as ``content_router.net.*`` on disk — a 2-layer
    ``Linear → SiLU → Linear`` parameterised exactly like ``FreqRouter`` /
    ``ContentRouter`` in ``networks/lora_anima/network.py``.
    """
    source = str(file_metadata.get("ss_chimera_content_router_source", "input")).strip().lower()
    # ``"crossattn"`` is the pre-rename spelling; accept it alongside the
    # current ``"crossattn_emb"`` so older chimera checkpoints still load.
    if source not in ("crossattn", "crossattn_emb"):
        return None
    try:
        cr_w0 = weights_sd["content_router.net.0.weight"]
        cr_b0 = weights_sd["content_router.net.0.bias"]
        cr_w2 = weights_sd["content_router.net.2.weight"]
        cr_b2 = weights_sd["content_router.net.2.bias"]
    except KeyError as exc:
        raise ValueError(
            f"{file_path}: ss_chimera_content_router_source={source!r} but "
            f"checkpoint is missing ContentRouter weight key {exc} "
            "(expected content_router.net.{0,2}.weight/bias)."
        ) from exc
    if cr_w2.shape[0] != K_c:
        raise ValueError(
            f"{file_path}: ContentRouter output dim {cr_w2.shape[0]} != K_c={K_c}."
        )
    ln_flag = str(
        file_metadata.get("ss_chimera_content_router_layer_norm", "false")
    ).strip().lower() == "true"
    return {
        "input_dim": int(cr_w0.shape[1]),
        "layer_norm": ln_flag,
        "K_c": int(K_c),
        "sd": {
            "net.0.weight": cr_w0,
            "net.0.bias": cr_b0,
            "net.2.weight": cr_w2,
            "net.2.bias": cr_b2,
        },
    }


def _parse_chimera_dual_a(
    weights_sd: Dict[str, torch.Tensor],
) -> Optional[dict]:
    """Group ChimeraHydra dual-A keys. Returns None if no dual-A prefixes
    are present.

    Dual-A on-disk shape per Linear (q/k/v already defused):

      * ``prefix.lora_down_c.weight`` (r, in)    content A
      * ``prefix.lora_down_f.weight`` (r, in)    freq    A
      * ``prefix.lora_ups_c.{i}.weight`` (out, r) for i in 0..K_c-1
      * ``prefix.lora_ups_f.{j}.weight`` (out, r) for j in 0..K_f-1
      * ``prefix.router.weight`` (K_c, r)
      * ``prefix.router.bias``   (K_c,)
      * ``prefix.alpha``                            optional
      * ``prefix.inv_scale``                        optional

    Distinct from the single-A chimera path handled by ``_parse_hydra``
    (which sees ``lora_down.weight`` + ``lora_ups.{i}.weight``). The two
    formats never coexist on the same prefix, so detection is by key
    suffix only. Discriminator: any ``.lora_down_c.weight`` key in the
    state dict ⇒ dual-A chimera.

    Per-pool K is derived from the highest expert index seen in
    ``lora_ups_{c,f}.{i}.weight`` — must match the
    ``ss_num_experts_content`` / ``ss_num_experts_freq`` stamps captured
    by ``load_adapter`` (cross-check performed there).
    """
    modules: Dict[str, dict] = {}
    for key, value in weights_sd.items():
        if key.startswith(("reft_", "freq_router.", "content_router.")):
            continue
        parts = key.split(".")
        prefix = parts[0]
        rest = ".".join(parts[1:])
        mod = modules.setdefault(prefix, {})
        if rest == "lora_down_c.weight":
            mod["lora_down_c"] = value
        elif rest == "lora_down_f.weight":
            mod["lora_down_f"] = value
        elif rest.startswith("lora_ups_c.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            mod.setdefault("lora_ups_c", {})[idx] = value
        elif rest.startswith("lora_ups_f.") and rest.endswith(".weight"):
            idx = int(rest.split(".")[1])
            mod.setdefault("lora_ups_f", {})[idx] = value
        elif rest == "alpha":
            mod["alpha"] = value
        elif rest == "inv_scale":
            mod["inv_scale"] = value
        elif rest == "router.weight":
            mod["router_w"] = value
        elif rest == "router.bias":
            mod["router_b"] = value

    # Only return prefixes that have BOTH lora_down_c and at least one
    # lora_ups_c — the discriminator above is permissive (a single key
    # would seed an entry). A well-formed dual-A module always has both.
    dual_only: Dict[str, dict] = {
        prefix: mod
        for prefix, mod in modules.items()
        if "lora_down_c" in mod and "lora_ups_c" in mod
    }
    if not dual_only:
        return None
    K_c = max(max(m["lora_ups_c"].keys()) + 1 for m in dual_only.values())
    K_f = max(
        (max(m["lora_ups_f"].keys()) + 1 for m in dual_only.values() if "lora_ups_f" in m),
        default=0,
    )
    return {"num_experts_content": K_c, "num_experts_freq": K_f, "modules": dual_only}


def _attach_single_a_chimera_metadata(
    hydra: dict,
    weights_sd: Dict[str, torch.Tensor],
    file_metadata: Dict[str, str],
    file_path: str,
) -> None:
    """Decorate ``hydra`` with ``hydra["chimera"]`` when the file is a
    legacy single-A ChimeraHydra checkpoint.

    No-op (returns silently) when ``ss_use_chimera_hydra`` is not set.
    Same Hydra-MoE on-disk shape (shared ``lora_down`` + per-expert
    ``lora_ups.{i}``) plus a network-level ``freq_router.net.*`` block
    and a K_c-narrowed per-Linear content router. K_c + K_f =
    num_experts; the FreqRouter input is ``concat(FEI, sinusoidal(σ))``
    with dims from the chimera-specific metadata stamps. See
    ``networks/lora_modules/chimera.py`` and
    ``networks/lora_anima/network.py::FreqRouter``.
    """
    is_chimera = (
        str(file_metadata.get("ss_use_chimera_hydra", "")).strip().lower() == "true"
    )
    if not is_chimera:
        return
    try:
        K_c = int(file_metadata["ss_num_experts_content"])
        K_f = int(file_metadata["ss_num_experts_freq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{file_path}: ss_use_chimera_hydra=true but "
            "ss_num_experts_content / ss_num_experts_freq are "
            f"missing or malformed ({exc}) — checkpoint is bad."
        ) from exc
    if K_c + K_f != hydra["num_experts"]:
        raise ValueError(
            f"{file_path}: chimera K_c={K_c} + K_f={K_f} != "
            f"num_experts={hydra['num_experts']} (from lora_ups). "
            "Checkpoint is inconsistent."
        )

    try:
        chimera_fei_dim = int(
            file_metadata.get("ss_chimera_fei_feature_dim", 0)
        )
        chimera_sigma_dim = int(
            file_metadata.get("ss_chimera_sigma_feature_dim", 0)
        )
        chimera_sigma_low_div = float(
            file_metadata.get("ss_chimera_fei_sigma_low_div", 4.0)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{file_path}: malformed chimera σ/FEI feature dims ({exc})."
        ) from exc
    if chimera_fei_dim + chimera_sigma_dim <= 0:
        raise ValueError(
            f"{file_path}: chimera FreqRouter requires "
            "fei_feature_dim + sigma_feature_dim > 0 "
            f"(got FEI={chimera_fei_dim}, σ={chimera_sigma_dim})."
        )

    freq_router_mode, freq_router_fei_tau = _parse_freq_router_mode(file_metadata)
    if freq_router_mode == "fei":
        # Hardwired-FEI freq gate: no FreqRouter MLP on disk. π_f =
        # normalize(FEI ** (1/τ)) is computed in the pre-hook. The FEI
        # band count IS the expert count, so K_f must equal fei_dim.
        if chimera_fei_dim != K_f:
            raise ValueError(
                f"{file_path}: freq_router_mode='fei' requires "
                f"fei_feature_dim == K_f (got FEI={chimera_fei_dim}, K_f={K_f})."
            )
        freq_router_sd = None
    else:
        # FreqRouter MLP: Linear → SiLU → Linear → softmax/τ. Keys
        # mirror ``torch.nn.Sequential`` indices (SiLU at index 1
        # carries no params).
        try:
            fr_w0 = weights_sd["freq_router.net.0.weight"]
            fr_b0 = weights_sd["freq_router.net.0.bias"]
            fr_w2 = weights_sd["freq_router.net.2.weight"]
            fr_b2 = weights_sd["freq_router.net.2.bias"]
        except KeyError as exc:
            raise ValueError(
                f"{file_path}: chimera checkpoint is missing FreqRouter "
                f"weight key {exc} (expected freq_router.net.{{0,2}}.weight"
                f"/bias)."
            ) from exc

        if fr_w2.shape[0] != K_f:
            raise ValueError(
                f"{file_path}: FreqRouter output dim {fr_w2.shape[0]} != K_f={K_f}."
            )

        expected_in = chimera_fei_dim + chimera_sigma_dim
        if fr_w0.shape[1] != expected_in:
            raise ValueError(
                f"{file_path}: FreqRouter input dim {fr_w0.shape[1]} "
                f"!= FEI({chimera_fei_dim}) + σ({chimera_sigma_dim})."
            )
        freq_router_sd = {
            "net.0.weight": fr_w0,
            "net.0.bias": fr_b0,
            "net.2.weight": fr_w2,
            "net.2.bias": fr_b2,
        }

    content_router = _parse_chimera_content_router(
        weights_sd, file_metadata, file_path, K_c
    )
    hydra["chimera"] = {
        "num_experts_content": K_c,
        "num_experts_freq": K_f,
        "fei_feature_dim": chimera_fei_dim,
        "sigma_feature_dim": chimera_sigma_dim,
        "fei_sigma_low_div": chimera_sigma_low_div,
        # τ is not stamped — both the FreqRouter and the live
        # GlobalRouter default to 1.0 from cfg.router_tau, and the
        # production chimera.toml does not override it. If a future
        # checkpoint stamps ss_router_tau, plumb it here.
        "router_tau": float(file_metadata.get("ss_router_tau", 1.0)),
        # Freq routing mode: "learned" (FreqRouter MLP) or "fei" (hardwired
        # π_f = normalize(FEI ** (1/freq_router_fei_tau))).
        "freq_router_mode": freq_router_mode,
        "freq_router_fei_tau": freq_router_fei_tau,
        "freq_router_sd": freq_router_sd,
        # Network-level ContentRouter (chimera content_router_source
        # ="crossattn"). None ⇒ the per-Linear softmax over pooled
        # lx_c is in use (default); a dict ⇒ pool the post-LLM-
        # adapter crossattn_emb and broadcast π_c.
        "content_router_source": (
            "crossattn" if content_router is not None else "input"
        ),
        "content_router": content_router,
    }


def _finalize_dual_a_chimera(
    chimera_dual: dict,
    weights_sd: Dict[str, torch.Tensor],
    file_metadata: Dict[str, str],
    file_path: str,
) -> None:
    """Validate + enrich the dual-A ChimeraHydra bundle returned by
    ``_parse_chimera_dual_a``.

    Cross-checks the per-pool K against ``ss_num_experts_content`` /
    ``ss_num_experts_freq`` stamps, pulls the FreqRouter MLP weights
    + chimera FEI/σ dims out of metadata, attaches the optional
    network-level ContentRouter state. Mutates ``chimera_dual`` in
    place. Raises ``ValueError`` on any inconsistency — dual-A files
    must be self-consistent or the inference math goes off-spec.
    """
    is_chimera_dual_flagged = (
        str(file_metadata.get("ss_use_chimera_hydra", "")).strip().lower() == "true"
    )
    if not is_chimera_dual_flagged:
        raise ValueError(
            f"{file_path}: found chimera dual-A keys (lora_down_c / "
            "lora_down_f) but metadata is missing ss_use_chimera_hydra=true. "
            "Checkpoint is inconsistent."
        )
    try:
        K_c = int(file_metadata["ss_num_experts_content"])
        K_f = int(file_metadata["ss_num_experts_freq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{file_path}: chimera dual-A checkpoint missing or malformed "
            f"ss_num_experts_content / ss_num_experts_freq ({exc})."
        ) from exc
    if K_c != chimera_dual["num_experts_content"]:
        raise ValueError(
            f"{file_path}: ss_num_experts_content={K_c} != "
            f"max(lora_ups_c idx)+1={chimera_dual['num_experts_content']}."
        )
    if K_f != chimera_dual["num_experts_freq"]:
        raise ValueError(
            f"{file_path}: ss_num_experts_freq={K_f} != "
            f"max(lora_ups_f idx)+1={chimera_dual['num_experts_freq']}."
        )

    try:
        chimera_fei_dim = int(
            file_metadata.get("ss_chimera_fei_feature_dim", 0)
        )
        chimera_sigma_dim = int(
            file_metadata.get("ss_chimera_sigma_feature_dim", 0)
        )
        # Stamp name (network.py:3065): ``ss_chimera_fei_sigma_low_div``.
        chimera_sigma_low_div = float(
            file_metadata.get("ss_chimera_fei_sigma_low_div", 4.0)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{file_path}: malformed chimera σ/FEI feature dims ({exc})."
        ) from exc
    if chimera_fei_dim + chimera_sigma_dim <= 0:
        raise ValueError(
            f"{file_path}: chimera FreqRouter requires "
            "fei_feature_dim + sigma_feature_dim > 0 "
            f"(got FEI={chimera_fei_dim}, σ={chimera_sigma_dim})."
        )

    freq_router_mode, freq_router_fei_tau = _parse_freq_router_mode(file_metadata)
    if freq_router_mode == "fei":
        # Hardwired-FEI freq gate — no FreqRouter MLP keys on disk.
        if chimera_fei_dim != K_f:
            raise ValueError(
                f"{file_path}: freq_router_mode='fei' requires "
                f"fei_feature_dim == K_f (got FEI={chimera_fei_dim}, K_f={K_f})."
            )
        chimera_dual["freq_router_sd"] = None
    else:
        try:
            fr_w0 = weights_sd["freq_router.net.0.weight"]
            fr_b0 = weights_sd["freq_router.net.0.bias"]
            fr_w2 = weights_sd["freq_router.net.2.weight"]
            fr_b2 = weights_sd["freq_router.net.2.bias"]
        except KeyError as exc:
            raise ValueError(
                f"{file_path}: chimera dual-A checkpoint is missing FreqRouter "
                f"weight key {exc} (expected freq_router.net.{{0,2}}.weight/bias)."
            ) from exc
        if fr_w2.shape[0] != K_f:
            raise ValueError(
                f"{file_path}: FreqRouter output dim {fr_w2.shape[0]} != K_f={K_f}."
            )
        expected_in = chimera_fei_dim + chimera_sigma_dim
        if fr_w0.shape[1] != expected_in:
            raise ValueError(
                f"{file_path}: FreqRouter input dim {fr_w0.shape[1]} "
                f"!= FEI({chimera_fei_dim}) + σ({chimera_sigma_dim})."
            )
        chimera_dual["freq_router_sd"] = {
            "net.0.weight": fr_w0,
            "net.0.bias": fr_b0,
            "net.2.weight": fr_w2,
            "net.2.bias": fr_b2,
        }

    chimera_dual["fei_feature_dim"] = chimera_fei_dim
    chimera_dual["sigma_feature_dim"] = chimera_sigma_dim
    chimera_dual["fei_sigma_low_div"] = chimera_sigma_low_div
    chimera_dual["router_tau"] = float(file_metadata.get("ss_router_tau", 1.0))
    chimera_dual["freq_router_mode"] = freq_router_mode
    chimera_dual["freq_router_fei_tau"] = freq_router_fei_tau
    # Network-level ContentRouter (chimera dual-A with
    # ``content_router_source="crossattn"``). None ⇒ per-Linear softmax
    # over pooled lx_c (default).
    chimera_dual["content_router"] = _parse_chimera_content_router(
        weights_sd, file_metadata, file_path, K_c
    )
    chimera_dual["content_router_source"] = (
        "crossattn" if chimera_dual["content_router"] is not None else "input"
    )
    # Centered-gate: both pools' combine subtracts 1/K (λ is baked into the
    # saved ups, so this alone reproduces the trained forward). Matches
    # ChimeraHydraInferenceModule with ``centered_gate=True``.
    chimera_dual["centered_gate"] = (
        str(file_metadata.get("ss_chimera_centered_gate", "")).strip().lower()
        == "true"
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _make_content_router_llm_adapter_hook(
    content_router: dict,
    router_tau: float,
    router_state: dict,
):
    """Forward hook on ``diffusion_model.llm_adapter`` that fires the
    network-level ContentRouter (chimera with
    ``content_router_source="crossattn"``).

    Captures the post-LLM-adapter context ``(B, L_text, D)``, zero-pads to
    ``_T5_PAD_LEN`` (matches training where T5 tokenizes with ``padding=
    "max_length"=512`` and the cached crossattn_emb is fixed at 512), RMS-
    pools over the sequence dim, optionally parameterless-LayerNorms over
    D, runs the saved ``Linear → SiLU → Linear`` MLP, and writes ``π_c``
    to ``router_state["pi_c"]``. Per-Linear chimera hooks then broadcast
    π_c instead of running their own pooled softmax — same contract as
    ``π_f``.

    Same fp32 pin as ``_make_chimera_pre_hook`` and the training-side
    ``ContentRouter.forward``: softmax(logits/τ) at small τ underflows in
    bf16, and the network-level router carries the only signal sourcing
    π_c — losing precision here propagates straight to every chimera
    Linear's content gate.
    """
    cr_state: dict = {
        "net0_w": content_router["sd"]["net.0.weight"],
        "net0_b": content_router["sd"]["net.0.bias"],
        "net2_w": content_router["sd"]["net.2.weight"],
        "net2_b": content_router["sd"]["net.2.bias"],
        "layer_norm": bool(content_router["layer_norm"]),
        "K_c": int(content_router["K_c"]),
        "input_dim": int(content_router["input_dim"]),
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if cr_state["device"] == x.device:
            return
        for k in ("net0_w", "net0_b", "net2_w", "net2_b"):
            cr_state[k] = cr_state[k].to(device=x.device, dtype=torch.float32)
        cr_state["device"] = x.device

    @torch._dynamo.disable
    def content_router_hook(module, inputs, output):
        # ComfyUI's Anima.LLMAdapter.forward returns the post-T5 features
        # directly (B, L_text, D); pad_to_512 happens in
        # ``preprocess_text_embeds`` AFTER this call. Re-do the same pad
        # so the RMS-pool denominator matches training.
        ctx = output if torch.is_tensor(output) else output[0]
        if ctx.dim() != 3:
            return  # defensive: shape contract broken — skip this step
        _ensure_on_device(ctx)

        D = int(ctx.shape[-1])
        if D != cr_state["input_dim"]:
            # Stamped input_dim must match the live adapter's hidden size;
            # mismatch means the ContentRouter was trained against a
            # different DiT than the one ComfyUI loaded. Fail loud rather
            # than silently emit a misshaped Linear matmul.
            raise RuntimeError(
                f"ChimeraHydra ContentRouter: adapter output D={D} != "
                f"stamped input_dim={cr_state['input_dim']} — checkpoint "
                "was trained against a different DiT hidden size."
            )

        L = int(ctx.shape[1])
        if L < _T5_PAD_LEN:
            pad = _T5_PAD_LEN - L
            ctx = torch.nn.functional.pad(ctx, (0, 0, 0, pad))
        elif L > _T5_PAD_LEN:
            ctx = ctx[:, :_T5_PAD_LEN, :]

        # RMS-pool over the sequence dim → (B, D). fp32 to match training-
        # side ContentRouter.forward (which casts to fp32 before the LN +
        # MLP). The padding tail is zeros so the denominator is 512 for
        # both training and inference — same as the network saw.
        x32 = ctx.float()
        pooled = x32.pow(2).mean(dim=1).sqrt()

        if cr_state["layer_norm"]:
            # Parameterless LN over D — same as ``torch.nn.LayerNorm(D,
            # elementwise_affine=False)`` used training-side. Inlined to
            # avoid building a stateful module per inference run.
            mean = pooled.mean(dim=-1, keepdim=True)
            var = pooled.var(dim=-1, keepdim=True, unbiased=False)
            pooled = (pooled - mean) * torch.rsqrt(var + 1e-5)

        h = torch.nn.functional.linear(pooled, cr_state["net0_w"], cr_state["net0_b"])
        h = torch.nn.functional.silu(h)
        logits = torch.nn.functional.linear(
            h, cr_state["net2_w"], cr_state["net2_b"]
        )
        pi_c = torch.softmax(logits / router_tau, dim=-1)
        if pi_c.shape[-1] != cr_state["K_c"]:
            raise RuntimeError(
                f"ChimeraHydra: ContentRouter emitted K_c="
                f"{pi_c.shape[-1]}, expected {cr_state['K_c']}."
            )
        router_state["pi_c"] = pi_c

    return content_router_hook


def _make_chimera_pre_hook(
    router_state: dict,
    freq_router_sd: Optional[Dict[str, torch.Tensor]],
    fei_feature_dim: int,
    sigma_feature_dim: int,
    fei_sigma_low_div: float,
    router_tau: float,
    K_f: int,
    freq_router_mode: str = "learned",
    freq_router_fei_tau: float = 1.0,
):
    """Pre-hook for ChimeraHydra: capture σ, compute FEI, emit ``π_f``.

    Same wrapping contract as ``_make_router_pre_hook`` (attached to
    ``diffusion_model._forward_pre_hooks`` via ``add_object_patch``). Once
    per denoising step it computes FEI on the latent and writes
    ``router_state["pi_f"]`` (shape ``(B, K_f)``); each per-Linear chimera
    hook then concatenates ``π_f`` onto its own K_c content gate.

    Two modes:

    * ``"learned"`` — evaluate the network-level FreqRouter MLP on
      ``concat(FEI, sinusoidal(σ))`` → softmax/τ. Concat order matches
      ``networks/lora_anima/network.py::set_fei``'s chimera branch
      (``[FEI, sinusoidal(σ)]``); reversing it scrambles the input.
    * ``"fei"`` — hardwire ``π_f = normalize(FEI ** (1/freq_router_fei_tau))``
      (``fei_temperature``). No FreqRouter MLP, no σ-features; the FEI
      band-simplex IS the gate (K_f == fei bands). Matches training-side
      ``set_fei`` under ``freq_router_mode="fei"``.
    """
    fei_mode = str(freq_router_mode).lower() == "fei"
    fr_state: dict = {"device": None}
    if not fei_mode:
        fr_state.update(
            {
                "net0_w": freq_router_sd["net.0.weight"],
                "net0_b": freq_router_sd["net.0.bias"],
                "net2_w": freq_router_sd["net.2.weight"],
                "net2_b": freq_router_sd["net.2.bias"],
            }
        )

    def _ensure_fr_on_device(x: torch.Tensor) -> None:
        if fei_mode or fr_state["device"] == x.device:
            return
        for k in ("net0_w", "net0_b", "net2_w", "net2_b"):
            fr_state[k] = fr_state[k].to(device=x.device, dtype=torch.float32)
        fr_state["device"] = x.device

    @torch._dynamo.disable
    def chimera_pre_hook(module, args):
        if len(args) >= 2 and args[1] is not None:
            # fp32 normalize once per step (see _make_router_pre_hook).
            router_state["sigma"] = args[1].detach().float()
        if len(args) >= 1 and args[0] is not None:
            x = args[0].detach()
            if x.dim() == 5:
                x = x.squeeze(2)
            h_lat = int(x.shape[-2])
            w_lat = int(x.shape[-1])
            sigma_low = fei_sigma_low(h_lat, w_lat, fei_sigma_low_div)
            fei = compute_fei_2band(x, sigma_low)  # (B, 2), fp32
            router_state["fei"] = fei
            _ensure_fr_on_device(fei)

            if fei_mode:
                # Hardwired-FEI gate: π_f = normalize(FEI**(1/τ)) over the
                # first K_f bands. No MLP, no σ — mirrors training set_fei.
                pi_f = fei_temperature(fei[:, :K_f], freq_router_fei_tau)
                router_state["pi_f"] = pi_f
                if pi_f.shape[-1] != K_f:
                    raise RuntimeError(
                        f"ChimeraHydra: hardwired-FEI gate emitted K_f="
                        f"{pi_f.shape[-1]}, expected {K_f}."
                    )
                return

            # Build router input: cat([FEI[:, :fei_dim], σ-sin features]).
            # Either slice may be empty.
            parts = []
            if fei_feature_dim > 0:
                parts.append(fei[:, :fei_feature_dim])
            if sigma_feature_dim > 0:
                sigma = router_state.get("sigma")
                if sigma is None:
                    return
                sigma_feat = sigma_sinusoidal_features(sigma, sigma_feature_dim)
                if sigma_feat.shape[0] == 1 and fei.shape[0] != 1:
                    sigma_feat = sigma_feat.expand(fei.shape[0], -1)
                parts.append(sigma_feat)
            if not parts:
                return
            router_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

            # FreqRouter: Linear → SiLU → Linear → softmax/τ. fp32 for the
            # same precision reason as GlobalRouter — softmax(logits / τ)
            # at small τ underflows in bf16.
            h = torch.nn.functional.linear(
                router_in, fr_state["net0_w"], fr_state["net0_b"]
            )
            h = torch.nn.functional.silu(h)
            logits = torch.nn.functional.linear(
                h, fr_state["net2_w"], fr_state["net2_b"]
            )
            router_state["pi_f"] = torch.softmax(logits / router_tau, dim=-1)
            # (Defensive) shape check — if K_f drifted, fail loud rather
            # than silently emit a misshaped gate cat downstream.
            if router_state["pi_f"].shape[-1] != K_f:
                raise RuntimeError(
                    f"ChimeraHydra: FreqRouter emitted K_f="
                    f"{router_state['pi_f'].shape[-1]}, expected {K_f}."
                )

    return chimera_pre_hook


def _make_chimera_hook(params: dict, strength: float, router_state: dict):
    """Per-Linear hook for ChimeraHydra (dual-pool additive routing).

    Differs from ``_make_hydra_hook``:

      * The per-Linear router has K_c outputs (content pool only); its
        input is pooled rank-R lx with NO σ/FEI columns.
      * π_f arrives precomputed in ``router_state["pi_f"]`` (shape
        ``(B, K_f)``); the hook concatenates ``[π_c, π_f]`` and runs the
        standard Hydra einsum/bmm over the full E = K_c + K_f experts.
      * σ-band partition is unsupported by construction — chimera trains
        with ``specialize_experts_by_sigma_buckets=False``.

    Matches ``networks/lora_modules/chimera.py::ChimeraHydraLoRAModule
    .forward`` (minus the T-LoRA mask, which is training-only — see
    ``[[project_tlora_inference_full_rank]]``).
    """
    state = {
        "lora_down": params["lora_down"],
        "lora_ups": params["lora_ups"],  # (E, out, rank)
        "router_w": params.get("router_w"),  # (K_c, rank), None under global router
        "router_b": params.get("router_b"),  # (K_c,), None under global router
        "inv_scale": params.get("inv_scale"),  # (in_dim,) or None
        "scale": params["scale"],
        "K_c": int(params["num_experts_content"]),
        "K_f": int(params["num_experts_freq"]),
        # When True, π_c is broadcast from the network-level ContentRouter
        # via ``router_state["pi_c"]`` (set by the llm_adapter forward_hook)
        # and the per-Linear softmax over pooled lx is skipped. Matches
        # ChimeraHydraInferenceModule with ``use_global_content_router=True``.
        "global_content_router": bool(params.get("global_content_router", False)),
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        for k in ("lora_down", "lora_ups", "router_w", "router_b", "inv_scale"):
            if state[k] is not None:
                state[k] = state[k].to(device=x.device, dtype=torch.float32)
        state["device"] = x.device

    def chimera_hook(module, inputs, output):
        x = inputs[0]
        _ensure_on_device(x)

        x_lora = x.float()
        if state["inv_scale"] is not None:
            x_lora = x_lora * state["inv_scale"]

        # down projection (B, *, rank), fp32 — feeds the content router
        # AND the gate-weighted bmm.
        lx = torch.nn.functional.linear(x_lora, state["lora_down"])

        B = lx.shape[0]
        if state["global_content_router"]:
            # π_c broadcast from the network-level ContentRouter (run once
            # per step in the llm_adapter forward_hook). Falls back to
            # uniform 1/K_c if the hook hasn't fired yet — matches the
            # _content_routing_weights placeholder init on
            # ChimeraHydraInferenceModule.
            pi_c = router_state.get("pi_c")
            if pi_c is None:
                pi_c = torch.full(
                    (B, state["K_c"]),
                    1.0 / max(state["K_c"], 1),
                    device=lx.device,
                    dtype=lx.dtype,
                )
            else:
                pi_c = pi_c.to(dtype=lx.dtype)
                if pi_c.dim() == 1:
                    pi_c = pi_c.unsqueeze(0)
                if pi_c.shape[0] == 1 and B != 1:
                    pi_c = pi_c.expand(B, -1)
        else:
            # Content router: pooled rank-R → softmax → π_c (B, K_c). No σ/FEI
            # — chimera deliberately starves the content router of frequency
            # information (see proposal §"Why HydraLoRA's auto-specialization
            # argument gets stronger").
            if lx.dim() >= 3:
                pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
            else:
                pooled = lx
            logits_c = torch.nn.functional.linear(
                pooled, state["router_w"], state["router_b"]
            )
            pi_c = torch.softmax(logits_c, dim=-1)  # (B, K_c)

        pi_f = router_state.get("pi_f")
        if pi_f is None:
            # FreqRouter pre-hook didn't fire (e.g. compile cache miss
            # before the first step) — fall back to uniform 1/K_f. Matches
            # ChimeraHydraLoRAModule's placeholder buffer init.
            pi_f = torch.full(
                (B, state["K_f"]),
                1.0 / max(state["K_f"], 1),
                device=lx.device,
                dtype=lx.dtype,
            )
        else:
            pi_f = pi_f.to(dtype=lx.dtype)
            if pi_f.dim() == 1:
                pi_f = pi_f.unsqueeze(0)
            if pi_f.shape[0] == 1 and B != 1:
                pi_f = pi_f.expand(B, -1)

        gate = torch.cat([pi_c, pi_f], dim=-1)  # (B, K_c + K_f)

        combined = torch.einsum("be,eor->bor", gate, state["lora_ups"])

        orig_shape = lx.shape
        lx_3d = lx.reshape(B, -1, orig_shape[-1])
        delta = torch.bmm(lx_3d, combined.transpose(1, 2)).reshape(*orig_shape[:-1], -1)
        return output + (delta * (state["scale"] * strength)).to(output.dtype)

    return chimera_hook


def _make_chimera_dual_a_hook(params: dict, strength: float, router_state: dict):
    """Per-Linear hook for ChimeraHydra dual-A (two independent A's per
    Linear, one per pool).

    Differs from ``_make_chimera_hook`` (single-A legacy):

      * Two down projections (``lora_down_c`` and ``lora_down_f``) instead
        of one shared ``lora_down`` — the two pools see disjoint latents.
      * Two B stacks (``lora_up_c_stack`` / ``lora_up_f_stack``), each
        gated by its own pool's gate, summed at the output.
      * Content router pools ``lx_c`` (the content branch's latent), NOT
        a shared ``lx``. Mirrors
        ``ChimeraHydraInferenceModule._compute_content_gate``.
      * The training-time ``lambda_c`` / ``lambda_f`` scalars are baked
        into the saved ``lora_down_{c,f}`` / ``lora_up_{c,f}`` via the
        sqrt-split in ``_convert_chimera_dual_a_to_hydra`` — no extra
        scaling factor at inference. ``alpha`` / ``inv_scale`` are
        passed through for the standard SmoothQuant rebalance only.

    Matches ``networks/lora_modules/chimera.py::ChimeraHydraInferenceModule
    .forward`` (T-LoRA's content mask stays training-only — see
    ``[[project_tlora_inference_full_rank]]``).
    """
    state = {
        "lora_down_c": params["lora_down_c"],
        "lora_down_f": params["lora_down_f"],
        "lora_up_c_stack": params["lora_up_c_stack"],  # (K_c, out, rank)
        "lora_up_f_stack": params["lora_up_f_stack"],  # (K_f, out, rank)
        "router_w": params.get("router_w"),  # (K_c, rank), None under global router
        "router_b": params.get("router_b"),  # (K_c,), None under global router
        "inv_scale": params.get("inv_scale"),  # (in_dim,) or None
        "K_c": int(params["num_experts_content"]),
        "K_f": int(params["num_experts_freq"]),
        # When True, π_c is broadcast from the network-level ContentRouter
        # via ``router_state["pi_c"]`` (set by the llm_adapter forward_hook)
        # and the per-Linear pooled-lx_c softmax is skipped. Matches
        # ChimeraHydraInferenceModule with ``use_global_content_router=True``.
        "global_content_router": bool(params.get("global_content_router", False)),
        # Centered-gate: subtract 1/K per pool before each combine (parity with
        # ChimeraHydraInferenceModule.forward — λ is baked into the saved ups).
        "centered_gate": bool(params.get("centered_gate", False)),
        "device": None,
    }

    def _ensure_on_device(x: torch.Tensor) -> None:
        if state["device"] == x.device:
            return
        for k in (
            "lora_down_c",
            "lora_down_f",
            "lora_up_c_stack",
            "lora_up_f_stack",
            "router_w",
            "router_b",
            "inv_scale",
        ):
            if state[k] is not None:
                state[k] = state[k].to(device=x.device, dtype=torch.float32)
        state["device"] = x.device

    def chimera_dual_a_hook(module, inputs, output):
        x = inputs[0]
        _ensure_on_device(x)

        x_lora = x.float()
        if state["inv_scale"] is not None:
            x_lora = x_lora * state["inv_scale"]

        # Two independent down projections (B, *, rank) — content + freq.
        # No shared lx: routers and ups for each pool see distinct latents,
        # which is the whole point of going dual-A (input-side ortho via
        # the SVD partition at init).
        lx_c = torch.nn.functional.linear(x_lora, state["lora_down_c"])
        lx_f = torch.nn.functional.linear(x_lora, state["lora_down_f"])

        B = lx_c.shape[0]
        if state["global_content_router"]:
            # π_c broadcast from the network-level ContentRouter (run once
            # per step in the llm_adapter forward_hook). Uniform 1/K_c
            # fallback if the hook hasn't fired yet, matching
            # ChimeraHydraInferenceModule's placeholder buffer.
            pi_c = router_state.get("pi_c")
            if pi_c is None:
                pi_c = torch.full(
                    (B, state["K_c"]),
                    1.0 / max(state["K_c"], 1),
                    device=lx_c.device,
                    dtype=lx_c.dtype,
                )
            else:
                pi_c = pi_c.to(dtype=lx_c.dtype)
                if pi_c.dim() == 1:
                    pi_c = pi_c.unsqueeze(0)
                if pi_c.shape[0] == 1 and B != 1:
                    pi_c = pi_c.expand(B, -1)
        else:
            # Content router on pooled lx_c. RMS pool over the sequence dim
            # matches ``_compute_content_gate``; using lx_c (not lx_f or x)
            # is load-bearing per the chimera proposal — pooling lx_f would
            # cross-couple the two pools and defeat the input-separation
            # argument.
            if lx_c.dim() >= 3:
                pooled_c = lx_c.reshape(B, -1, lx_c.shape[-1]).pow(2).mean(dim=1).sqrt()
            else:
                pooled_c = lx_c
            logits_c = torch.nn.functional.linear(
                pooled_c, state["router_w"], state["router_b"]
            )
            pi_c = torch.softmax(logits_c, dim=-1)  # (B, K_c)

        pi_f = router_state.get("pi_f")
        if pi_f is None:
            # FreqRouter pre-hook didn't fire (compile cache miss before
            # first step or wrapper bypass) — fall back to uniform 1/K_f.
            # Matches ChimeraHydraInferenceModule's placeholder buffer.
            pi_f = torch.full(
                (B, state["K_f"]),
                1.0 / max(state["K_f"], 1),
                device=lx_c.device,
                dtype=lx_c.dtype,
            )
        else:
            pi_f = pi_f.to(dtype=lx_c.dtype)
            if pi_f.dim() == 1:
                pi_f = pi_f.unsqueeze(0)
            if pi_f.shape[0] == 1 and B != 1:
                pi_f = pi_f.expand(B, -1)

        # Centered-gate parity: subtract per-pool 1/K so a uniform gate
        # contributes 0 (mirrors training's recentered combine).
        if state["centered_gate"]:
            pi_c = pi_c - (1.0 / max(state["K_c"], 1))
            pi_f = pi_f - (1.0 / max(state["K_f"], 1))

        # Gate-weighted per-pool combined ups (B, out, rank). Two einsums
        # because the two pools have different K — keeps the math 1:1
        # with the training/inference module rather than padding to a
        # joint stack.
        comb_c = torch.einsum("bc,cor->bor", pi_c, state["lora_up_c_stack"])
        comb_f = torch.einsum("bf,for->bor", pi_f, state["lora_up_f_stack"])

        orig_shape = lx_c.shape
        lx_c_3d = lx_c.reshape(B, -1, orig_shape[-1])
        lx_f_3d = lx_f.reshape(B, -1, orig_shape[-1])
        out_c = torch.bmm(lx_c_3d, comb_c.transpose(1, 2))
        out_f = torch.bmm(lx_f_3d, comb_f.transpose(1, 2))
        delta = (out_c + out_f).reshape(*orig_shape[:-1], -1)
        return output + (delta * strength).to(output.dtype)

    return chimera_dual_a_hook


# ---------------------------------------------------------------------------
# Application (dual-A path; single-A lives in adapter.py inside
# _apply_hydra_live_to_model)
# ---------------------------------------------------------------------------


def _apply_chimera_dual_a_to_model(
    model, chimera_data: dict, strength: float
) -> int:
    """Install live-routing forward hooks on each ChimeraHydra dual-A
    Linear.

    Counterpart to ``_apply_hydra_live_to_model``'s chimera branch but
    for the dual-A on-disk format (post-c4851b6 chimera). Same FreqRouter
    pre-hook (one network-level π_f per step on
    ``concat(FEI, sinusoidal(σ))``) — only the per-Linear math changes
    (two A's + two B stacks, summed at the output). Mutually exclusive
    with the legacy single-A chimera path; the loader picks one based
    on the key shape and produces exactly one of
    ``hydra["chimera"]`` / ``chimera_dual_a``.

    Returns number of hooks installed.
    """
    import comfy.lora

    if strength == 0:
        return 0

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})

    # Shared routing state — same dict the single-A chimera path uses.
    # FreqRouter pre-hook writes ``pi_f`` once per denoising step; every
    # per-Linear hook reads it.
    sigma_state: dict = {}

    diffusion_model = model.get_model_object("diffusion_model")
    router_pre_hook = _make_chimera_pre_hook(
        sigma_state,
        chimera_data["freq_router_sd"],
        fei_feature_dim=int(chimera_data["fei_feature_dim"]),
        sigma_feature_dim=int(chimera_data["sigma_feature_dim"]),
        fei_sigma_low_div=float(chimera_data["fei_sigma_low_div"]),
        router_tau=float(chimera_data["router_tau"]),
        K_f=int(chimera_data["num_experts_freq"]),
        freq_router_mode=str(chimera_data.get("freq_router_mode", "learned")),
        freq_router_fei_tau=float(chimera_data.get("freq_router_fei_tau", 1.0)),
    )
    new_pre_hooks = OrderedDict(diffusion_model._forward_pre_hooks)
    new_pre_hooks[id(router_pre_hook)] = router_pre_hook
    model.add_object_patch("diffusion_model._forward_pre_hooks", new_pre_hooks)

    # ChimeraHydra global ContentRouter (content_router_source="crossattn"):
    # install a forward_hook on ``diffusion_model.llm_adapter`` that pools
    # the post-T5 features and writes π_c into ``sigma_state["pi_c"]``. Per-
    # Linear hooks below are flagged ``global_content_router=True`` so they
    # broadcast π_c instead of running their own pooled-lx_c softmax. No-op
    # when the per-Linear router is in use (default).
    global_cr = chimera_data.get("content_router")
    if global_cr is not None:
        if not hasattr(diffusion_model, "llm_adapter"):
            raise RuntimeError(
                "ChimeraHydra content_router_source='crossattn' requires "
                "diffusion_model.llm_adapter (Anima DiT). Loaded model has "
                "no llm_adapter attribute."
            )
        cr_hook = _make_content_router_llm_adapter_hook(
            global_cr,
            router_tau=float(chimera_data["router_tau"]),
            router_state=sigma_state,
        )
        new_adapter_hooks = OrderedDict(diffusion_model.llm_adapter._forward_hooks)
        new_adapter_hooks[id(cr_hook)] = cr_hook
        model.add_object_patch(
            "diffusion_model.llm_adapter._forward_hooks", new_adapter_hooks
        )

    K_c = int(chimera_data["num_experts_content"])
    K_f = int(chimera_data["num_experts_freq"])
    global_cr_on = global_cr is not None

    patched = 0
    skipped: list[str] = []
    for prefix, mod in chimera_data["modules"].items():
        # Per-Linear router_w/router_b are absent under the global
        # ContentRouter — ChimeraHydraInferenceModule constructs
        # ``self.router = None`` under ``use_global_content_router=True``,
        # so the saved state_dict skips those keys entirely.
        required: tuple[str, ...]
        if global_cr_on:
            required = ("lora_down_c", "lora_down_f", "lora_ups_c")
        else:
            required = (
                "lora_down_c", "lora_down_f", "lora_ups_c", "router_w", "router_b",
            )
        missing = [k for k in required if k not in mod]
        if missing:
            skipped.append(f"{prefix}: missing {missing}")
            continue
        # K_f == 0 is structurally legal (degenerates to "content only"),
        # but the training cfg refuses it — so a missing lora_ups_f stack
        # under K_f > 0 indicates a malformed checkpoint.
        if K_f > 0 and "lora_ups_f" not in mod:
            skipped.append(f"{prefix}: missing lora_ups_f under K_f={K_f}")
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

        ups_c_dict = mod["lora_ups_c"]
        ups_c_stacked = torch.stack(
            [ups_c_dict[i] for i in sorted(ups_c_dict.keys())], dim=0
        )
        if ups_c_stacked.shape[0] != K_c:
            skipped.append(
                f"{prefix}: lora_ups_c stack size {ups_c_stacked.shape[0]} != K_c={K_c}"
            )
            continue
        if K_f > 0:
            ups_f_dict = mod["lora_ups_f"]
            ups_f_stacked = torch.stack(
                [ups_f_dict[i] for i in sorted(ups_f_dict.keys())], dim=0
            )
            if ups_f_stacked.shape[0] != K_f:
                skipped.append(
                    f"{prefix}: lora_ups_f stack size {ups_f_stacked.shape[0]} != K_f={K_f}"
                )
                continue
        else:
            ups_f_stacked = torch.empty(
                0, ups_c_stacked.shape[1], ups_c_stacked.shape[2]
            )

        rank = mod["lora_down_c"].shape[0]
        if not global_cr_on:
            if mod["router_w"].shape != (K_c, rank):
                skipped.append(
                    f"{prefix}: content router shape {tuple(mod['router_w'].shape)} "
                    f"!= (K_c={K_c}, rank={rank})"
                )
                continue

        params = {
            "lora_down_c": mod["lora_down_c"],
            "lora_down_f": mod["lora_down_f"],
            "lora_up_c_stack": ups_c_stacked,
            "lora_up_f_stack": ups_f_stacked,
            "router_w": None if global_cr_on else mod["router_w"],
            "router_b": None if global_cr_on else mod["router_b"],
            "inv_scale": mod.get("inv_scale"),
            "num_experts_content": K_c,
            "num_experts_freq": K_f,
            "global_content_router": global_cr_on,
            "centered_gate": bool(chimera_data.get("centered_gate", False)),
        }
        hook = _make_chimera_dual_a_hook(params, strength, sigma_state)

        new_hooks = OrderedDict(linear._forward_hooks)
        new_hooks[id(hook)] = hook
        model.add_object_patch(f"{module_path}._forward_hooks", new_hooks)
        patched += 1

    if skipped:
        logger.warning(
            f"ChimeraHydra dual-A skipped {len(skipped)} prefix(es); "
            f"first few: {skipped[:5]}"
        )
    cr_tag = (
        f", ContentRouter=crossattn(in={global_cr['input_dim']}, "
        f"LN={'on' if global_cr['layer_norm'] else 'off'})"
        if global_cr_on
        else ""
    )
    if str(chimera_data.get("freq_router_mode", "learned")) == "fei":
        freq_desc = (
            f"freq=hardwired-FEI(τ={chimera_data.get('freq_router_fei_tau', 1.0):g})"
        )
    else:
        freq_desc = (
            f"FreqRouter input=FEI({chimera_data['fei_feature_dim']}) + "
            f"σ({chimera_data['sigma_feature_dim']}), "
            f"τ={chimera_data['router_tau']:g}"
        )
    logger.info(
        f"ChimeraHydra dual-A live-routing installed {patched} hooks "
        f"(strength={strength}, K_c={K_c} + K_f={K_f}, {freq_desc}, "
        f"σ_low_div={chimera_data['fei_sigma_low_div']:g}{cr_tag})"
    )
    return patched
