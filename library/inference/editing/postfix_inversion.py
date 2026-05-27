"""Per-image inversion of the orthogonal postfix tail.

Probe instrument for ``docs/proposal/postfix_residual_per_image_inversion.md``.
Optimizes a K-dim scale vector ``s`` so that

    ψ = concat(T5(tags), Q @ diag(s))      (S=512, D, last K rows replaced)

minimizes flow-matching loss against the frozen DiT for one image, where
``Q ∈ R^{K×D}`` is a fixed row-orthonormal basis (typically the top-K right
singular vectors of the cached T5 corpus). The K trainable parameters live in
``s``; ``Q`` is frozen.

The body of the loop is lifted from ``archive/inversion/invert_reference.py``
with three changes:

1. Prefix is the actual cached ``{stem}_anima_te.safetensors`` (the tags-
   conditioned T5 output for this image), not an encoded ``--template``.
2. The K trainable slots become ``Q @ diag(s)`` (K params per image) instead
   of a free ``(K, D)`` tensor (K·D params).
3. Splice position is ``end_of_sequence`` — same as the default postfix path.

Not a deployable adapter. The output is the K-vector ``s`` and a loss CSV,
which downstream analysis (PCA / clustering / lane-discipline / multi-seed
functional cosine — see proposal §Probe metrics) consumes.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512


# region Orthonormal basis builders
#
# Lifted verbatim from the (now archived) postfix method
# (``_archive/postfix/networks/methods/postfix.py``) so the inversion probe
# stays self-contained — it is the only live consumer of the cond+ortho basis
# construction. Both functions depend on stdlib + torch + safetensors only.


def _build_svd_te_basis(
    cache_dir: str,
    K: int,
    D: int,
    num_files: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Top-K right singular vectors of a sample of cached adapter outputs,
    row-shuffled deterministically.

    Reads `*_anima_te.safetensors` under ``cache_dir``, masks padding via
    `attn_mask_v0`, accumulates non-padding rows into an (M, D) matrix, runs
    full SVD, and returns the top-K rows of V_h (the K right singular vectors
    with the largest singular values). The K rows are row-orthonormal (V_h has
    orthonormal rows by construction).

    Row-shuffle (deterministic from `seed`) breaks the "slot-0 is the principal
    direction" inductive bias that would otherwise let the optimizer collapse
    its budget onto the top slot.
    """
    if K > D:
        raise ValueError(
            f"cond mode requires K ({K}) ≤ D ({D}); cannot build K orthonormal "
            "rows in a D-dim space"
        )

    from safetensors.torch import load_file as _load_file

    files = sorted(
        glob.glob(
            os.path.join(cache_dir, "**", "*_anima_te.safetensors"),
            recursive=True,
        )
    )
    if not files:
        raise FileNotFoundError(
            f"ortho_basis='svd_te' requires cached *_anima_te.safetensors files "
            f"under {cache_dir!r} (run `make preprocess-te` first)"
        )

    rng = torch.Generator().manual_seed(int(seed))
    if len(files) > num_files:
        idx = torch.randperm(len(files), generator=rng)[:num_files].tolist()
        files = [files[i] for i in sorted(idx)]

    chunks: list[torch.Tensor] = []
    for path in files:
        sd = _load_file(path)
        emb = sd["crossattn_emb_v0"].float()  # (S, D)
        mask = sd["attn_mask_v0"].bool()       # (S,)
        if emb.shape[-1] != D:
            raise ValueError(
                f"cached embed dim {emb.shape[-1]} != requested D={D} (file: {path})"
            )
        if mask.any():
            chunks.append(emb[mask])

    if not chunks:
        raise RuntimeError(f"no non-padding tokens found across {len(files)} cached files")

    A = torch.cat(chunks, dim=0)  # (M, D)
    # full_matrices=False → V_h: (min(M, D), D); top-K rows are the K right
    # singular vectors with the largest singular values.
    _U, _S, V_h = torch.linalg.svd(A, full_matrices=False)
    if V_h.shape[0] < K:
        raise RuntimeError(
            f"svd_te: only {V_h.shape[0]} singular vectors available (< K={K}); "
            "use more cached files or smaller K"
        )
    top = V_h[:K].contiguous()  # (K, D), row-orthonormal

    # Deterministic row-shuffle: scrambles the "slot k = k-th principal
    # direction" ordering so the optimizer can't latch onto slot 0.
    perm = torch.randperm(K, generator=rng)
    return top[perm].contiguous()


def _make_orthonormal_basis(
    K: int,
    D: int,
    kind: str = "random",
    *,
    te_cache_dir: Optional[str] = None,
    svd_num_files: int = 256,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build a (K, D) row-orthonormal basis (K rows, D-dim each).

    QR on a (D, K) Gaussian matrix gives Q with orthonormal columns; transpose
    to get K row-orthonormal vectors in R^D. Requires K ≤ D.

    Supports two basis kinds:
      - ``"random"``: QR of a Gaussian (D, K) matrix.
      - ``"svd_te"``: top-K right singular vectors of cached
        ``_anima_te.safetensors`` adapter outputs under ``te_cache_dir``,
        row-shuffled with ``seed``. See ``_build_svd_te_basis``.
    """
    if K > D:
        raise ValueError(
            f"cond mode requires K ({K}) ≤ D ({D}); cannot build K orthonormal "
            "rows in a D-dim space"
        )
    if kind == "random":
        M = torch.randn(D, K, generator=generator)
        Q, _R = torch.linalg.qr(M)  # Q: (D, K), columns orthonormal
        return Q.T.contiguous()  # (K, D), rows orthonormal
    if kind == "svd_te":
        if te_cache_dir is None:
            raise ValueError(
                "ortho_basis='svd_te' requires te_cache_dir kwarg (path to a directory "
                "of cached *_anima_te.safetensors files, typically post_image_dataset/lora)"
            )
        return _build_svd_te_basis(
            te_cache_dir, K, D, num_files=svd_num_files, seed=seed
        )
    raise NotImplementedError(
        f"ortho_basis={kind!r}: only 'random' and 'svd_te' are implemented."
    )


# endregion


# region Basis


def load_or_build_basis(
    K: int,
    D: int,
    *,
    kind: str = "svd_te",
    te_cache_dir: Optional[str] = None,
    basis_path: Optional[str] = None,
    svd_num_files: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Return a frozen ``(K, D)`` row-orthonormal basis ``Q``.

    Caching contract: if ``basis_path`` exists AND has the right shape AND was
    built for the same ``kind``/``seed`` metadata, load from disk. Otherwise
    compute fresh and save to ``basis_path`` (when set). The SVD over a 256-
    file sample is the expensive piece — caching avoids redoing it across
    sweep modes and seeds.
    """
    if basis_path and os.path.exists(basis_path):
        cached = torch.load(basis_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict):
            Q = cached.get("basis")
            meta_kind = cached.get("kind")
            meta_seed = cached.get("seed")
        else:
            Q = cached
            meta_kind, meta_seed = None, None
        if Q is not None and Q.shape == (K, D):
            if meta_kind is not None and meta_kind != kind:
                logger.warning(
                    f"basis cache at {basis_path} was built with kind={meta_kind!r} "
                    f"but caller requested {kind!r}; rebuilding"
                )
            elif meta_seed is not None and int(meta_seed) != int(seed):
                logger.warning(
                    f"basis cache at {basis_path} was built with seed={meta_seed} "
                    f"but caller requested {seed}; rebuilding"
                )
            else:
                logger.info(
                    f"Loaded cached basis: {basis_path} (K={K}, D={D}, kind={meta_kind or kind})"
                )
                return Q.float().contiguous()

    if kind == "svd_te":
        if te_cache_dir is None:
            raise ValueError(
                "basis kind 'svd_te' requires te_cache_dir (a directory of cached "
                "_anima_te.safetensors files)"
            )
        Q = _build_svd_te_basis(te_cache_dir, K, D, num_files=svd_num_files, seed=seed)
        logger.info(
            f"Built SVD-of-cached-TE basis: K={K} D={D} from {te_cache_dir} "
            f"(num_files={svd_num_files}, seed={seed})"
        )
    elif kind == "random":
        gen = torch.Generator().manual_seed(int(seed))
        Q = _make_orthonormal_basis(K, D, kind="random", generator=gen)
        logger.info(f"Built random orthonormal basis: K={K} D={D} (seed={seed})")
    else:
        raise ValueError(f"unknown basis kind {kind!r}: expected 'svd_te' or 'random'")

    if basis_path:
        os.makedirs(os.path.dirname(basis_path) or ".", exist_ok=True)
        torch.save(
            {"basis": Q.float().contiguous(), "kind": kind, "seed": int(seed)},
            basis_path,
        )
        logger.info(f"Cached basis to {basis_path}")
    return Q.float().contiguous()


# endregion


# region Prefix


def load_cached_prefix(te_path: str, *, device: torch.device) -> torch.Tensor:
    """Load ``T5(tags)`` prefix from ``{stem}_anima_te.safetensors``.

    Returns a ``(1, MAX_SEQ_LEN, D)`` bf16 tensor, zero-padded to MAX_SEQ_LEN.
    Padding rows (where ``attn_mask_v0`` is False) are zeroed explicitly so the
    splice region is guaranteed-clean.
    """
    sd = load_file(te_path)
    if "crossattn_emb_v0" in sd:
        emb = sd["crossattn_emb_v0"].float()
    elif "crossattn_emb" in sd:
        emb = sd["crossattn_emb"].float()
    else:
        raise KeyError(
            f"{te_path}: missing 'crossattn_emb_v0' / 'crossattn_emb' "
            f"(keys: {list(sd.keys())[:8]}...)"
        )

    if "attn_mask_v0" in sd:
        mask = sd["attn_mask_v0"].bool()
        if mask.shape[0] == emb.shape[0]:
            emb = emb.clone()
            emb[~mask] = 0.0
    if emb.shape[0] < MAX_SEQ_LEN:
        emb = F.pad(emb, (0, 0, 0, MAX_SEQ_LEN - emb.shape[0]))
    elif emb.shape[0] > MAX_SEQ_LEN:
        emb = emb[:MAX_SEQ_LEN]

    return emb.unsqueeze(0).to(device=device, dtype=torch.bfloat16)


def assemble_emb(prefix_emb: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Concat ``[prefix[:, :S-K], tail]`` — end_of_sequence splice.

    Mirrors ``PostfixNetwork.append_postfix`` for ``splice_position='end_of_sequence'``.
    Autograd flows through ``tail``; prefix is constant.
    """
    K = tail.shape[-2]
    S = prefix_emb.shape[1]
    return torch.cat(
        [prefix_emb[:, : S - K, :], tail.to(prefix_emb.dtype)],
        dim=1,
    )


# endregion


# region Optimization core


def sample_sigmas(
    batch_size: int,
    device: torch.device,
    *,
    sigma_sampling: str = "uniform",
    sigmoid_scale: float = 1.0,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
) -> torch.Tensor:
    """Sigma sampler — uniform or sigmoid, rescaled to ``[sigma_min, sigma_max]``.

    Extends the rescale-to-[sigma_min, 1.0] convention from
    ``archive/inversion/invert_embedding.py:sample_sigmas`` with an upper
    bound, so callers can restrict supervision to either end of the trajectory
    (e.g. ``sigma_max=0.25`` for low-σ-only inversion).
    """
    if sigma_sampling == "sigmoid":
        sigmas = torch.sigmoid(sigmoid_scale * torch.randn(batch_size, device=device))
    elif sigma_sampling == "uniform":
        sigmas = torch.rand(batch_size, device=device)
    else:
        raise ValueError(f"unknown sigma_sampling {sigma_sampling!r}")
    lo = max(0.0, min(1.0, sigma_min))
    hi = max(0.0, min(1.0, sigma_max))
    if hi <= lo:
        raise ValueError(f"sigma_max ({sigma_max}) must be > sigma_min ({sigma_min})")
    if lo > 0.0 or hi < 1.0:
        sigmas = lo + (hi - lo) * sigmas
    return sigmas


def fm_loss_step(
    anima,
    latents: torch.Tensor,
    emb_full: torch.Tensor,
    sigmas: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """One flow-matching loss eval — identical math to invert_reference.py."""
    n_t = sigmas.shape[0]
    lat = latents.expand(n_t, -1, -1, -1)
    noise = torch.randn_like(lat)
    # Cast sv to lat.dtype (bf16) so the mix stays bf16 — sigmas defaults to fp32
    # from torch.rand, and bf16 * fp32 promotes to a transient fp32 latent.
    sv = sigmas.view(-1, 1, 1, 1).to(lat.dtype)
    noisy = (1.0 - sv) * lat + sv * noise
    noisy_5d = noisy.unsqueeze(2)
    emb = emb_full.expand(n_t, -1, -1)
    pm = padding_mask.expand(n_t, -1, -1, -1)
    timesteps = sigmas.to(torch.bfloat16)
    pred = anima(noisy_5d, timesteps, emb, padding_mask=pm).squeeze(2)
    target = noise - lat
    return F.mse_loss(pred.float(), target.float())


# endregion


# region VR (variance-reduced FM via AsymFlow §5.2 control variate)


def _build_vr_pool(
    anima,
    *,
    prefix_emb: torch.Tensor,
    latents: torch.Tensor,
    padding_mask: torch.Tensor,
    K: int,
    D: int,
    sigma_lo: float,
    sigma_hi: float,
    sigma_sampling: str,
    sigmoid_scale: float,
    fei_sigma_low_div: float,
    pool_size: int,
    device: torch.device,
    seed: int,
) -> list[dict]:
    """Pre-sample (σ, noise) and precompute z for each — done once per image.

    The reference forward at s=0 doesn't depend on the optimization state, so
    its cost is paid upfront and the per-step inner loop only needs one
    with-grad forward at the current s. Reusing a finite pool of (σ, noise)
    pairs across steps is a known trick for low-budget per-image optimization
    (DDIM inversion, textual inversion) — it concentrates supervision on a
    fixed sample set and increases signal-to-noise vs fresh σ each step.
    """
    from library.runtime.fei import fei_sigma_low, gaussian_blur_2d

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    sigma_low = fei_sigma_low(h_lat, w_lat, fei_sigma_low_div)
    x0_L = gaussian_blur_2d(latents.float(), sigma_low).to(latents.dtype)

    # Reference embed: prefix + zeros tail (s=0).
    tail0 = torch.zeros(1, K, D, device=device, dtype=prefix_emb.dtype)
    emb_ref = assemble_emb(prefix_emb, tail0)

    g = torch.Generator(device=device).manual_seed(int(seed) + 0xA51F1011)

    pool: list[dict] = []
    with torch.no_grad():
        for _ in range(pool_size):
            # sample_sigmas uses the global torch RNG; we want a private stream
            # for the pool so it's deterministic regardless of optimizer seeding.
            if sigma_sampling == "sigmoid":
                u = torch.randn(1, device=device, generator=g)
                sig = torch.sigmoid(sigmoid_scale * u)
            else:
                sig = torch.rand(1, device=device, generator=g)
            lo = max(0.0, min(1.0, sigma_lo))
            hi = max(0.0, min(1.0, sigma_hi))
            if hi > lo and (lo > 0.0 or hi < 1.0):
                sig = lo + (hi - lo) * sig

            noise = torch.randn(
                latents.shape, device=device, dtype=latents.dtype, generator=g
            )
            sv = sig.view(-1, 1, 1, 1).to(latents.dtype)
            x_t_L = (1.0 - sv) * x0_L + sv * noise
            x_t_L_5d = x_t_L.unsqueeze(2)
            ts = sig.to(torch.bfloat16)
            ref_pred = anima(
                x_t_L_5d, ts, emb_ref, padding_mask=padding_mask
            ).squeeze(2)
            z = ref_pred.float() - (noise.float() - x0_L.float())
            pool.append({"sigma": sig, "noise": noise, "z": z})

    return pool


def _vr_loss_step(
    anima,
    latents: torch.Tensor,
    emb_full: torch.Tensor,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
    padding_mask: torch.Tensor,
    z: torch.Tensor,
    lambda_ema: Optional[float],
) -> tuple[torch.Tensor, float]:
    """One VR-FM loss eval using the cached (σ, noise, z).

    Returns ``(loss, lambda_batch)``. Caller is responsible for updating
    ``lambda_ema`` from ``lambda_batch``. If ``lambda_ema is None`` (first
    call), uses ``lambda_batch`` directly so the first step gets the
    locally-optimal mix.
    """
    sv = sigmas.view(-1, 1, 1, 1).to(latents.dtype)
    noisy = ((1.0 - sv) * latents + sv * noise).unsqueeze(2)
    ts = sigmas.to(torch.bfloat16)
    pred = anima(noisy, ts, emb_full, padding_mask=padding_mask).squeeze(2)
    y = pred.float() - (noise.float() - latents.float())

    # λ_batch = −cov(y_det, z) / var(z) — pixel-wise sums, single scalar.
    with torch.no_grad():
        y_d = y.detach()
        cov = (y_d * z).sum()
        var = (z * z).sum().clamp_min(1e-12)
        lambda_batch = float(-(cov / var).item())

    lam = lambda_batch if lambda_ema is None else lambda_ema
    diff = y + lam * z
    return diff.pow(2).mean(), lambda_batch


# endregion


# region Public entrypoint


@dataclass
class TailInversionConfig:
    """Per-image optimization knobs. Defaults match the proposal."""

    K: int = 48
    steps: int = 100
    lr: float = 0.01
    lr_schedule: str = "cosine"  # "cosine" or "constant"
    grad_accum: int = 4
    timesteps_per_step: int = 1
    sigma_sampling: str = "uniform"  # "uniform" or "sigmoid"
    sigmoid_scale: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    lambda_zero: float = 0.0  # ‖s‖² regularization weight
    init_std: float = 0.0  # 0 → zero-init; >0 → N(0, init_std²)
    log_every: int = 10
    # Tail parameterization. ``ortho_tail`` (default) is the K-scalar ``s`` over a
    # frozen orthonormal basis Q — one tail shared across all blocks and all t.
    # ``soft_tokens`` swaps in the SoftREPA parameterization (per-block × per-t
    # free tokens, no ortho, no caption-conditioning) to measure how much that
    # extra freedom lowers the per-image inversion floor. The optimization loop,
    # VR control variate, σ sampling, and logging are identical across both.
    parameterization: str = "ortho_tail"  # "ortho_tail" or "soft_tokens"
    # AsymFlow §5.2 control-variate loss adapted for per-image inversion.
    # Reference forward is the same frozen DiT with s=0 (postfix tail zeroed),
    # evaluated on FEI-low-passed latents. (σ, noise, z) tuples are pre-sampled
    # into a pool of vr_pool_size to amortize the extra reference forwards.
    vr_enabled: bool = False
    vr_pool_size: int = 32
    vr_lambda_beta: float = 0.2  # bumped from training's 0.01 (~50-step horizon)
    vr_fei_sigma_low_div: float = 8.0  # matches FeRA σ_low default
    metadata: dict = field(default_factory=dict)


@dataclass
class TailInversionResult:
    """Output of one inversion run — the things the analyzer reads."""

    s: Optional[torch.Tensor]  # (K,) float32 CPU for ortho_tail; None for soft_tokens
    best_loss: float  # best (fm_loss + reg) across optimization steps
    best_fm_loss: float  # fm component at the best-loss step
    best_step: int
    final_s_l2: float  # final ‖params‖₂ (the K-vector, or the whole bank)
    final_lambda_ema: Optional[float] = None  # VR λ EMA at end of optimization
    history: list[dict] = field(default_factory=list)
    # soft_tokens parameterization saves a {tokens, t_offsets.weight} bank here
    # at the best-loss step (CPU fp32); None for ortho_tail.
    bank: Optional[dict] = None


# region Parameterizations
#
# Two ways to fill the K postfix slots, sharing one optimization loop. Each
# object owns: its trainable leaves, how to turn the current state into the
# crossattn embedding handed to the frozen DiT (``make_emb``), an optional ‖·‖²
# regularizer, an L2 readout for the CSV, and a best-step snapshot. ``make_emb``
# is the load-bearing seam — ``ortho_tail`` splices once at the input (all blocks
# see the same tail), ``soft_tokens`` sets per-block × per-t tokens that its own
# Block.forward hooks splice during the DiT forward.


class _OrthoTailParam:
    """``tail = Q · diag(s)`` — K trainable scalars over a frozen orthonormal Q."""

    kind = "ortho_tail"

    def __init__(
        self,
        *,
        prefix_emb: torch.Tensor,
        basis_Q: torch.Tensor,
        device: torch.device,
        init_std: float,
        lambda_zero: float,
    ):
        self.prefix_emb = prefix_emb
        self.Q = basis_Q.to(device=device, dtype=torch.float32).contiguous()
        self.K, self.D = self.Q.shape
        self.lambda_zero = lambda_zero
        if init_std <= 0.0:
            s_init = torch.zeros(self.K, dtype=torch.float32, device=device)
        else:
            s_init = torch.randn(self.K, dtype=torch.float32, device=device) * init_std
        self.s = torch.nn.Parameter(s_init)
        self._best: Optional[torch.Tensor] = None

    def trainable(self) -> list:
        return [self.s]

    def make_emb(self, sigmas: torch.Tensor) -> torch.Tensor:
        # tail independent of σ; sigmas accepted for a uniform make_emb() seam.
        tail = (self.Q * self.s.unsqueeze(-1)).unsqueeze(0)  # (1, K, D) fp32
        return assemble_emb(self.prefix_emb, tail)

    def reg(self, device: torch.device) -> torch.Tensor:
        if self.lambda_zero > 0.0:
            return self.lambda_zero * self.s.pow(2).sum()
        return torch.zeros((), device=device, dtype=torch.float32)

    def param_l2(self) -> float:
        return float(self.s.detach().norm().item())

    def snapshot_best(self) -> None:
        self._best = self.s.detach().clone().cpu()

    def result_fields(self) -> dict:
        return {"s": self._best, "bank": None}


class _SoftTokensParam:
    """SoftREPA bank — per-block × per-t free tokens, spliced by the network's
    own Block.forward hooks. ``net`` is built + ``apply_to``-d once by the caller
    (so the monkey-patch + any torch.compile ordering is handled outside); this
    object resets its params in-place per image."""

    kind = "soft_tokens"

    def __init__(
        self,
        *,
        net,
        prefix_emb: torch.Tensor,
        device: torch.device,
        init_std: float,
        lambda_zero: float,
    ):
        self.net = net
        self.prefix_emb = prefix_emb
        self.lambda_zero = lambda_zero
        self.K = net.num_tokens
        self.D = net.embed_dim
        # Real (non-padding) token count per sample — only consumed by the
        # front_of_padding splice; harmless for end_of_sequence.
        self.seqlens = (
            (prefix_emb.abs().sum(dim=-1) > 0).sum(dim=1).to(torch.int32)
        )  # (B,)
        # Fresh start per image: re-init the bank in place (same Parameter
        # objects → caller can build one optimizer per image over them).
        with torch.no_grad():
            if init_std <= 0.0:
                net.tokens.zero_()
            else:
                net.tokens.normal_(0.0, init_std)
            net.t_offsets.weight.zero_()
        net._step_layer_tokens = None  # ensure s=0 baseline for the VR pool build
        self._best: Optional[dict] = None

    def trainable(self) -> list:
        return [self.net.tokens, self.net.t_offsets.weight]

    def make_emb(self, sigmas: torch.Tensor) -> torch.Tensor:
        # Sets _step_layer_tokens (with grad) for this σ; the per-block hooks
        # splice it during the DiT forward, so we hand back the bare prefix.
        self.net.append_postfix(self.prefix_emb, self.seqlens, timesteps=sigmas)
        return self.prefix_emb

    def reg(self, device: torch.device) -> torch.Tensor:
        if self.lambda_zero > 0.0:
            return self.lambda_zero * (
                self.net.tokens.pow(2).sum()
                + self.net.t_offsets.weight.pow(2).sum()
            )
        return torch.zeros((), device=device, dtype=torch.float32)

    def param_l2(self) -> float:
        with torch.no_grad():
            sq = (
                self.net.tokens.detach().pow(2).sum()
                + self.net.t_offsets.weight.detach().pow(2).sum()
            )
            return float(sq.sqrt().item())

    def snapshot_best(self) -> None:
        self._best = {
            "tokens": self.net.tokens.detach().clone().cpu().float(),
            "t_offsets.weight": self.net.t_offsets.weight.detach().clone().cpu().float(),
        }

    def result_fields(self) -> dict:
        return {"s": None, "bank": self._best}


def _build_parameterization(
    config: TailInversionConfig,
    *,
    prefix_emb: torch.Tensor,
    basis_Q: Optional[torch.Tensor],
    soft_tokens_net,
    device: torch.device,
):
    """Pick + construct the parameterization object from the config."""
    kind = config.parameterization
    if kind == "ortho_tail":
        if basis_Q is None:
            raise ValueError("parameterization='ortho_tail' requires basis_Q")
        K, D = basis_Q.shape
        if K != config.K:
            raise ValueError(f"basis_Q rows ({K}) != config.K ({config.K})")
        if prefix_emb.shape[-1] != D:
            raise ValueError(
                f"prefix_emb dim ({prefix_emb.shape[-1]}) != basis_Q dim ({D})"
            )
        return _OrthoTailParam(
            prefix_emb=prefix_emb,
            basis_Q=basis_Q,
            device=device,
            init_std=config.init_std,
            lambda_zero=config.lambda_zero,
        )
    if kind == "soft_tokens":
        if soft_tokens_net is None:
            raise ValueError(
                "parameterization='soft_tokens' requires a pre-built + applied "
                "soft_tokens_net (build it before torch.compile — see the "
                "build_anima compile-after-apply invariant)"
            )
        if prefix_emb.shape[-1] != soft_tokens_net.embed_dim:
            raise ValueError(
                f"prefix_emb dim ({prefix_emb.shape[-1]}) != soft_tokens_net "
                f"embed_dim ({soft_tokens_net.embed_dim})"
            )
        return _SoftTokensParam(
            net=soft_tokens_net,
            prefix_emb=prefix_emb,
            device=device,
            init_std=config.init_std,
            lambda_zero=config.lambda_zero,
        )
    raise ValueError(
        f"unknown parameterization {kind!r}: expected 'ortho_tail' or 'soft_tokens'"
    )


# endregion


def invert_tail(
    anima,
    *,
    prefix_emb: torch.Tensor,
    latents: torch.Tensor,
    basis_Q: Optional[torch.Tensor] = None,
    config: TailInversionConfig,
    device: torch.device,
    seed: int = 0,
    log_path: Optional[str] = None,
    soft_tokens_net=None,
) -> TailInversionResult:
    """Optimize the postfix tail for one image; return the best-loss state + diagnostics.

    The trainable object depends on ``config.parameterization`` (see
    ``_build_parameterization``): ``ortho_tail`` optimizes the K-scalar ``s`` over
    ``basis_Q``; ``soft_tokens`` optimizes a pre-built+applied ``soft_tokens_net``
    bank. The loop below (VR pool, σ sampling, grad-accum, logging, best-tracking)
    is identical across both — only ``param.make_emb`` / ``param.reg`` differ.

    Args:
        anima: frozen DiT (already ``requires_grad_(False)``-ed; ``torch.compile``
            on the outside is fine — shapes are static for fixed K + image size).
        prefix_emb: ``(1, MAX_SEQ_LEN, D)`` bf16 cached T5 output for this image.
        latents: ``(B=1, C, H/8, W/8)`` bf16 cached VAE latents for this image.
        basis_Q: ``(K, D)`` row-orthonormal fp32 basis (frozen) — ``ortho_tail`` only.
        config: optimization knobs (incl. ``parameterization``).
        device: cuda or cpu.
        seed: RNG seed for sigma sampling + param init.
        log_path: optional CSV path for per-step loss / ‖params‖₂ / grad norm.
        soft_tokens_net: pre-built + ``apply_to``-d ``SoftTokensNetwork`` —
            required for ``parameterization='soft_tokens'``, ignored otherwise.
    """
    cfg = config

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    param = _build_parameterization(
        cfg,
        prefix_emb=prefix_emb,
        basis_Q=basis_Q,
        soft_tokens_net=soft_tokens_net,
        device=device,
    )
    K, D = param.K, param.D

    optimizer = torch.optim.AdamW(
        param.trainable(), lr=cfg.lr, weight_decay=0.0, fused=(device.type == "cuda")
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.01
        )
        if cfg.lr_schedule == "cosine"
        else None
    )

    h_lat, w_lat = latents.shape[-2], latents.shape[-1]
    padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "loss",
                "fm_loss",
                "reg",
                "best_loss",
                "lr",
                "grad_norm",
                "s_l2",
                "lambda_ema",
                "lambda_batch",
            ],
        )
        csv_writer.writeheader()

    vr_pool: Optional[list[dict]] = None
    lambda_ema: Optional[float] = None
    last_lambda_batch: float = float("nan")
    if cfg.vr_enabled:
        # Pool build is no-grad. The default offloader runs with
        # supports_backward=True, where a forward only submits half the
        # CPU↔GPU cycle and the backward hooks complete it on loss.backward().
        # Chaining vr_pool_size no-grad forwards therefore leaves blocks
        # inverted from their prepared state after the first iteration and
        # the next forward mm's against a CPU-side weight ("mat2 is on cpu").
        # Switch to forward-only swap for the pool, then restore training-mode
        # swap before the optimization loop.
        swap_active = (
            getattr(anima, "offloader", None) is not None
            and getattr(anima, "blocks_to_swap", 0)
        )
        if swap_active:
            anima.switch_block_swap_for_inference()
        try:
            vr_pool = _build_vr_pool(
                anima,
                prefix_emb=prefix_emb,
                latents=latents,
                padding_mask=padding_mask,
                K=K,
                D=D,
                sigma_lo=cfg.sigma_min,
                sigma_hi=cfg.sigma_max,
                sigma_sampling=cfg.sigma_sampling,
                sigmoid_scale=cfg.sigmoid_scale,
                fei_sigma_low_div=cfg.vr_fei_sigma_low_div,
                pool_size=cfg.vr_pool_size,
                device=device,
                seed=seed,
            )
        finally:
            if swap_active:
                anima.switch_block_swap_for_training()
        logger.info(
            f"VR pool built: {cfg.vr_pool_size} (σ, noise, z) tuples, "
            f"β={cfg.vr_lambda_beta}"
        )

    best_loss = float("inf")
    best_fm_loss = float("inf")
    have_best = False
    best_step = 0
    history: list[dict] = []

    # timesteps_per_step now multiplies into grad_accum instead of growing the
    # per-forward σ batch — keeps memory at batch=1 regardless of how many σ
    # samples you want to average per optimizer step. Mean gradient over the
    # M·N samples is unchanged (each backward is scaled by 1/microsteps).
    microsteps = max(1, cfg.grad_accum * cfg.timesteps_per_step)

    pbar = tqdm(range(cfg.steps), desc=f"Inverting tail K={K}", leave=False)
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_fm = 0.0
        accum_reg = 0.0
        for _ in range(microsteps):
            # σ first, then build the emb from it — soft_tokens' make_emb is
            # σ-dependent (per-t bucket); ortho_tail's ignores σ. make_emb is the
            # only place the two parameterizations diverge inside the loop.
            if vr_pool is not None:
                idx = int(torch.randint(len(vr_pool), (1,)).item())
                entry = vr_pool[idx]
                emb_full = param.make_emb(entry["sigma"])
                fm_loss, lambda_batch = _vr_loss_step(
                    anima,
                    latents,
                    emb_full,
                    entry["sigma"],
                    entry["noise"],
                    padding_mask,
                    entry["z"],
                    lambda_ema,
                )
                lambda_ema = (
                    lambda_batch
                    if lambda_ema is None
                    else (1.0 - cfg.vr_lambda_beta) * lambda_ema
                    + cfg.vr_lambda_beta * lambda_batch
                )
                last_lambda_batch = lambda_batch
            else:
                sigmas = sample_sigmas(
                    1,
                    device,
                    sigma_sampling=cfg.sigma_sampling,
                    sigmoid_scale=cfg.sigmoid_scale,
                    sigma_min=cfg.sigma_min,
                    sigma_max=cfg.sigma_max,
                )
                emb_full = param.make_emb(sigmas)
                fm_loss = fm_loss_step(
                    anima, latents, emb_full, sigmas, padding_mask
                )
            reg = param.reg(device)
            loss = fm_loss + reg
            (loss / microsteps).backward()
            accum_loss += loss.item()
            accum_fm += fm_loss.item()
            accum_reg += float(reg.detach().item())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            param.trainable(), max_norm=1.0
        ).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = accum_loss / microsteps
        fm_val = accum_fm / microsteps
        reg_val = accum_reg / microsteps
        s_l2 = param.param_l2()  # ‖s‖ (ortho_tail) or ‖bank‖ (soft_tokens)
        lr_now = optimizer.param_groups[0]["lr"]

        if loss_val < best_loss:
            best_loss = loss_val
            best_fm_loss = fm_val
            param.snapshot_best()
            have_best = True
            best_step = step

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            pbar.set_postfix(
                loss=f"{loss_val:.6f}",
                best=f"{best_loss:.6f}",
                s_l2=f"{s_l2:.3f}",
                lr=f"{lr_now:.2e}",
            )

        row = {
            "step": step,
            "loss": loss_val,
            "fm_loss": fm_val,
            "reg": reg_val,
            "best_loss": best_loss,
            "lr": lr_now,
            "grad_norm": grad_norm,
            "s_l2": s_l2,
            "lambda_ema": float("nan") if lambda_ema is None else float(lambda_ema),
            "lambda_batch": last_lambda_batch,
        }
        history.append(row)
        if csv_writer is not None:
            csv_writer.writerow(
                {
                    "step": step,
                    "loss": f"{loss_val:.6f}",
                    "fm_loss": f"{fm_val:.6f}",
                    "reg": f"{reg_val:.6f}",
                    "best_loss": f"{best_loss:.6f}",
                    "lr": f"{lr_now:.2e}",
                    "grad_norm": f"{grad_norm:.6f}",
                    "s_l2": f"{s_l2:.6f}",
                    "lambda_ema": f"{row['lambda_ema']:.6f}",
                    "lambda_batch": f"{row['lambda_batch']:.6f}",
                }
            )

    if csv_file is not None:
        csv_file.close()

    assert have_best, "optimization produced no best step"
    fields = param.result_fields()
    return TailInversionResult(
        s=fields["s"],
        best_loss=best_loss,
        best_fm_loss=best_fm_loss,
        best_step=best_step,
        final_s_l2=param.param_l2(),
        final_lambda_ema=None if lambda_ema is None else float(lambda_ema),
        history=history,
        bank=fields["bank"],
    )


# endregion


# region IO


def save_tail_s(
    save_path: str,
    s: torch.Tensor,
    *,
    K: int,
    D: int,
    basis_kind: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save a single per-image ``s`` vector as a safetensors file.

    Schema is intentionally minimal — analysis works on ``s`` alone, and the
    basis is reconstructed from the cached file at analyzer time. Metadata
    pins the run config so a downstream consumer can verify (K, basis kind,
    optimization budget, etc.) match expectations.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    state = {"s": s.detach().clone().cpu().float().contiguous()}
    meta = {
        "ss_artifact": "postfix_tail_s",
        "ss_K": str(K),
        "ss_D": str(D),
        "ss_basis_kind": basis_kind,
    }
    if metadata:
        for k, v in metadata.items():
            meta[str(k)] = str(v)
    save_file(state, save_path, metadata=meta)


def save_soft_tokens_bank(
    save_path: str,
    bank: dict,
    *,
    n_layers: int,
    num_tokens: int,
    embed_dim: int,
    n_t_buckets: int,
    splice_position: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save a per-image soft-tokens bank (``tokens`` + ``t_offsets.weight``).

    Schema matches ``SoftTokensNetwork.state_dict_for_save`` /
    ``create_network_from_weights`` so the same loader reads a probe bank as a
    deployable one — though this is a per-image inversion artifact, not a
    generalizing adapter. Metadata pins the parameterization for the analyzer.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    state = {
        "tokens": bank["tokens"].detach().clone().cpu().float().contiguous(),
        "t_offsets.weight": bank["t_offsets.weight"]
        .detach()
        .clone()
        .cpu()
        .float()
        .contiguous(),
    }
    meta = {
        "ss_artifact": "postfix_tail_soft_tokens_bank",
        "ss_n_layers": str(n_layers),
        "ss_num_tokens": str(num_tokens),
        "ss_embed_dim": str(embed_dim),
        "ss_n_t_buckets": str(n_t_buckets),
        "ss_splice_position": splice_position,
    }
    if metadata:
        for k, v in metadata.items():
            meta[str(k)] = str(v)
    save_file(state, save_path, metadata=meta)


def load_tail_s(s_path: str) -> tuple[torch.Tensor, dict]:
    """Inverse of ``save_tail_s``: returns ``(s: (K,) fp32, metadata: dict)``."""
    from safetensors import safe_open

    with safe_open(s_path, framework="pt") as f:
        s = f.get_tensor("s").float().contiguous()
        meta = f.metadata() or {}
    return s, dict(meta)


def splice_tail_into_te_cache(
    te_in_path: str,
    te_out_path: str,
    *,
    s: torch.Tensor,
    Q: torch.Tensor,
    variant_index: int = 0,
) -> None:
    """Bake ``Q @ diag(s)`` into the last K rows of a cached T5 prefix.

    Reads ``{te_in_path}``'s ``crossattn_emb_v{variant_index}`` (or
    ``crossattn_emb`` for single-variant caches), overwrites positions
    ``[S-K, S)`` with the postfix tail, and writes a minimal single-variant
    cache to ``{te_out_path}``. The output schema is just ``crossattn_emb``
    — exactly what ``edit.py``'s ``--cached_embed`` single-variant branch
    expects, so dry-mode reconstruction picks up the spliced prefix without
    any downstream code changes.

    Args:
        te_in_path: Source ``{stem}_anima_te.safetensors`` cache.
        te_out_path: Destination — created fresh, single-variant.
        s: ``(K,)`` scales, fp32 or bf16, on any device.
        Q: ``(K, D)`` row-orthonormal basis matching the original inversion.
        variant_index: Which cached variant to splice into. v0 (pristine
            caption) is what dry-mode normally uses.
    """
    from safetensors import safe_open

    K, D = Q.shape
    if s.shape != (K,):
        raise ValueError(f"s shape {tuple(s.shape)} != (K={K},)")

    with safe_open(te_in_path, framework="pt") as f:
        keys = set(f.keys())
        has_variants = "num_variants" in keys
        if has_variants:
            suf = f"_v{variant_index}"
            crossattn_key = f"crossattn_emb{suf}"
        else:
            crossattn_key = "crossattn_emb"
        if crossattn_key not in keys:
            raise KeyError(
                f"{te_in_path}: missing {crossattn_key!r} "
                f"(keys: {sorted(keys)[:8]}...)"
            )
        emb = f.get_tensor(crossattn_key).float()  # (S, D)

    S = emb.shape[0]
    if emb.shape[-1] != D:
        raise ValueError(
            f"cached emb dim {emb.shape[-1]} != basis Q dim {D} (file: {te_in_path})"
        )
    if S < K:
        raise ValueError(f"cached emb seq len {S} < K={K} (file: {te_in_path})")

    tail = (Q.float().cpu() * s.float().cpu().unsqueeze(-1))  # (K, D)
    spliced = emb.clone()
    spliced[S - K : S, :] = tail
    spliced_bf16 = spliced.to(torch.bfloat16).contiguous()

    os.makedirs(os.path.dirname(te_out_path) or ".", exist_ok=True)
    save_file(
        {"crossattn_emb": spliced_bf16},
        te_out_path,
        metadata={
            "ss_artifact": "postfix_tail_spliced_te_cache",
            "ss_source_te": os.path.basename(te_in_path),
            "ss_variant_index": str(variant_index),
            "ss_K": str(K),
            "ss_D": str(D),
            "ss_splice_position": "end_of_sequence",
        },
    )


# endregion
