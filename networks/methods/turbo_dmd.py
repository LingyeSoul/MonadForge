"""Turbo Anima — DP-DMD distillation harness.

Owns two plain ``LoRANetwork`` instances (student + fake) on one frozen Anima
DiT. Both call ``apply_to(unet)`` which chains them onto every targeted
Linear's forward — at runtime the chain order is::

    linear(x) -> fake.forward -> student.forward -> original_linear.forward

Each LoRA module short-circuits at ``not self.enabled`` (see
``lora_modules/lora.py::LoRAModule.forward``), so view-toggling is just
``set_enabled(bool)`` on each network — O(num_modules) Python loop, negligible
vs a DiT forward.

Used by ``scripts/distill_turbo/distill.py``. Inference loads the saved
``anima_turbo.safetensors`` through the standard LoRA path (no inference-side
turbo code) — the student LoRA is just a normal LoRA with CFG=4 baked in.

This harness is method-agnostic (two view-toggled LoRA stacks); the shipped
objective driving it is DP-DMD.

Docs: ``docs/structure/dpdmd.md`` (structure), ``docs/experimental/dpdmd.md`` (ops).
Paper: Wu, Li, Zhang, Ma, "Diversity-Preserved Distribution Matching
Distillation" (arXiv:2602.03139).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.lora_anima.factory import create_network
from networks.lora_anima.network import LoRANetwork

logger = logging.getLogger(__name__)

View = Literal["teacher", "student", "fake"]


class PooledTokenDiscriminator(nn.Module):
    """Pooled-token GAN head over frozen-teacher block features (FastGen v0).

    FastGen's ``Discriminator_ImageDiT`` un-flattens each tapped block's tokens
    back to ``(B, D, H_p, W_p)`` and runs a conv head. Under Anima's native-shape
    bucketing the patch grid is per-bucket and, with ``compile_blocks``, the block
    output is the fake-5D ``(B, 1, L, 1, D)`` layout — so the spatial reshape is
    the fragile part the proposal flags. This v0 sidesteps it: **mean-pool each
    tap's token output over every axis between batch and channel** → ``(B, D)``,
    then a 2-layer MLP per tap → a per-tap logit. The pool is shape-agnostic, so
    it works identically on the eager ``(B, T, H, W, D)`` grid and the compiled
    ``(B, 1, L, 1, D)`` layout. Logits for all taps are concatenated to
    ``(B, num_taps)``; runs in fp32 for GAN-loss stability.

    Tiny by design (~``inner_dim²/2`` params/tap, ≈2M at D=2048) and discarded at
    save — pure training scaffolding, exactly like the fake/critic LoRA.
    """

    def __init__(
        self, *, inner_dim: int, num_taps: int, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim is not None else inner_dim // 2
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, h),
                nn.LeakyReLU(0.2),
                nn.Linear(h, 1),
            )
            for _ in range(num_taps)
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.heads):
            raise ValueError(
                f"PooledTokenDiscriminator expected {len(self.heads)} feature "
                f"tensors, got {len(feats)}"
            )
        logits = []
        for head, f in zip(self.heads, feats):
            # Pool over every axis between batch (0) and channel (-1): handles
            # both (B, T, H, W, D) eager and (B, 1, L, 1, D) native-flatten.
            pooled = f.float().mean(dim=tuple(range(1, f.ndim - 1)))  # (B, D)
            logits.append(head(pooled))
        return torch.cat(logits, dim=1)  # (B, num_taps)


def gan_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    """Softplus hinge — generator wants the disc to score its samples as real."""
    return F.softplus(-fake_logits).mean()


def gan_loss_discriminator(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Softplus hinge — push real logits up, fake logits down (FastGen common_loss)."""
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def load_step_expert_student(
    unet,
    weights_sd: dict[str, torch.Tensor],
    metadata: dict[str, str],
    *,
    multiplier: float = 1.0,
) -> LoRANetwork:
    """Rebuild a per-step-expert turbo student as a router-free, kept-live net.

    Per-step-expert checkpoints can't go through ``merge_to_dit`` (K up-heads
    don't fold into one DiT weight) nor the shared ``create_network_from_weights``
    key-sniff (the ``.lora_ups.{k}.weight`` layout collides with Hydra's). The
    file keeps the training-runtime fused-qkv key layout, so we build a fresh
    ``StepExpertLoRAModule`` network on the same fused DiT (uniform rank from
    metadata) and ``load_state_dict`` matches directly — no split→re-fuse.

    Caller is responsible for ``apply_to``? No: this applies + loads + freezes,
    returning a net ready for ``set_step_index`` per denoise step. Stash it on
    the model so the sampler loop can find it.
    """
    K = int(metadata.get("ss_turbo_step_expert_K", "0") or "0")
    if K <= 1:
        raise RuntimeError(
            "load_step_expert_student called on a checkpoint without "
            "ss_turbo_step_expert_K > 1 — not a per-step-expert turbo file."
        )
    rank = int(metadata.get("ss_turbo_student_rank", "0") or "0")
    if rank <= 0:
        raise RuntimeError(
            "per-step-expert turbo checkpoint missing ss_turbo_student_rank "
            "metadata — cannot rebuild the student at the right rank."
        )
    # scale = alpha / rank is fixed at module construction (load_state_dict only
    # overwrites the alpha buffer, not the cached scale), so build with the
    # trained alpha. Shipped config uses alpha == rank (scale 1.0); fall back to
    # rank when the stamp is absent on older checkpoints.
    alpha = float(metadata.get("ss_turbo_student_alpha", str(rank)) or rank)

    network = create_network(
        multiplier=multiplier,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=unet,
        step_expert_K=K,
    )
    network.apply_to([], unet, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.unexpected_keys:
        logger.warning(
            f"step-expert turbo: unexpected keys in state dict: "
            f"{info.unexpected_keys[:5]}..."
        )
    if info.missing_keys:
        # Zero-init heads that never received a checkpoint tensor would silently
        # contribute nothing; surface the count so a rank/target mismatch is loud.
        logger.warning(
            f"step-expert turbo: {len(info.missing_keys)} missing keys "
            f"(first: {info.missing_keys[:5]})"
        )
    network.set_step_index(0)
    logger.info(
        f"step-expert turbo: router-free kept-live attached "
        f"({len(network.unet_loras)} modules, K={K} heads, rank={rank})"
    )
    return network


class TurboDMDNetwork:
    """Two LoRA stacks on one frozen DiT, view-toggleable per forward.

    Not a ``nn.Module`` — it's a thin coordinator that holds two real
    ``LoRANetwork`` instances. The DiT itself is owned by the caller and
    stays frozen.
    """

    def __init__(
        self,
        unet,
        *,
        student_rank: int,
        fake_rank: int,
        student_alpha: float | None = None,
        fake_alpha: float | None = None,
        use_custom_down_autograd: bool = False,
        channel_scaling_alpha: float = 0.0,
        student_step_expert_K: int = 0,
        gan_feature_indices: set[int] | None = None,
        gan_disc_hidden: int | None = None,
    ) -> None:
        self.unet = unet
        self.student_rank = int(student_rank)
        self.fake_rank = int(fake_rank)
        # SmoothQuant-style per-input-channel rebalance absorbed into each
        # ``lora_down`` (bit-equivalent at init, merges out cleanly). 0.0 = off,
        # 0.5 = sqrt-balance. Applied to both student and fake — it only
        # conditions the LoRA gradient on the DiT's outlier-DC input channels,
        # so it leaves the DP-DMD objective untouched. The shipped calibration
        # (networks/calibration/channel_stats.safetensors) is σ-insensitive, so
        # it transfers to the student's 2-step grid and the fake's random τ.
        self.channel_scaling_alpha = float(channel_scaling_alpha)
        # Per-step expert: when > 1 the student's adapted Linears become
        # StepExpertLoRAModule (shared down + K step-indexed up-heads); head k
        # is trained only by step-k's gradient (see set_student_step + the
        # detach in scripts/distill_turbo/distill.py). The fake/critic stays a
        # plain single-head LoRA. 0/1 = the shipped single-head student.
        self.student_step_expert_K = int(student_step_expert_K)

        # Plain LoRA on both (LoRANetworkCfg defaults: no MoE/ortho/T-LoRA).
        # alpha = rank by default (scale 1.0) per the LoRA-family convention.
        # use_custom_down_autograd is forwarded for config compat but a deprecated
        # no-op in the factory (fp32-bottleneck path removed 2026-06-10).
        # step_expert_K rides **kwargs; >1 flips resolve_network_spec to
        # StepExpertLoRAModule for the student only (the fake never sees it).
        _student_kwargs: dict = {}
        if self.student_step_expert_K > 1:
            _student_kwargs["step_expert_K"] = self.student_step_expert_K
        self.student: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.student_rank,
            network_alpha=student_alpha
            if student_alpha is not None
            else self.student_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            use_custom_down_autograd=use_custom_down_autograd,
            channel_scaling_alpha=self.channel_scaling_alpha,
            **_student_kwargs,
        )
        self.fake: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.fake_rank,
            network_alpha=fake_alpha if fake_alpha is not None else self.fake_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            use_custom_down_autograd=use_custom_down_autograd,
            channel_scaling_alpha=self.channel_scaling_alpha,
        )

        # Apply student-first so the runtime chain is
        # ``linear -> fake -> student -> original`` (additive, but a stable order).
        self.student.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )
        self.fake.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )

        logger.info(
            f"TurboDMDNetwork: student rank={self.student_rank} "
            f"({len(self.student.unet_loras)} modules), "
            f"fake rank={self.fake_rank} "
            f"({len(self.fake.unet_loras)} modules)"
        )

        # Start in teacher view. LoRAModule defaults enabled=True, and diff-only
        # set_view short-circuits when view==self._view, so we MUST disable both
        # stacks explicitly here — else the set_view invariant (cur state matches
        # _VIEW_FLAGS[_view]) breaks once either stack carries nonzero weights.
        self.student.set_enabled(False)
        self.fake.set_enabled(False)
        self._view: View = "teacher"

        # GAN feature tap: off unless gan_feature_indices is given. The disc reads
        # frozen-teacher block activations via the DiT's first-class feature-tap
        # (forward_mini_train_dit's return_block_features / return_features_early),
        # NOT a forward hook; the caller hands the feature dict to self.disc through
        # features_in_order. Taps arrive in native-flatten (B,1,L,1,D) under compile
        # (pooled shape-agnostically); early-exit runs only blocks[0..k].
        self.disc: PooledTokenDiscriminator | None = None
        self.gan_feature_indices: list[int] = []
        if gan_feature_indices:
            self.gan_feature_indices = sorted(gan_feature_indices)
            self.disc = PooledTokenDiscriminator(
                inner_dim=unet.model_channels,
                num_taps=len(self.gan_feature_indices),
                hidden_dim=gan_disc_hidden,
            )
            logger.info(
                f"TurboDMDNetwork: GAN disc attached (taps={self.gan_feature_indices}, "
                f"inner_dim={unet.model_channels}, "
                f"{sum(p.numel() for p in self.disc.parameters()):,} params)"
            )

    @property
    def gan_feature_set(self) -> set[int] | None:
        """Tap indices as a set for ``return_block_features`` (None when GAN off)."""
        return set(self.gan_feature_indices) if self.gan_feature_indices else None

    def features_in_order(self, feats: dict[int, torch.Tensor]) -> list[torch.Tensor]:
        """Reorder a model feature-dict to tap-index order for the disc.

        ``feats`` is the dict returned by the teacher feature-tap forward
        (``return_block_features``); raises if a tap is missing (mis-wired call).
        """
        try:
            return [feats[i] for i in self.gan_feature_indices]
        except KeyError as e:
            raise RuntimeError(
                f"GAN feature tap {e} missing from teacher forward output — "
                "the forward was not run with return_block_features=gan_feature_set."
            ) from e

    def disc_params(self):
        """Trainable params for the discriminator optimizer."""
        if self.disc is None:
            return []
        return [p for p in self.disc.parameters() if p.requires_grad]

    def set_disc_requires_grad(self, flag: bool) -> None:
        """Toggle disc grad: False during the student (gen) step so the GAN-gen
        gradient only reaches x_pred; True during the disc update."""
        if self.disc is not None:
            self.disc.requires_grad_(flag)

    # Per-view (student_on, fake_on) target states; lookup makes the
    # "flip only what changed" diff explicit.
    _VIEW_FLAGS: dict[str, tuple[bool, bool]] = {
        "teacher": (False, False),
        "student": (True, False),
        "fake": (False, True),
    }

    def set_view(self, view: View) -> None:
        """Flip per-network enabled flags so the next DiT forward acts as
        the named view.

        - ``teacher``: both LoRA stacks off, DiT delivers base velocity.
        - ``student``: student on, fake off — produces v_student for x_pred.
        - ``fake``: fake on, student off — fake's score estimate at τ_DM.

        Short-circuits when already in the target view (consecutive teacher
        forwards in the CA + DM branches don't repay the ~O(num_modules)
        attribute-write loop, and dynamo doesn't get a chance to invalidate
        guards it would have re-validated anyway).
        """
        if view == self._view:
            return
        try:
            want_student, want_fake = self._VIEW_FLAGS[view]
        except KeyError as e:
            raise ValueError(
                f"Unknown view {view!r}; expected teacher/student/fake"
            ) from e
        cur_student, cur_fake = self._VIEW_FLAGS[self._view]
        if want_student != cur_student:
            self.student.set_enabled(want_student)
        if want_fake != cur_fake:
            self.fake.set_enabled(want_fake)
        self._view = view

    @property
    def view(self) -> View:
        return self._view

    def set_student_step(self, i: int) -> None:
        """Select the student's step-``i`` up-head before its forward.

        No-op when per-step-expert is off (the plain student modules have no
        ``set_step``). The detach between the diversity step-0 backward and the
        DMD steps-1..N chain (distill.py) means head 0 only ever sees the
        diversity gradient and head k only step-k's DMD gradient — the shared
        ``lora_down`` is trained by both. Mirror of ``LoRANetwork.set_step_index``;
        kept as a coordinator method so the training loop never reaches into the
        student network directly.
        """
        if self.student_step_expert_K > 1:
            self.student.set_step_index(i)

    def student_params(self):
        """Trainable params for the student optimizer."""
        return [p for p in self.student.parameters() if p.requires_grad]

    def fake_params(self):
        """Trainable params for the fake optimizer."""
        return [p for p in self.fake.parameters() if p.requires_grad]

    def freeze_dit(self) -> None:
        """Set ``requires_grad=False`` on every base DiT param.

        Must be called AFTER both ``apply_to``'s — the LoRA networks add
        sub-modules to ``unet`` via ``add_module(lora.lora_name, lora)``, so
        a wholesale ``unet.requires_grad_(False)`` BEFORE apply would still
        be undone by the LoRA modules' own requires_grad=True params (good),
        but a wholesale call AFTER would zero those too (bad). We selectively
        walk only ``unet`` params whose name doesn't start with a LoRA prefix.
        """
        lora_prefixes = tuple(
            set(m.lora_name for m in self.student.unet_loras)
            | set(m.lora_name for m in self.fake.unet_loras)
        )
        n_frozen = 0
        for name, param in self.unet.named_parameters():
            if name.startswith(lora_prefixes):
                continue
            param.requires_grad_(False)
            n_frozen += 1
        logger.info(f"freeze_dit: {n_frozen} base params frozen")

    def save_student(
        self,
        file: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Serialize only the student LoRA in the standard plain-LoRA layout.

        Output is loadable by ``inference.py --lora_weight <file>`` — the
        fake network is training scaffolding and never shipped.
        """
        sd = self.student.state_dict()
        # Strip any non-LoRA keys defensively (plain LoRA shouldn't have any, but
        # the LoRANetwork may carry non-load-bearing buffers).
        sd = {k: v for k, v in sd.items() if ".lora_" in k or ".alpha" in k}

        if self.student_step_expert_K > 1:
            self._save_student_step_expert(sd, file, dtype, metadata)
            return

        from networks.lora_save import save_network_weights

        save_network_weights(
            sd,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant="standard",
        )
        logger.info(f"saved student LoRA → {file}  ({len(sd)} keys)")

    def _save_student_step_expert(
        self,
        sd: dict[str, torch.Tensor],
        file: str,
        dtype: torch.dtype,
        metadata: dict[str, str] | None,
    ) -> None:
        """Write the per-step-expert student in its bespoke layout.

        Multi-head keys (``…lora_ups.{k}.weight``) are NOT plain-LoRA loadable,
        so the standard defuse-qkv save pipeline (which expects one
        ``.lora_up.weight`` per ``.lora_down.weight``) can't run. We keep the
        fused-qkv key layout the training-runtime DiT uses verbatim — the CLI /
        ComfyUI step-expert loaders rebuild the network on the same fused DiT and
        ``load_state_dict`` matches directly (no split→re-fuse round trip). The
        ``ss_turbo_per_step_expert`` metadata stamp drives loader detection; the
        keys deliberately reuse ``.lora_ups.`` so a stock loader fails loudly
        rather than silently mis-merging only head 0.
        """
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes
        from networks.lora_modules.lora import bake_inv_scale

        # Fold any per-channel scaling into lora_down (no-op when absent) so the
        # on-disk delta acts on raw inputs.
        bake_inv_scale(sd)

        if dtype is not None:
            sd = {k: v.detach().clone().to("cpu").to(dtype) for k, v in sd.items()}

        meta = dict(metadata or {})
        model_hash, legacy_hash = precalculate_safetensors_hashes(sd, meta)
        meta["sshs_model_hash"] = model_hash
        meta["sshs_legacy_hash"] = legacy_hash

        save_file(sd, file, meta)
        n_heads = self.student_step_expert_K
        logger.info(
            f"saved step-expert student LoRA → {file}  ({len(sd)} keys, "
            f"K={n_heads} up-heads/Linear; kept-live only — not plain-LoRA mergeable)"
        )
