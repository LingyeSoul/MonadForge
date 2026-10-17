"""Register-token adapter for the Anima DiT — Phase-0.5 bench harness.

Gives the frozen DiT K non-decoded **register tokens** on the *self-attention*
sequence: a true residual-carrying scratchpad (concat once at the block-stack
entry, carried through all blocks, stripped before unpatchify/decode). This is
the DSR "diffusion register" (arXiv:2605.05206) retrofitted onto a frozen
pretrained DiT on a small adapter budget — the open, Anima-specific question
(`_archive/proposals/headroom_register_tokens.md`, RQ3-first redesign in
`bench/headroom/README.md`).

Two arms adjudicate *why* the DSR sink is load-bearing (bench README §"Where
this leaves RQ2/RQ3"):

* **Arm A** — K *fixed-zero* registers + self-attn QKV-LoRA. The base can learn
  to *route* into a content-free sink slot, but the slot carries no learnable
  state. Tests Theory 1: the sink is just a softmax mass-drain.
* **Arm B** — K *learnable* register embeddings + the same QKV-LoRA. The slot can
  carry state. Tests Theory 2: the sink is a scratchpad the rest of the canvas
  reads back (what `sink_intervention.py` measured — clamping ~7 tokens re-plans
  the whole image).

Implementation notes (the load-bearing invariants):

* **Native-flatten, no compile.** We force ``anima._native_flatten = True`` so the
  block stream is ``(B, 1, seq, 1, D)`` and appending K on the seq axis is clean.
  We do NOT torch.compile — a bench runs eager, which sidesteps the whole
  ``EDGE_TOKEN_BANDS``/``_derive_token_budget`` +K threading. (Shipping would need
  that; a bench does not.)
* **Rope-exempt registers for free.** We extend ``rope_cos_sin`` by K rows of
  ``cos=1, sin=0`` — identity rotation. No attention-path slicing (which would
  fight the ``mark_dynamic`` seq-axis invariant); the register keys are simply
  unrotated. See models.py::apply_rotary_pos_emb_qk.
* **Strip before unpatchify.** We wrap ``_run_blocks`` and drop the K tail tokens
  from its return, so ``_unflatten_native_shape``/``final_layer`` see the original
  seq. The dim-2 singleton is untouched (registers live on the flattened seq
  axis, dim 2).
* **QKV-LoRA zero-init up** → the LoRA is a no-op at step 0; the only step-0
  perturbation is the K registers' base-projected attention mass.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegisterAdapter(nn.Module):
    def __init__(
        self,
        anima,
        num_registers: int = 16,
        arm: str = "B",
        lora_rank: int = 8,
        lora_alpha: Optional[float] = None,
        target_blocks: Optional[list[int]] = None,
        init_std: float = 0.02,
        qkv_mode: str = "lora",
    ) -> None:
        super().__init__()
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        if qkv_mode not in ("lora", "unfrozen"):
            raise ValueError(f"qkv_mode must be 'lora' or 'unfrozen', got {qkv_mode!r}")
        # Hide the DiT from nn.Module registration (a list is not a Module) so
        # self.parameters() stays adapter-only and doesn't drag in the frozen DiT.
        self._anima_box = [anima]
        self.K = int(num_registers)
        self.arm = arm
        self.D = int(anima.model_channels)

        if arm == "B":
            self.register = nn.Parameter(torch.randn(self.K, self.D) * init_std)
        else:  # arm A: content-free sink slot, non-trainable
            self.register_buffer("register", torch.zeros(self.K, self.D))

        n_blocks = len(anima.blocks)
        if target_blocks is None:
            target_blocks = list(range(n_blocks))
        self.target_blocks = list(target_blocks)
        self.qkv_mode = qkv_mode
        self.lora_rank = int(lora_rank)
        self.scale = (lora_alpha if lora_alpha is not None else lora_rank) / lora_rank

        # The self-attn QKV surface is trained one of two ways on the target
        # blocks. **lora** = low-rank additive adapter (cheap, the RQ3 budget).
        # **unfrozen** = a full-rank trainable ΔW carried as a separate fp32
        # parameter (base weight stays frozen — same reachability as unfreezing
        # qkv_proj, but zero-init so it's a step-0 no-op and merges cleanly).
        # This is the proposal's K36-b8-QKV arm ("max reachability — the arm the
        # negative most likely mispriced"), _archive/proposals/headroom_register_tokens.md.
        self.lora = nn.ModuleDict()
        self.qkv_delta = nn.ParameterDict()
        for bi in self.target_blocks:
            qkv = anima.blocks[bi].self_attn.qkv_proj
            if qkv_mode == "lora":
                down = nn.Linear(qkv.in_features, self.lora_rank, bias=False)
                up = nn.Linear(self.lora_rank, qkv.out_features, bias=False)
                nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
                nn.init.zeros_(up.weight)  # LoRA is a no-op until trained
                self.lora[str(bi)] = nn.ModuleDict({"down": down, "up": up})
            else:  # unfrozen: full-rank ΔW, zero-init (no-op at step 0)
                self.qkv_delta[str(bi)] = nn.Parameter(
                    torch.zeros(qkv.out_features, qkv.in_features)
                )

        self._applied = False
        self._orig_run_blocks = None
        self._orig_qkv_fwd: dict[int, object] = {}
        self._orig_native_flatten = None
        # eval hook: last-forward norms (mean over the 5D block-output stream)
        self.last_reg_norm: Optional[float] = None
        self.last_patch_norm: Optional[float] = None
        self.last_patch_sink_ratio: Optional[float] = None  # topk_patch / median
        self.last_reg_max: Optional[float] = None
        self.last_reg_ratio: Optional[float] = None  # max register / median patch

    @property
    def anima(self):
        return self._anima_box[0]

    # -- trainable surface ---------------------------------------------------
    def trainable_parameters(self):
        params = list(self.lora.parameters()) + list(self.qkv_delta.parameters())
        if self.arm == "B":
            params.append(self.register)
        return params

    def register_parameters(self):
        """The register embedding(s) only — for a per-group (higher) lr."""
        return [self.register] if self.arm == "B" else []

    def qkv_parameters(self):
        """The QKV-adapter surface only (LoRA or full-rank ΔW)."""
        return list(self.lora.parameters()) + list(self.qkv_delta.parameters())

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # -- apply / remove ------------------------------------------------------
    def apply_to(self) -> "RegisterAdapter":
        if self._applied:
            return self
        anima = self.anima
        self._orig_native_flatten = anima._native_flatten
        anima._native_flatten = True  # eager native-flatten path

        adapter = self
        self._orig_run_blocks = anima._run_blocks

        def wrapped_run_blocks(
            x_padded,
            t_embedding_B_T_D,
            crossattn_emb,
            attn_params,
            capture_blocks=None,
            feature_sink=None,
            stop_after_block=None,
            **block_kwargs,
        ):
            # x_padded: (B, 1, seq, 1, D)
            B = x_padded.shape[0]
            seq = x_padded.shape[2]
            if adapter.K > 0:
                reg = adapter.register.to(dtype=x_padded.dtype, device=x_padded.device)
                reg = reg.view(1, 1, adapter.K, 1, adapter.D).expand(B, -1, -1, -1, -1)
                x_ext = torch.cat([x_padded, reg], dim=2)  # (B, 1, seq+K, 1, D)

                rope = block_kwargs.get("rope_cos_sin")
                if rope is not None:
                    cos, sin = rope  # each (seq, 1, 1, D_head)
                    pad_shape = (adapter.K,) + tuple(cos.shape[1:])
                    cos_ext = torch.cat([cos, cos.new_ones(pad_shape)], dim=0)
                    sin_ext = torch.cat([sin, sin.new_zeros(pad_shape)], dim=0)
                    block_kwargs = {**block_kwargs, "rope_cos_sin": (cos_ext, sin_ext)}
            else:  # K=0 — LoRA-only arm, the register-free drift control
                x_ext = x_padded

            out = adapter._orig_run_blocks(
                x_ext,
                t_embedding_B_T_D,
                crossattn_emb,
                attn_params,
                capture_blocks=capture_blocks,
                feature_sink=feature_sink,
                stop_after_block=stop_after_block,
                **block_kwargs,
            )
            # relocation-crossover readout: register vs patch residual ‖x‖.
            # The sink is SPARSE (~7 of 4200 tokens), so mean patch norm won't
            # show it move — track the top-k/median outlier ratio too.
            with torch.no_grad():
                pt = out[:, :, :seq, :, :].float().norm(dim=-1).flatten()  # (seq,)
                med = pt.median().clamp_min(1e-6)
                k = max(1, int(0.002 * pt.numel()))  # ~top 0.2% = the sink set
                topk_patch = pt.topk(k).values.mean()
                adapter.last_patch_norm = pt.mean().item()
                adapter.last_patch_sink_ratio = (topk_patch / med).item()
                if adapter.K > 0:
                    rt = out[:, :, seq:, :, :].float().norm(dim=-1).flatten()  # (K,)
                    adapter.last_reg_norm = rt.mean().item()
                    adapter.last_reg_max = rt.max().item()
                    adapter.last_reg_ratio = (rt.max() / med).item()
                else:
                    adapter.last_reg_norm = 0.0
                    adapter.last_reg_max = 0.0
                    adapter.last_reg_ratio = 0.0
            return out[:, :, :seq, :, :]  # strip registers before unpatchify

        anima._run_blocks = wrapped_run_blocks

        # self-attn QKV adapter on the target blocks (LoRA or full-rank ΔW).
        # ΔW is fp32; under the training autocast F.linear runs it in the block
        # dtype, matching the LoRA path (whose fp32 Linear weights autocast too).
        for bi in self.target_blocks:
            qkv = anima.blocks[bi].self_attn.qkv_proj
            self._orig_qkv_fwd[bi] = qkv.forward

            if self.qkv_mode == "lora":
                lora = self.lora[str(bi)]

                def make_fwd(orig, lora):
                    def fwd(x):
                        return orig(x) + lora["up"](lora["down"](x)) * adapter.scale

                    return fwd

                qkv.forward = make_fwd(qkv.forward, lora)
            else:
                delta = self.qkv_delta[str(bi)]

                def make_fwd_unfrozen(orig, delta):
                    def fwd(x):
                        return orig(x) + F.linear(x, delta)

                    return fwd

                qkv.forward = make_fwd_unfrozen(qkv.forward, delta)

        self._applied = True
        return self

    def remove(self) -> None:
        if not self._applied:
            return
        anima = self.anima
        anima._run_blocks = self._orig_run_blocks
        for bi, orig in self._orig_qkv_fwd.items():
            anima.blocks[bi].self_attn.qkv_proj.forward = orig
        anima._native_flatten = self._orig_native_flatten
        self._orig_qkv_fwd = {}
        self._applied = False

    # -- persistence ---------------------------------------------------------
    def config(self) -> dict:
        return {
            "num_registers": self.K,
            "arm": self.arm,
            "qkv_mode": self.qkv_mode,
            "lora_rank": self.lora_rank,
            "target_blocks": self.target_blocks,
            "scale": self.scale,
            "D": self.D,
        }
