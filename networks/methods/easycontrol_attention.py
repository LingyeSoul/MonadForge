"""LSE-decomposed extended self-attention for EasyControl's two-stream forward.

Self-contained attention math shared by the EasyControl network
(``networks/methods/easycontrol.py``) and BYG (``networks/methods/byg.py``):
the target stream attends over the concatenation ``[target_k ; cond_k]`` with a
per-block scalar logit bias (``b_cond``) on the cond rows, without ever
materializing the ``[B, H, S_t, S_t + S_c]`` attention matrix.

Split out of ``easycontrol.py`` (2026-06-08) — pure attention, zero coupling to
``EasyControlNetwork``, and separately benched
(``bench/easycontrol/step0_equivalence.py`` /
``bench/easycontrol/step1p5_lse_equivalence.py``).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class _ExtendedSelfAttnLSEFunc(torch.autograd.Function):
    """LSE-decomposed extended self-attention with a per-block scalar logit bias.

    Mathematically equivalent to::

        joint_out = softmax([Q@K_t^T·s ; Q@K_c^T·s + b]) @ [V_t; V_c]

    but never materializes the ``[B, H, S_q, S_t+S_c]`` attention matrix. Two
    memory-efficient FA2 forwards on the disjoint key tiles, then a Python
    LSE-arithmetic combine::

        α = exp(lse_t  - joint_lse)
        β = exp(lse_c+b - joint_lse)        joint_lse = logaddexp(lse_t, lse_c + b)
        joint_out = α · out_t + β · out_c

    Forward correctness (vs. masked SDPA) is identity, modulo fp32 ulp.

    Backward correctness is more subtle. FA2's stock ``FlashAttnFunc.backward``
    only consumes ``dout`` and silently discards the upstream gradient on
    ``softmax_lse``. A plain "two FA + Python combine via flash_attn_func"
    therefore drops the *path-2* gradient that flows from the loss back through
    α/β into ``q``/``k_t``/``k_c`` (the contribution scales as α·β·(out_c−out_t)
    in dout-space; negligible at init when β≈4.5e-5 from b_cond=-10, but grows
    as b_cond rises during training).

    To recover the joint-softmax gradient exactly, this Function bypasses the
    stock autograd and calls ``_wrapped_flash_attn_forward / _backward``
    directly. The trick: feeding ``softmax_lse = joint_lse`` (target) and
    ``softmax_lse = joint_lse - b`` (cond) into the per-tile FA backward causes
    FA to compute joint-softmax probabilities ``exp(L_t·s)/Z`` and
    ``exp(L_c·s + b)/Z`` respectively, so per-tile contributions sum to the
    correct joint gradient on q/k/v. ``b_cond``'s gradient is computed
    analytically from α, β, out_t, out_c, dout.
    """

    @staticmethod
    def forward(ctx, q, k_t, v_t, k_c, v_c, b_cond, softmax_scale):
        from networks import attention_dispatch as anima_attention

        if anima_attention._wrapped_flash_attn_forward is None:
            raise RuntimeError(
                "_ExtendedSelfAttnLSEFunc requires flash-attn to be installed"
            )
        fa_fwd = anima_attention._wrapped_flash_attn_forward

        out_t, lse_t, _, rng_state_t = fa_fwd(
            q,
            k_t,
            v_t,
            0.0,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        out_c, lse_c, _, rng_state_c = fa_fwd(
            q,
            k_c,
            v_c,
            0.0,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

        # LSE arithmetic combine. (FA returns lse in fp32 regardless of
        # input dtype, so b_cond — also fp32 — adds without promotion.)
        b_fp32 = b_cond.to(lse_c.dtype)
        lse_c_adj = lse_c + b_fp32
        joint_lse = torch.logaddexp(lse_t, lse_c_adj)
        alpha = (lse_t - joint_lse).exp()  # [B, H, S_q] fp32
        beta = (lse_c_adj - joint_lse).exp()  # [B, H, S_q] fp32

        # out_t/out_c are [B, S_q, H, D] (BLHD); broadcast α/β over D.
        alpha_bd = alpha.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
        beta_bd = beta.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
        joint_out = alpha_bd * out_t + beta_bd * out_c

        ctx.save_for_backward(
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            joint_out,
            joint_lse,
            alpha,
            beta,
            out_t,
            out_c,
            b_fp32,
            rng_state_t,
            rng_state_c,
        )
        ctx.softmax_scale = softmax_scale
        ctx.b_cond_orig_dtype = b_cond.dtype
        return joint_out

    @staticmethod
    def backward(ctx, dout):
        from networks import attention_dispatch as anima_attention

        fa_bwd = anima_attention._wrapped_flash_attn_backward
        (
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            joint_out,
            joint_lse,
            alpha,
            beta,
            out_t,
            out_c,
            b_fp32,
            rng_state_t,
            rng_state_c,
        ) = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale

        dout = dout.contiguous()

        # Tile 1 (target) — feed JOINT lse and JOINT out so that FA computes
        # per-key softmax mass = exp(L_t·s - joint_lse) = exp(L_t·s) / Z, which
        # is the joint-softmax probability on target keys; and uses joint_out
        # as the "softmax output" reference (V_t - joint_out is the correct
        # second term).
        dq_t = torch.empty_like(q)
        dk_t = torch.empty_like(k_t)
        dv_t = torch.empty_like(v_t)
        fa_bwd(
            dout,
            q,
            k_t,
            v_t,
            joint_out,
            joint_lse,
            dq_t,
            dk_t,
            dv_t,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng_state=rng_state_t,
        )

        # Tile 2 (cond) — feed (joint_lse - b) so per-key mass becomes
        # exp(L_c·s - (joint_lse - b)) = exp(L_c·s + b) / Z, the joint-softmax
        # probability on cond keys (with the bias).
        effective_lse_c = joint_lse - b_fp32
        dq_c = torch.empty_like(q)
        dk_c = torch.empty_like(k_c)
        dv_c = torch.empty_like(v_c)
        fa_bwd(
            dout,
            q,
            k_c,
            v_c,
            joint_out,
            effective_lse_c,
            dq_c,
            dk_c,
            dv_c,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng_state=rng_state_c,
        )

        dq = dq_t + dq_c

        # b_cond gradient — analytical from the LSE arithmetic.
        #   ∂joint_out/∂b = α · β · (out_c − out_t)            [B, S_q, H, D]
        #   ∂L/∂b         = sum (α · β · ⟨out_c − out_t, dout⟩_D)
        # Reduction in fp32 for stability (α, β are fp32; bf16 inner can lose
        # ulps on long S_q reductions).
        inner_bsh = ((out_c.float() - out_t.float()) * dout.float()).sum(
            dim=-1
        )  # [B, S_q, H]
        inner_bhq = inner_bsh.transpose(1, 2)  # [B, H, S_q]
        db_scalar = (alpha * beta * inner_bhq).sum()
        db_cond = db_scalar.to(ctx.b_cond_orig_dtype)
        # Match b_cond's original 0-d shape.
        if b_fp32.dim() == 0:
            db_cond = db_cond.reshape(())

        return dq, dk_t, dv_t, dk_c, dv_c, db_cond, None


class _ExtendedSelfAttnLSEFunc3(torch.autograd.Function):
    """Three-tile LSE-decomposed extended self-attention (FlowBender).

    Generalizes :class:`_ExtendedSelfAttnLSEFunc` to a SECOND extra key tile —
    the FlowBender feedback stream — so the target attends over
    ``[target_k ; cond_k ; feedback_k]`` with two independent per-block scalar
    logit biases (``b_cond`` on the cond rows, ``b_feedback`` on the feedback
    rows). Equivalent to::

        softmax([Q@K_t^T·s ; Q@K_c^T·s + b_c ; Q@K_f^T·s + b_f]) @ [V_t;V_c;V_f]

    Three memory-efficient FA2 forwards on the disjoint key tiles, then the
    LSE-arithmetic combine with three weights α/β/γ that sum to 1::

        joint_lse = logaddexp(logaddexp(lse_t, lse_c+b_c), lse_f+b_f)
        α,β,γ     = exp(lse_t-J), exp(lse_c+b_c-J), exp(lse_f+b_f-J)
        joint_out = α·out_t + β·out_c + γ·out_f

    Backward mirrors the two-tile derivation. Each tile's FA backward is fed the
    JOINT lse minus that tile's bias so FA computes the joint-softmax mass on
    that tile's keys; per-tile dq sum to the joint dq. The bias gradients use
    the general softmax-Jacobian form (which reduces to the two-tile
    ``α·β·(out_c−out_t)`` when γ→0)::

        ∂L/∂b_i = Σ w_i · ⟨out_i − joint_out, dout⟩    (w_t=α, w_c=β, w_f=γ)
    """

    @staticmethod
    def forward(
        ctx, q, k_t, v_t, k_c, v_c, k_f, v_f, b_cond, b_feedback, softmax_scale
    ):
        from networks import attention_dispatch as anima_attention

        if anima_attention._wrapped_flash_attn_forward is None:
            raise RuntimeError(
                "_ExtendedSelfAttnLSEFunc3 requires flash-attn to be installed"
            )
        fa_fwd = anima_attention._wrapped_flash_attn_forward

        def _fa(k, v):
            return fa_fwd(
                q,
                k,
                v,
                0.0,
                softmax_scale,
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )

        out_t, lse_t, _, rng_t = _fa(k_t, v_t)
        out_c, lse_c, _, rng_c = _fa(k_c, v_c)
        out_f, lse_f, _, rng_f = _fa(k_f, v_f)

        bc = b_cond.to(lse_c.dtype)
        bf = b_feedback.to(lse_f.dtype)
        lse_c_adj = lse_c + bc
        lse_f_adj = lse_f + bf
        joint_lse = torch.logaddexp(torch.logaddexp(lse_t, lse_c_adj), lse_f_adj)
        alpha = (lse_t - joint_lse).exp()  # [B, H, S_q] fp32
        beta = (lse_c_adj - joint_lse).exp()
        gamma = (lse_f_adj - joint_lse).exp()

        a_bd = alpha.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
        b_bd = beta.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
        g_bd = gamma.transpose(1, 2).unsqueeze(-1).to(out_f.dtype)
        joint_out = a_bd * out_t + b_bd * out_c + g_bd * out_f

        ctx.save_for_backward(
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            k_f,
            v_f,
            joint_out,
            joint_lse,
            alpha,
            beta,
            gamma,
            out_t,
            out_c,
            out_f,
            bc,
            bf,
            rng_t,
            rng_c,
            rng_f,
        )
        ctx.softmax_scale = softmax_scale
        ctx.b_cond_orig_dtype = b_cond.dtype
        ctx.b_feedback_orig_dtype = b_feedback.dtype
        ctx.b_cond_is_0d = b_cond.dim() == 0
        ctx.b_feedback_is_0d = b_feedback.dim() == 0
        return joint_out

    @staticmethod
    def backward(ctx, dout):
        from networks import attention_dispatch as anima_attention

        fa_bwd = anima_attention._wrapped_flash_attn_backward
        (
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            k_f,
            v_f,
            joint_out,
            joint_lse,
            alpha,
            beta,
            gamma,
            out_t,
            out_c,
            out_f,
            bc,
            bf,
            rng_t,
            rng_c,
            rng_f,
        ) = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        dout = dout.contiguous()

        def _bwd(k, v, eff_lse, rng):
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            fa_bwd(
                dout,
                q,
                k,
                v,
                joint_out,
                eff_lse,
                dq,
                dk,
                dv,
                0.0,
                softmax_scale,
                False,
                -1,
                -1,
                0.0,
                None,
                False,
                rng_state=rng,
            )
            return dq, dk, dv

        # Each tile feeds joint_lse - b_tile so FA's per-key mass is the joint
        # softmax probability on that tile (b_target = 0).
        dq_t, dk_t, dv_t = _bwd(k_t, v_t, joint_lse, rng_t)
        dq_c, dk_c, dv_c = _bwd(k_c, v_c, joint_lse - bc, rng_c)
        dq_f, dk_f, dv_f = _bwd(k_f, v_f, joint_lse - bf, rng_f)
        dq = dq_t + dq_c + dq_f

        # Bias gradients (general softmax-Jacobian form, fp32 reduction).
        #   ∂L/∂b_i = Σ w_i · ⟨out_i − joint_out, dout⟩
        jo = joint_out.float()
        do = dout.float()

        def _db(w_bhq, out_i, orig_dtype, is_0d):
            inner_bsh = ((out_i.float() - jo) * do).sum(dim=-1)  # [B, S_q, H]
            db = (w_bhq * inner_bsh.transpose(1, 2)).sum().to(orig_dtype)
            return db.reshape(()) if is_0d else db

        db_cond = _db(beta, out_c, ctx.b_cond_orig_dtype, ctx.b_cond_is_0d)
        db_feedback = _db(gamma, out_f, ctx.b_feedback_orig_dtype, ctx.b_feedback_is_0d)

        return dq, dk_t, dv_t, dk_c, dv_c, dk_f, dv_f, db_cond, db_feedback, None


_LSE_FALLBACK_WARNED = False


def _warn_lse_fallback_once(reason: str) -> None:
    """One-shot warning when we can't use the LSE-decomposed path."""
    global _LSE_FALLBACK_WARNED
    if _LSE_FALLBACK_WARNED:
        return
    _LSE_FALLBACK_WARNED = True
    logger.warning(
        f"EasyControl: falling back to masked-SDPA path ({reason}). The math "
        f"kernel materializes a [B, H, S_t, S_t+S_c] attention matrix per "
        f"block (~1 GB / block at bf16), which can OOM on real hardware. "
        f"Install flash-attn and use attn_mode='flash' for the LSE-decomposed "
        f"path."
    )


def _extended_target_attention(
    target_q,
    target_k,
    target_v,
    cond_k,
    cond_v,
    *,
    b_param,
    scale,
    attn_params,
    feedback_k=None,
    feedback_v=None,
    b_feedback_param=None,
):
    """Run target's extended self-attention over [target_k; cond_k] — or, when
    a FlowBender feedback tile is supplied, over [target_k; cond_k; feedback_k].

    Inputs are BSHD: target_q/k/v ``[B, S_t, H, D]``, cond_k/v ``[B, S_c, H, D]``,
    optional feedback_k/v ``[B, S_f, H, D]``. Returns ``[B, S_t, H*D]`` ready for
    output_proj. Uses ``_ExtendedSelfAttnLSEFunc`` (2-tile) or
    ``_ExtendedSelfAttnLSEFunc3`` (3-tile) when flash-attn + flash mode is
    available; falls back to masked-SDPA (math kernel; OOM risk) otherwise.
    """
    from networks import attention_dispatch as anima_attention

    # dtype matching mirrors the original Attention.forward casting policy.
    if target_q.dtype != target_v.dtype:
        if (
            not attn_params.supports_fp32 or attn_params.requires_same_dtype
        ) and torch.is_autocast_enabled():
            target_q = target_q.to(target_v.dtype)
            target_k = target_k.to(target_v.dtype)
    cond_k = cond_k.to(target_k.dtype)
    cond_v = cond_v.to(target_v.dtype)
    has_feedback = feedback_k is not None
    if has_feedback:
        feedback_k = feedback_k.to(target_k.dtype)
        feedback_v = feedback_v.to(target_v.dtype)

    if scale is None:
        scale = target_q.shape[-1] ** -0.5

    use_lse = (
        anima_attention._wrapped_flash_attn_forward is not None
        and attn_params.attn_mode == "flash"
    )
    if use_lse:
        if has_feedback:
            out = _ExtendedSelfAttnLSEFunc3.apply(
                target_q.contiguous(),
                target_k.contiguous(),
                target_v.contiguous(),
                cond_k.contiguous(),
                cond_v.contiguous(),
                feedback_k.contiguous(),
                feedback_v.contiguous(),
                b_param,
                b_feedback_param,
                scale,
            )
        else:
            out = _ExtendedSelfAttnLSEFunc.apply(
                target_q.contiguous(),
                target_k.contiguous(),
                target_v.contiguous(),
                cond_k.contiguous(),
                cond_v.contiguous(),
                b_param,
                scale,
            )
        # out: [B, S_t, H, D] → [B, S_t, H*D]
        B, S_t = out.shape[0], out.shape[1]
        return out.reshape(B, S_t, -1)

    # Fallback: masked extended SDPA. Materializes the full attention matrix
    # in the math kernel — only used when FA is unavailable.
    if attn_params.attn_mode == "flash":
        _warn_lse_fallback_once("flash-attn import failed at module load")
    else:
        _warn_lse_fallback_once(
            f"attn_mode={attn_params.attn_mode!r} unsupported by LSE path"
        )

    B, S_t = target_q.shape[0], target_q.shape[1]
    S_c = cond_k.shape[1]
    k_tiles = [target_k, cond_k]
    v_tiles = [target_v, cond_v]
    b = b_param.to(target_q.dtype)
    bias_tiles = [
        torch.zeros(S_t, device=target_q.device, dtype=target_q.dtype),
        b.expand(S_c),
    ]
    if has_feedback:
        S_f = feedback_k.shape[1]
        k_tiles.append(feedback_k)
        v_tiles.append(feedback_v)
        bias_tiles.append(b_feedback_param.to(target_q.dtype).expand(S_f))
    k_s = torch.cat(k_tiles, dim=1).transpose(1, 2)
    v_s = torch.cat(v_tiles, dim=1).transpose(1, 2)
    q_s = target_q.transpose(1, 2)
    attn_bias = torch.cat(bias_tiles, dim=0).view(1, 1, 1, -1)
    out = F.scaled_dot_product_attention(
        q_s, k_s, v_s, attn_mask=attn_bias, scale=scale
    )
    return out.transpose(1, 2).reshape(B, S_t, -1)
