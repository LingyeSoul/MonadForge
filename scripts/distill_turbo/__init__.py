"""Turbo Anima — Decoupled DMD2 distillation.

Trains a 4-step LoRA student against the 28-step CFG=4 Anima teacher, using
Liu et al.'s Decoupled-Hybrid schedule (arXiv:2511.22677, Table 1 row 4) on
top of a co-LoRA fake score model.

Docs:     ``docs/experimental/dmd2-decoupled.md`` (usage / ops),
          ``docs/structure/dmd2-decoupled.md`` (math / walkthrough),
          ``docs/proposal/dmd2_decoupled_improvements.md`` (decision log).
Config:   ``configs/methods/turbo.toml`` (CLI flags override TOML values).

Entry point:

* ``python -m scripts.distill_turbo.distill`` — train the student LoRA.

Module map:

* :mod:`scripts.distill_turbo.distill`    — main loop (sample t, build x_t,
  student/CA/DM/fake forwards, optimizer steps, save).
* :mod:`scripts.distill_turbo.config`     — TOML loader, argparser, CLI/TOML
  precedence resolver, schema validation.
* :mod:`scripts.distill_turbo.primitives` — re-noising, τ samplers, scheduler
  factory, pad-tensor cache, dataloader collate.
* :mod:`scripts.distill_turbo.ca_band`    — CA-branch FEI band-deficit
  reweighting (item 2; see ``item2_plan.md``).
* :mod:`scripts.distill_turbo.warmup`     — fake (critic) head-start loop
  that runs before the main training loop.
* :mod:`scripts.distill_turbo.metrics`    — GPU-side accumulators + single-sync
  log flush.

One frozen DiT serves three roles via per-network ``set_enabled`` toggling:

    teacher view  — both LoRA stacks off (base velocity)
    student view  — student on, fake off (v_student for x_pred)
    fake view     — student off, fake on (s_fake_cond_dm)

Before the main loop, an optional fake (critic) head-start runs
``fake_warmup_steps`` fake-only updates (step 5 alone) against the student's
init ≈ teacher x_pred distribution, so the critic is calibrated before the
student LR warmup ramps to full strength — this removes the early
grad_signal_rms spike (~step 50). The student is untouched during it.

Per training step (single-call DMD2 — no inference sampler unroll at train
time, gradient is one ODE step from the sampled generator-t):

    1.  v_student = student(x_t, t, c)        # grad to student params
        x_pred    = x_t - t · v_student       # endpoint estimate

    2.  CA branch (τ_CA > t)                  # paper's CFG-bake engine
        v_real_cond_ca   = teacher(x_τ_ca, τ_CA, c)        # no_grad
        v_real_uncond_ca = teacher(x_τ_ca, τ_CA, c_null)   # no_grad
        Δ_cfg = v_real_cond_ca - v_real_uncond_ca

    3.  DM branch (τ_DM ∈ [0, 1])             # regularizer
        v_real_cond_dm = teacher(x_τ_dm, τ_DM, c)          # no_grad
        v_fake_cond_dm = fake   (x_τ_dm, τ_DM, c)          # no_grad
        Δ_dm = v_real_cond_dm - v_fake_cond_dm

    4.  α_eff ramps 1.0 → α over alpha_warmup_steps         # CA warmup
        The DiT predicts velocity v = ε − x0, so the x0-prediction gap the
        DMD2 update acts on converts with a +τ factor (per branch):
            x0_real − x0_fake          = −τ_dm·Δ_dm
            CFG-baked x0 shift          = −τ_ca·(α−1)·Δ_cfg
        We want x_pred to move TOWARD x0_real / the CFG-baked endpoint, so the
        surrogate-loss gradient on x_pred must be +(τ_dm·Δ_dm + τ_ca·(α−1)·Δ_cfg);
        gradient descent then steps x_pred along the negative of that — the
        desired direction.
            grad_signal  = τ_dm·Δ_dm + τ_ca·(α_eff − 1)·Δ_cfg
            loss_student = (grad_signal · x_pred).mean()
            loss_student.backward()  → student.step()

    5.  Fake update — flow-matching loss on student's x_pred distribution:
        τ_fake ~ U[0,1]
        x_t_fake = (1-τ_fake)·x_pred.detach() + τ_fake·ε_fake
        v_fake   = fake(x_t_fake, τ_fake, c)                # grad to fake params
        target   = ε_fake - x_pred.detach()                 # flow-matching target
        fake_loss = MSE(v_fake, target)  → fake.step()

Output: ``output/ckpt/anima_turbo.safetensors`` — a normal plain-LoRA file
loadable by the standard inference path at ``--infer_steps 4 --cfg 1.0``.
"""
