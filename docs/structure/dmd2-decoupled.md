# Decoupled DMD2: CFG-augmentation as the spear, distribution-matching as the shield

A few-step distillation that compresses the 28-step, CFG=4 Anima teacher into a
**4-step student LoRA**. Both the student *and* the auxiliary "fake" score model are
plain LoRAs on **one frozen DiT** — the base weights do the heavy lifting for all
three roles (teacher / student / fake), and only the rank-$r$ adapters train.

This is a port of Liu et al., *"CFG Augmentation as the Spear, Distribution Matching
as the Shield"* (arXiv:2511.22677), Table 1 row 4 (Decoupled-Hybrid). The output is a
normal LoRA file: no inference-side code, you just run it at 4 steps with CFG=1.

> This doc is the **structural walkthrough** — the gradient decomposition, the
> velocity↔x0 conversion that makes it work on a flow-matching DiT, the schedule, and
> the co-LoRA capacity argument. For the **usage / ops / decision-log** reference
> (config knobs, `make` targets, metrics to watch, current status), see
> **`docs/experimental/dmd2-decoupled.md`**.

---

## 1. Why distill at all, and why DMD2

The teacher is good but slow: 28 sampling steps × 2 forwards per step (cond + uncond
for CFG=4) = 56 DiT forwards per image. A distilled student that matches it at 4
steps × 1 forward (CFG baked in) is a ~14× inference cut.

The naive distillation target — "regress the student's few-step trajectory onto the
teacher's many-step trajectory" — is brittle: it locks the student to one solver and
one step count, and small errors compound across the compressed trajectory. **DMD**
(Distribution Matching Distillation) instead matches *distributions*: it asks the
student's output distribution to look like the teacher's, scored by a moving "fake"
model that learns what the student currently produces. DMD2 is the score-model
variant; Decoupled DMD is the observation that the DMD gradient **factors into two
terms with different jobs**, and that scheduling them separately is what makes it
fast and stable.

---

## 2. The gradient decomposition (the whole idea)

The DMD-in-practice gradient on the student parameters $\theta$ decomposes
algebraically into two terms:

$$
\nabla_\theta \mathcal{L}_\text{DMD}
= -\,\mathbb{E}\Big[\big(\underbrace{\Delta_\text{real-fake}}_{\text{DM, the shield}}
+ (\alpha-1)\underbrace{\Delta_\text{cfg}}_{\text{CA, the spear}}\big)\cdot
\frac{\partial G_\theta(z_t)}{\partial \theta}\Big]
$$

$$
\Delta_\text{real-fake} = s^\text{real}_\text{cond}(x_\tau) - s^\text{fake}_\text{cond}(x_\tau)
\qquad
\Delta_\text{cfg} = s^\text{real}_\text{cond}(x_\tau) - s^\text{real}_\text{uncond}(x_\tau)
$$

The paper's two empirical claims, which the whole design leans on:

- **CA (CFG-augmentation) is the engine.** Training with $\Delta_\text{cfg}$ alone
  converts a multi-step model into a usable few-step generator *quickly* — almost all
  of the few-step ability comes from "baking the CFG pattern" $(\alpha-1)(s_\text{cond}
  - s_\text{uncond})$ into the student. But it then collapses into artifacts after a
  few thousand iterations.
- **DM (distribution-matching) is a regularizer, not the engine.** Training with
  $\Delta_\text{real-fake}$ alone is unstable; its job in the combined loss is to
  *cancel the artifacts CA introduces*. (The paper shows simpler regularizers —
  mean/variance matching, GAN — also stabilize CA; DM is just a well-behaved choice.)

Spear and shield. The student is converted to few-step by CA and kept on-manifold by
DM.

---

## 3. Three roles, one frozen DiT

```
                          frozen Anima DiT  (no grad, ~5 GB bf16)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   teacher view              student view                fake view
   both LoRAs OFF            student ON, fake OFF         fake ON, student OFF
   → base velocity           → v_student → x_pred         → v_fake_cond_dm
   (CFG'd at α externally)   (the generator G_θ)          (the score tracker)
```

`networks/methods/turbo_dmd.py::TurboDMDNetwork` owns two ordinary `LoRANetwork`
instances. Both `apply_to(unet)`, chaining onto every targeted Linear's forward:

```
linear(x)  →  fake.forward  →  student.forward  →  original_linear.forward
```

Each `LoRAModule` short-circuits at `not self.enabled`, so a *view switch* is just
`set_enabled(bool)` on each network — an O(num_modules) Python flag flip, negligible
beside a DiT forward. `set_view` flips only what changed, and short-circuits when the
target view is already active (the CA branch fires two consecutive teacher forwards).

**`freeze_dit` runs after both `apply_to`'s** and walks only the base params (names
not prefixed by a LoRA module), so it freezes the backbone without zeroing the LoRA
grads. The DiT is never an optimizer target; there are two AdamW states, one per LoRA.

### Co-LoRA capacity: why $r_\text{fake} \ge r_\text{student}$

This is the counter-intuitive bit. Usual LoRA intuition says smaller rank =
better-regularized. Here the fake is **not a generator — it's a score tracker.** Its
DM term $(s^\text{real}_\text{cond} - s^\text{fake}_\text{cond})$ is the corrective
signal; if the fake *under-fits* the student's current output distribution, DM produces
noisy gradients and stops canceling CA's artifacts (the paper's Fig 2 row 1:
high-frequency checkerboard that compounds over training). The student LoRA at rank
$r_s$ defines a manifold of perturbed scores; the fake needs to be at least
$r_s$-expressive to track it pointwise. So $r_\text{fake} \ge r_\text{student}$ is a
capacity floor on the regularizer, not a stability prior. (The script warns if
`fake_rank < student_rank`.)

---

## 4. The velocity↔x0 conversion (what makes it work on Anima)

The paper is written for score/ε-prediction. **Anima predicts velocity**
$v = \varepsilon - x_0$ on the flow-matching path $x_t = (1-t)\,x_0 + t\,\varepsilon$
(see `docs/findings/asymflow_parameterization.md`). Everything has to be re-derived in
velocity/x0 space or the signs and scales are wrong.

**Single-call generator.** Per step we sample a generator timestep $t$, build
$x_t = (1-t)\,x_0 + t\,\varepsilon$ from a *dataset* latent, and convert the student's
velocity to its clean-endpoint estimate:

$$
v_\text{student} = G_\theta(x_t, t, c), \qquad x_\text{pred} = x_t - t\,v_\text{student}
$$

This is the DMD2 single-call generator. **We do not unroll the 4-step inference sampler
at train time** — the gradient is one ODE step from the sampled $t$.

**Re-noising primitive.** To score $x_\text{pred}$ at a fresh noise level $\tau$:

$$
x_\tau = (1-\tau)\,x_\text{pred} + \tau\,\varepsilon, \qquad \varepsilon \sim \mathcal N(0, I)
$$

— the same forward path, applied to the *predicted* clean image instead of a dataset
latent (`renoise()` in `scripts/distill_turbo/primitives.py`).

**Sign / scale of the update.** The DM and CA deltas are *velocity* gaps; the
distribution-matching update acts in **x0 space**. Converting a velocity gap to its x0
gap at level $\tau$ picks up a $-\tau$ factor:

$$
x_0^\text{real} - x_0^\text{fake} = -\tau_\text{dm}\,\Delta_\text{dm},
\qquad
\text{CFG-baked x0 shift} = -\tau_\text{ca}\,(\alpha-1)\,\Delta_\text{cfg}
$$

We want $x_\text{pred}$ to move **toward** $x_0^\text{real}$ and the CFG-baked
endpoint, so the surrogate-loss gradient *on* $x_\text{pred}$ must be the positive
combination; gradient descent then steps along its negative — the desired direction.
Each branch carries **its own** renoise level $\tau$:

$$
\texttt{grad\_signal} = \tau_\text{dm}\,\Delta_\text{dm}
+ \tau_\text{ca}\,(\alpha_\text{eff}-1)\,\Delta_\text{cfg}
$$

and the DMD2 grad trick assembles a dummy scalar whose $\partial/\partial x_\text{pred}$
equals it:

```python
loss_student = (grad_signal.detach() * x_pred).mean()   # ∂/∂x_pred = grad_signal
loss_student.backward()                                  # walks x_pred → v_student → θ
```

> **The sign of this term is load-bearing and was once inverted.** Before the
> 2026-05-27 fix the student gradient pointed the wrong way (anti-distill). The tell
> is subtle: inverted-sign runs look like *"base 4-step blur / never trained,"* **not**
> a blow-up. See [[project_turbo_dmd_sign_fix]].

---

## 5. The Decoupled-Hybrid schedule (Table 1 row 4)

The two terms re-noise to **different timesteps**, sampled per step:

### CA branch — $\tau_\text{CA} > t$ (focused spear)

$$
\tau_\text{CA} \sim \mathcal U\big(\max(t + \delta,\ \cdot),\ 1.0\big), \qquad \delta = \texttt{tau\_ca\_min\_gap}
$$

CA re-noises **strictly noisier** than the generator's current step. Two reasons this
is correctness work, not just gradient concentration:

1. **It focuses CA on still-unresolved content** — the engine works on what actually
   needs converting.
2. **It dodges the training–inference drift.** Anima's iterative inference state
   matches the training $x_t$ distribution through ~step 25, then drifts at steps
   26–28 (the same SNR-t boundary drift DCW corrects). At small generator-$t$ the
   student operates near that drifted regime, so evaluating CA there would compute the
   teacher's CFG gap on *out-of-distribution* states. Renoising **up** into
   $\tau > t$ puts the CA evaluation back where the teacher's score is well-calibrated.

Testable prediction (and the reason for the guard): CA-branch variance blows up if
$\tau_\text{CA}$ is allowed to drop near small generator-$t$. The implementation
clamps $\tau_\text{CA} \ge t + \delta$ and **skips the CA branch entirely when
$t > \texttt{tau\_ca\_skip\_above\_t}$** (default 0.95) — at near-clean $t$ the
$\mathcal U(t, 1)$ interval collapses to noise. That step trains on DM alone.

### DM branch — $\tau_\text{DM} \sim \mathcal U[0, 1]$ (comprehensive shield)

DM spans the full noise range so it can correct global artifacts (color drift,
oversaturation) that the student inherits regardless of the current step.

### CFG-bake warmup

$$
\alpha_\text{eff} = \alpha \cdot w + 1\cdot(1-w), \qquad w = \min\!\big(1, \tfrac{\text{step}+1}{\texttt{alpha\_warmup\_steps}}\big)
$$

At step 0, $\alpha_\text{eff} = 1$ → the CA term vanishes → the student starts inside
the regime DM can regularize, *before* the large $(\alpha-1)=3$ spear kicks in. This
is structural, not a band-aid: the LoRA student has less capacity than the paper's
full-fine-tune student, and $(\alpha-1)=3$ from step 0 can NaN it before any image
structure forms.

---

## 6. The fake update (keeping the shield sharp)

The fake learns to denoise the student's *current* output distribution by plain
flow-matching regression:

$$
\tau_\text{fake} \sim \mathcal U[0,1], \quad
x_\tau^\text{fake} = (1-\tau_\text{fake})\,x_\text{pred}^\text{detach} + \tau_\text{fake}\,\varepsilon,
\quad
\mathcal L_\text{fake} = \big\|\,\text{fake}(x_\tau^\text{fake}, \tau_\text{fake}, c) - (\varepsilon - x_\text{pred}^\text{detach})\,\big\|^2
$$

Run `fake_steps_per_student_step` (default 2) inner steps against the **same**
$x_\text{pred}^\text{detach}$, resampling $(\tau, \varepsilon)$ each time. Standard DMD2
practice: the fake's target distribution is *moving* as the student sharpens, so the
fake is given extra SGD iterations (and a hotter LR, `fake_lr` > `student_lr`) to stay
ahead. The fake's cosine LR schedule anneals over `iterations × fake_steps_per_student_step`,
not the student's count.

---

## 7. DM-grad normalization: two policies, not additive

There are two ways to weight the DM term, and they are **alternatives**, not a stack.

- **(a) τ-damping** — use $\tau_\text{dm}\,\Delta_\text{dm}$ directly (the
  $\tau$-weighting that co-landed with the sign fix).
- **(b) DMD per-sample x0-norm** — original DMD normalizes the DM gradient per-sample
  for scale-invariance. The DM x0-gap is $x_0^\text{real} - x_\tau = -\tau\,v^\text{real}_\text{cond,dm}$,
  whose per-sample magnitude is $\text{denom} = \tau\cdot\overline{|v^\text{real}|}$:

  ```python
  denom   = (tau_dm_e * v_real_cond_dm).abs().mean(dim=(1,2,3), keepdim=True).clamp_min(norm_floor)
  grad_dm = (tau_dm_e * delta_dm) / denom        # ≈ Δ_dm / mean|v_real|
  ```

**The subtlety:** because $\text{denom} \approx \tau\cdot\overline{|v^\text{real}|}$, the
$\tau$ **cancels across the bulk of the range** — (b) is therefore $\approx$ "drop the
$\tau$-weight and magnitude-normalize." The `clamp_min(norm_floor)` only bites for
$\tau < \texttt{norm\_floor}/\overline{|v^\text{real}|}$ (a thin sliver). So:

| Policy | What it is |
|---|---|
| (a) | $\tau$-damping only |
| (b) | DMD magnitude-normalization ($\tau$ roughly cancels) — **shipped default** |
| (c) | both ≈ (b) with $\tau$ re-multiplied in — **do not ship believing it composes** |

**Why (b) wins (settled A/B, 2026-05-28).** (a)'s $\tau$-damp × *raw* magnitude
over-weights high-$|v^\text{real}|$ samples (high-frequency structure, text, off-mode
tails) and over-pulls them to the dominant mode — DMD mode-seeking, visible as
near-identical outputs across seeds. (b)'s per-sample normalization gives every sample
a unit-scale direction → even distribution-matching pressure → preserves the tails
(seed diversity) and fine structure (**text rendering** improved). The effective-LR
confound is ruled out: (b) runs ~2× the DM-grad magnitude, and more DM pressure would
mean *more* mode-seeking, yet diversity went up — so scale-invariance is the driver,
not the magnitude bump. The per-step health scalars are blind to between-seed
diversity; the multi-seed 4-step sample check was the decisive signal. The
$\tau$-weighting was therefore **harmful, not inert**. See
[[project_turbo_dmd_x0_norm_wins]]. The CA engine keeps its own $\tau_\text{ca}$
weighting either way — normalization is DM-term only.

---

## 8. Masked loss (student-only)

With `use_masked_loss=true` the per-image foreground mask multiplies the **student
DMD2 gradient** so distribution-matching focuses on the subject:

```python
loss_student = (grad_signal * x_pred * mask).mean()   # background latents → zero student push
```

The fake/critic regression is left **full-frame** (it still needs to model the whole
distribution). Normalization stays `/numel` (no renorm by mask area, matching
`apply_masked_loss`), so a masked run sees a lower effective gradient by design.

---

## 9. What's frozen at inference, and why it ships as a plain LoRA

`TurboDMDNetwork.save_student` serializes **only the student** in the standard
plain-LoRA layout (`save_variant="standard"`); the fake is training scaffolding and
never shipped. The student is an ordinary rank-$r$ LoRA with **CFG=4 baked in** — load
it through the normal inference path and run `--infer_steps 4 --cfg 1.0`. No turbo
code runs at inference.

Consequences of the plain-LoRA bake (the load-bearing constraint):

- **Composes** with concept LoRAs linearly (ranks add), same model surgery as
  LCM-LoRA + style LoRA.
- **Cannot carry** anything needing a step-size/per-t input at inference (Shortcut /
  MeanFlow Δt-conditioning; timestep-conditioned T-LoRA whose mask is training-only,
  [[project_tlora_inference_full_rank]]). A plain LoRA must average antagonistic
  per-$t$ corrections — true multi-stride robustness is out of scope.
- **Incompatible with Spectrum** (Chebyshev cache assumes ≥16 steps).

---

## 10. Minimal mental model

1. **Two LoRAs on one frozen DiT.** Student = generator $G_\theta$; fake = score
   tracker. View-toggle by enabling/disabling each per forward.
2. **The DMD gradient factors into CA (spear) + DM (shield).** CA bakes CFG and
   converts to few-step fast; DM cancels CA's artifacts.
3. **Single-call generator in velocity space:** $x_\text{pred} = x_t - t\,v_\text{student}$,
   then re-noise to a fresh $\tau$ per branch. The x0-space update picks up a $\tau$
   factor; its sign must point $x_\text{pred}$ toward $x_0^\text{real}$.
4. **Decoupled-Hybrid schedule:** $\tau_\text{CA} > t$ (focused, dodges inference
   drift), $\tau_\text{DM} \in [0,1]$ (global). $\alpha_\text{eff}$ warms up $1\to\alpha$.
5. **Fake stays ahead** with a hotter LR + extra inner steps on the moving
   $x_\text{pred}$ distribution; $r_\text{fake} \ge r_\text{student}$.
6. **DM grad uses x0-norm (b), not τ-damping (a)** — preserves seed diversity and
   text; the two are alternatives, never stacked.
7. **Ships as a plain student LoRA**, CFG baked in, run at 4 steps / CFG=1.

---

*(A schematic for `docs/structure_images/dmd2_decoupled.png` — the three-role
frozen-DiT diagram plus the spear/shield gradient split — is still to be drawn; the
ASCII diagrams above are the interim reference.)*
