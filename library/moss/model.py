"""MOSS-Image DiT + scheduler, vendored from ``pipeline_moss_image.py``.

Source: OpenMOSS-Team/MOSS-Image-{Base,SFT} (``pipeline_moss_image.py``,
https://github.com/OpenMOSS/MOSS-Image), Apache License 2.0, Copyright the
OpenMOSS team. Vendored so nothing loads with ``trust_remote_code``. State-dict
names are unchanged, so ``from_pretrained`` reads the published ``transformer/``
folder as-is.

Changes from the reference, all additive:

- ``forward`` is the reference per-sample loop, op for op. It is split into
  ``time_conditioning`` / ``embed_sample`` / ``run_layers`` / ``head`` so the
  sampler (Spectrum) and the trainer can reach the pieces.
- ``forward_batched``: every sample in one forward, text padded to the longest
  and excluded by a key mask; ``skip_layers`` per sample, so cond and the
  skip-22 uncond share a batch.
- ``gradient_checkpointing``: per-block activation checkpointing in
  ``run_layers`` while training.
- ``pack_latents`` / ``unpack_latents``: the FLUX.2 VAE ↔ DiT latent boundary
  (2×2 patchify + the VAE's BatchNorm statistics).
"""

from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# The reference sampler's defaults (``MOSSImagePipeline.__call__``).
DEFAULT_STEPS = 50
DEFAULT_CFG = 3.0
DEFAULT_UNCOND_SKIP = (22,)
DEFAULT_SHIFT = 5.0


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _validate_skip_layers(layers, depth, name="skip_layers"):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, key_mask=None):
    """``key_mask``: bool ``(B, L)``, True = attend. None = the reference path."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    attn_mask = None if key_mask is None else key_mask[:, None, None, :]
    return (
        F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=attn_mask)
        .transpose(1, 2)
        .flatten(2)
    )


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, key_mask=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, key_mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, key_mask=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, key_mask)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, key_mask=None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        attn = _attention(q, k, self.proj["v"](y), rope, self.heads, key_mask)
        x = x + self.proj["o"](attn) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = (
            torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        )
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        """``(1, H·W + text_length, head_dim)``, fp32. Image tokens first."""
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        image = (
            torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1)
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    """MOSS-Image DiT with training-compatible state-dict names and SDPA attention.

    The sequence is ``[image (H·W) | text (S)]`` through every block; only the
    image tokens reach the head.
    """

    @register_to_config
    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, and even freq_dim."
            )
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False

    def load_training_state_dict(self, state_dict, *, prefix=""):
        """Load a consolidated raw/EMA model mapping, strictly; no silent key drops."""
        if prefix:
            if any(not key.startswith(prefix) for key in state_dict):
                raise ValueError("All keys must start with the explicit prefix.")
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
        return self.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------ pieces

    def time_conditioning(self, timestep, batch):
        """``timestep`` is ``1000·σ`` (scalar or ``(B,)``) → ``mod (B,6,D)``, ``head (B,2,D)``."""
        times = torch.as_tensor(
            timestep, device=self.time_embedding[0].weight.device, dtype=torch.float32
        ).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype))
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype))
        return mod.view(batch, 6, -1), head.view(batch, 2, -1)

    def embed_sample(self, latent, context):
        """One sample: ``latent (1,C,H,W)``, ``context (1,S,text_dim)`` at its true
        length → ``x (1, H·W+S, D)``, ``rope (1, H·W+S, head_dim)``."""
        _, _, height, width = latent.shape
        video = latent.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        # Match the reference activation dtype under fp32 weights + autocast.
        rope = self.rope(height, width, context.shape[1], latent.device).to(video.dtype)
        text = self.text_embedder(context, rope[:, height * width :])
        return torch.cat((video, text), 1), rope

    def run_layers(self, x, rope, mod, skip_layers=frozenset(), key_mask=None):
        """The 42 blocks. ``skip_layers``: one frozenset for the whole batch, or a
        sequence of frozensets, one per sample (rows skipping a block keep ``x``)."""
        per_sample = not isinstance(skip_layers, (frozenset, set))
        ckpt = self.gradient_checkpointing and torch.is_grad_enabled()
        for index, layer in enumerate(self.layers):
            if per_sample:
                active = [b for b, skips in enumerate(skip_layers) if index not in skips]
                if not active:
                    continue
                if len(active) < x.shape[0]:
                    rows = torch.tensor(active, device=x.device)
                    sub = layer(
                        x[rows], rope[rows], mod[rows], None if key_mask is None else key_mask[rows]
                    )
                    x = x.index_copy(0, rows, sub)
                    continue
            elif index in skip_layers:
                continue
            if ckpt:
                x = checkpoint(layer, x, rope, mod, key_mask, use_reentrant=False)
            else:
                x = layer(x, rope, mod, key_mask)
        return x

    def head(self, features, head_mod, height, width):
        """Image-token hidden state after the last block ``(B, H·W, D)`` →
        velocity ``(B, C, H, W)``. Spectrum forecasts ``features`` and calls this."""
        x = self.final_norm_video(features.to(self.final_norm_video.weight.dtype))
        x = x * (1 + head_mod[:, 1, None]) + head_mod[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)

    # ---------------------------------------------------------------- forwards

    def _check_inputs(self, hidden_states, encoder_hidden_states, mask):
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch = hidden_states.shape[0]
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        if mask is None:
            mask = torch.ones(
                encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        if not torch.all(mask.bool().any(1)):
            raise ValueError("Each sample needs at least one valid text token.")
        return mask.bool()

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
        return_features=False,
    ):
        """The reference per-sample loop. ``return_features`` returns the
        image-token hidden state ``(B, H·W, D)`` before ``head`` instead."""
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        mask = self._check_inputs(hidden_states, encoder_hidden_states, encoder_attention_mask)
        batch, _, height, width = hidden_states.shape
        mod, head = self.time_conditioning(timestep, batch)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index]]
            x, rope = self.embed_sample(hidden_states[index : index + 1], context)
            x = self.run_layers(x, rope, mod[index : index + 1], skip_layers)
            features = x[:, : height * width]
            if return_features:
                outputs.append(features)
            else:
                outputs.append(self.head(features, head[index : index + 1], height, width))
        sample = torch.cat(outputs)
        if return_features:
            return sample
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)

    def forward_batched(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        skip_layers=None,
        return_features=False,
    ):
        """All samples in one forward. Text is right-padded to the longest valid
        length and masked out of every key set (text refiner and joint blocks), so
        padding never reaches a valid token. Not bit-exact with ``forward`` (the
        masked SDPA picks a different kernel); within bf16 noise.

        ``skip_layers``: None, one iterable for the whole batch, or a list with one
        iterable per sample (e.g. ``[(), (22,)]`` for cond + skip-22 uncond).
        Returns the velocity ``(B, C, H, W)`` (or features ``(B, H·W, D)``).
        """
        depth = len(self.layers)
        mask = self._check_inputs(hidden_states, encoder_hidden_states, encoder_attention_mask)
        batch, _, height, width = hidden_states.shape
        if skip_layers is not None and len(skip_layers) and not isinstance(skip_layers[0], int):
            if len(skip_layers) != batch:
                raise ValueError("Per-sample skip_layers needs one entry per sample.")
            skips = [_validate_skip_layers(s, depth) for s in skip_layers]
            if len(set(skips)) == 1:
                skips = skips[0]
        else:
            skips = _validate_skip_layers(skip_layers, depth)
        mod, head = self.time_conditioning(timestep, batch)
        n_img = height * width
        lengths = mask.sum(1)
        s_max = int(lengths.max())
        device = hidden_states.device

        context = encoder_hidden_states.new_zeros(batch, s_max, encoder_hidden_states.shape[-1])
        text_mask = torch.zeros(batch, s_max, dtype=torch.bool, device=device)
        ropes = []
        for b in range(batch):
            n = int(lengths[b])
            context[b, :n] = encoder_hidden_states[b, mask[b]]
            text_mask[b, :n] = True
            rope = self.rope(height, width, n, device)
            if n < s_max:
                rope = F.pad(rope, (0, 0, 0, s_max - n))
            ropes.append(rope)
        video = hidden_states.flatten(2).transpose(1, 2)
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
        rope = torch.cat(ropes).to(video.dtype)
        text = self.text_embedder(context, rope[:, n_img:], text_mask)
        x = torch.cat((video, text), 1)
        key_mask = torch.cat((text_mask.new_ones(batch, n_img), text_mask), 1)
        if bool(text_mask.all()):
            key_mask = None
        x = self.run_layers(x, rope, mod, skips, key_mask)
        features = x[:, :n_img]
        if return_features:
            return features
        return self.head(features, head, height, width)


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration using the training project's V2, nearest-grid schedule."""

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if num_inference_steps < 1 or self.config.shift <= 0 or self.config.training_shift <= 0:
            raise ValueError("Steps and shifts must be positive.")
        sigmas = snapped_sigmas(
            num_inference_steps,
            self.config.shift,
            self.config.training_shift,
            self.config.num_train_timesteps,
        )
        # Our sigmas have already been shifted and snapped. Avoid a second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


def snapped_sigmas(num_steps, shift=DEFAULT_SHIFT, training_shift=1.0, total=1000):
    """The inference σ grid (descending, no terminal 0): ``shift``-warped linspace
    snapped to the nearest σ of the training grid, duplicates collapsed — so a
    high step count can return fewer than ``num_steps`` values."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper
    )
    return torch.unique_consecutive(training[nearest])


# ------------------------------------------------------------- VAE boundary


def _bn_stats(vae, like):
    bn, eps = vae.bn, vae.config.batch_norm_eps
    mean = bn.running_mean.to(like).view(1, -1, 1, 1)
    std = (bn.running_var + eps).sqrt().to(like).view(1, -1, 1, 1)
    return mean, std


def unpack_latents(z, vae):
    """DiT latent ``(B,128,h,w)`` → FLUX.2 VAE latent ``(B,32,2h,2w)``, in the VAE's
    dtype: BN de-normalise, then 2×2 un-patchify (the reference decode)."""
    z = z.to(dtype=vae.dtype)
    mean, std = _bn_stats(vae, z)
    z = z * std + mean
    b, c, h, w = z.shape
    return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def pack_latents(z, vae):
    """FLUX.2 VAE latent ``(B,32,H,W)`` → DiT latent ``(B,128,H/2,W/2)``: the exact
    inverse of ``unpack_latents`` (FLUX.2's ``_patchify_latents`` + BN normalise)."""
    b, c, h, w = z.shape
    z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean, std = _bn_stats(vae, z)
    return (z - mean) / std


@torch.no_grad()
def decode(z, vae):
    """DiT latent → image tensor in ``[-1, 1]``, ``(B,3,16h,16w)``."""
    return vae.decode(unpack_latents(z, vae), return_dict=False)[0].clamp(-1, 1)


@torch.no_grad()
def encode(pixels, vae):
    """Image tensor in ``[-1, 1]`` ``(B,3,H,W)``, sides multiples of 16 → DiT latent
    ``(B,128,H/16,W/16)``, fp32. Distribution mode, as FLUX.2 encodes references."""
    dist = vae.encode(pixels.to(device=vae.device, dtype=vae.dtype)).latent_dist
    return pack_latents(dist.mode(), vae).float()
