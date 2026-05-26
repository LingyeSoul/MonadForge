"""Tests for ``GenerationRequest`` — the typed front door for ``generate()``.

Covers:

* the always-emitted core defaults match ``inference.parse_args`` (drift guard:
  the dataclass mirrors the parser in two places, this pins them together)
* scalar / path / LoRA fields thread through to the namespace
* store_true flags emit only when ``True``
* ``extra_argv`` is appended and can override a structured field
* required-arg placeholders let a prompt-only request build a namespace, while a
  prompt-less request still fails the parser's validation
* the export is reachable from both ``library.inference`` and ``anima_lora``
"""

from __future__ import annotations

import inference
import pytest

from library.inference.generation import resolve_seed
from library.inference.request import GenerationRequest

# Minimal argv covering the parser's two required args + a prompt, used as the
# "bare parser defaults" baseline.
_BASELINE_ARGV = ["--text_encoder", "te", "--save_path", "out.png", "--prompt", "p"]

# Fields GenerationRequest always emits with a parser-mirroring default.
_CORE_DEFAULT_FIELDS = [
    "negative_prompt",
    "image_size",
    "infer_steps",
    "guidance_scale",
    "flow_shift",
    "sampler",
    "attn_mode",
    "output_type",
]


def test_core_defaults_match_parser():
    """A default request must reproduce the parser's defaults verbatim."""
    baseline = inference.parse_args(_BASELINE_ARGV)
    args = GenerationRequest(prompt="p").to_args()
    for fld in _CORE_DEFAULT_FIELDS:
        assert getattr(args, fld) == getattr(baseline, fld), f"drift on {fld!r}"


def test_scalar_and_path_fields_thread_through():
    args = GenerationRequest(
        prompt="a fox",
        negative_prompt="blurry",
        image_size=(832, 1248),
        infer_steps=30,
        guidance_scale=4.0,
        flow_shift=2.5,
        sampler="er_sde",
        seed=123,
        dit="d.safetensors",
        vae="v.safetensors",
        text_encoder="te.safetensors",
        save_path="out/x.png",
        vae_chunk_size=64,
        pooled_text_proj="mod.safetensors",
    ).to_args()

    assert args.prompt == "a fox"
    assert args.negative_prompt == "blurry"
    assert args.image_size == [832, 1248]  # parser yields a list
    assert args.infer_steps == 30
    assert args.guidance_scale == 4.0
    assert args.flow_shift == 2.5
    assert args.sampler == "er_sde"
    assert args.seed == 123
    assert args.dit == "d.safetensors"
    assert args.vae == "v.safetensors"
    assert args.text_encoder == "te.safetensors"
    assert args.save_path == "out/x.png"
    assert args.vae_chunk_size == 64
    assert args.pooled_text_proj == "mod.safetensors"


def test_lora_list_and_multiplier_list():
    args = GenerationRequest(
        prompt="p",
        lora_weight=["a.safetensors", "b.safetensors"],
        lora_multiplier=[0.8, 0.5],
    ).to_args()
    assert args.lora_weight == ["a.safetensors", "b.safetensors"]
    assert args.lora_multiplier == [0.8, 0.5]


def test_lora_multiplier_scalar():
    """A bare float multiplier emits a single nargs token."""
    args = GenerationRequest(
        prompt="p", lora_weight=["a.safetensors"], lora_multiplier=0.7
    ).to_args()
    assert args.lora_multiplier == [0.7]


def test_unset_lora_falls_through_to_parser_default():
    args = GenerationRequest(prompt="p").to_args()
    assert args.lora_weight is None
    assert args.lora_multiplier == 1.0  # parser default (scalar)


def test_store_true_flags():
    on = GenerationRequest(
        prompt="p",
        no_metadata=True,
        vae_disable_cache=True,
        text_encoder_cpu=True,
    ).to_args()
    assert on.no_metadata and on.vae_disable_cache and on.text_encoder_cpu

    off = GenerationRequest(prompt="p").to_args()
    assert not off.no_metadata
    assert not off.vae_disable_cache
    assert not off.text_encoder_cpu


def test_extra_argv_reaches_unmodelled_knobs():
    """The long tail (spectrum/dcw/…) rides through extra_argv verbatim."""
    args = GenerationRequest(
        prompt="p",
        extra_argv=["--spectrum", "--dcw", "--dcw_lambda", "0.01"],
    ).to_args()
    assert args.spectrum is True
    assert args.dcw is True
    assert args.dcw_lambda == 0.01


def test_extra_argv_overrides_structured_field():
    """extra_argv is appended last, so it wins on conflict."""
    args = GenerationRequest(
        prompt="p", infer_steps=30, extra_argv=["--infer_steps", "12"]
    ).to_args()
    assert args.infer_steps == 12


def test_required_arg_placeholders_let_request_build():
    """A prompt-only request (no text_encoder/save_path) still builds a namespace."""
    args = GenerationRequest(prompt="p").to_args()
    # placeholders, not None — the parser's required= is satisfied.
    assert args.text_encoder and args.save_path


def test_missing_prompt_fails_parser_validation():
    """No prompt/from_file/interactive → the parser raises, same as the CLI."""
    with pytest.raises(ValueError):
        GenerationRequest().to_args()


def test_to_args_accepts_injected_parser():
    seen: dict = {}

    def fake_parse(argv):
        seen["argv"] = argv
        return "sentinel"

    req = GenerationRequest(prompt="p", seed=7)
    assert req.to_args(parse_args=fake_parse) == "sentinel"
    assert "--prompt" in seen["argv"] and "p" in seen["argv"]
    assert "--seed" in seen["argv"] and "7" in seen["argv"]


def test_frozen_request_is_reusable_across_seeds():
    """Each to_args() builds an independent namespace, and the frozen request is
    never written back — so resolving a seed on one namespace can't leak to
    another (or to the request)."""
    req = GenerationRequest(prompt="p")  # seed unset
    a, b = req.to_args(), req.to_args()
    a.seed = resolve_seed(a)  # caller-side seed resolution (generate() reads it)
    assert isinstance(a.seed, int)  # a got a concrete seed
    assert b.seed is None  # the other namespace is untouched
    assert req.seed is None  # the request never changed


def test_exported_from_packages():
    from anima_lora import GenerationRequest as FromTop
    from library.inference import GenerationRequest as FromLib

    assert FromTop is GenerationRequest
    assert FromLib is GenerationRequest
