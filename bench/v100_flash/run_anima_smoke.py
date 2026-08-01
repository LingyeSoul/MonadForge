"""Run the issue-43 real-weight Anima FP16 LoRA training smoke test."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from bench.v100_flash._validation import (
    environment_manifest,
    require_v100,
    resolve_device,
    sha256_file,
    tensor_stats,
    write_json,
)

DEFAULT_SHAPES = "58x68,90x130,118x140"
EXPECTED_TOKENS = (986, 2925, 4130)
EXPECTED_GRADIENT_TENSORS = 392


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        height, separator, width = item.strip().lower().partition("x")
        if not separator:
            raise argparse.ArgumentTypeError(f"invalid latent shape {item!r}; use HxW")
        shape = (int(height), int(width))
        if shape[0] <= 0 or shape[1] <= 0 or shape[0] % 2 or shape[1] % 2:
            raise argparse.ArgumentTypeError(
                f"latent shape {item!r} must contain positive even dimensions"
            )
        shapes.append(shape)
    if not shapes:
        raise argparse.ArgumentTypeError("at least one latent shape is required")
    return shapes


def _dtype_counts(parameters) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in parameters:
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def _finite_parameter_state(named_parameters) -> tuple[bool, list[str]]:
    checks: list[tuple[str, torch.Tensor]] = []
    for name, parameter in named_parameters:
        if parameter.requires_grad:
            checks.append((name, torch.isfinite(parameter.detach()).all()))
    if not checks:
        return False, ["no trainable parameters"]
    flags = torch.stack([check for _name, check in checks])
    if bool(flags.all().item()):
        return True, []
    bad = [name for (name, _check), finite in zip(checks, flags.tolist()) if not finite]
    return False, bad


def _finite_gradient_state(named_parameters) -> dict[str, Any]:
    present: list[tuple[str, torch.Tensor]] = []
    missing: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        else:
            present.append((name, torch.isfinite(parameter.grad.detach()).all()))

    bad: list[str] = []
    if present:
        flags = torch.stack([check for _name, check in present])
        if not bool(flags.all().item()):
            bad = [
                name
                for (name, _check), finite in zip(present, flags.tolist())
                if not finite
            ]
    return {
        "trainable_count": len(present) + len(missing),
        "present_count": len(present),
        "missing_count": len(missing),
        "missing_names": missing,
        "nonfinite_count": len(bad),
        "nonfinite_names": bad,
        "finite": bool(present) and not missing and not bad,
    }


def _load_crossattn(path: Path, device: torch.device) -> torch.Tensor:
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    crossattn = state.get("crossattn_emb")
    if crossattn is None:
        raise KeyError(f"{path} does not contain crossattn_emb")
    if crossattn.ndim != 2 or crossattn.shape[-1] != 1024:
        raise ValueError(f"unexpected crossattn_emb shape: {tuple(crossattn.shape)}")
    return crossattn.unsqueeze(0).to(device=device, dtype=torch.float16).contiguous()


def _acceptance(
    expectation: str,
    rows: list[dict[str, Any]],
    *,
    steps: int,
    required_tokens: set[int],
) -> tuple[bool | None, list[str]]:
    if expectation == "none":
        return None, []
    failures: list[str] = []
    if expectation == "known-bad":
        reproduced = any(row.get("nonfinite_failure") for row in rows)
        if not reproduced:
            failures.append(
                "published wheel did not reproduce a non-finite Anima failure"
            )
        return not failures, failures

    if len(rows) != steps:
        failures.append(f"ran {len(rows)} steps, expected {steps}")
    failed_rows = [row for row in rows if not row.get("passed")]
    if failed_rows:
        failures.append(f"{len(failed_rows)} Anima optimizer steps failed")
    covered = {int(row["tokens"]) for row in rows if row.get("passed")}
    missing = sorted(required_tokens - covered)
    if missing:
        failures.append(
            f"native token lengths not covered by a passing step: {missing}"
        )
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--crossattn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--wheel-sha")
    parser.add_argument("--dit-sha")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shapes", type=_parse_shapes, default=_parse_shapes(DEFAULT_SHAPES)
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--attn-mode", choices=("flash", "torch"), default="flash")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-dynamic-seq",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamo-backend", choices=("inductor", "eager"), default="inductor"
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )
    parser.add_argument("--blocks-to-swap", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--debug-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--expect", choices=("none", "known-bad", "fixed"), default="none"
    )
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.rank < 1:
        parser.error("--rank must be positive")
    if args.blocks_to_swap < 0 or args.blocks_to_swap > 26:
        parser.error("--blocks-to-swap must be between 0 and 26")
    if args.expect == "fixed" and args.steps < len(args.shapes):
        parser.error("fixed-candidate steps must cover every requested shape")

    device = resolve_device(args.device)
    require_v100(device)
    torch.cuda.set_device(device)
    dit_path = args.dit.resolve()
    crossattn_path = args.crossattn.resolve()
    if not dit_path.is_file():
        raise SystemExit(f"DiT checkpoint not found: {dit_path}")
    if not crossattn_path.is_file():
        raise SystemExit(f"cross-attention sidecar not found: {crossattn_path}")

    shape_tokens = {shape: (shape[0] // 2) * (shape[1] // 2) for shape in args.shapes}
    if (
        args.shapes == _parse_shapes(DEFAULT_SHAPES)
        and tuple(shape_tokens.values()) != EXPECTED_TOKENS
    ):
        raise AssertionError(
            f"default native-shape token mapping changed: {shape_tokens}"
        )

    from library.anima import weights as anima_utils
    from networks.lora_anima.factory import create_network

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = anima_utils.load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=torch.float16,
        v100_flash_stability="safe" if args.attn_mode == "flash" else "off",
        debug_finite_checks=args.debug_finite,
    )
    model.to(device=device, dtype=torch.float16)
    model.reset_mod_guidance()
    model.enable_fp32_residual()
    load_seconds = time.perf_counter() - load_started

    network = create_network(
        multiplier=1.0,
        network_dim=args.rank,
        network_alpha=float(args.rank),
        vae=None,
        text_encoders=[],
        unet=model,
        neuron_dropout=0.0,
        channel_scaling_alpha=0.0,
        lora_fp32_compute="true",
        use_custom_down_autograd="true",
    )
    network.apply_to(
        text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    network.to(device=device)
    network.prepare_grad_etc(None, model)
    from library.runtime.harness import (
        compile_blocks_for_training,
        place_dit_for_training,
    )

    place_dit_for_training(model, device, blocks_to_swap=args.blocks_to_swap)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()

    if args.compile:
        token_counts = sorted(set(shape_tokens.values()))
        compile_blocks_for_training(
            model,
            network,
            backend=args.dynamo_backend,
            mode=args.compile_mode,
            n_token_families=len(token_counts),
            seq_range=(token_counts[0], token_counts[-1]),
            dynamic_seq=args.compile_dynamic_seq,
            grad_ckpt=args.gradient_checkpointing,
        )
    model.train()
    network.train()

    trainable = [
        parameter for parameter in network.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise SystemExit("LoRA network has no trainable parameters")
    if len(trainable) != EXPECTED_GRADIENT_TENSORS:
        raise SystemExit(
            f"expected {EXPECTED_GRADIENT_TENSORS} trainable LoRA tensors, "
            f"found {len(trainable)}"
        )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    crossattn = _load_crossattn(crossattn_path, device)
    if not bool(torch.isfinite(crossattn).all().item()):
        raise SystemExit("cross-attention sidecar contains non-finite values")

    rows: list[dict[str, Any]] = []
    for step in range(args.steps):
        height, width = args.shapes[step % len(args.shapes)]
        tokens = shape_tokens[(height, width)]
        torch.manual_seed(args.seed + step + 1)
        noisy = torch.randn(1, 16, 1, height, width, device=device, dtype=torch.float16)
        target = torch.randn_like(noisy)
        # Native-shape training uses an all-valid zero mask; this is the same
        # concat_padding_mask input the main trainer creates for unpadded latents.
        padding_mask = torch.zeros(
            1, 1, height, width, device=device, dtype=torch.float16
        )
        timestep = torch.tensor(
            [0.15 + 0.7 * ((step + 1) / (args.steps + 1))],
            device=device,
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)
        if args.blocks_to_swap:
            model.prepare_block_swap_before_forward()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        row: dict[str, Any] = {
            "step": step,
            "latent_shape": [height, width],
            "tokens": tokens,
            "token_residue": tokens % 16,
            "timestep": float(timestep.item()),
        }
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                output = model.forward_mini_train_dit(
                    noisy,
                    timestep,
                    crossattn,
                    padding_mask=padding_mask,
                    skip_pooled_text_proj=True,
                )
                loss = (output.float() - target.float()).square().mean()
            row["output"] = tensor_stats(output)
            row["loss"] = float(loss.detach().item())
            row["loss_finite"] = bool(torch.isfinite(loss.detach()).item())
            if not row["output"]["finite"] or not row["loss_finite"]:
                raise FloatingPointError("non-finite Anima output or loss")
            loss.backward()
            gradient_state = _finite_gradient_state(network.named_parameters())
            row["gradients"] = gradient_state
            if not gradient_state["finite"]:
                raise FloatingPointError("non-finite LoRA gradients")
            optimizer.step()
            parameters_finite, bad_parameters = _finite_parameter_state(
                network.named_parameters()
            )
            row["parameters_finite"] = parameters_finite
            row["nonfinite_parameters"] = bad_parameters
            if not parameters_finite:
                raise FloatingPointError(
                    "non-finite LoRA parameters after optimizer step"
                )
            row["passed"] = True
        except Exception as exc:  # noqa: BLE001 - preserve the first failing boundary.
            row["passed"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            row["nonfinite_failure"] = isinstance(exc, FloatingPointError) or (
                "non-finite" in str(exc).lower()
            )
        torch.cuda.synchronize(device)
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        rows.append(row)
        print(
            f"step={step} tokens={tokens} passed={row['passed']} "
            f"elapsed_ms={row['elapsed_ms']}"
        )
        del noisy, target, padding_mask
        if not row["passed"]:
            break

    required_tokens = set(shape_tokens.values())
    accepted, failures = _acceptance(
        args.expect,
        rows,
        steps=args.steps,
        required_tokens=required_tokens,
    )
    torch.cuda.synchronize(device)
    report = {
        "schema_version": 1,
        "candidate": args.candidate,
        "source_sha": args.source_sha,
        "wheel_sha": args.wheel_sha,
        "dit": str(dit_path),
        "dit_sha256": args.dit_sha,
        "crossattn": str(crossattn_path),
        "crossattn_sha256": sha256_file(crossattn_path),
        "environment": environment_manifest(device),
        "configuration": {
            "attn_mode": args.attn_mode,
            "v100_flash_stability": "safe" if args.attn_mode == "flash" else "off",
            "debug_finite_checks": args.debug_finite,
            "base_dtype": "torch.float16",
            "fp32_residual": True,
            "gradient_checkpointing": args.gradient_checkpointing,
            "torch_compile": args.compile,
            "compile_dynamic_seq": args.compile_dynamic_seq,
            "dynamo_backend": args.dynamo_backend,
            "compile_mode": args.compile_mode,
            "blocks_to_swap": args.blocks_to_swap,
            "lora_rank": args.rank,
            "lora_alpha": args.rank,
            "lora_fp32_compute": True,
            "use_custom_down_autograd": True,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "shapes": [list(shape) for shape in args.shapes],
            "shape_tokens": {
                f"{h}x{w}": tokens for (h, w), tokens in shape_tokens.items()
            },
            "steps_requested": args.steps,
            "seed": args.seed,
        },
        "model_load_seconds": round(load_seconds, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "trainable_parameter_tensors": len(trainable),
        "trainable_dtype_counts": _dtype_counts(trainable),
        "crossattn_stats": tensor_stats(crossattn),
        "steps": rows,
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 3
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / (1024**2), 3
        ),
        "expectation": args.expect,
        "accepted": accepted,
        "acceptance_failures": failures,
    }
    write_json(args.output, report)
    print(f"wrote {args.output}")
    print(f"accepted={accepted} failures={len(failures)}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if accepted is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
