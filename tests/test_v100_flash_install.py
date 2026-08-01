from __future__ import annotations

from pathlib import Path

import pytest

from bench.v100_flash.replay_capture import DEFAULT_FULL_REPEATS
from bench.v100_flash.replay_capture import _acceptance as capture_acceptance
from scripts.v100_flash.install import (
    build_parser as build_install_parser,
    build_environment,
    executable_path,
    validate_wheel_name,
)
from scripts.v100_flash.validate import (
    FLASH_PRESET,
    SDPA_PRESET,
    _remove_presets,
    _write_presets,
    build_parser as build_validate_parser,
    compare_aligned_benchmarks,
)


def test_v100_build_environment_is_fixed_and_clears_extra_mma(tmp_path: Path):
    cuda_home = tmp_path / "cuda"
    gcc = Path("/usr/bin/gcc-14")
    gxx = Path("/usr/bin/g++-14")
    env = build_environment(
        {
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/usr/lib",
            "MMA_NATIVE": "1",
            "MMA_884": "1",
            "ATTENTION_DEBUG": "1",
        },
        cuda_home=cuda_home,
        gcc=gcc,
        gxx=gxx,
    )

    assert env["MAX_JOBS"] == "2"
    assert env["NVCC_THREADS"] == "2"
    assert env["CC"] == str(gcc)
    assert env["CXX"] == str(gxx)
    assert env["CUDAHOSTCXX"] == str(gxx)
    assert env["CUDA_HOME"] == str(cuda_home)
    assert env["PATH"].startswith(f"{cuda_home / 'bin'}:")
    assert env["LD_LIBRARY_PATH"].startswith(f"{cuda_home / 'lib64'}:")
    assert "MMA_NATIVE" not in env
    assert "MMA_884" not in env
    assert "ATTENTION_DEBUG" not in env


def test_v100_installer_preserves_virtualenv_python_symlink(tmp_path: Path):
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    assert executable_path(venv_python) == venv_python.absolute()
    assert executable_path(venv_python) != venv_python.resolve()


def test_v100_wheel_must_be_cp313_linux():
    validate_wheel_name(Path("flash_attn_v100-26.6-cp313-cp313-linux_x86_64.whl"))


def test_v100_wheel_rejects_old_cp312():
    try:
        validate_wheel_name(Path("flash_attn_v100-26.6-cp312-cp312-linux_x86_64.whl"))
    except RuntimeError as exc:
        assert "cp313" in str(exc)
    else:
        raise AssertionError("cp312 wheel was accepted")


def test_strict_capture_replay_uses_ten_repeats():
    assert DEFAULT_FULL_REPEATS == 10


def test_v100_presets_only_change_attention_backend():
    assert 'attn_mode = "flash"' in FLASH_PRESET
    assert 'attn_mode = "torch"' in SDPA_PRESET
    assert (
        FLASH_PRESET.replace('attn_mode = "flash"', 'attn_mode = "torch"')
        == SDPA_PRESET
    )


def test_v100_presets_are_revoked_before_revalidation(tmp_path: Path):
    written = _write_presets(tmp_path)

    assert all(Path(path).is_file() for path in written)
    assert sorted(_remove_presets(tmp_path)) == sorted(written)
    assert not any(Path(path).exists() for path in written)


def test_v100_presets_preserve_user_owned_collisions(tmp_path: Path):
    flash = tmp_path / "configs/custom/presets/V100_flash.toml"
    sdpa = flash.with_name("V100_sdpa.toml")
    flash.parent.mkdir(parents=True)
    flash.write_text('attn_mode = "flash"\n# user tuning\n', encoding="utf-8")
    sdpa.write_text('attn_mode = "torch"\n# user tuning\n', encoding="utf-8")

    assert _remove_presets(tmp_path) == []
    with pytest.raises(RuntimeError, match="user-owned"):
        _write_presets(tmp_path)

    assert "user tuning" in flash.read_text(encoding="utf-8")
    assert "user tuning" in sdpa.read_text(encoding="utf-8")


def test_v100_cli_requires_machine_local_inputs():
    with pytest.raises(SystemExit):
        build_install_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_validate_parser().parse_args([])

    install = build_install_parser().parse_args(["--cuda-home", "/opt/cuda-12.9"])
    validate = build_validate_parser().parse_args(
        [
            "--capture",
            "/data/first_failure.pt",
            "--dit",
            "/models/anima.safetensors",
            "--performance-baseline",
            "/data/tail-matrix.json",
        ]
    )
    assert install.cuda_home == Path("/opt/cuda-12.9")
    assert validate.capture == Path("/data/first_failure.pt")


def test_aligned_regression_requires_both_lengths():
    def report(first: float, second: float):
        return {
            "aligned_benchmarks": [
                {
                    "length": 4112,
                    "forward_median_ms": first,
                    "backward_median_ms": first,
                    "peak_allocated_mib": first,
                    "peak_reserved_mib": first,
                },
                {
                    "length": 4128,
                    "forward_median_ms": second,
                    "backward_median_ms": second,
                    "peak_allocated_mib": second,
                    "peak_reserved_mib": second,
                },
            ]
        }

    baseline = report(100.0, 100.0)
    one_slow = compare_aligned_benchmarks(report(106.0, 104.0), baseline)
    both_slow = compare_aligned_benchmarks(report(106.0, 106.0), baseline)

    assert one_slow["accepted"] is True
    assert both_slow["accepted"] is False
    assert both_slow["material_regressions"]["forward_median_ms"] is True


def test_fixed_capture_requires_all_prefix_paths_but_separates_compile_failure():
    finite = {"finite": True}
    replays = {
        "torch_sdpa_fp16": {"stats": finite},
        "raw_flash_eager": {"ok": True, "stats": finite},
        "compat_flash_eager": {"ok": True, "stats": finite},
        "compat_flash_compiled": {"ok": True, "stats": finite},
    }
    prefixes = [
        {
            "length": length,
            "sdpa": finite,
            "paths": {
                "raw_flash_eager": {"stats": finite},
                "compat_flash_eager": {"stats": finite},
                "compat_flash_compiled": {"stats": finite},
            },
        }
        for length in range(4112, 4129)
    ]

    accepted, numeric, integration, numeric_failures, integration_failures = (
        capture_acceptance("fixed", replays, prefixes)
    )
    assert accepted and numeric and integration
    assert not numeric_failures and not integration_failures

    prefixes[3]["paths"]["compat_flash_compiled"] = {
        "ok": False,
        "error": "graph break",
    }
    accepted, numeric, integration, numeric_failures, integration_failures = (
        capture_acceptance("fixed", replays, prefixes)
    )
    assert accepted is False
    assert numeric is True
    assert integration is False
    assert not numeric_failures
    assert "4115" in integration_failures[0]
