from __future__ import annotations

import json
import locale
import os
import subprocess
import sys

import pytest


def test_select_bitsandbytes_cuda_fallback_uses_latest_same_major():
    from library.runtime.compat import select_bitsandbytes_cuda_fallback

    assert (
        select_bitsandbytes_cuda_fallback("2.12.0+cu132", {"118", "128", "129", "130"})
        == "130"
    )


def test_select_bitsandbytes_cuda_fallback_skips_exact_and_cross_major():
    from library.runtime.compat import select_bitsandbytes_cuda_fallback

    assert select_bitsandbytes_cuda_fallback("2.12.0+cu132", {"130", "132"}) is None
    assert select_bitsandbytes_cuda_fallback("2.12.0+cu132", {"129", "140"}) is None


def test_prepare_python_child_env_uses_locale_for_nested_windows_processes(
    monkeypatch,
):
    from library.runtime import compat

    monkeypatch.setattr(compat, "installed_bitsandbytes_cuda_fallback", lambda: "130")
    env = {"PYTHONUTF8": "1"}

    compat.prepare_python_child_env(env, platform="win32")

    assert env["PYTHONUTF8"] == "0"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["BNB_CUDA_VERSION"] == "130"


def test_prepare_python_child_env_respects_explicit_bitsandbytes_override(
    monkeypatch,
):
    from library.runtime import compat

    monkeypatch.setattr(compat, "installed_bitsandbytes_cuda_fallback", lambda: "130")
    env = {"BNB_CUDA_VERSION": "128"}

    compat.prepare_python_child_env(env, platform="win32")

    assert env["BNB_CUDA_VERSION"] == "128"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows locale behavior")
def test_prepared_windows_child_has_utf8_stdio_and_locale_subprocess_decoder():
    from library.runtime.compat import prepare_python_child_env

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    prepare_python_child_env(env)
    code = (
        "import json,locale,sys; "
        "print(json.dumps({'utf8_mode':sys.flags.utf8_mode,"
        "'stdout':sys.stdout.encoding,'locale':locale.getencoding()}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    child = json.loads(result.stdout)

    assert child["utf8_mode"] == 0
    assert child["stdout"].lower().replace("-", "") == "utf8"
    assert child["locale"].lower() == locale.getencoding().lower()
