"""Regression guards for the frontend dependency trust boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "webui" / "frontend"


def test_frontend_lock_has_no_keyv_and_pinned_integrity():
    lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))
    packages = lock["packages"]
    for name, info in packages.items():
        if not name:
            continue
        assert "keyv" not in name.lower()
        assert str(info.get("resolved", "")).startswith("https://")
        assert str(info.get("integrity", "")).startswith("sha512-")


def test_frontend_lock_checker_passes_without_installing():
    checker = ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs"
    result = subprocess.run(
        ["node", str(checker)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "lock verified" in result.stdout


def test_build_entrypoints_use_clean_ignore_scripts_install():
    linux = (ROOT / "setup-linux.sh").read_text(encoding="utf-8")
    windows = (ROOT / "build-webui-win.bat").read_text(encoding="utf-8")
    # The shared checker owns the actual npm ci invocation; these entrypoints
    # must request a clean repair instead of trusting node_modules existence.
    assert "ensure_npm_deps.mjs" in linux and "--clean" in linux
    assert "ensure_npm_deps.mjs" in windows and "--clean" in windows
    checker = (ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs").read_text(
        encoding="utf-8"
    )
    assert '"ci"' in checker
    assert '"--ignore-scripts"' in checker


def test_lock_checker_enforces_keyv_alias_and_native_probe_boundaries():
    checker = (ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs").read_text(
        encoding="utf-8"
    )
    assert "lock.lockfileVersion !== 3" in checker
    assert "allowedLifecyclePackages" in checker
    assert "npm:" in checker and "workspace:" in checker
    assert "@keyv" in checker
    assert "transformSync" in checker
    assert "r.rollup" in checker
    assert "--offline" in checker


def test_lock_checker_resolves_windows_npm_without_command_shim():
    checker = (ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs").read_text(
        encoding="utf-8"
    )
    assert "process.execPath" in checker
    assert "node_modules/npm/bin/npm-cli.js" in checker
    assert "shell: true" in checker
