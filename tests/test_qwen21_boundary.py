"""Keep Qwen isolated while shared request/adapter helpers remain reusable."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN21 = ROOT / "library/qwen21"
QWEN_WEBUI_MODULES = (
    "webui.api.qwen21",
    "webui.services.qwen21_service",
    "webui.services.qwen21_workspace",
)
_QWEN21_HOMES = (
    "library/qwen21/",
    "scripts/qwen21/",
    "gui/qwen21/",
    "project/qwen21_lora/",
    "tests/",
)
_WEBUI_BOUNDARY = {module.replace(".", "/") + ".py" for module in QWEN_WEBUI_MODULES}
_SHARED = (
    "library.env",
    "library.lines",
    "library.runtime.offloading",
    "library.runtime.blockswap",
    "library.runtime.device",
    "library.runtime.dynamo",
)
_SCANNED = (
    "anima_lora",
    "library",
    "networks",
    "scripts",
    "gui",
    "bench",
    "webui/api",
    "webui/services",
    "train.py",
)


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _python_files():
    for entry in _SCANNED:
        path = ROOT / entry
        yield from [path] if path.is_file() else path.rglob("*.py")


def test_anima_side_never_imports_qwen21():
    offenders = []
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(_QWEN21_HOMES) or rel in _WEBUI_BOUNDARY:
            continue
        if any(n.startswith("library.qwen21") for n in _imports(path)):
            offenders.append(rel)
    assert not offenders, f"Anima code imports library.qwen21: {offenders}"


def test_qwen21_uses_only_shared_library_helpers():
    offenders = []
    for path in QWEN21.rglob("*.py"):
        for name in _imports(path):
            if not name.startswith("library.") or name.startswith("library.qwen21"):
                continue
            if not any(name == root or name.startswith(root + ".") for root in _SHARED):
                offenders.append(f"{path.name}: {name}")
    assert not offenders, f"library.qwen21 reaches into Anima: {offenders}"


def test_qwen_configuration_and_web_forms_do_not_import_torch():
    modules = ", ".join((*QWEN_WEBUI_MODULES, "library.qwen21.requests"))
    code = f"import sys; import {modules}; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_webui_is_torch_free():
    code = "import sys, webui.server; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    assert result.returncode == 0, "webui.server imports torch"
