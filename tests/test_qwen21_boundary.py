"""Keep Qwen isolated from Anima and keep its configuration surfaces torch-free."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_HOMES = ("library/qwen21/", "scripts/qwen21/", "tests/")
WEBUI_BOUNDARY = {"webui/api/qwen21.py", "webui/services/qwen21_service.py"}
SHARED = {
    "library.env",
    "library.runtime.device",
    "library.runtime.offloading",
    "library.runtime.dynamo",
}


def imports(path: Path) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            result.add(node.module)
    return result


def test_qwen_uses_only_shared_anima_helpers():
    offenders = []
    for path in (ROOT / "library/qwen21").glob("*.py"):
        for module in imports(path):
            if (
                module.startswith("library.")
                and not module.startswith("library.qwen21")
                and module not in SHARED
            ):
                offenders.append((str(path), module))
    assert not offenders


def test_anima_does_not_import_qwen():
    offenders = []
    for directory in (
        "anima_lora",
        "library",
        "networks",
        "scripts",
        "webui/api",
        "webui/services",
    ):
        for path in (ROOT / directory).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(QWEN_HOMES) or relative in WEBUI_BOUNDARY:
                continue
            if any(module.startswith("library.qwen21") for module in imports(path)):
                offenders.append(relative)
    assert not offenders


def test_qwen_configuration_and_web_forms_do_not_import_torch():
    code = "import sys; import library.qwen21.requests, webui.services.qwen21_service; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
