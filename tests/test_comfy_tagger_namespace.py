"""Regression coverage for ComfyUI sibling-node namespace contamination."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAGGER_NODE = ROOT / "custom_nodes" / "comfyui-anima-tagger" / "nodes.py"


def test_tagger_node_evicts_incomplete_sibling_vendor(tmp_path):
    """The live repo must win after another node cached a partial ``library``.

    Run in a child interpreter because the fix intentionally replaces global
    ``library``/``networks`` entries in ``sys.modules``.
    """
    sibling_vendor = tmp_path / "sibling_vendor"
    runtime = sibling_vendor / "library" / "runtime"
    runtime.mkdir(parents=True)
    (sibling_vendor / "library" / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "fei.py").write_text("SIBLING = True\n", encoding="utf-8")

    code = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        import types
        from pathlib import Path

        root = Path({json.dumps(str(ROOT))})
        sibling = Path({json.dumps(str(sibling_vendor))})
        node_path = Path({json.dumps(str(TAGGER_NODE))})

        sys.path.insert(0, str(sibling))
        import library
        import library.runtime
        assert Path(library.__file__).resolve().is_relative_to(sibling.resolve())

        comfy = types.ModuleType("comfy")
        model_management = types.ModuleType("comfy.model_management")
        model_management.get_torch_device = lambda: "cpu"
        comfy.model_management = model_management
        sys.modules["comfy"] = comfy
        sys.modules["comfy.model_management"] = model_management

        spec = importlib.util.spec_from_file_location("anima_tagger_node_test", node_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        resolved_library = Path(sys.modules["library"].__file__).resolve()
        assert resolved_library.is_relative_to(root.resolve()), resolved_library
        assert module.AnimaTagger.__module__ == "library.captioning.anima_tagger"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
