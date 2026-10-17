"""Allowlisted JSON RPC for a separately registered curation interpreter.

The WebUI never imports anime-tools into a protected training environment.
Only fixed service operations are accepted; stdin contains data, never code.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

OPERATIONS = frozenset({
    "get_settings", "get_roots", "save_roots", "save_settings", "get_schema",
    "dataset", "item", "caption", "parse", "compose", "exclude", "groups",
    "file_path", "thumbnail", "prepare", "report", "proposals", "proposal",
    "prepare_replay", "remember_values",
})


def in_curation_runtime(function):
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        from library.runtime.profiles import resolve_runtime
        runtime = resolve_runtime("curation")
        if Path(os.path.abspath(runtime.python)) == Path(os.path.abspath(sys.executable)):
            return function(*args, **kwargs)
        from scripts.curation.context import CurationUnavailable
        if function.__name__ not in OPERATIONS:
            raise ValueError("Unknown curation RPC operation")
        root = Path(__file__).resolve().parents[2]
        payload = json.dumps({"operation": function.__name__, "args": args, "kwargs": kwargs})
        try:
            response = subprocess.run(
                [str(runtime.python), "-m", "scripts.curation.bridge"],
                cwd=root, env={**os.environ, **runtime.env}, input=payload,
                text=True, encoding="utf-8", capture_output=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CurationUnavailable(f"Curation runtime could not answer: {exc}") from exc
        try:
            result = json.loads(response.stdout)
        except ValueError as exc:
            raise CurationUnavailable(f"Curation runtime failed: {response.stderr[-1500:]}") from exc
        if not isinstance(result, dict) or not ({"value", "error"} & result.keys()):
            raise CurationUnavailable("Curation runtime returned an invalid RPC response")
        if "error" in result:
            error = result["error"]
            cls = {"FileNotFoundError": FileNotFoundError, "CurationUnavailable": CurationUnavailable}.get(error["type"], ValueError)
            raise cls(error["message"])
        if response.returncode:
            raise CurationUnavailable(f"Curation runtime exited with status {response.returncode}")
        value = result["value"]
        if function.__name__ == "file_path":
            return Path(value)
        if function.__name__ == "thumbnail":
            import base64
            return base64.b64decode(value)
        return value
    return invoke


def main() -> int:
    import contextlib
    from webui.services import curation_service
    try:
        request = json.load(sys.stdin)
        operation = request.get("operation")
        if operation not in OPERATIONS:
            raise ValueError("Unknown curation RPC operation")
        function = getattr(curation_service, operation)
        # The selected child is already the registered runtime. Calling the
        # underlying function also avoids recursively resolving runtime config.
        with contextlib.redirect_stdout(sys.stderr):
            value = function.__wrapped__(*request.get("args", []), **request.get("kwargs", {}))
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, bytes):
            import base64
            value = base64.b64encode(value).decode("ascii")
        print(json.dumps({"value": value}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
