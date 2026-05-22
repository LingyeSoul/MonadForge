"""Local training-job daemon (Phase 1 of the daemon plan).

A single localhost process: a FIFO serial job queue + worker thread that spawns
``accelerate launch … train.py`` subprocesses (detached, so a console ctrl-C
can't reach them) and follows each run by tailing its Phase-0
``progress.jsonl``. Exposes a small stdlib HTTP API (no framework, no auth,
``127.0.0.1`` only) consumed by the CLI (``make daemon*``), the ComfyUI trainer
node, and (Phase 3) an MCP server.

    python -m scripts.daemon [port]     # run the daemon (normally detached)

Public surface lives in submodules: ``client.DaemonClient`` / ``ensure_daemon``
for callers, ``manager.JobManager`` + ``server.serve`` for the process itself.
"""
