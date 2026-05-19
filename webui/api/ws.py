"""WebSocket endpoint skeleton for real-time task logs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_log_ws(websocket: WebSocket, task_id: str):
    """Stream task output lines over WebSocket.

    Phase 2 will wire this to the task_service subprocess manager.
    For now, it accepts the connection and sends a placeholder.
    """
    await websocket.accept()
    try:
        await websocket.send_json({"type": "connected", "task_id": task_id})
        # Placeholder — Phase 2 will loop over subprocess stdout lines
        await websocket.send_json(
            {
                "type": "log",
                "line": f"[Phase 2] Task {task_id} log streaming not yet implemented.",
            }
        )
        await websocket.send_json({"type": "done", "exit_code": 0})
    except Exception:
        logger.exception("WebSocket error for task %s", task_id)
    finally:
        await websocket.close()
