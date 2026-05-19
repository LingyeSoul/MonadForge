"""WebSocket endpoint for real-time task log streaming."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from webui.services.task_service import task_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_log_ws(websocket: WebSocket, task_id: str):
    """Stream task output lines over WebSocket.

    Subscribes to the task_service subscriber queue and forwards
    each message to the client until the task completes or the
    client disconnects.
    """
    await websocket.accept()
    queue = task_service.subscribe(task_id)
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            # Stop after a terminal message
            if msg.get("type") in ("done", "cancelled", "error"):
                break
    except WebSocketDisconnect:
        logger.debug("Client disconnected from task %s", task_id)
    except Exception:
        logger.exception("WebSocket error for task %s", task_id)
    finally:
        task_service.unsubscribe(task_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
