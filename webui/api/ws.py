"""WebSocket endpoint for real-time task log streaming."""

from __future__ import annotations

import logging
import asyncio

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
    # Hydrate a historical/unknown task off the event loop first: the daemon
    # lookups behind it are blocking, and doing them here would stall every
    # other request for up to two 15 s timeouts.
    await task_service.ensure_task_loaded(task_id)
    queue = task_service.subscribe(task_id)

    async def forward():
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            # Stop after a terminal message
            if msg.get("type") in ("done", "cancelled", "error"):
                break

    async def wait_disconnect():
        # A queued or paused task may emit nothing for hours. Receive the
        # disconnect independently so its subscriber is released immediately.
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

    workers = [asyncio.create_task(forward()), asyncio.create_task(wait_disconnect())]
    try:
        done, _pending = await asyncio.wait(
            workers, return_when=asyncio.FIRST_COMPLETED
        )
        for worker in done:
            worker.result()
    except WebSocketDisconnect:
        logger.debug("Client disconnected from task %s", task_id)
    except Exception:
        logger.exception("WebSocket error for task %s", task_id)
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        task_service.unsubscribe(task_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
