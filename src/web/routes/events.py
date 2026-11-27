from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from web import broadcaster


router = APIRouter()
_ALWAYS_DELIVER_TYPES = frozenset({"connected", "resync"})


def _matches_job(event_line: str, job_id: str | None) -> bool:
    if not job_id:
        return True
    try:
        payload = json.loads(event_line)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("type") in _ALWAYS_DELIVER_TYPES:
        return True
    return str(payload.get("job_id", "")) == job_id


async def _event_stream(
    request: Request,
    job_id: str | None,
    queue: asyncio.Queue[str],
) -> AsyncIterator[str]:
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event_line = await asyncio.wait_for(queue.get(), timeout=15.0)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if _matches_job(event_line, job_id):
                yield f"data: {event_line}\n\n"
    finally:
        broadcaster.unsubscribe(queue)


@router.get("/events")
async def get_events(
    request: Request,
    job_id: str | None = None,
) -> StreamingResponse:
    try:
        queue = await broadcaster.subscribe()
    except broadcaster.SubscriberLimitError as exc:
        raise HTTPException(
            status_code=503,
            detail="实时事件流连接数已达上限，请关闭多余标签页后重试。",
        ) from exc
    return StreamingResponse(
        _event_stream(request, job_id, queue),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
