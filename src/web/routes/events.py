from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from web import broadcaster


router = APIRouter()


def _matches_job(event_line: str, job_id: str | None) -> bool:
    if not job_id:
        return True
    try:
        payload = json.loads(event_line)
    except json.JSONDecodeError:
        return False
    return str(payload.get("job_id", "")) == job_id


async def _event_stream(
    request: Request,
    job_id: str | None,
) -> AsyncIterator[str]:
    queue = await broadcaster.subscribe()
    yield 'data: {"type": "connected"}\n\n'
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
    return StreamingResponse(
        _event_stream(request, job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
