from __future__ import annotations

import asyncio
import contextlib
import json


_subscribers: list[asyncio.Queue[str]] = []


async def subscribe() -> asyncio.Queue[str]:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
    _subscribers.append(queue)
    return queue


def unsubscribe(q: asyncio.Queue[str]) -> None:
    with contextlib.suppress(ValueError):
        _subscribers.remove(q)


def publish(event_line: str) -> None:
    line = str(event_line).strip()
    if not line:
        return
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        event = None
    if isinstance(event, dict):
        job_id = str(event.get("job_id", "") or "")
        if job_id:
            progress = {
                "stage": event.get("stage"),
                "phase": event.get("phase"),
                "video": event.get("video"),
                "ts": event.get("ts"),
                "extra": dict(event.get("extra", {}))
                if isinstance(event.get("extra"), dict)
                else {},
            }
            extra = progress["extra"]
            for key in (
                "translated",
                "expected",
                "content_chars",
                "reasoning_chars",
                "current",
                "total",
                "attempt",
            ):
                if key in extra:
                    progress[key] = extra[key]
            try:
                from web import pipeline_manager

                loop = asyncio.get_running_loop()
                loop.create_task(pipeline_manager.update_job_progress(job_id, progress))
            except RuntimeError:
                pass
    for queue in list(_subscribers):
        try:
            queue.put_nowait(line)
        except asyncio.QueueFull:
            continue
        except RuntimeError:
            continue


async def _handle_tcp_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        while True:
            raw_line = await reader.readline()
            if not raw_line:
                break
            publish(raw_line.decode("utf-8", errors="replace").rstrip("\r\n"))
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def tcp_listener(host: str, port: int) -> None:
    server = await asyncio.start_server(_handle_tcp_client, host, port)
    try:
        async with server:
            await server.serve_forever()
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()
        raise
