from __future__ import annotations

import asyncio
import contextlib
import json


_subscribers: list[asyncio.Queue[str]] = []
# Strong refs for fire-and-forget progress tasks so they aren't GC'd before
# completing (loop.create_task drops the ref otherwise).
_pending_progress_tasks: set[asyncio.Task[None]] = set()


async def subscribe() -> asyncio.Queue[str]:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
    _subscribers.append(queue)
    await queue.put('{"type": "connected"}')
    return queue


def unsubscribe(q: asyncio.Queue[str]) -> None:
    with contextlib.suppress(ValueError):
        _subscribers.remove(q)


def _is_stale_run(event: dict) -> bool:
    """Did a previous execution of this job send this?

    Both defences are needed: dropping it here keeps the browser from showing a
    cancelled run's progress under the retry, and `update_job_progress` makes
    the same check inside the state lock so the stored job cannot be rewritten
    by it either. An event without a run id cannot be judged and is kept.
    """
    job_id = str(event.get("job_id", "") or "")
    run_id = str(event.get("run_id", "") or "")
    if not job_id or not run_id:
        return False
    try:
        from web import pipeline_manager

        active = pipeline_manager.active_run_id(job_id)
    except Exception:
        return False
    return bool(active) and active != run_id


def publish(event_line: str) -> None:
    line = str(event_line).strip()
    if not line:
        return
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        event = None
    if isinstance(event, dict):
        if _is_stale_run(event):
            return
        job_id = str(event.get("job_id", "") or "")
        if job_id and event.get("stage") != "timing_summary":
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
                task = loop.create_task(
                    pipeline_manager.update_job_progress(
                        job_id,
                        progress,
                        run_id=str(event.get("run_id", "") or ""),
                    )
                )
                _pending_progress_tasks.add(task)
                task.add_done_callback(_pending_progress_tasks.discard)
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
