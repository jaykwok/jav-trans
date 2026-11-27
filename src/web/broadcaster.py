from __future__ import annotations

import asyncio
import contextlib
import json
import logging

log = logging.getLogger(__name__)
_TCP_STREAM_LIMIT_BYTES = 4 * 1024 * 1024
MAX_SUBSCRIBERS = 64
MAX_TCP_CLIENTS = 64
RESYNC_EVENT_LINE = '{"type": "resync"}'
_CRITICAL_PHASES = frozenset({"done", "error", "blocked"})
_CONTROL_TYPES = frozenset({"connected", "resync"})

_subscribers: list[asyncio.Queue[str]] = []
_active_tcp_clients = 0
# Strong refs for background progress persistence tasks so they aren't GC'd
# before completing and can be cancelled cleanly on shutdown.
_pending_progress_tasks: set[asyncio.Task[None]] = set()


class SubscriberLimitError(RuntimeError):
    """Raised when the SSE subscriber cap has been reached."""


async def subscribe() -> asyncio.Queue[str]:
    if len(_subscribers) >= MAX_SUBSCRIBERS:
        raise SubscriberLimitError(f"SSE subscriber limit ({MAX_SUBSCRIBERS}) reached")
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
    cancelled run's progress under the retry, and `apply_job_progress_memory`
    enforces the same check on the stored job. An event without a run id cannot
    be judged and is kept.
    """
    job_id = str(event.get("job_id", "") or "")
    run_id = str(event.get("run_id", "") or "")
    if not job_id or not run_id:
        return False
    try:
        from web import pipeline_manager

        active = pipeline_manager.active_run_id(job_id)
    except Exception as exc:
        log.debug("Could not check active_run_id for job %s: %s", job_id, exc)
        return False
    return bool(active) and active != run_id


def _on_progress_task_done(task: asyncio.Task[None]) -> None:
    _pending_progress_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("Background job progress persistence failed", exc_info=exc)


def _is_critical_event(event: dict | None) -> bool:
    if not isinstance(event, dict):
        return False
    if event.get("type") in _CONTROL_TYPES:
        return True
    if event.get("stage") == "timing_summary":
        return True
    return event.get("phase") in _CRITICAL_PHASES


def _classify_queued_line(line: str) -> tuple[bool, bool]:
    """Return ``(is_progress, is_critical)`` for a queued SSE line."""
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return False, False
    if not isinstance(parsed, dict):
        return False, False
    return parsed.get("phase") == "progress", _is_critical_event(parsed)


def _enqueue_with_backpressure(
    queue: asyncio.Queue[str],
    line: str,
    *,
    is_critical: bool,
) -> None:
    """Enqueue ``line``, shedding oldest progress events rather than terminal ones."""
    try:
        queue.put_nowait(line)
        return
    except asyncio.QueueFull:
        pass
    except RuntimeError:
        return

    items: list[str] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    maxsize = queue.maxsize if queue.maxsize > 0 else len(items) + 2
    has_resync = RESYNC_EVENT_LINE in items

    def _drop_one(*, allow_non_progress: bool) -> bool:
        for idx, queued_line in enumerate(items):
            is_progress, queued_critical = _classify_queued_line(queued_line)
            if is_progress:
                items.pop(idx)
                return True
            if allow_non_progress and not queued_critical:
                items.pop(idx)
                return True
        if allow_non_progress:
            for idx, queued_line in enumerate(items):
                if queued_line not in {'{"type": "connected"}', RESYNC_EVENT_LINE}:
                    items.pop(idx)
                    return True
        return False

    dropped = _drop_one(allow_non_progress=is_critical)
    if not dropped and not is_critical:
        # Every queued item is critical; prefer preserving critical events and
        # signal a resync only if space already exists.
        if not has_resync and len(items) < maxsize:
            items.append(RESYNC_EVENT_LINE)
    else:
        if dropped and not has_resync and maxsize >= 2:
            if len(items) + 2 > maxsize:
                _drop_one(allow_non_progress=is_critical)
            if len(items) + 2 <= maxsize:
                items.append(RESYNC_EVENT_LINE)
                has_resync = True
        if len(items) < maxsize:
            items.append(line)

    for queued_line in items:
        with contextlib.suppress(asyncio.QueueFull, RuntimeError):
            queue.put_nowait(queued_line)


def publish(event_line: str) -> None:
    line = str(event_line).replace("\r", " ").replace("\n", " ").strip()
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
            from web import pipeline_manager

            updated, stage_changed = pipeline_manager.apply_job_progress_memory(
                job_id,
                progress,
                run_id=str(event.get("run_id", "") or ""),
            )
            if updated:
                task = pipeline_manager.schedule_progress_persist(
                    job_id,
                    immediate=stage_changed,
                )
                if task is not None and task not in _pending_progress_tasks:
                    _pending_progress_tasks.add(task)
                    task.add_done_callback(_on_progress_task_done)

    is_critical = _is_critical_event(event if isinstance(event, dict) else None)
    for queue in list(_subscribers):
        _enqueue_with_backpressure(queue, line, is_critical=is_critical)


async def cancel_pending_progress_tasks() -> None:
    """Cancel and await any pending progress persistence tasks during shutdown."""
    tasks = list(_pending_progress_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _handle_tcp_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    global _active_tcp_clients
    if _active_tcp_clients >= MAX_TCP_CLIENTS:
        log.warning(
            "Event TCP client rejected: active client limit (%d) reached",
            MAX_TCP_CLIENTS,
        )
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return
    _active_tcp_clients += 1
    try:
        while True:
            try:
                raw_line = await reader.readline()
            except ValueError as exc:
                log.warning(
                    "Event TCP stream line exceeded %d bytes: %s",
                    _TCP_STREAM_LIMIT_BYTES,
                    exc,
                )
                break
            if not raw_line:
                break
            try:
                publish(raw_line.decode("utf-8", errors="replace").rstrip("\r\n"))
            except Exception:
                log.exception("Failed to publish stage event from TCP stream")
    except (ConnectionError, OSError) as exc:
        log.debug("Event TCP client disconnected: %s", exc)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Unhandled error in event TCP client handler")
    finally:
        _active_tcp_clients = max(0, _active_tcp_clients - 1)
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def start_tcp_listener(host: str, port: int) -> tuple[asyncio.Server, asyncio.Task[None]]:
    """Bind the event socket before returning so callers never race startup."""
    server = await asyncio.start_server(
        _handle_tcp_client,
        host,
        port,
        limit=_TCP_STREAM_LIMIT_BYTES,
    )

    async def _serve() -> None:
        try:
            async with server:
                await server.serve_forever()
        except asyncio.CancelledError:
            server.close()
            await server.wait_closed()
            raise

    task = asyncio.create_task(_serve(), name=f"sse-tcp-listener-{port}")
    return server, task
