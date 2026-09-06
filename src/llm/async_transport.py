"""One cancellable operation per network request.

A translation request spends most of its life inside a single blocking read.
The model thinks for minutes before the first token, and a reasoning model goes
quiet again mid-stream; checking a cancel event between stream events never
reaches those gaps, because the thread is parked in `recv()`. That is why 取消
used to be noticed only when the provider finally spoke - or, on a request that
never answered, not at all.

Another checkpoint cannot fix that. Each request instead runs as a single
asyncio Task on a private loop owned by the calling worker thread, covering the
whole lifetime: waiting for the connection and the response headers, receiving
the body, parsing it, and closing the request. Cancelling that task aborts the
socket operation itself, wherever it happens to be parked.

Two things this deliberately is not:

- It does not wrap a *synchronous* request in a thread and cancel the future.
  That cancels the wrapper and leaves the socket read exactly where it was.
- It does not touch a shared client. The unit of cancellation is one request,
  so cancelling job A cannot disturb job B's in-flight request.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from llm.errors import TranslationCancelledError
from llm.transport_util import _cancel_requested


T = TypeVar("T")

# How often the watcher looks at the (threading) cancel event. The event is set
# from another thread and has no async counterpart, so this is a poll; 50 ms is
# below the threshold where a user notices, and costs nothing next to a request
# that runs for minutes.
_CANCEL_POLL_S = 0.05

# How long to give a cancelled request to unwind before asking again, and how
# many times to ask before leaving it to the loop teardown.
_CANCEL_RETRY_S = 0.5
_CANCEL_ATTEMPTS = 8


async def _watch_for_cancel(
    cancel_event: threading.Event | None,
    task: "asyncio.Task[Any]",
    poll_interval_s: float,
) -> None:
    while not task.done():
        if _cancel_requested(cancel_event):
            await _cancel_until_done(task)
            return
        await asyncio.sleep(poll_interval_s)


async def _cancel_until_done(task: "asyncio.Task[Any]") -> None:
    """Ask again if the first cancel did not land.

    A single `cancel()` is enough against every stall the acceptance tests
    reproduce (no response headers, no first event, silence mid-stream). It is
    not guaranteed in general: a coroutine that catches CancelledError somewhere
    in the HTTP stack below us would keep running, and the caller would then
    wait out the request's own timeout - the exact hang this exists to prevent.
    Re-asking costs nothing, and the gap is wide enough that a task already
    unwinding gets to finish its cleanup.
    """
    for _ in range(_CANCEL_ATTEMPTS):
        task.cancel()
        if task.done():
            return
        try:
            await asyncio.wait({task}, timeout=_CANCEL_RETRY_S)
        except asyncio.CancelledError:
            return
        if task.done():
            return


async def _run_with_watcher(
    operation: Callable[[], Awaitable[T]],
    cancel_event: threading.Event | None,
    poll_interval_s: float,
) -> T:
    loop = asyncio.get_running_loop()
    # The task exists before anything opens a socket, so there is no window in
    # which the request is running but not yet cancellable.
    task = loop.create_task(_call_operation(operation))
    watcher = loop.create_task(_watch_for_cancel(cancel_event, task, poll_interval_s))
    try:
        return await task
    finally:
        watcher.cancel()
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass


async def _call_operation(operation: Callable[[], Awaitable[T]]) -> T:
    return await operation()


def run_cancellable_request(
    operation: Callable[[], Awaitable[T]],
    *,
    cancel_event: threading.Event | None = None,
    poll_interval_s: float = _CANCEL_POLL_S,
) -> T:
    """Run one async request from a synchronous worker, cancellable throughout.

    `operation` must perform the *whole* request - connect, headers, body, parse
    and close - so that cancelling it releases everything it acquired. Raises
    `TranslationCancelledError` if the cancel event is set before or during the
    request; that exception is never retried by the callers above.
    """
    # Cancel-before-register: a cancel that arrived while this thread was
    # queuing must take effect now, not after one more request is issued.
    if _cancel_requested(cancel_event):
        raise TranslationCancelledError("任务已取消")
    try:
        return asyncio.run(
            _run_with_watcher(operation, cancel_event, poll_interval_s)
        )
    except asyncio.CancelledError as exc:
        raise TranslationCancelledError("任务已取消") from exc


async def close_quietly(resource: Any) -> None:
    """Close a stream/client without letting cleanup change the outcome.

    Idempotent by contract: callers close in a `finally` that may run after the
    object already closed itself. A failure here is a leak to report, never a
    reason to retry a request that was cancelled or already answered.
    """
    if resource is None:
        return
    closer = getattr(resource, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if asyncio.iscoroutine(result):
            # Cancellation is what got us here on the cancel path; shield the
            # close so it is not cancelled before it releases the socket.
            await asyncio.shield(asyncio.ensure_future(result))
    except (asyncio.CancelledError, Exception):
        pass
