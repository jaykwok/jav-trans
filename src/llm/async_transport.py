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

Reusing one private event loop per worker thread avoids repeated IOCP /
socketpair churn on Windows while preserving strict cross-task isolation: each
worker thread owns its own loop and each request owns its own HTTP client, so
cancelling job A never disturbs job B's in-flight request.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import inspect
import logging
import threading
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from llm.errors import TranslationCancelledError
from llm.transport_util import _cancel_requested

logger = logging.getLogger(__name__)


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

_THREAD_LOOP = threading.local()
_LOOP_HOLDERS_LOCK = threading.Lock()
_LOOP_HOLDERS: "weakref.WeakSet[_ThreadLoopHolder]" = weakref.WeakSet()


class _ThreadLoopHolder:
    """Owns one worker thread's private event loop and closes it on thread exit."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop: asyncio.AbstractEventLoop | None = loop
        with _LOOP_HOLDERS_LOCK:
            _LOOP_HOLDERS.add(self)

    def close(self) -> None:
        loop = self.loop
        self.loop = None
        if loop is None or loop.is_closed():
            return
        if loop.is_running():
            return
        try:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending:
                for task in pending:
                    task.cancel()
                with contextlib.suppress(Exception):
                    loop.run_until_complete(asyncio.wait(pending, timeout=0.5))
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            with contextlib.suppress(Exception):
                loop.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def _close_all_thread_loops() -> None:
    with _LOOP_HOLDERS_LOCK:
        holders = list(_LOOP_HOLDERS)
    for holder in holders:
        with contextlib.suppress(Exception):
            holder.close()


atexit.register(_close_all_thread_loops)


def _acquire_thread_loop() -> asyncio.AbstractEventLoop:
    holder: _ThreadLoopHolder | None = getattr(_THREAD_LOOP, "holder", None)
    if (
        holder is not None
        and holder.loop is not None
        and not holder.loop.is_closed()
        and not holder.loop.is_running()
    ):
        return holder.loop
    loop = asyncio.new_event_loop()
    _THREAD_LOOP.holder = _ThreadLoopHolder(loop)
    return loop


def _discard_thread_loop(loop: asyncio.AbstractEventLoop) -> None:
    holder: _ThreadLoopHolder | None = getattr(_THREAD_LOOP, "holder", None)
    if holder is not None and holder.loop is loop:
        _THREAD_LOOP.holder = None
        holder.close()
    elif not loop.is_closed() and not loop.is_running():
        with contextlib.suppress(Exception):
            loop.close()


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
    logger.warning(
        "Async translation request did not unwind after %d cancel attempts",
        _CANCEL_ATTEMPTS,
    )


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
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Cancel watcher failed during cleanup: %s", exc, exc_info=True)


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
    loop = _acquire_thread_loop()
    reusable = True
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            _run_with_watcher(operation, cancel_event, poll_interval_s)
        )
    except asyncio.CancelledError as exc:
        if _cancel_requested(cancel_event):
            raise TranslationCancelledError("任务已取消") from exc
        logger.warning(
            "Async translation request was cancelled without cancel_event set",
            exc_info=True,
        )
        raise RuntimeError("异步翻译请求被意外中断（CancelledError）") from exc
    finally:
        try:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.wait(pending, timeout=1.0))
                if any(not task.done() for task in pending):
                    reusable = False
            if reusable:
                loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            reusable = False
        finally:
            asyncio.set_event_loop(None)
            if not reusable:
                _discard_thread_loop(loop)


async def close_quietly(resource: Any, *, timeout_s: float = 5.0) -> None:
    """Close a stream/client without letting cleanup change the outcome.

    Idempotent by contract: callers close in a `finally` that may run after the
    object already closed itself. A failure here is a leak to report, never a
    reason to retry a request that was cancelled or already answered.
    """
    if resource is None:
        return
    closer = getattr(resource, "aclose", None)
    if not callable(closer):
        closer = getattr(resource, "close", None)
    if not callable(closer):
        return
    try:
        result = closer()
        if inspect.isawaitable(result):
            close_task = asyncio.ensure_future(result)
            try:
                await asyncio.wait_for(asyncio.shield(close_task), timeout=timeout_s)
            except asyncio.TimeoutError:
                close_task.cancel()
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(close_task, timeout=0.5)
            except asyncio.CancelledError:
                if not close_task.done():
                    try:
                        await asyncio.wait_for(close_task, timeout=0.5)
                    except BaseException:
                        if not close_task.done():
                            close_task.cancel()
                            with contextlib.suppress(BaseException):
                                await asyncio.wait_for(close_task, timeout=0.2)
                else:
                    with contextlib.suppress(BaseException):
                        close_task.exception()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.debug("Ignoring error while closing %s: %s", type(resource).__name__, exc)
