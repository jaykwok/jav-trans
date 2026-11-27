"""A single, ordered writer with explicit durability acknowledgements."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
from typing import Callable

from web.models import JobState

log = logging.getLogger(__name__)

_CACHEABLE_TERMINAL_STATUSES = frozenset(
    {"done", "failed", "cancelled", "publish_failed", "export_failed"}
)


class JobPersistenceError(RuntimeError):
    """The requested state has not been confirmed on disk."""


class _CommitGate:
    """Order abandonment against the single irreversible filesystem replace."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._started = False

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._cancelled = True
            return True

    def commit(self, replace: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                raise JobPersistenceError("保存请求已超时或取消，未提交这次任务变更。")
            self._started = True
        replace()


class _Receipt(Future):
    def __init__(self, gate: _CommitGate | None) -> None:
        super().__init__()
        self.gate = gate


@dataclass
class _Write:
    sequence: int
    snapshot: list[JobState] | Callable[[], list[JobState]]
    barrier: bool
    write: Callable[[list[JobState]], None]
    gate: _CommitGate | None = None
    receipts: list[_Receipt] = field(default_factory=list)


def _job_cache_key(job: JobState) -> tuple:
    return (
        job.revision,
        job.run_id,
        job.status,
        job.current_stage,
        job.output_target,
        job.output_generation,
        job.error,
        job.storage_error,
        job.pending_status,
        job.subtitle_cue_count,
        tuple(job.artifacts),
    )


def _copy_job_lightweight(job: JobState) -> JobState:
    """Snapshot a job without deep-copying immutable ``spec`` payloads."""
    return job.model_copy(
        update={
            "progress": copy.deepcopy(job.progress) if job.progress else {},
            "artifacts": list(job.artifacts),
        }
    )


class JobStoreWriter:
    _PACE_S = 2.0
    _CONFIRM_TIMEOUT_S = 10.0

    def __init__(self, write: Callable[[list[JobState]], None]) -> None:
        self._write = write
        self._ready = threading.Condition()
        self._queue: deque[_Write] = deque()
        self._submitted = 0
        self._written = 0
        self._last_write = 0.0
        self._thread: threading.Thread | None = None
        self._latest: Future | None = None
        self._frozen_cache: dict[str, tuple[tuple, JobState]] = {}

    def _freeze_snapshot(self, snapshot: list[JobState]) -> list[JobState]:
        """Freeze caller-owned models while reusing cached immutable terminal records."""
        frozen: list[JobState] = []
        live_ids: set[str] = set()
        for job in snapshot:
            live_ids.add(job.id)
            if job.revision > 0 and job.status in _CACHEABLE_TERMINAL_STATUSES:
                key = _job_cache_key(job)
                cached = self._frozen_cache.get(job.id)
                if cached is not None and cached[0] == key:
                    frozen.append(cached[1])
                    continue
                copied = _copy_job_lightweight(job)
                self._frozen_cache[job.id] = (key, copied)
                frozen.append(copied)
            else:
                self._frozen_cache.pop(job.id, None)
                frozen.append(_copy_job_lightweight(job))
        if len(self._frozen_cache) > len(live_ids):
            for stale_id in list(self._frozen_cache):
                if stale_id not in live_ids:
                    self._frozen_cache.pop(stale_id, None)
        return frozen

    def enqueue(
        self,
        snapshot: list[JobState] | Callable[[], list[JobState]],
        *,
        confirm: bool,
        write: Callable[[list[JobState]], None] | None = None,
        guarded_write: Callable[[list[JobState], Callable], None] | None = None,
    ) -> Future:
        gate = _CommitGate() if guarded_write is not None else None
        receipt = _Receipt(gate)
        if guarded_write is not None:
            assert gate is not None
            effective_write: Callable[[list[JobState]], None] = (
                lambda frozen, _gw=guarded_write, _commit=gate.commit: _gw(frozen, _commit)
            )
        else:
            effective_write = write or self._write
        with self._ready:
            frozen: list[JobState] | Callable[[], list[JobState]] = (
                snapshot if callable(snapshot) else self._freeze_snapshot(snapshot)
            )
            self._submitted += 1
            operation = _Write(
                self._submitted,
                frozen,
                confirm,
                effective_write,
                gate=gate,
                receipts=[receipt],
            )
            # A barrier may absorb preceding progress, but can never itself be
            # replaced by progress or another barrier.
            if self._queue and not self._queue[-1].barrier:
                preceding = self._queue.pop()
                operation.receipts = preceding.receipts + operation.receipts
                for preceding_receipt in preceding.receipts:
                    preceding_receipt.gate = gate
            self._queue.append(operation)
            self._latest = receipt
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="job-store-writer",
                    daemon=True,
                )
                self._thread.start()
            self._ready.notify_all()
        return receipt

    def submit(self, snapshot: list[JobState], *, confirm: bool) -> Future:
        receipt = self.enqueue(snapshot, confirm=confirm)
        if confirm:
            self._wait(receipt)
        return receipt

    def _wait(self, receipt: Future) -> None:
        try:
            try:
                receipt.result(timeout=self._CONFIRM_TIMEOUT_S)
            except TimeoutError as exc:
                gate = getattr(receipt, "gate", None)
                if gate is not None and not gate.cancel():
                    # The atomic replace already won. Do not report failure
                    # and let callers roll back a transaction that committed.
                    receipt.result()
                    return
                raise JobPersistenceError("保存任务状态超时；尚未确认保存，请重试保存。") from exc
        except JobPersistenceError:
            raise
        except Exception as exc:
            raise JobPersistenceError("任务状态保存失败；请检查磁盘后重试保存。") from exc

    @staticmethod
    async def _settle_commit(wrapped: asyncio.Future) -> None:
        # Once replace started, cancellation cannot undo it. Finish the short
        # commit and its caller's state update before releasing the mutation gate.
        while True:
            try:
                await asyncio.shield(wrapped)
                return
            except asyncio.CancelledError:
                if wrapped.cancelled():
                    raise

    async def confirm(self, receipt: Future | None) -> None:
        if receipt is None:
            return
        wrapped = asyncio.wrap_future(receipt)
        try:
            try:
                await asyncio.wait_for(asyncio.shield(wrapped), timeout=self._CONFIRM_TIMEOUT_S)
            except (TimeoutError, asyncio.CancelledError) as exc:
                gate = getattr(receipt, "gate", None)
                if gate is not None and not gate.cancel():
                    await self._settle_commit(wrapped)
                    return
                # The abandoned writer may still be preparing its temp file.
                # Consume its eventual rejection; no late exception is lost.
                wrapped.add_done_callback(
                    lambda result: None if result.cancelled() else result.exception()
                )
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise JobPersistenceError("保存任务状态超时；尚未确认保存，请重试保存。") from exc
        except JobPersistenceError:
            raise
        except Exception as exc:
            raise JobPersistenceError("任务状态保存失败；请检查磁盘后重试保存。") from exc

    def flush(self) -> None:
        with self._ready:
            receipt = self._latest
            for operation in self._queue:
                operation.barrier = True
            self._ready.notify_all()
        if receipt is not None:
            self._wait(receipt)

    def _run(self) -> None:
        while True:
            with self._ready:
                while not self._queue:
                    if not self._ready.wait(30.0) and not self._queue:
                        self._thread = None
                        return
                if not any(operation.barrier for operation in self._queue):
                    remaining = self._PACE_S - (time.monotonic() - self._last_write)
                    if remaining > 0:
                        self._ready.wait(remaining)
                        continue
                operation = self._queue.popleft()
            try:
                if operation.gate is not None and operation.gate.is_cancelled():
                    raise JobPersistenceError("保存请求已超时或取消，未提交这次任务变更。")
                snapshot = (
                    operation.snapshot()
                    if callable(operation.snapshot)
                    else operation.snapshot
                )
                operation.write(snapshot)
            except Exception as exc:
                log.exception("Failed to persist job state")
                for receipt in operation.receipts:
                    if not receipt.done():
                        receipt.set_exception(exc)
            else:
                with self._ready:
                    self._written = operation.sequence
                for receipt in operation.receipts:
                    if not receipt.done():
                        receipt.set_result(operation.sequence)
            finally:
                self._last_write = time.monotonic()
