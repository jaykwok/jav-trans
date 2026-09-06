"""One owner for every managed process, handle and cleanup retry.

Three subsystems start OS processes that hold real resources: the ASR GPU
worker, the local llama-server, and the media children (ffmpeg/ffprobe). Each
had grown its own copy of the same machinery - a pending table, a retry entry
point, a backoff loop, and its own reading of "has this been released?" - and
every round of review found the same class of bug in a different copy: a stale
retry acting on the resource that replaced its target, two retries closing one
handle, a status that said "released" while a process was still resident.

So the machinery lives here once, and the subsystems keep only what is genuinely
theirs: how to start their process, and how to stop it.

Two ideas carry the whole module:

* **Identity is `(resource_id, generation)` and it never moves.** A record names
  one concrete process. It is never re-pointed at a replacement, so a cleanup
  request raised for generation 3 can never act on generation 4 - the record it
  names is simply no longer here.
* **"Process gone" and "handle closed" are separate confirmations.** A process
  that may still be resident holds the GPU; a Job Object handle nobody could
  close does not (whatever was inside it is gone). Conflating them either frees
  a card that is still occupied, or blocks one that is free.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# "gone" is the only value that means released. An error while asking is
# "unknown", which is *not* an exit: it is the one answer that would otherwise
# let a caller start a second process on the same card.
ProcessState = str  # "live" | "alive" | "gone" | "unknown"

RETRY_MIN_S = 2.0
RETRY_MAX_S = 60.0


@dataclass
class StopOutcome:
    """What a stopper actually achieved. No optional fields, no None-means-ok."""

    process_state: ProcessState
    job_handle: Any = None
    detail: str = ""


# A stopper performs the blocking part of a stop: terminate, wait, close the
# handle. It is called *outside* the registry lock and must report what it
# confirmed rather than what it attempted.
Stopper = Callable[[Any, Any], StopOutcome]


@dataclass
class StopResult:
    resource_id: str
    generation: int
    stopped: bool
    process_state: ProcessState
    handle_open: bool
    detail: str = ""
    #

    @property
    def released(self) -> bool:
        """Nothing of this resource is left: process gone, handle closed."""
        return self.stopped and not self.handle_open


@dataclass
class _Record:
    resource_id: str
    generation: int
    kind: str
    description: str
    process: Any
    job_handle: Any
    stopper: Stopper
    holds_gpu: bool
    state: str  # "live" | "stopping"
    process_state: ProcessState = "alive"
    attempts: int = 0
    last_error: str = ""
    since: float = field(default_factory=time.time)
    last_attempt_at: float = 0.0
    next_retry_at: float = 0.0
    claimed: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.resource_id, self.generation)

    @property
    def handle_open(self) -> bool:
        return bool(self.job_handle)

    def view(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "generation": self.generation,
            "kind": self.kind,
            # Never the command line: the arguments carry media paths.
            "description": self.description,
            "pid": getattr(self.process, "pid", None),
            "state": self.state,
            "process_state": self.process_state,
            "handle_open": self.handle_open,
            "holds_gpu": self.holds_gpu,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "since": self.since,
            "next_retry_in_s": (
                max(0.0, self.next_retry_at - time.time()) if self.next_retry_at else 0.0
            ),
            # A stop was asked for and has not been confirmed.
            "blocked": self.state == "stopping" and self.process_state != "gone",
        }


_LOCK = threading.RLock()
_RECORDS: dict[tuple[str, int], _Record] = {}
_WATCHDOG: threading.Thread | None = None
_ID_SEQ = itertools.count(1)


class ResourceIdentityError(RuntimeError):
    """An identity was reused. Always a bug in the caller, never recoverable."""


def new_resource_id(prefix: str) -> str:
    """Allocate an identity that will not be handed out again in this process.

    Owners used to name themselves - `"llamacpp-server"`, and a generation that
    restarted at 0 for every instance - so two objects created one after the
    other claimed *the same* identity for their first process. A cleanup queued
    against the first one then named, and terminated, the second one's server.
    Uniqueness cannot be left to each owner's own numbering: it is a property of
    the registry, so the registry hands the ids out.
    """
    with _LOCK:
        return f"{prefix}#{next(_ID_SEQ)}"


def _backoff_s(attempts: int) -> float:
    return min(RETRY_MAX_S, RETRY_MIN_S * (2 ** max(0, attempts - 1)))


def register(
    *,
    resource_id: str,
    generation: int,
    kind: str,
    process: Any,
    job_handle: Any = None,
    stopper: Stopper,
    holds_gpu: bool = False,
    description: str = "",
    state: str = "live",
    process_state: ProcessState = "alive",
    detail: str = "",
) -> None:
    """Take ownership of a resource under an identity that will never move.

    Refuses an identity that is already taken. Overwriting one silently drops
    the previous record - which is the only reference to a process or handle
    somebody is still waiting to clean up - and leaves a queued cleanup request
    pointing at whatever took its place.
    """
    with _LOCK:
        taken = _RECORDS.get((resource_id, generation))
        if taken is not None:
            raise ResourceIdentityError(
                f"resource identity already registered: {resource_id} "
                f"generation={generation} (kind={taken.kind}, state={taken.state}). "
                "Allocate a new id with new_resource_id() instead of reusing one."
            )
        record = _Record(
            resource_id=resource_id,
            generation=generation,
            kind=kind,
            description=description,
            process=process,
            job_handle=job_handle,
            stopper=stopper,
            holds_gpu=holds_gpu,
            state=state,
            process_state=process_state,
            last_error=detail,
        )
        if state == "stopping":
            record.attempts = 1
            record.next_retry_at = time.time() + _backoff_s(1)
        _RECORDS[record.key] = record
    if state == "stopping":
        _start_watchdog()


def schedule_stop(resource_id: str, generation: int) -> None:
    """Publish stop intent without waiting for the owner's lifecycle lock.

    Admission must see the pending owner even if close() has not yet reached
    its OS stopper. The watchdog uses the same exclusive request_stop path.
    """
    with _LOCK:
        record = _RECORDS.get((resource_id, generation))
        if record is None:
            return
        if record.state == "live":
            record.state = "stopping"
            record.next_retry_at = time.time() + RETRY_MIN_S
    _start_watchdog()


def attach_handle(resource_id: str, generation: int, job_handle: Any) -> None:
    """Bind a Job Object handle to a resource that is already registered."""
    with _LOCK:
        record = _RECORDS.get((resource_id, generation))
        if record is not None:
            record.job_handle = job_handle


def confirm_gone(resource_id: str, generation: int) -> bool:
    """The resource exited on its own and left nothing behind.

    Refuses to drop a record that still holds an unclosed handle: "the process
    is gone" and "there is nothing left to close" are different claims.
    """
    with _LOCK:
        record = _RECORDS.get((resource_id, generation))
        if record is None:
            return True
        record.process_state = "gone"
        if record.handle_open:
            record.state = "stopping"
            return False
        _RECORDS.pop(record.key, None)
        return True


def request_stop(resource_id: str, generation: int) -> StopResult:
    """The single cleanup entry point: stop *this* resource, if it is still here.

    Used by every caller - the owner finishing normally, the manual 重试清理
    button, and the backoff watchdog - because they are the same operation. A
    request that names a resource which is no longer registered succeeds without
    touching anything: that is what stops a late retry from killing the process
    that took its place, and what keeps 重试清理 from escalating into "terminate
    whatever is running now".
    """
    with _LOCK:
        record = _RECORDS.get((resource_id, generation))
        if record is None:
            return StopResult(
                resource_id=resource_id,
                generation=generation,
                stopped=True,
                process_state="gone",
                handle_open=False,
                detail="not registered",
            )
        if record.claimed:
            # Somebody else is inside the blocking part right now. Reporting
            # "still going" is honest; joining them would double-close.
            return StopResult(
                resource_id=resource_id,
                generation=generation,
                stopped=False,
                process_state=record.process_state,
                handle_open=record.handle_open,
                detail="cleanup already in progress",
            )
        record.claimed = True
        record.state = "stopping"
        record.last_attempt_at = time.time()
        stopper = record.stopper
        process = record.process
        job_handle = record.job_handle

    try:
        outcome = stopper(process, job_handle)
        if not isinstance(outcome, StopOutcome):  # pragma: no cover - defensive
            raise TypeError(f"stopper returned {outcome!r}, expected StopOutcome")
    except BaseException as exc:
        # An error is an unknown, never an exit.
        with _LOCK:
            if _RECORDS.get(record.key) is record:
                record.process_state = "unknown"
                record.attempts += 1
                record.last_error = f"stopper raised: {exc!r}"
                record.next_retry_at = time.time() + _backoff_s(record.attempts)
            record.claimed = False
        _start_watchdog()
        if isinstance(exc, Exception):
            return StopResult(
                resource_id=resource_id,
                generation=generation,
                stopped=False,
                process_state="unknown",
                handle_open=bool(job_handle),
                detail=f"stopper raised: {exc!r}",
            )
        raise

    # Commit and release the claim in one lock section: a claim released before
    # the result is written is a window where another retry takes the record and
    # acts on the handle that was just closed.
    with _LOCK:
        record.process_state = outcome.process_state
        record.job_handle = outcome.job_handle
        stopped = outcome.process_state == "gone"
        # Remove *this* record, never "whatever is filed under this key". The
        # key alone is not proof of identity, and a result belonging to a
        # resource that is no longer registered must not evict its successor.
        owned = _RECORDS.get(record.key) is record
        if stopped and not record.handle_open:
            if owned:
                _RECORDS.pop(record.key, None)
            record.claimed = False
            record.last_error = ""
            record.next_retry_at = 0.0
        else:
            record.attempts += 1
            record.last_error = outcome.detail or record.last_error
            record.next_retry_at = time.time() + _backoff_s(record.attempts)
            record.claimed = False
        result = StopResult(
            resource_id=resource_id,
            generation=generation,
            stopped=stopped,
            process_state=outcome.process_state,
            handle_open=bool(outcome.job_handle),
            detail=outcome.detail,
        )
    if not result.released:
        _start_watchdog()
    return result


def retry_pending(kind: str | None = None) -> dict[str, Any]:
    """Retry every resource waiting for cleanup. Blocking - call from a thread.

    Only resources that are *already* waiting: a live one is not a target, which
    is the difference between "retry the cleanup" and "stop what is running".
    """
    with _LOCK:
        targets = [
            (record.resource_id, record.generation)
            for record in _RECORDS.values()
            if record.state == "stopping" and (kind is None or record.kind == kind)
        ]
    released = 0
    for resource_id, generation in targets:
        if request_stop(resource_id, generation).released:
            released += 1
    return {"retried": len(targets), "released": released, "pending": len(pending(kind))}


def pending(kind: str | None = None) -> list[dict[str, Any]]:
    """Resources that were asked to stop and have not confirmed."""
    with _LOCK:
        return [
            record.view()
            for record in _RECORDS.values()
            if record.state == "stopping" and (kind is None or record.kind == kind)
        ]


def live(kind: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        return [
            record.view()
            for record in _RECORDS.values()
            if record.state == "live" and (kind is None or record.kind == kind)
        ]


def is_registered(resource_id: str, generation: int) -> bool:
    with _LOCK:
        return (resource_id, generation) in _RECORDS


def gpu_blocked() -> bool:
    """Does anything still hold the GPU after being asked to let go?

    Asked of every owner of the card, not one subsystem's private flag. A leaked
    handle does not count: whatever was inside that job is confirmed gone, so it
    holds no VRAM - it is a handle to close, tracked and retried separately.
    """
    with _LOCK:
        return any(
            record.holds_gpu
            and record.state == "stopping"
            and record.process_state != "gone"
            for record in _RECORDS.values()
        )


def open_handles() -> list[dict[str, Any]]:
    """Handles we tried to close and could not.

    A live resource's handle is not one of these: it belongs to a running
    process and closing it would kill that process tree.
    """
    with _LOCK:
        return [
            record.view()
            for record in _RECORDS.values()
            if record.handle_open and record.state == "stopping"
        ]


def snapshot() -> dict[str, Any]:
    """One status shape for the page, the queue and the diagnostics panel."""
    pending_records = pending()
    handles = open_handles()
    blocked = gpu_blocked()
    next_retry = [
        record["next_retry_in_s"] for record in pending_records if record["next_retry_in_s"]
    ]
    return {
        "blocked": blocked,
        # 正在清理 / 清理失败 / 已释放 - derived from the records, not stored
        # twice. "released" here means every owner confirmed, not just the ASR
        # worker's own state.
        "state": (
            "cleanup_failed" if blocked else "cleaning" if pending_records else "released"
        ),
        "pending": pending_records,
        "handle_open": bool(handles),
        "open_job_handles": len(handles),
        "next_retry_in_s": min(next_retry) if next_retry else 0.0,
    }


def forget_all() -> None:
    """Drop every record. Tests only - it releases nothing."""
    with _LOCK:
        _RECORDS.clear()


def _start_watchdog() -> None:
    """Retry pending cleanups on a bounded backoff, so recovery needs no user.

    One loop for every subsystem: a second watchdog per backend was how the
    backoff drifted apart from the state it was supposed to be retrying.
    """
    global _WATCHDOG
    with _LOCK:
        if _WATCHDOG is not None and _WATCHDOG.is_alive():
            return
        _WATCHDOG = threading.Thread(
            target=_watchdog_loop, name="resource-cleanup-watchdog", daemon=True
        )
        _WATCHDOG.start()


def _watchdog_loop() -> None:
    while True:
        with _LOCK:
            due = [
                (record.resource_id, record.generation, record.next_retry_at)
                for record in _RECORDS.values()
                if record.state == "stopping" and not record.claimed
            ]
        if not due:
            return
        wait_s = max(0.05, min(at for _id, _gen, at in due) - time.time())
        time.sleep(min(wait_s, RETRY_MAX_S))
        now = time.time()
        # Re-read after sleeping: the resource may have been released while this
        # thread waited, and a request raised against the identity captured
        # before the sleep is the one thing that cannot touch its replacement.
        with _LOCK:
            ready = [
                (record.resource_id, record.generation)
                for record in _RECORDS.values()
                if record.state == "stopping"
                and not record.claimed
                and record.next_retry_at <= now
            ]
        for resource_id, generation in ready:
            try:
                request_stop(resource_id, generation)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[WARN] cleanup retry raised for {resource_id}: {exc!r}")
