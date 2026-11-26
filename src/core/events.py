from __future__ import annotations

import json
import logging
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol, TypedDict

logger = logging.getLogger(__name__)
_ERROR_LOG_INTERVAL_S = 10.0
_RECONNECT_INTERVAL_S = 1.0


class StageEvent(TypedDict):
    ts: str
    job_id: str
    # Which execution of that job produced this. Empty from producers that do
    # not run under the Web manager (the CLI), and from anything older; the
    # receiver treats empty as "cannot tell" and does not filter it.
    run_id: str
    video: str
    stage: str
    phase: Literal["start", "done", "skip", "blocked", "degraded", "progress"]
    extra: dict[str, Any]


class _Sink(Protocol):
    def write(self, event: StageEvent) -> None: ...

    def close(self) -> None: ...


class _FileSink:
    def __init__(self, path: str) -> None:
        target = Path(path).expanduser()
        if target.parent != Path("."):
            target.parent.mkdir(parents=True, exist_ok=True)
        self._writer = target.open("a", encoding="utf-8", buffering=1)

    def write(self, event: StageEvent) -> None:
        self._writer.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        self._writer.close()


class _TcpSink:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._socket: socket.socket | None = None
        self._last_reconnect_attempt = 0.0
        self._connect()

    def _connect(self) -> None:
        self._close_socket()
        self._socket = socket.create_connection((self._host, self._port), timeout=1.0)

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def write(self, event: StageEvent) -> None:
        encoded = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        if self._socket is None:
            now = time.monotonic()
            if now - self._last_reconnect_attempt < _RECONNECT_INTERVAL_S:
                raise ConnectionError("event TCP sink temporarily disconnected")
            self._last_reconnect_attempt = now
            self._connect()
        try:
            assert self._socket is not None
            self._socket.sendall(encoded)
        except OSError:
            # One immediate reconnect attempt recovers seamlessly if the listener
            # recycled a connection without silently blackholing the whole session.
            self._close_socket()
            self._last_reconnect_attempt = time.monotonic()
            self._connect()
            assert self._socket is not None
            self._socket.sendall(encoded)

    def close(self) -> None:
        self._close_socket()


class _MemorySink:
    def __init__(self, events: list[StageEvent]) -> None:
        self._events = events

    def write(self, event: StageEvent) -> None:
        self._events.append(
            {
                "ts": str(event.get("ts", "")),
                "job_id": str(event.get("job_id", "")),
                "run_id": str(event.get("run_id", "")),
                "video": str(event.get("video", "")),
                "stage": str(event.get("stage", "")),
                "phase": event.get("phase", "progress"),
                "extra": dict(event.get("extra", {})),
            }
        )

    def close(self) -> None:
        return


_sink: _Sink | None = None
_lock = threading.Lock()
_thread_local = threading.local()
_memory_events: list[StageEvent] = []
_last_emit_error_log = 0.0


def _current_job_id() -> str:
    return str(getattr(_thread_local, "job_id", "") or "")


def _current_run_id() -> str:
    return str(getattr(_thread_local, "run_id", "") or "")


def set_current_job_id(job_id: str) -> None:
    _thread_local.job_id = job_id


def set_current_run(job_id: str, run_id: str = "") -> None:
    """Bind this thread to one execution of one job.

    Producers capture the run when they start and emit under it. They must not
    look the current run up at send time: a thread left over from a cancelled
    run would then stamp its late events with the retry's id and impersonate it.
    """
    _thread_local.job_id = job_id
    _thread_local.run_id = run_id


@contextmanager
def scoped_run_identity(
    job_id: str,
    run_id: str = "",
    *,
    video: str | None = None,
) -> Iterator[None]:
    """Bind `(job_id, run_id, video)` on `_thread_local` and restore on exit."""
    prev_job_id = getattr(_thread_local, "job_id", "")
    prev_run_id = getattr(_thread_local, "run_id", "")
    prev_video = getattr(_thread_local, "video", "")
    set_current_run(job_id, run_id)
    if video is not None:
        _thread_local.video = video
    try:
        yield
    finally:
        _thread_local.job_id = prev_job_id
        _thread_local.run_id = prev_run_id
        _thread_local.video = prev_video


def configure_sink(sink_spec: str) -> None:
    global _memory_events, _sink

    spec = (sink_spec or "").strip()
    new_sink: _Sink | None = None

    try:
        if not spec:
            _memory_events = []
            new_sink = None
        elif spec == "memory":
            _memory_events = []
            new_sink = _MemorySink(_memory_events)
        elif spec.startswith("file:"):
            path = spec[len("file:") :].strip()
            if path:
                new_sink = _FileSink(path)
        elif spec.startswith("tcp:"):
            address = spec[len("tcp:") :].strip()
            host, port_text = address.rsplit(":", 1)
            new_sink = _TcpSink(host, int(port_text))
    except Exception as exc:
        logger.warning("Failed to configure event sink %r: %s", spec, exc)
        new_sink = None

    with _lock:
        old_sink = _sink
        _sink = new_sink

    if old_sink is not None:
        try:
            old_sink.close()
        except Exception as exc:
            logger.debug("Ignoring error closing previous event sink: %s", exc)


def emit(event: StageEvent) -> None:
    global _last_emit_error_log
    with _lock:
        sink = _sink
        if sink is None:
            return
        try:
            sink.write(event)
        except Exception as exc:
            now = time.monotonic()
            if now - _last_emit_error_log >= _ERROR_LOG_INTERVAL_S:
                _last_emit_error_log = now
                logger.warning("Event sink write failed: %s", exc)


def get_memory_events() -> list[StageEvent]:
    with _lock:
        return [
            {
                "ts": event["ts"],
                "job_id": event["job_id"],
                "run_id": event.get("run_id", ""),
                "video": event["video"],
                "stage": event["stage"],
                "phase": event["phase"],
                "extra": dict(event.get("extra", {})),
            }
            for event in _memory_events
        ]
