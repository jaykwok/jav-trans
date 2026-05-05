from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict


class StageEvent(TypedDict):
    ts: str
    job_id: str
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
        self._socket = socket.create_connection((host, port), timeout=1.0)

    def write(self, event: StageEvent) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        self._socket.sendall(line.encode("utf-8"))

    def close(self) -> None:
        self._socket.close()


class _MemorySink:
    def __init__(self, events: list[StageEvent]) -> None:
        self._events = events

    def write(self, event: StageEvent) -> None:
        self._events.append(
            {
                "ts": str(event.get("ts", "")),
                "job_id": str(event.get("job_id", "")),
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


def _current_job_id() -> str:
    return str(getattr(_thread_local, "job_id", "") or "")


def set_current_job_id(job_id: str) -> None:
    _thread_local.job_id = job_id


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
    except Exception:
        new_sink = None

    with _lock:
        old_sink = _sink
        _sink = new_sink

    if old_sink is not None:
        try:
            old_sink.close()
        except Exception:
            pass


def emit(event: StageEvent) -> None:
    with _lock:
        sink = _sink
        if sink is None:
            return
        try:
            sink.write(event)
        except Exception:
            pass


def get_memory_events() -> list[StageEvent]:
    with _lock:
        return [
            {
                "ts": event["ts"],
                "job_id": event["job_id"],
                "video": event["video"],
                "stage": event["stage"],
                "phase": event["phase"],
                "extra": dict(event.get("extra", {})),
            }
            for event in _memory_events
        ]
