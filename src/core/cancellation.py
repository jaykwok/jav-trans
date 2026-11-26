"""The pipeline's cancellation signal, low enough for every stage to raise it.

`PipelineCancelledError` used to live in `main`, which sits *above* every stage
module. A stage helper that wants to stop on a cancel request - the media
subprocess runner, for one - cannot import it from there without a cycle, so it
either invented its own exception type (which the Web layer would report as a
failure rather than a cancellation) or had no way to stop at all. `main` keeps
re-exporting these names, so `main.PipelineCancelledError` stays the same class.
"""

from __future__ import annotations

import threading


class PipelineCancelledError(RuntimeError):
    pass


def cancel_requested(cancel_event: threading.Event | None) -> bool:
    try:
        return bool(cancel_event is not None and cancel_event.is_set())
    except Exception:
        return False


def raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_requested(cancel_event):
        raise PipelineCancelledError("任务已取消")


__all__ = [
    "PipelineCancelledError",
    "cancel_requested",
    "raise_if_cancelled",
]
