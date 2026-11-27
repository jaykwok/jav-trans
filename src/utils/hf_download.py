"""Refresh Hub connections between downloads without closing an active transfer."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import threading

from core.config import NETWORK_PROXY_ENV_KEYS

logger = logging.getLogger(__name__)

_SESSION_ENV_KEYS = NETWORK_PROXY_ENV_KEYS + ("HF_ENDPOINT", "HF_TOKEN_PATH")

_LOCK = threading.Lock()
_COND = threading.Condition(_LOCK)
_active_readers = 0
_writer_active = False
_waiting_writers = 0
_thread_reader_depth: dict[int, int] = {}
_proxy_signature: tuple[str, ...] | None = None


def _current_session_signature() -> tuple[str, ...]:
    return tuple(os.environ.get(key, "") for key in _SESSION_ENV_KEYS)


def _abort_xet_session_best_effort() -> None:
    try:
        from huggingface_hub.utils._xet import abort_xet_session
    except ImportError:
        return
    try:
        abort_xet_session()
    except Exception:
        logger.debug("Ignoring error aborting idle Xet session", exc_info=True)


def _reset_hub_sessions(close_session) -> None:
    # Xet keeps a separate Rust connection pool. Holding the exclusive writer
    # phase guarantees aborting that idle session cannot interrupt a transfer.
    _abort_xet_session_best_effort()
    close_session()


@contextmanager
def download_session():
    """Use current proxy/endpoint/token settings for the next download operation.

    Concurrent downloads sharing the same environment signature proceed in
    parallel (reader lock). When proxy, ``HF_ENDPOINT``, or ``HF_TOKEN_PATH``
    changes, the caller waits for active downloads to finish before taking an
    exclusive writer phase to reset the shared HTTP and Xet sessions.
    """
    global _active_readers, _proxy_signature, _waiting_writers, _writer_active
    from huggingface_hub.utils import close_session

    tid = threading.get_ident()
    acquired_outer = False
    with _COND:
        if _thread_reader_depth.get(tid, 0) > 0:
            _thread_reader_depth[tid] += 1
        else:
            while True:
                signature = _current_session_signature()
                if (
                    _proxy_signature == signature
                    and not _writer_active
                    and _waiting_writers == 0
                ):
                    _active_readers += 1
                    _thread_reader_depth[tid] = 1
                    acquired_outer = True
                    break
                if _proxy_signature != signature:
                    _waiting_writers += 1
                    try:
                        while _writer_active or _active_readers > 0:
                            _COND.wait()
                        _writer_active = True
                    finally:
                        _waiting_writers -= 1
                    try:
                        latest_signature = _current_session_signature()
                        if latest_signature != _proxy_signature:
                            _reset_hub_sessions(close_session)
                            _proxy_signature = latest_signature
                    except BaseException:
                        _writer_active = False
                        _COND.notify_all()
                        raise
                    _writer_active = False
                    _active_readers += 1
                    _thread_reader_depth[tid] = 1
                    acquired_outer = True
                    _COND.notify_all()
                    break
                _COND.wait()
    try:
        yield
    finally:
        with _COND:
            depth = _thread_reader_depth.get(tid, 0)
            if depth <= 1:
                _thread_reader_depth.pop(tid, None)
                if acquired_outer:
                    _active_readers = max(0, _active_readers - 1)
                    if _active_readers == 0:
                        _COND.notify_all()
            else:
                _thread_reader_depth[tid] = depth - 1
