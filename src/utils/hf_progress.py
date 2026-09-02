from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub.utils.tqdm import tqdm as _base_tqdm

# 模块级 fallback，跨线程可见（download 是串行的，无并发风险）
_override_job_id: str = ""


def set_current_job_id(job_id: str) -> None:
    global _override_job_id
    _override_job_id = job_id


def _event_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _current_job_id() -> str:
    # 先试线程本地（主线程调用路径有效）
    try:
        from core import events

        private_getter = getattr(events, "_current_job_id", None)
        if callable(private_getter):
            tid = str(private_getter() or "")
            if tid:
                return tid
    except Exception:
        pass
    # fallback：模块级变量（download worker 线程走这里）
    return _override_job_id


def _current_video() -> str:
    try:
        from core import events

        thread_local = getattr(events, "_thread_local", None)
        return str(getattr(thread_local, "video", "") or "")
    except Exception:
        return ""


def _emit(phase: str, extra: dict[str, Any]) -> None:
    try:
        from core import events

        events.emit(
            {
                "ts": _event_ts(),
                "job_id": _current_job_id(),
                "video": _current_video(),
                "stage": "model_download",
                "phase": phase,
                "extra": extra,
            }
        )
    except Exception:
        pass


def _display_file(desc: Any) -> str:
    raw = str(desc or "").strip()
    if not raw:
        return ""
    return Path(raw.replace("(＃)", "")).name or raw


def _mb(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / (1024.0 * 1024.0), 2)
    except (TypeError, ValueError):
        return None


class HfDownloadProgressTqdm(_base_tqdm):  # type: ignore[misc, valid-type]
    """Passed as ``tqdm_class=`` to snapshot_download/hf_hub_download so every
    per-file progress bar emits a ``model_download`` stage event instead of
    only printing to stdout."""

    def __init__(self, *args, **kwargs):
        self._hf_progress_unit = kwargs.get("unit", "it")
        self._hf_progress_file = _display_file(kwargs.get("desc", ""))
        self._hf_progress_total = kwargs.get("total")
        self._hf_progress_started = time.perf_counter()
        self._hf_progress_last_emit = 0.0
        self._hf_progress_last_pct = -1
        self._hf_progress_finished = False
        super().__init__(*args, **kwargs)
        if self._hf_should_emit:
            extra: dict[str, Any] = {"file": self._hf_progress_file}
            size_mb = _mb(self._hf_progress_total)
            if size_mb is not None:
                extra["size_mb"] = size_mb
            _emit("start", extra)

    @property
    def _hf_should_emit(self) -> bool:
        return bool(self._hf_progress_file) and self._hf_progress_unit == "B"

    def update(self, n=1):
        result = super().update(n)
        self._hf_emit_progress()
        return result

    def close(self):
        try:
            return super().close()
        finally:
            if self._hf_should_emit and not self._hf_progress_finished:
                self._hf_progress_finished = True
                _emit(
                    "done",
                    {
                        "file": self._hf_progress_file,
                        "elapsed_s": round(
                            time.perf_counter() - self._hf_progress_started,
                            2,
                        ),
                    },
                )

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None and self._hf_should_emit:
            self._hf_progress_finished = True
            _emit(
                "error",
                {
                    "file": self._hf_progress_file,
                    "elapsed_s": round(
                        time.perf_counter() - self._hf_progress_started,
                        2,
                    ),
                    "error": str(exc_value or exc_type.__name__),
                },
            )
        return super().__exit__(exc_type, exc_value, traceback)

    def _hf_emit_progress(self) -> None:
        if not self._hf_should_emit or self._hf_progress_finished:
            return
        now = time.perf_counter()
        total = self.total or self._hf_progress_total
        current = float(getattr(self, "n", 0) or 0)
        pct: int | None = None
        if total:
            pct = max(0, min(100, int(current * 100 / float(total))))
            if pct == self._hf_progress_last_pct and now - self._hf_progress_last_emit < 1.0:
                return
            if pct != 100 and now - self._hf_progress_last_emit < 0.5:
                return
            self._hf_progress_last_pct = pct
        elif now - self._hf_progress_last_emit < 1.0:
            return
        self._hf_progress_last_emit = now

        extra: dict[str, Any] = {"file": self._hf_progress_file}
        if pct is not None:
            extra["pct"] = pct
        rate = None
        try:
            rate = self.format_dict.get("rate")
        except Exception:
            rate = None
        speed_mb = _mb(rate)
        if speed_mb is not None:
            extra["speed_mb"] = speed_mb
        _emit("progress", extra)


def tqdm_class() -> type:
    """The tqdm subclass to pass as ``tqdm_class=`` to huggingface_hub's
    ``snapshot_download``/``hf_hub_download``. Both accept the kwarg directly
    on the huggingface-hub version this project pins, so no monkeypatching of
    huggingface_hub internals or version-compatibility fallback is needed."""
    return HfDownloadProgressTqdm
