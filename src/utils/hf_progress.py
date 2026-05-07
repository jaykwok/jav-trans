from __future__ import annotations

import importlib
import inspect
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_install_lock = threading.Lock()
_installed = False
_original_tqdm: type | None = None
_progress_tqdm: type | None = None
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


def _make_progress_tqdm(base_tqdm: type) -> type:
    class HfDownloadProgressTqdm(base_tqdm):  # type: ignore[misc, valid-type]
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

    return HfDownloadProgressTqdm


def install() -> None:
    global _installed, _original_tqdm, _progress_tqdm
    if _installed:
        return
    with _install_lock:
        if _installed:
            return
        try:
            tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")
            base_tqdm = getattr(tqdm_module, "tqdm")
            _original_tqdm = base_tqdm
            _progress_tqdm = _make_progress_tqdm(base_tqdm)
            setattr(tqdm_module, "tqdm", _progress_tqdm)

            for module_name, attr_name in (
                ("huggingface_hub.file_download", "tqdm"),
                ("huggingface_hub._snapshot_download", "hf_tqdm"),
            ):
                try:
                    module = importlib.import_module(module_name)
                    setattr(module, attr_name, _progress_tqdm)
                except Exception:
                    continue
        except Exception:
            _progress_tqdm = None
        finally:
            _installed = True


def snapshot_download_kwargs(snapshot_download: Callable[..., Any]) -> dict[str, Any]:
    install()
    if _progress_tqdm is None:
        return {}
    try:
        parameters = inspect.signature(snapshot_download).parameters
    except (TypeError, ValueError):
        return {}
    if "tqdm_class" not in parameters:
        return {}
    return {"tqdm_class": _progress_tqdm}


class _FallbackToken:
    def __init__(self, file: str) -> None:
        self.file = file
        self.started = time.perf_counter()


def fallback_start(file: str) -> _FallbackToken:
    token = _FallbackToken(file)
    _emit("start", {"file": file})
    return token


def fallback_done(token: _FallbackToken) -> None:
    _emit(
        "done",
        {
            "file": token.file,
            "elapsed_s": round(time.perf_counter() - token.started, 2),
        },
    )


def fallback_error(token: _FallbackToken, exc: BaseException) -> None:
    _emit(
        "error",
        {
            "file": token.file,
            "elapsed_s": round(time.perf_counter() - token.started, 2),
            "error": str(exc),
        },
    )
