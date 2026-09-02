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


def propagate_job_id_to_current_thread(job_id: str) -> None:
    """Re-establish job_id on a worker thread a pool just spun up.

    `core.events`'s job_id is thread-local by design, which a fresh
    ``ThreadPoolExecutor`` worker (e.g. the batch-translation pool in
    llm/engine.py) does not inherit from whoever submitted the work -- unlike
    the module-level ``_override_job_id`` fallback above, a thread-local is
    the right tool here because concurrent jobs can each have a translation
    pool active at once (see test_pipeline_workers_overlap), and a shared
    global would race between them. Call this as the first thing inside a
    callable crossing that boundary, with the job_id captured on the
    correctly-tagged thread that submitted it.
    """
    try:
        from core import events

        events.set_current_job_id(job_id)
    except Exception:
        pass


def _event_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def current_job_id() -> str:
    """The job_id this module would currently emit events under.

    Public so a caller about to hand work to a new thread (e.g.
    translator.py, before submitting to a ThreadPoolExecutor) can capture it
    and hand it to `propagate_job_id_to_current_thread` on the other side.
    """
    return _current_job_id()


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


# huggingface_hub's xet_get hard-codes these two suffixes onto its bars'
# `desc` (see file_download.py's `reconstruction_desc`/`transfer_desc`);
# strip them so the label shown is the plain filename, matching every other
# download path.
_XET_DESC_SUFFIXES = (": reconstructing file", ": downloading bytes")


def _display_file(desc: Any) -> str:
    raw = str(desc or "").strip()
    if not raw:
        return ""
    for suffix in _XET_DESC_SUFFIXES:
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
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
    progress bar emits a ``model_download`` stage event instead of only
    printing to stdout.

    huggingface_hub drives two progress bars for one logical download, not
    one -- but which twin is the reliable one depends on how the pair was
    built, so this class picks per exact bar name rather than by one blanket
    rule:

    * ``snapshot_download`` (many files, name ``"huggingface_hub
      .snapshot_download"`` + ``.transfer``): the reconstruction bar's total
      is the real, stable file size aggregated across the snapshot; the
      ``.transfer`` twin's total is not a real denominator -- huggingface_hub's
      own ``_update_transfer_bar`` keeps re-inflating it by 1.25x as more bytes
      arrive, "since network bytes are hard to predict (dedup/compression)".
      Emitting both made the UI bar swing between two independent numbers for
      the same download, so the ``.transfer`` twin is suppressed.
    * a standalone Xet-accelerated ``hf_hub_download`` (one file, name
      ``"huggingface_hub.xet_get"`` + ``.transfer``): both bars are built with
      the *same* real total (``expected_size``, known upfront from the HTTP
      HEAD) -- but huggingface_hub's own docstring says "reconstruction lags
      behind transfer", and reconstruction only advances in batches as chunks
      are flushed to disk. Suppressing ``.transfer`` here left the UI showing
      0% for most of a multi-GB download while the network was actually
      near-saturated, so it is the reconstruction bar that gets suppressed
      instead and the (here, reliable) ``.transfer`` twin is shown.
    """

    _AGGREGATE_LABELS = {"huggingface_hub.snapshot_download": "模型文件"}
    _SUPPRESSED_NAMES = {
        "huggingface_hub.snapshot_download.transfer",
        "huggingface_hub.xet_get",
    }

    def __init__(self, *args, **kwargs):
        self._hf_progress_unit = kwargs.get("unit", "it")
        self._hf_progress_name = str(kwargs.get("name") or "")
        self._hf_progress_is_suppressed = self._hf_progress_name in self._SUPPRESSED_NAMES
        label = self._AGGREGATE_LABELS.get(self._hf_progress_name)
        self._hf_progress_file = label or _display_file(kwargs.get("desc", ""))
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
        if self._hf_progress_is_suppressed:
            return False
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
        # snapshot_download's reconstruct total starts at 0 and grows as each
        # file registers, so the size shown at "start" (0) would otherwise
        # never update again -- re-report it as it settles.
        size_mb = _mb(total)
        if size_mb is not None:
            extra["size_mb"] = size_mb
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
