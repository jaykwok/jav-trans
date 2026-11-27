"""Per-run isolated FileHandler logger lifecycle for ASR and translation stages."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from core import events
from core.job_context import JobContext
from pipeline.ids import sanitize_job_id

if TYPE_CHECKING:
    from pipeline.artifacts import AsrArtifacts

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def _new_isolated_logger(name: str) -> logging.Logger:
    """Create a standalone Logger instance outside ``logging.Logger.manager``.

    Global ``logging.getLogger(name)`` keeps every per-job name alive forever
    in ``loggerDict`` and caused concurrent retries with matching timestamp/stem
    to share a logger and close each other's active ``FileHandler``.
    """
    logger = logging.Logger(name, level=logging.INFO)
    logger.propagate = False
    return logger


def run_log_dir(ctx: JobContext, *, project_root: Path) -> Path:
    raw = (ctx.run_log_dir or "./logs/jobs").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def run_log_component(value: str, *, max_chars: int) -> str:
    safe = sanitize_job_id((value or "run").replace(" ", "_")) or "run"
    if len(safe) <= max_chars:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8", errors="ignore")).hexdigest()[:8]
    keep = max(1, max_chars - len(digest) - 1)
    return f"{safe[:keep]}-{digest}"


def setup_run_logger(
    job_id: str,
    backend_label: str,
    ctx: JobContext,
    *,
    project_root: Path,
    run_log_root: Path | None = None,
) -> tuple[logging.Logger, Path]:
    resolved_root = (
        run_log_root
        if run_log_root is not None
        else run_log_dir(ctx, project_root=project_root)
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # Leave headroom below legacy Windows MAX_PATH. The root can itself be
    # user-configured and deep, so fixed per-component limits are insufficient.
    fixed_chars = len(str(resolved_root)) + 2 + len(stamp) + 1 + len(".run.log")
    available_chars = max(24, 240 - fixed_chars)
    job_chars = min(48, max(12, available_chars - 12))
    backend_chars = min(40, max(12, available_chars - job_chars))
    safe_job_id = run_log_component(job_id, max_chars=job_chars)
    safe_backend = run_log_component(backend_label, max_chars=backend_chars)

    candidate_roots: list[Path] = [resolved_root]
    temp_dir_raw = str(getattr(ctx, "job_temp_dir", "") or "").strip()
    if temp_dir_raw:
        fallback_root = Path(temp_dir_raw) / "log"
        if fallback_root != resolved_root:
            candidate_roots.append(fallback_root)

    logger = _new_isolated_logger(f"jav-sub.run.{job_id}.{stamp}")
    log_path = resolved_root / safe_job_id / f"{stamp}_{safe_backend}.run.log"
    last_error: OSError | None = None
    for root in candidate_roots:
        log_dir = root / safe_job_id
        candidate_path = log_dir / f"{stamp}_{safe_backend}.run.log"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(candidate_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(handler)
            return logger, candidate_path
        except OSError as exc:
            last_error = exc

    logging.getLogger(__name__).warning(
        "Could not open run log for job %s under %s: %s",
        job_id,
        resolved_root,
        last_error,
    )
    logger.addHandler(logging.NullHandler())
    return logger, log_path


def reopen_snapshot_run_logger(artifacts: AsrArtifacts) -> None:
    """Attach a fresh append-mode isolated run logger for the translation stage.

    Because ``run_asr_alignment`` always closes its ``FileHandler`` on return
    (and translation retries reload from a JSON snapshot), every translation
    stage opens the run log cleanly in append mode using its own non-global
    ``logging.Logger`` instance and closes it in ``finally``.
    """
    if isinstance(artifacts.logger, logging.Logger) or not artifacts.run_log_path:
        return
    log_path = Path(artifacts.run_log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    except OSError:
        # Diagnostics are not worth failing a run that is otherwise fine.
        return
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger = _new_isolated_logger(f"jav-sub.run.append.{log_path.stem}")
    logger.addHandler(handler)
    artifacts.logger = logger
    events._thread_local.run_logger = logger


def close_run_logger(logger: logging.Logger | None) -> None:
    if logger is None:
        return
    for handler in list(logger.handlers):
        try:
            handler.flush()
        except OSError:
            pass
        try:
            handler.close()
        except OSError:
            pass
        logger.removeHandler(handler)


def clear_thread_run_logger(logger: logging.Logger | None) -> None:
    if logger is not None and getattr(events._thread_local, "run_logger", None) is logger:
        try:
            delattr(events._thread_local, "run_logger")
        except AttributeError:
            pass


def close_artifacts_logger(artifacts: AsrArtifacts) -> None:
    logger = artifacts.logger if isinstance(artifacts.logger, logging.Logger) else None
    close_run_logger(logger)
    clear_thread_run_logger(logger)
    artifacts.logger = None
