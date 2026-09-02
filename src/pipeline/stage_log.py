from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from core import events
from rich.table import Table


# The labels here must match what the ASR pipeline actually emits through
# `on_stage`. When the boundary chain was retired on 2026-07-31 its five labels
# stopped being emitted and `切分` took their place, so every one of these
# entries was matching nothing and the new stage was reported as nothing at all.
_ASR_PROGRESS_RE = re.compile(
    r"(?P<label>切分|音频切块|ASR 文本转写|字幕时间轴)"
    r"\s+(?P<current>\d+)/(?P<total>\d+)"
)
# gpu_worker.py's "still alive" ping while the worker sits idle (e.g. blocked
# on a model download). It echoes the *last real* stage message inside itself
# for diagnostics, which means _ASR_PROGRESS_RE matches it too -- without
# excluding it, every heartbeat tick got misread as a fresh stage-progress
# update, so the UI showed audio_chunking "progressing" once every
# ASR_STAGE_WORKER_HEARTBEAT_S (10s default) while it was actually just
# waiting on something else entirely.
ASR_STAGE_HEARTBEAT_PREFIX = "阶段心跳"
_STAGE_LOG_RE = re.compile(
    r"^stage_(?P<phase>start|done|skip|blocked|degraded)\s+(?P<stage>[A-Za-z0-9_]+)(?:\s+(?P<extra>.*))?$"
)
_ASR_STAGE_MAP = {
    "切分": "audio_chunking",
    "音频切块": "audio_chunk_export",
    "ASR 文本转写": "asr_text_transcribe",
    "字幕时间轴": "subtitle_timing",
}

_TIMING_SUMMARY_ROWS = (
    ("音频准备", "audio_prepare_s", "pipeline"),
    ("ASR Worker 启动与传输", "asr_worker_overhead_s", "derived"),
    ("静音分析与切块", "split_s", "asr"),
    ("ASR 模型加载", "asr_model_load_s", "asr"),
    ("ASR 文本转写", "asr_text_transcribe_s", "asr"),
    ("ASR 模型卸载", "asr_model_unload_s", "asr"),
    ("字幕时间轴", "alignment_s", "asr"),
    ("字幕分段", "subtitle_segment_s", "asr"),
    ("字幕 Cue Plan", "subtitle_cue_plan_s", "pipeline"),
    ("翻译上下文", "translation_context_s", "pipeline"),
    ("翻译", "translation_s", "pipeline"),
    ("输出写入", "write_output_s", "pipeline"),
    ("其他", "other_s", "derived"),
    ("总计", "pipeline_total_s", "pipeline"),
)

# Below this a residual is measurement noise between stage boundaries, not a
# stage anyone can act on, and printing it would only add a row of zeros.
_TIMING_RESIDUAL_MIN_S = 0.05


def _event_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _coerce_event_value(value: str) -> object:
    stripped = value.strip().rstrip(",")
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        if any(marker in stripped for marker in (".", "e", "E")):
            return float(stripped.rstrip("s"))
        return int(stripped)
    except ValueError:
        return stripped


def _parse_stage_extra(raw_extra: str | None) -> dict:
    extra: dict[str, object] = {}
    if not raw_extra:
        return extra
    for token in raw_extra.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key:
            continue
        extra[key] = _coerce_event_value(value)
    return extra


def _emit_stage_event(
    video_path: str | None,
    stage: str,
    phase: str,
    extra: dict | None = None,
) -> None:
    if phase not in {"start", "done", "skip", "blocked", "degraded", "progress"}:
        return
    video = (
        os.path.basename(video_path)
        if video_path
        else str(getattr(events._thread_local, "video", "") or "")
    )
    events.emit(
        {
            "ts": _event_ts(),
            "job_id": events._current_job_id(),
            "video": video,
            "stage": stage,
            "phase": phase,
            "extra": dict(extra or {}),
        }
    )


def _emit_stage_log_event(video_path: str | None, message: str) -> None:
    match = _STAGE_LOG_RE.match(message)
    if not match:
        return
    phase = match.group("phase")
    if phase == "blocked":
        phase = "blocked"
    _emit_stage_event(
        video_path,
        match.group("stage"),
        phase,
        _parse_stage_extra(match.group("extra")),
    )


def _parse_asr_stage_event(message: str) -> tuple[str, dict] | None:
    if message.startswith(ASR_STAGE_HEARTBEAT_PREFIX):
        return None
    match = _ASR_PROGRESS_RE.search(message)
    if not match:
        return None
    raw_label = match.group("label")
    stage = _ASR_STAGE_MAP.get(raw_label)
    if stage is None:
        return None
    current = int(match.group("current"))
    total = int(match.group("total"))
    extra = {
        "label": raw_label,
        "current": current,
        "total": total,
    }
    return stage, extra


def _log_stage(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.info(message)
    _emit_stage_log_event(None, message)


def _log_timing_snapshot(
    logger: logging.Logger | None,
    stage_timings: dict,
    asr_details: dict,
) -> None:
    if logger is None:
        return
    asr_stage_timings = asr_details.get("stage_timings", {}) if asr_details else {}
    labels = (
        ("audio_prepare_s", "audio_prepare"),
        ("audio_extract_s", "audio_extract"),
        ("split_s", "audio_chunking"),
        ("asr_model_load_s", "asr_model_load"),
        ("asr_text_transcribe_s", "asr_text_transcribe"),
        ("asr_model_unload_s", "asr_model_unload"),
        ("alignment_s", "subtitle_timing"),
        ("alignment_model_unload_s", "subtitle_timing_model_unload"),
        ("subtitle_segment_s", "subtitle_segment"),
        ("asr_alignment_total_s", "asr_subtitle_timing_total"),
        ("translation_handoff_snapshot_s", "translation_handoff_snapshot"),
        ("subtitle_cue_plan_s", "subtitle_cue_plan"),
        ("translation_context_s", "translation_context"),
        ("translation_s", "translation"),
        ("write_output_s", "write_output"),
        ("pipeline_total_s", "pipeline_total"),
    )
    for key, label in labels:
        value = stage_timings.get(key, asr_stage_timings.get(key))
        if value is not None:
            logger.info("timing %s=%.2fs", label, float(value))


def _format_asr_stage_label(raw_label: str) -> str:
    mapping = {
        "切分": "音频切分",
        "音频切块": "音频切块",
        "ASR 文本转写": "ASR 转写",
        "字幕时间轴": "字幕时间轴",
    }
    return mapping.get(raw_label, raw_label)


def _add_timing_row(table: Table, label: str, seconds: float | None) -> None:
    if seconds is None:
        return
    table.add_row(label, f"{seconds:.2f}s")


def _timing_summary_rows(stage_timings: dict, asr_details: dict) -> list[dict]:
    asr_stage_timings = asr_details.get("stage_timings", {}) if asr_details else {}
    asr_skipped = (
        "asr_alignment_total_s" in stage_timings
        and float(stage_timings.get("asr_alignment_total_s") or 0.0) == 0.0
    )
    rows: list[dict] = []
    for label, key, source in _TIMING_SUMMARY_ROWS:
        if source == "derived":
            # Filled in below, once the measured rows and the total are known.
            continue
        if source == "asr":
            value = 0.0 if asr_skipped else asr_stage_timings.get(key)
        else:
            value = stage_timings.get(key)
        if value is None:
            continue
        rows.append({"label": label, "key": key, "seconds": round(float(value), 2)})

    def _insert_before(anchor_key: str, label: str, key: str, seconds: float) -> None:
        entry = {"label": label, "key": key, "seconds": round(seconds, 2)}
        position = next(
            (index for index, row in enumerate(rows) if row["key"] == anchor_key),
            len(rows),
        )
        rows.insert(position, entry)

    # The ASR stage runs in a separate GPU-owning process. The pipeline's own
    # measurement of that window is longer than the sum the worker reports from
    # inside itself, and the difference - process startup, env handoff, moving
    # results back - is real time that belonged in no row.
    worker_window_s = stage_timings.get("asr_alignment_total_s")
    worker_inner_s = asr_stage_timings.get("asr_alignment_total_s")
    if not asr_skipped and worker_window_s is not None and worker_inner_s is not None:
        overhead = float(worker_window_s) - float(worker_inner_s)
        if overhead >= _TIMING_RESIDUAL_MIN_S:
            label = next(
                (
                    row_label
                    for row_label, row_key, _source in _TIMING_SUMMARY_ROWS
                    if row_key == "asr_worker_overhead_s"
                ),
                "ASR Worker 启动与传输",
            )
            # Ahead of the stages it wraps, which is where it happens.
            _insert_before("split_s", label, "asr_worker_overhead_s", overhead)

    # The listed stages still do not tile the run: quality reporting, cleanup and
    # the audio handoff sit between them. Without this row the table shows parts
    # that visibly fail to add up to the total it prints right underneath, which
    # reads as a broken measurement rather than unlisted work.
    total_s = stage_timings.get("pipeline_total_s")
    if total_s is not None:
        measured = sum(
            float(row["seconds"]) for row in rows if row["key"] != "pipeline_total_s"
        )
        residual = float(total_s) - measured
        if residual >= _TIMING_RESIDUAL_MIN_S:
            label = next(
                (
                    row_label
                    for row_label, row_key, _source in _TIMING_SUMMARY_ROWS
                    if row_key == "other_s"
                ),
                "其他",
            )
            _insert_before("pipeline_total_s", label, "other_s", residual)
    return rows


def _print_timing_summary(console, stage_timings: dict, asr_details: dict) -> None:
    table = Table(title="阶段耗时", show_lines=False)
    table.add_column("阶段")
    table.add_column("耗时", justify="right")

    rows = _timing_summary_rows(stage_timings, asr_details)
    for row in rows:
        _add_timing_row(table, str(row["label"]), float(row["seconds"]))

    console.print(table)
    _emit_stage_event(
        None,
        "timing_summary",
        "done",
        {
            "title": "阶段耗时",
            "rows": rows,
            "total_s": next(
                (
                    row["seconds"]
                    for row in rows
                    if row["key"] == "pipeline_total_s"
                ),
                None,
            ),
        },
    )
