from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from core import events
from rich.table import Table


_ASR_PROGRESS_RE = re.compile(
    r"(?P<label>音频切块|ASR 文本转写|Alignment 对齐|Alignment 局部细化|Alignment 降级重试)\s+(?P<current>\d+)/(?P<total>\d+)"
)
_STAGE_LOG_RE = re.compile(
    r"^stage_(?P<phase>start|done|skip|blocked|degraded)\s+(?P<stage>[A-Za-z0-9_]+)(?:\s+(?P<extra>.*))?$"
)
_ASR_STAGE_MAP = {
    "ASR 文本转写": "asr_text_transcribe",
    "Alignment 对齐": "alignment",
    "Alignment 局部细化": "alignment",
    "Alignment 降级重试": "alignment",
}


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
        ("asr_model_load_s", "asr_model_load"),
        ("asr_text_transcribe_s", "asr_text_transcribe"),
        ("asr_model_unload_s", "asr_model_unload"),
        ("alignment_s", "alignment"),
        ("alignment_model_unload_s", "alignment_model_unload"),
        ("subtitle_merge_s", "subtitle_merge"),
        ("asr_alignment_total_s", "asr_alignment_total"),
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
        "音频切块": "音频切块",
        "ASR 文本转写": "ASR 转写",
        "Alignment 对齐": "Alignment 对齐",
        "Alignment 局部细化": "Alignment 局部细化",
        "Alignment 降级重试": "Alignment 降级重试",
    }
    return mapping.get(raw_label, raw_label)


def _add_timing_row(table: Table, label: str, seconds: float | None) -> None:
    if seconds is None:
        return
    table.add_row(label, f"{seconds:.2f}s")


def _print_timing_summary(console, stage_timings: dict, asr_details: dict) -> None:
    table = Table(title="阶段耗时", show_lines=False)
    table.add_column("阶段")
    table.add_column("耗时", justify="right")

    asr_stage_timings = asr_details.get("stage_timings", {}) if asr_details else {}
    for label, key in (
        ("音频准备", "audio_prepare_s"),
        ("ASR 模型加载", "asr_model_load_s"),
        ("ASR 文本转写", "asr_text_transcribe_s"),
        ("ASR 模型卸载", "asr_model_unload_s"),
        ("Alignment 对齐", "alignment_s"),
        ("Alignment 模型卸载", "alignment_model_unload_s"),
        ("字幕合并", "subtitle_merge_s"),
        ("ASR+Alignment", "asr_alignment_total_s"),
        ("翻译上下文", "translation_context_s"),
        ("翻译", "translation_s"),
        ("输出写入", "write_output_s"),
        ("总计", "pipeline_total_s"),
    ):
        seconds = stage_timings.get(key, asr_stage_timings.get(key))
        _add_timing_row(table, label, seconds)

    console.print(table)
