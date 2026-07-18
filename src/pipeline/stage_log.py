from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from core import events
from rich.table import Table


_ASR_PROGRESS_RE = re.compile(
    r"(?P<label>边界缓存|语音岛检测|外边界精修|语义切分判断|"
    r"Pre-ASR CueQC|音频切块|ASR 文本转写|字幕时间轴)"
    r"\s+(?P<current>\d+)/(?P<total>\d+)"
)
_STAGE_LOG_RE = re.compile(
    r"^stage_(?P<phase>start|done|skip|blocked|degraded)\s+(?P<stage>[A-Za-z0-9_]+)(?:\s+(?P<extra>.*))?$"
)
_ASR_STAGE_MAP = {
    "边界缓存": "boundary_cache",
    "语音岛检测": "speech_island_scorer",
    "外边界精修": "outer_edge_refiner",
    "语义切分判断": "semantic_split_model",
    "Pre-ASR CueQC": "pre_asr_cueqc",
    "音频切块": "audio_chunk_export",
    "ASR 文本转写": "asr_text_transcribe",
    "字幕时间轴": "subtitle_timing",
}

_TIMING_SUMMARY_ROWS = (
    ("音频准备", "audio_prepare_s", "pipeline"),
    ("语音边界与音频切块", "split_s", "asr"),
    ("ASR 模型加载", "asr_model_load_s", "asr"),
    ("ASR 文本转写", "asr_text_transcribe_s", "asr"),
    ("ASR 模型卸载", "asr_model_unload_s", "asr"),
    ("字幕时间轴", "alignment_s", "asr"),
    ("字幕分段", "subtitle_segment_s", "asr"),
    ("字幕 Cue Plan", "subtitle_cue_plan_s", "pipeline"),
    ("翻译上下文", "translation_context_s", "pipeline"),
    ("翻译", "translation_s", "pipeline"),
    ("输出写入", "write_output_s", "pipeline"),
    ("总计", "pipeline_total_s", "pipeline"),
)


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
        "边界缓存": "边界缓存",
        "语音岛检测": "语音岛检测",
        "外边界精修": "外边界精修",
        "语义切分判断": "语义切分判断",
        "内部切点精修": "内部切点精修",
        "Pre-ASR CueQC": "Pre-ASR CueQC",
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
        if source == "asr":
            value = 0.0 if asr_skipped else asr_stage_timings.get(key)
        else:
            value = stage_timings.get(key)
        if value is None:
            continue
        rows.append({"label": label, "key": key, "seconds": round(float(value), 2)})
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
