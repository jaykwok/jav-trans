from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TeacherSegment:
    start: float
    end: float
    score: float | None = None


@dataclass(frozen=True)
class LabelRecord:
    audio_id: str
    source: str
    duration_s: float
    text: str
    teacher_segments: dict[str, list[TeacherSegment]]
    frame_hop_s: float
    speech_frames: list[int]
    label_quality: str


@dataclass(frozen=True)
class AudioAudit:
    audio_id: str
    source: str
    duration_s: float
    sample_rate: int
    rms_dbfs: float
    head_rms_dbfs: float
    tail_rms_dbfs: float
    text_chars: int
    text: str


def clamp_segments(
    segments: Iterable[TeacherSegment | Mapping[str, Any] | Sequence[float]],
    *,
    duration_s: float,
) -> list[TeacherSegment]:
    clamped: list[TeacherSegment] = []
    for item in segments:
        try:
            if isinstance(item, TeacherSegment):
                raw_start, raw_end, score = item.start, item.end, item.score
            elif isinstance(item, Mapping):
                raw_start = item.get("start", 0.0)
                raw_end = item.get("end", 0.0)
                raw_score = item.get("score")
                score = None if raw_score is None else float(raw_score)
            else:
                raw_start, raw_end = item[0], item[1]
                score = None
            start = max(0.0, min(float(raw_start), duration_s))
            end = max(0.0, min(float(raw_end), duration_s))
        except (TypeError, ValueError, IndexError):
            continue
        if end <= start:
            continue
        clamped.append(TeacherSegment(start=start, end=end, score=score))
    return clamped


def frame_count(duration_s: float, frame_hop_s: float) -> int:
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    if frame_hop_s <= 0:
        raise ValueError("frame_hop_s must be positive")
    return int(math.ceil(duration_s / frame_hop_s)) if duration_s > 0 else 0


def segments_to_frame_labels(
    segments: Iterable[TeacherSegment | Mapping[str, Any] | Sequence[float]],
    *,
    duration_s: float,
    frame_hop_s: float,
) -> list[int]:
    count = frame_count(duration_s, frame_hop_s)
    labels = [0] * count
    for segment in clamp_segments(segments, duration_s=duration_s):
        start_index = max(0, int(math.floor(segment.start / frame_hop_s)))
        end_index = min(count, int(math.ceil(segment.end / frame_hop_s)))
        for index in range(start_index, end_index):
            frame_start = index * frame_hop_s
            frame_end = min(duration_s, frame_start + frame_hop_s)
            if frame_end > segment.start and segment.end > frame_start:
                labels[index] = 1
    return labels


def build_teacher_record(
    *,
    audio_id: str,
    source: str,
    duration_s: float,
    text: str = "",
    teacher_segments: Mapping[str, Iterable[TeacherSegment | Mapping[str, Any] | Sequence[float]]],
    frame_hop_s: float = 0.02,
    min_speech_teachers: int = 2,
) -> LabelRecord:
    if min_speech_teachers <= 0:
        raise ValueError("min_speech_teachers must be positive")
    normalized = {
        str(name): clamp_segments(segments, duration_s=duration_s)
        for name, segments in teacher_segments.items()
    }
    count = frame_count(duration_s, frame_hop_s)
    votes = [0] * count
    any_teacher_speech = False
    for segments in normalized.values():
        labels = segments_to_frame_labels(
            segments,
            duration_s=duration_s,
            frame_hop_s=frame_hop_s,
        )
        any_teacher_speech = any_teacher_speech or any(labels)
        for index, value in enumerate(labels):
            votes[index] += int(value)

    speech_frames = [1 if vote >= min_speech_teachers else 0 for vote in votes]
    if any(speech_frames):
        label_quality = "teacher_agree"
    elif any_teacher_speech:
        label_quality = "teacher_conflict"
    else:
        label_quality = "negative"
    return LabelRecord(
        audio_id=audio_id,
        source=source,
        duration_s=float(duration_s),
        text=text,
        teacher_segments=normalized,
        frame_hop_s=float(frame_hop_s),
        speech_frames=speech_frames,
        label_quality=label_quality,
    )


def build_supervised_record(
    *,
    audio_id: str,
    source: str,
    duration_s: float,
    text: str = "",
    speech_segments: Iterable[TeacherSegment | Mapping[str, Any] | Sequence[float]],
    frame_hop_s: float = 0.02,
) -> LabelRecord:
    segments = clamp_segments(speech_segments, duration_s=duration_s)
    return LabelRecord(
        audio_id=audio_id,
        source=source,
        duration_s=float(duration_s),
        text=text,
        teacher_segments={"supervised": segments},
        frame_hop_s=float(frame_hop_s),
        speech_frames=segments_to_frame_labels(
            segments,
            duration_s=duration_s,
            frame_hop_s=frame_hop_s,
        ),
        label_quality="supervised",
    )


def audit_audio(
    *,
    audio_id: str,
    source: str,
    audio: np.ndarray,
    sample_rate: int,
    text: str = "",
    edge_window_s: float = 0.5,
) -> AudioAudit:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1, dtype=np.float32)
    duration_s = float(len(samples) / sample_rate) if sample_rate > 0 else 0.0
    edge_samples = max(1, int(edge_window_s * sample_rate)) if sample_rate > 0 else 1
    return AudioAudit(
        audio_id=audio_id,
        source=source,
        duration_s=duration_s,
        sample_rate=int(sample_rate),
        rms_dbfs=_rms_dbfs(samples),
        head_rms_dbfs=_rms_dbfs(samples[:edge_samples]),
        tail_rms_dbfs=_rms_dbfs(samples[-edge_samples:]),
        text_chars=len(text or ""),
        text=text or "",
    )


def label_record_to_dict(record: LabelRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["teacher_segments"] = {
        name: [asdict(segment) for segment in segments]
        for name, segments in record.teacher_segments.items()
    }
    return payload


def label_record_from_dict(payload: Mapping[str, Any]) -> LabelRecord:
    teacher_segments = {
        str(name): [
            TeacherSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                score=None if item.get("score") is None else float(item["score"]),
            )
            for item in list(items or [])
        ]
        for name, items in dict(payload.get("teacher_segments") or {}).items()
    }
    return LabelRecord(
        audio_id=str(payload.get("audio_id") or ""),
        source=str(payload.get("source") or ""),
        duration_s=float(payload.get("duration_s") or 0.0),
        text=str(payload.get("text") or ""),
        teacher_segments=teacher_segments,
        frame_hop_s=float(payload.get("frame_hop_s") or 0.02),
        speech_frames=[int(value) for value in list(payload.get("speech_frames") or [])],
        label_quality=str(payload.get("label_quality") or ""),
    )


def write_jsonl(path: Path, rows: Iterable[LabelRecord | Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = label_record_to_dict(row) if isinstance(row, LabelRecord) else dict(row)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[LabelRecord]:
    rows: list[LabelRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(label_record_from_dict(json.loads(line)))
    return rows


def _rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64, copy=False)))))
    return float(20.0 * np.log10(max(rms, 1e-12)))
