from __future__ import annotations

import json
import math
import re
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
    frame_weights: list[float] | None = None
    boundary_metadata: dict[str, Any] | None = None


DEFAULT_TRAINABLE_LABEL_QUALITIES = frozenset({"supervised", "teacher_agree", "negative"})


@dataclass(frozen=True)
class AudioAudit:
    audio_id: str
    source: str
    duration_s: float
    sample_rate: int
    rms_dbfs: float
    head_rms_dbfs: float
    tail_rms_dbfs: float
    head_silence_s: float
    tail_silence_s: float
    text_chars: int
    text: str


def normalize_audio_16k_mono(
    audio: np.ndarray | Sequence[float],
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if int(sample_rate) != 16000:
        from scipy import signal

        divisor = math.gcd(int(sample_rate), 16000)
        samples = signal.resample_poly(
            samples,
            16000 // divisor,
            int(sample_rate) // divisor,
        ).astype("float32", copy=False)
    return np.ascontiguousarray(samples, dtype=np.float32), 16000


def decode_audio_bytes_16k_mono(audio_bytes: bytes | bytearray) -> tuple[np.ndarray, int]:
    import io

    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return normalize_audio_16k_mono(data, int(sample_rate))


def stable_hf_audio_id(*, dataset_name: str, split: str, index: int) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("_")
    split_key = re.sub(r"[^A-Za-z0-9._-]+", "_", split).strip("_")
    return f"{prefix}-{split_key}-{index:06d}"


def sample_hf_audio_16k_mono(example: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    audio_obj = example.get("ogg") or example.get("audio")
    if isinstance(audio_obj, (bytes, bytearray)):
        return decode_audio_bytes_16k_mono(audio_obj)
    if isinstance(audio_obj, Mapping):
        audio_bytes = audio_obj.get("bytes")
        if isinstance(audio_bytes, (bytes, bytearray)):
            return decode_audio_bytes_16k_mono(audio_bytes)
        array = audio_obj.get("array")
        sample_rate = int(audio_obj.get("sampling_rate") or 16000)
        if array is None:
            raise ValueError("audio sample has no bytes or array")
        return normalize_audio_16k_mono(np.asarray(array, dtype=np.float32), sample_rate)
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = getattr(samples, "data")
        sample_rate = int(getattr(samples, "sample_rate"))
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        samples_array = np.asarray(data, dtype=np.float32)
        if samples_array.ndim == 2 and samples_array.shape[0] <= 8:
            samples_array = samples_array.mean(axis=0, dtype=np.float32)
        return normalize_audio_16k_mono(samples_array, sample_rate)
    raise ValueError("expected an 'ogg' or 'audio' field decoded by datasets")


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
            elif hasattr(item, "start") and hasattr(item, "end"):
                raw_start = getattr(item, "start")
                raw_end = getattr(item, "end")
                raw_score = getattr(item, "score", None)
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
    if duration_s <= 0:
        return 0
    return int(math.ceil((duration_s / frame_hop_s) - 1e-9))


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


def build_weighted_teacher_record(
    *,
    audio_id: str,
    source: str,
    duration_s: float,
    text: str = "",
    teacher_segments: Mapping[str, Iterable[TeacherSegment | Mapping[str, Any] | Sequence[float]]],
    frame_hop_s: float = 0.02,
    min_speech_teachers: int = 2,
    min_negative_gap_s: float = 0.5,
    boundary_dilation_s: float = 0.08,
    positive_weight: float = 0.5,
    negative_weight: float = 0.5,
) -> LabelRecord:
    """Build pseudo-labels with explicit ignore frames via frame_weights.

    Frames where enough teachers agree on speech become positive. Long gaps
    where no teacher detects speech become negative only when the clip also has
    an agreed speech region; the rest stays ignored with weight 0.
    """
    if min_speech_teachers <= 0:
        raise ValueError("min_speech_teachers must be positive")
    normalized = {
        str(name): clamp_segments(segments, duration_s=duration_s)
        for name, segments in teacher_segments.items()
    }
    count = frame_count(duration_s, frame_hop_s)
    votes = np.zeros(count, dtype=np.int32)
    for segments in normalized.values():
        labels = segments_to_frame_labels(
            segments,
            duration_s=duration_s,
            frame_hop_s=frame_hop_s,
        )
        votes[: len(labels)] += np.asarray(labels, dtype=np.int32)

    positive_mask = votes >= int(min_speech_teachers)
    speech_frames = positive_mask.astype(np.int32).tolist()
    weights = np.zeros(count, dtype=np.float32)
    weights[positive_mask] = max(0.0, float(positive_weight))

    if positive_mask.any():
        min_gap_frames = max(1, int(math.ceil(max(0.0, min_negative_gap_s) / frame_hop_s)))
        for start, end in _true_runs(votes == 0):
            if end - start >= min_gap_frames:
                weights[start:end] = max(0.0, float(negative_weight))

    if boundary_dilation_s > 0.0 and positive_mask.any():
        dilation_frames = int(math.ceil(boundary_dilation_s / frame_hop_s))
        for start, end in _true_runs(positive_mask):
            weights[max(0, start - dilation_frames) : min(count, start + dilation_frames)] = 0.0
            weights[max(0, end - dilation_frames) : min(count, end + dilation_frames)] = 0.0

    any_teacher_speech = bool((votes > 0).any())
    if not positive_mask.any() and not any_teacher_speech and not text.strip():
        weights[:] = max(0.0, float(negative_weight))

    weighted_positive = bool(np.logical_and(positive_mask, weights > 0.0).any())
    weighted_negative = bool(np.logical_and(np.logical_not(positive_mask), weights > 0.0).any())
    if weighted_positive:
        label_quality = "teacher_agree"
    elif any_teacher_speech or text.strip():
        label_quality = "teacher_conflict"
    elif weighted_negative:
        label_quality = "negative"
    else:
        label_quality = "teacher_conflict"

    return LabelRecord(
        audio_id=audio_id,
        source=source,
        duration_s=float(duration_s),
        text=text,
        teacher_segments=normalized,
        frame_hop_s=float(frame_hop_s),
        speech_frames=speech_frames,
        label_quality=label_quality,
        frame_weights=[float(value) for value in weights.tolist()],
    )


def with_frame_weights(
    record: LabelRecord,
    frame_weights: Sequence[float] | None,
) -> LabelRecord:
    weights = None if frame_weights is None else [float(value) for value in frame_weights]
    return LabelRecord(
        audio_id=record.audio_id,
        source=record.source,
        duration_s=record.duration_s,
        text=record.text,
        teacher_segments=record.teacher_segments,
        frame_hop_s=record.frame_hop_s,
        speech_frames=record.speech_frames,
        label_quality=record.label_quality,
        frame_weights=weights,
        boundary_metadata=record.boundary_metadata,
    )


def effective_frame_weights(record: LabelRecord) -> list[float]:
    if record.frame_weights is None:
        return [1.0] * len(record.speech_frames)
    return [float(value) for value in record.frame_weights]


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


def build_negative_record(
    *,
    audio_id: str,
    source: str,
    duration_s: float,
    text: str = "",
    frame_hop_s: float = 0.02,
) -> LabelRecord:
    return LabelRecord(
        audio_id=audio_id,
        source=source,
        duration_s=float(duration_s),
        text=text,
        teacher_segments={"negative": []},
        frame_hop_s=float(frame_hop_s),
        speech_frames=[0] * frame_count(duration_s, frame_hop_s),
        label_quality="negative",
    )


def build_weak_positive_record(
    *,
    audio_id: str,
    source: str,
    duration_s: float,
    text: str = "",
    frame_hop_s: float = 0.02,
    teacher_name: str = "weak_positive",
    trim_head_s: float = 0.0,
    trim_tail_s: float = 0.0,
) -> LabelRecord:
    start = max(0.0, min(float(trim_head_s), duration_s))
    end = max(start, duration_s - max(0.0, float(trim_tail_s)))
    segments = [TeacherSegment(start=start, end=end, score=1.0)] if end > start else []
    return build_teacher_record(
        audio_id=audio_id,
        source=source,
        duration_s=duration_s,
        text=text,
        teacher_segments={teacher_name: segments},
        frame_hop_s=frame_hop_s,
        min_speech_teachers=1,
    )


def is_default_trainable(record: LabelRecord) -> bool:
    return record.label_quality in DEFAULT_TRAINABLE_LABEL_QUALITIES


def default_trainable_records(records: Iterable[LabelRecord]) -> list[LabelRecord]:
    return [record for record in records if is_default_trainable(record)]


def audit_audio(
    *,
    audio_id: str,
    source: str,
    audio: np.ndarray,
    sample_rate: int,
    text: str = "",
    edge_window_s: float = 0.5,
) -> AudioAudit:
    samples, normalized_sample_rate = normalize_audio_16k_mono(audio, sample_rate)
    duration_s = float(len(samples) / normalized_sample_rate) if normalized_sample_rate > 0 else 0.0
    edge_samples = max(1, int(edge_window_s * normalized_sample_rate)) if normalized_sample_rate > 0 else 1
    head_silence_s, tail_silence_s = _edge_silence_s(
        samples,
        sample_rate=normalized_sample_rate,
    )
    return AudioAudit(
        audio_id=audio_id,
        source=source,
        duration_s=duration_s,
        sample_rate=normalized_sample_rate,
        rms_dbfs=_rms_dbfs(samples),
        head_rms_dbfs=_rms_dbfs(samples[:edge_samples]),
        tail_rms_dbfs=_rms_dbfs(samples[-edge_samples:]),
        head_silence_s=head_silence_s,
        tail_silence_s=tail_silence_s,
        text_chars=len(text or ""),
        text=text or "",
    )


def label_record_to_dict(record: LabelRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["teacher_segments"] = {
        name: [asdict(segment) for segment in segments]
        for name, segments in record.teacher_segments.items()
    }
    if record.frame_weights is None:
        payload.pop("frame_weights", None)
    if record.boundary_metadata is None:
        payload.pop("boundary_metadata", None)
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
        frame_weights=(
            None
            if payload.get("frame_weights") is None
            else [float(value) for value in list(payload.get("frame_weights") or [])]
        ),
        boundary_metadata=(
            None
            if payload.get("boundary_metadata") is None
            else dict(payload.get("boundary_metadata") or {})
        ),
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


def _edge_silence_s(
    samples: np.ndarray,
    *,
    sample_rate: int,
    threshold_dbfs: float = -45.0,
    frame_hop_s: float = 0.02,
) -> tuple[float, float]:
    if samples.size == 0 or sample_rate <= 0:
        return 0.0, 0.0
    frame_size = max(1, int(sample_rate * frame_hop_s))
    frame_total = int(math.ceil(samples.size / frame_size))
    silent_frames: list[bool] = []
    for index in range(frame_total):
        start = index * frame_size
        end = min(samples.size, start + frame_size)
        silent_frames.append(_rms_dbfs(samples[start:end]) <= threshold_dbfs)

    head_frames = 0
    for value in silent_frames:
        if not value:
            break
        head_frames += 1

    tail_frames = 0
    for value in reversed(silent_frames):
        if not value:
            break
        tail_frames += 1

    duration_s = samples.size / sample_rate
    return (
        min(duration_s, head_frames * frame_hop_s),
        min(duration_s, tail_frames * frame_hop_s),
    )


def _true_runs(values: Sequence[bool] | np.ndarray) -> list[tuple[int, int]]:
    array = np.asarray(values, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(array):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(array.size)))
    return runs
