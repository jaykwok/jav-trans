from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from audio.loading import load_audio_16k_mono
from vad.fusionvad_ja.dataset import (
    LabelRecord,
    effective_frame_weights,
    frame_count,
    is_default_trainable,
    read_jsonl,
)


@dataclass(frozen=True)
class TrainingExample:
    audio_id: str
    source: str
    label_quality: str
    duration_s: float
    frame_hop_s: float
    audio_path: str
    label_index: int
    speech_frame_count: int
    frame_count: int


@dataclass(frozen=True)
class DryRunBatch:
    audio_id: str
    audio_path: str
    start_frame: int
    frame_count: int
    audio_shape: tuple[int, ...]
    label_shape: tuple[int, ...]
    sample_rate: int
    speech_ratio: float


def load_manifest_audio_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list")
    mapping: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        audio_path = row.get("audio")
        if not audio_path:
            continue
        keys = [
            row.get("audio_id"),
            row.get("input"),
            Path(str(audio_path)).stem,
        ]
        input_value = row.get("input")
        if input_value:
            keys.append(Path(str(input_value)).stem)
        for key in keys:
            if key:
                mapping[str(key)] = str(audio_path)
    return mapping


def resolve_audio_path(
    record: LabelRecord,
    *,
    manifest_audio_map: Mapping[str, str],
    audio_root: Path | None,
    extension_hints: Iterable[str],
) -> Path | None:
    for key in (record.audio_id, f"{record.source}:{record.audio_id}"):
        value = manifest_audio_map.get(key)
        if value:
            return Path(value)
    candidate = Path(record.audio_id)
    if candidate.exists():
        return candidate
    if audio_root is None:
        return None
    for suffix in extension_hints:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        candidate = audio_root / f"{record.audio_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_training_examples(
    records: Iterable[LabelRecord],
    *,
    manifest_audio_map: Mapping[str, str],
    audio_root: Path | None = None,
    extension_hints: Iterable[str] = (".wav", ".flac", ".ogg", ".mp3", ".m4a"),
    trainable_only: bool = True,
) -> tuple[list[TrainingExample], list[dict[str, Any]]]:
    all_records = list(records)
    examples: list[TrainingExample] = []
    skipped: list[dict[str, Any]] = []
    for index, record in enumerate(all_records):
        if trainable_only and not is_default_trainable(record):
            continue
        expected_frames = frame_count(record.duration_s, record.frame_hop_s)
        if len(record.speech_frames) != expected_frames:
            skipped.append(
                {
                    "audio_id": record.audio_id,
                    "source": record.source,
                    "label_quality": record.label_quality,
                    "reason": "frame_count_mismatch",
                    "expected_frames": expected_frames,
                    "actual_frames": len(record.speech_frames),
                }
            )
            continue
        frame_weights = effective_frame_weights(record)
        if len(frame_weights) != expected_frames:
            skipped.append(
                {
                    "audio_id": record.audio_id,
                    "source": record.source,
                    "label_quality": record.label_quality,
                    "reason": "frame_weight_count_mismatch",
                    "expected_frames": expected_frames,
                    "actual_frames": len(frame_weights),
                }
            )
            continue
        audio_path = resolve_audio_path(
            record,
            manifest_audio_map=manifest_audio_map,
            audio_root=audio_root,
            extension_hints=extension_hints,
        )
        if audio_path is None:
            skipped.append(
                {
                    "audio_id": record.audio_id,
                    "source": record.source,
                    "label_quality": record.label_quality,
                    "reason": "missing_audio_path",
                }
            )
            continue
        examples.append(
            TrainingExample(
                audio_id=record.audio_id,
                source=record.source,
                label_quality=record.label_quality,
                duration_s=record.duration_s,
                frame_hop_s=record.frame_hop_s,
                audio_path=str(audio_path),
                label_index=index,
                speech_frame_count=sum(int(value) for value in record.speech_frames),
                frame_count=len(record.speech_frames),
            )
        )
    return examples, skipped


def dry_run_batches(
    records: list[LabelRecord],
    examples: list[TrainingExample],
    *,
    window_s: float = 2.0,
    max_batches: int = 8,
) -> list[DryRunBatch]:
    batches: list[DryRunBatch] = []
    for example in examples:
        if len(batches) >= max_batches:
            break
        record = records[example.label_index]
        audio, sample_rate = load_audio_16k_mono(example.audio_path)
        window_samples = max(1, int(round(window_s * sample_rate)))
        window_frames = max(1, int(math.ceil(window_s / record.frame_hop_s)))
        audio_window = _pad_or_trim(audio, window_samples)
        label_window = _pad_or_trim_labels(record.speech_frames, window_frames)
        batches.append(
            DryRunBatch(
                audio_id=example.audio_id,
                audio_path=example.audio_path,
                start_frame=0,
                frame_count=window_frames,
                audio_shape=tuple(audio_window.shape),
                label_shape=tuple(label_window.shape),
                sample_rate=sample_rate,
                speech_ratio=float(np.mean(label_window)) if label_window.size else 0.0,
            )
        )
    return batches


def write_training_manifest(
    *,
    path: Path,
    examples: Iterable[TrainingExample],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n")


def write_dry_run(path: Path, batches: Iterable[DryRunBatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(batch) for batch in batches], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_label_records(path: Path) -> list[LabelRecord]:
    return read_jsonl(path)


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] >= length:
        return np.ascontiguousarray(audio[:length], dtype=np.float32)
    padded = np.zeros(length, dtype=np.float32)
    padded[: audio.shape[0]] = audio
    return padded


def _pad_or_trim_labels(labels: list[int], length: int) -> np.ndarray:
    values = np.asarray(labels[:length], dtype=np.float32)
    if values.shape[0] >= length:
        return values
    padded = np.zeros(length, dtype=np.float32)
    padded[: values.shape[0]] = values
    return padded
