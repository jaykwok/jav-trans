#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.backends.qwen import QWEN_ASR_17B_REPO_ID  # noqa: E402
from audio.loading import load_audio_16k_mono  # noqa: E402
from boundary.ja import (  # noqa: E402
    LabelRecord,
    TeacherSegment,
    TrainingExample,
    build_supervised_record,
    frame_count,
    write_jsonl as write_label_jsonl,
    write_training_manifest,
)


DATASET_SCHEMA = "speech_boundary_ja_scorer_v6_native_dataset"
DEFAULT_SOURCE_SPECS = (
    "anime_nsfw=40=datasets/train/boundary-sources/japanese-anime-speech-v2-nsfw-60k/hf_audio_manifest.json",
    "anime_sfw=40=datasets/train/boundary-sources/japanese-anime-speech-v2-sfw-40k/hf_audio_manifest.json",
    "galgame=20=datasets/train/boundary-sources/galgame-asr-100k-ogg/manifest.jsonl",
)
MIX_KEYS = (
    "long_speech_chain",
    "positive_speech_timeline",
    "pure_hard_negative",
    "mixed_contrast",
    "split_stress",
)
DEFAULT_MIX = {
    "long_speech_chain": 0.0,
    "positive_speech_timeline": 0.40,
    "pure_hard_negative": 0.25,
    "mixed_contrast": 0.20,
    "split_stress": 0.15,
}
LONG_CHAIN_GAP_BUCKETS = (
    ("touch", 0.0, 0.0),
    ("micro_20_80ms", 0.02, 0.08),
    ("short_80_250ms", 0.08, 0.25),
    ("medium_250_600ms", 0.25, 0.60),
)
SAMPLE_RATE = 16000


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_display_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT)).replace("/", "\\")
    except ValueError:
        return str(raw)


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_stem(value: Any) -> str:
    raw = str(value or "sample")
    clean = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw)
    return clean.strip("._") or "sample"


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"manifest must be a JSON list or JSONL: {path}")
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest JSONL row must be an object: {path}:{line_number}")
        rows.append(dict(row))
    return rows


def parse_source_spec(spec: str) -> tuple[str, float, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"source spec must be name=weight=manifest: {spec!r}")
    name, weight, path = parts
    return safe_stem(name), float(weight), project_path(path)


def row_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def row_str_list(row: Mapping[str, Any], key: str) -> list[str]:
    raw = row.get(key) or []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item).strip()]
    return []


def candidate_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_name: str,
    source_manifest: Path,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("error"):
            skipped.append({"index": index, "source_group": source_name, "reason": "source_error"})
            continue
        audio_value = row.get("audio")
        if not audio_value:
            skipped.append({"index": index, "source_group": source_name, "reason": "missing_audio"})
            continue
        audio_path = project_path(str(audio_value))
        if not audio_path.exists():
            skipped.append(
                {
                    "index": index,
                    "source_group": source_name,
                    "reason": "audio_not_found",
                    "audio": repo_display_path(audio_path),
                }
            )
            continue
        duration_s = row_float(row, "duration_s", 0.0)
        if duration_s and duration_s < min_duration_s:
            skipped.append({"index": index, "source_group": source_name, "reason": "too_short"})
            continue
        if duration_s and max_duration_s > 0.0 and duration_s > max_duration_s:
            skipped.append({"index": index, "source_group": source_name, "reason": "too_long"})
            continue
        source_audio_id = str(row.get("audio_id") or audio_path.stem)
        item = dict(row)
        item["_source_group"] = source_name
        item["_source_manifest"] = repo_display_path(source_manifest)
        item["_audio_path"] = str(audio_path)
        item["_duration_s"] = duration_s
        item["_source_audio_id"] = source_audio_id
        item["_source_partition"] = partition_for_source_id(source_audio_id)
        candidates.append(item)
    return candidates, skipped


def load_source_groups(
    source_specs: Sequence[str],
    *,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[tuple[str, float, list[dict[str, Any]], Path]], list[dict[str, Any]]]:
    groups: list[tuple[str, float, list[dict[str, Any]], Path]] = []
    skipped: list[dict[str, Any]] = []
    for spec in source_specs:
        name, weight, path = parse_source_spec(spec)
        raw_rows = read_manifest(path)
        rows, row_skipped = candidate_rows(
            rows=raw_rows,
            source_name=name,
            source_manifest=path,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )
        skipped.extend(row_skipped)
        if not rows:
            raise ValueError(f"no valid source rows for {name}: {repo_display_path(path)}")
        groups.append((name, weight, rows, path))
    return groups, skipped


def load_negative_rows(paths: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw_path in paths:
        path = project_path(raw_path)
        for index, row in enumerate(read_manifest(path)):
            audio_value = row.get("audio")
            if not audio_value:
                skipped.append({"manifest": repo_display_path(path), "index": index, "reason": "missing_audio"})
                continue
            audio_path = project_path(str(audio_value))
            if not audio_path.exists():
                skipped.append(
                    {
                        "manifest": repo_display_path(path),
                        "index": index,
                        "reason": "audio_not_found",
                        "audio": repo_display_path(audio_path),
                    }
                )
                continue
            source_audio_id = str(row.get("audio_id") or audio_path.stem)
            item = dict(row)
            item["_audio_path"] = str(audio_path)
            item["_manifest"] = repo_display_path(path)
            item["_source_audio_id"] = source_audio_id
            item["_source_partition"] = partition_for_source_id(source_audio_id)
            rows.append(item)
    return rows, skipped


def allocate_counts(count: int, weights: Mapping[str, float]) -> dict[str, int]:
    total_weight = sum(max(0.0, float(value)) for value in weights.values())
    if count <= 0 or total_weight <= 0.0:
        raise ValueError("count and weights must be positive")
    raw = {name: count * max(0.0, float(weight)) / total_weight for name, weight in weights.items()}
    allocated = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = count - sum(allocated.values())
    order = sorted(raw, key=lambda name: (raw[name] - allocated[name], name), reverse=True)
    for name in order[:remaining]:
        allocated[name] += 1
    return allocated


def scorer_mix_from_args(args: argparse.Namespace) -> dict[str, float]:
    mix = dict(DEFAULT_MIX)
    for raw in getattr(args, "type_mix", []) or []:
        name, sep, value = str(raw).partition("=")
        if not sep:
            raise ValueError(f"--type-mix must be name=weight, got: {raw!r}")
        name = name.strip()
        if name not in MIX_KEYS:
            raise ValueError(f"unknown --type-mix name {name!r}; expected one of {', '.join(MIX_KEYS)}")
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"--type-mix weight must be numeric, got: {raw!r}") from exc
        if weight < 0.0:
            raise ValueError(f"--type-mix weight must be non-negative, got: {raw!r}")
        mix[name] = weight
    if sum(mix.values()) <= 0.0:
        raise ValueError("at least one scorer type mix weight must be positive")
    return {name: float(mix.get(name, 0.0)) for name in MIX_KEYS}


def choose_source_row(
    groups: Sequence[tuple[str, float, list[dict[str, Any]], Path]],
    rng: random.Random,
    *,
    source_partition: str | None = None,
) -> dict[str, Any]:
    eligible: list[tuple[str, float, list[dict[str, Any]], Path]] = []
    for name, weight, rows, path in groups:
        partition_rows = [
            row for row in rows if not source_partition or str(row.get("_source_partition") or "") == source_partition
        ]
        if partition_rows:
            eligible.append((name, weight, partition_rows, path))
    if not eligible:
        raise ValueError(f"no source rows available for source_partition={source_partition!r}")
    weights = [max(0.0, float(item[1])) for item in eligible]
    selected = rng.choices(range(len(eligible)), weights=weights, k=1)[0]
    _name, _weight, rows, _path = eligible[selected]
    return dict(rng.choice(rows))


def load_clip(
    row: Mapping[str, Any],
    *,
    rng: random.Random,
    min_speech_s: float,
    max_speech_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    audio_path = Path(str(row["_audio_path"]))
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected 16kHz audio after normalization, got {sample_rate}")
    if len(audio) < max(1, int(round(min_speech_s * SAMPLE_RATE))):
        raise ValueError(f"speech clip too short: {repo_display_path(audio_path)}")
    if max_speech_s > 0.0:
        max_samples = max(1, int(round(max_speech_s * SAMPLE_RATE)))
        if len(audio) > max_samples:
            start = rng.randint(0, len(audio) - max_samples)
            audio = np.ascontiguousarray(audio[start : start + max_samples], dtype=np.float32)
            source_start_s = start / SAMPLE_RATE
        else:
            source_start_s = 0.0
    else:
        source_start_s = 0.0
    source_audio_id = str(row.get("_source_audio_id") or row.get("audio_id") or audio_path.stem)
    return np.ascontiguousarray(audio, dtype=np.float32), {
        "source_group": str(row.get("_source_group") or row.get("source") or ""),
        "source_manifest": str(row.get("_source_manifest") or ""),
        "source_audio_id": source_audio_id,
        "source_partition": str(row.get("_source_partition") or partition_for_source_id(source_audio_id)),
        "source_audio_path": repo_display_path(audio_path),
        "source_start_s": source_start_s,
        "source_end_s": source_start_s + (len(audio) / SAMPLE_RATE),
        "source_text": str(row.get("text") or ""),
    }


def load_or_synthesize_negative(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    noise_rms: float,
    source_partition: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), {"negative_source": "empty"}
    eligible_rows = [
        row for row in rows if not source_partition or str(row.get("_source_partition") or "") == source_partition
    ]
    if eligible_rows:
        first = rng.randrange(len(eligible_rows))
        errors: list[str] = []
        for attempt in range(min(len(eligible_rows), 8)):
            row = eligible_rows[(first + attempt) % len(eligible_rows)]
            try:
                audio_path = Path(str(row["_audio_path"]))
                audio, sample_rate = load_audio_16k_mono(str(audio_path))
                if sample_rate != SAMPLE_RATE:
                    raise ValueError(f"expected 16kHz, got {sample_rate}")
                if len(audio) >= samples:
                    start = rng.randint(0, len(audio) - samples)
                    clipped = np.ascontiguousarray(audio[start : start + samples], dtype=np.float32)
                else:
                    repeat = int(math.ceil(samples / max(1, len(audio))))
                    clipped = np.tile(audio, repeat)[:samples].astype(np.float32, copy=False)
                    start = 0
                return clipped, {
                    "negative_source": "manifest",
                    "negative_audio_id": str(row.get("_source_audio_id") or row.get("audio_id") or audio_path.stem),
                    "negative_partition": str(row.get("_source_partition") or ""),
                    "negative_audio_path": repo_display_path(audio_path),
                    "negative_manifest": str(row.get("_manifest") or ""),
                    "negative_offset_s": start / SAMPLE_RATE,
                    "negative_source_label": str(row.get("source") or row.get("negative_source") or ""),
                    "negative_reason": str(row.get("reason") or row.get("manual_reason") or ""),
                    "negative_reason_tags": row_str_list(row, "reason_tags"),
                    "negative_audit_label": str(row.get("audit_label") or row.get("manual_label") or ""),
                }
            except Exception as exc:
                errors.append(str(exc))
        return synthesize_negative(samples=samples, np_rng=np_rng, noise_rms=noise_rms), {
            "negative_source": "synthetic_after_manifest_errors",
            "negative_partition": source_partition or "",
            "errors": errors[:3],
        }
    return synthesize_negative(samples=samples, np_rng=np_rng, noise_rms=noise_rms), {
        "negative_source": "synthetic",
        "negative_partition": source_partition or "",
    }


def synthesize_negative(*, samples: int, np_rng: np.random.Generator, noise_rms: float) -> np.ndarray:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    mode = int(np_rng.integers(0, 4))
    if mode == 0:
        return np.zeros(samples, dtype=np.float32)
    if mode == 1:
        return np_rng.normal(0.0, max(0.0, noise_rms), samples).astype(np.float32)
    if mode == 2:
        t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
        freq = float(np_rng.uniform(60.0, 180.0))
        return (max(0.0, noise_rms * 2.0) * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    noise = np_rng.normal(0.0, max(0.0, noise_rms), samples).astype(np.float32)
    envelope = np.linspace(0.2, 1.0, samples, dtype=np.float32)
    return (noise * envelope).astype(np.float32)


def mix_at_snr(signal: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(np.square(signal.astype(np.float64, copy=False))))) if signal.size else 0.0
    noise_rms = float(np.sqrt(np.mean(np.square(background.astype(np.float64, copy=False))))) if background.size else 0.0
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        return signal.astype(np.float32, copy=False)
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    return np.clip(signal + background * (target_noise_rms / max(noise_rms, 1e-12)), -1.0, 1.0).astype(np.float32)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak > 0.98:
        values = values * (0.98 / peak)
    return np.ascontiguousarray(np.clip(values, -1.0, 1.0), dtype=np.float32)


def expand_segments(
    segments: Sequence[TeacherSegment],
    *,
    duration_s: float,
    dilation_s: float,
) -> list[TeacherSegment]:
    expanded: list[TeacherSegment] = []
    for segment in segments:
        start = max(0.0, float(segment.start) - max(0.0, float(dilation_s)))
        end = min(float(duration_s), float(segment.end) + max(0.0, float(dilation_s)))
        if end > start:
            expanded.append(TeacherSegment(start=start, end=end, score=segment.score))
    merged: list[TeacherSegment] = []
    for segment in sorted(expanded, key=lambda item: (item.start, item.end)):
        if not merged or segment.start > merged[-1].end:
            merged.append(segment)
        else:
            prev = merged[-1]
            merged[-1] = TeacherSegment(start=prev.start, end=max(prev.end, segment.end), score=prev.score)
    return merged


def partition_for_source_id(source_audio_id: str) -> str:
    key = str(source_audio_id or "synthetic")
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def partition_for_source_ids(source_audio_ids: Sequence[str]) -> str:
    partitions = {partition_for_source_id(str(item)) for item in source_audio_ids if item}
    if not partitions:
        return partition_for_source_id("synthetic")
    if len(partitions) != 1:
        raise ValueError(f"source_audio_ids cross partitions: {sorted(partitions)}")
    return next(iter(partitions))


def attach_metadata(record: LabelRecord, metadata: Mapping[str, Any]) -> LabelRecord:
    return LabelRecord(
        audio_id=record.audio_id,
        source=record.source,
        duration_s=record.duration_s,
        text=record.text,
        teacher_segments=record.teacher_segments,
        frame_hop_s=record.frame_hop_s,
        speech_frames=record.speech_frames,
        label_quality=record.label_quality,
        frame_weights=record.frame_weights,
        boundary_metadata=dict(metadata),
    )


def build_record(
    *,
    audio_id: str,
    source: str,
    duration_s: float,
    text: str,
    speech_segments: Sequence[TeacherSegment],
    frame_hop_s: float,
    speech_label_dilation_s: float,
    metadata: Mapping[str, Any],
) -> LabelRecord:
    if speech_segments:
        label_segments = expand_segments(
            speech_segments,
            duration_s=duration_s,
            dilation_s=speech_label_dilation_s,
        )
        record = build_supervised_record(
            audio_id=audio_id,
            source=source,
            duration_s=duration_s,
            text=text,
            speech_segments=label_segments,
            frame_hop_s=frame_hop_s,
        )
    else:
        record = LabelRecord(
            audio_id=audio_id,
            source=source,
            duration_s=float(duration_s),
            text=text,
            teacher_segments={"supervised": []},
            frame_hop_s=float(frame_hop_s),
            speech_frames=[0] * frame_count(duration_s, frame_hop_s),
            label_quality="negative",
            frame_weights=[1.0] * frame_count(duration_s, frame_hop_s),
        )
    return attach_metadata(record, metadata)


def base_metadata(
    *,
    example_type: str,
    args: argparse.Namespace,
    source_audio_ids: Sequence[str],
    source_partition: str,
    source_mix: Mapping[str, Any],
    negative_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_partition = str(source_partition or partition_for_source_ids(source_audio_ids))
    if source_audio_ids:
        observed_partition = partition_for_source_ids(source_audio_ids)
        if observed_partition != source_partition:
            raise ValueError(
                f"source_partition {source_partition!r} does not match source ids partition {observed_partition!r}"
            )
    return {
        "dataset_schema": DATASET_SCHEMA,
        "native_example_type": example_type,
        "asr_repo_id": args.asr_repo_id,
        "feature_hash": args.feature_hash,
        "source_mix": dict(source_mix),
        "negative_source": dict(negative_source or {}),
        "speech_label_dilation_s": args.speech_label_dilation_s,
        "split_boundary_label_mode": args.split_boundary_label_mode,
        "split_boundary_radius_frames": args.split_boundary_radius_frames,
        "split_boundary_sigma_frames": args.split_boundary_sigma_frames,
        "negative_policy": "speech_negative_only_pre_asr_cueqc_handles_drop",
        "head_frame_weights_policy": "base_frame_weights_times_optional_head_frame_weights",
        "seed": args.seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_audio_ids": list(source_audio_ids),
        "source_partition": source_partition,
    }


def build_positive_example(
    *,
    index: int,
    source_groups: Sequence[tuple[str, float, list[dict[str, Any]], Path]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[np.ndarray, LabelRecord, dict[str, Any]]:
    row = choose_source_row(source_groups, rng)
    audio, detail = load_clip(row, rng=rng, min_speech_s=args.min_speech_s, max_speech_s=args.max_speech_s)
    duration_s = len(audio) / SAMPLE_RATE
    audio_id = f"scorer-v6native-pos-{index:06d}"
    segment = TeacherSegment(0.0, duration_s, score=1.0)
    metadata = base_metadata(
        example_type="positive_speech_timeline",
        args=args,
        source_audio_ids=[str(detail["source_audio_id"])],
        source_partition=str(detail["source_partition"]),
        source_mix={"speech": [detail]},
        negative_source=None,
    )
    metadata.update(
        {
            "actual_speech_segments": [{"start": 0.0, "end": duration_s}],
            "cut_point_segments": [],
        }
    )
    record = build_record(
        audio_id=audio_id,
        source="speech_boundary_ja_scorer_v6_native:positive_speech_timeline",
        duration_s=duration_s,
        text=str(detail.get("source_text") or ""),
        speech_segments=[segment],
        frame_hop_s=args.frame_hop_s,
        speech_label_dilation_s=args.speech_label_dilation_s,
        metadata=metadata,
    )
    return normalize_peak(audio), record, detail


def build_pure_negative_example(
    *,
    index: int,
    negative_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, LabelRecord, dict[str, Any]]:
    duration_s = rng.uniform(args.negative_min_s, args.negative_max_s)
    samples = max(1, int(round(duration_s * SAMPLE_RATE)))
    audio, negative_detail = load_or_synthesize_negative(
        negative_rows,
        samples=samples,
        rng=rng,
        np_rng=np_rng,
        noise_rms=args.noise_rms,
    )
    duration_s = len(audio) / SAMPLE_RATE
    audio_id = f"scorer-v6native-neg-{index:06d}"
    source_id = str(negative_detail.get("negative_audio_id") or audio_id)
    source_partition = str(negative_detail.get("negative_partition") or partition_for_source_id(source_id))
    metadata = base_metadata(
        example_type="pure_hard_negative",
        args=args,
        source_audio_ids=[source_id],
        source_partition=source_partition,
        source_mix={},
        negative_source=negative_detail,
    )
    metadata.update(
        {
            "actual_speech_segments": [],
            "cut_point_segments": [],
        }
    )
    record = build_record(
        audio_id=audio_id,
        source="speech_boundary_ja_scorer_v6_native:pure_hard_negative",
        duration_s=duration_s,
        text="",
        speech_segments=[],
        frame_hop_s=args.frame_hop_s,
        speech_label_dilation_s=0.0,
        metadata=metadata,
    )
    return normalize_peak(audio), record, negative_detail


def build_mixed_contrast_example(
    *,
    index: int,
    source_groups: Sequence[tuple[str, float, list[dict[str, Any]], Path]],
    negative_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, LabelRecord, dict[str, Any]]:
    row = choose_source_row(source_groups, rng)
    speech, detail = load_clip(row, rng=rng, min_speech_s=args.min_speech_s, max_speech_s=args.max_speech_s)
    source_partition = str(detail["source_partition"])
    lead_s = rng.uniform(args.context_gap_min_s, args.context_gap_max_s)
    tail_s = rng.uniform(args.context_gap_min_s, args.context_gap_max_s)
    lead_samples = int(round(lead_s * SAMPLE_RATE))
    tail_samples = int(round(tail_s * SAMPLE_RATE))
    total_samples = lead_samples + len(speech) + tail_samples
    background, negative_detail = load_or_synthesize_negative(
        negative_rows,
        samples=total_samples,
        rng=rng,
        np_rng=np_rng,
        noise_rms=args.noise_rms,
        source_partition=source_partition,
    )
    mixed = np.array(background, dtype=np.float32, copy=True)
    snr_db = rng.uniform(args.mixed_snr_db_min, args.mixed_snr_db_max)
    mixed[lead_samples : lead_samples + len(speech)] = mix_at_snr(
        speech,
        background[lead_samples : lead_samples + len(speech)],
        snr_db,
    )
    duration_s = len(mixed) / SAMPLE_RATE
    start_s = lead_samples / SAMPLE_RATE
    end_s = (lead_samples + len(speech)) / SAMPLE_RATE
    audio_id = f"scorer-v6native-mixed-{index:06d}"
    source_audio_ids = [str(detail["source_audio_id"])]
    if negative_detail.get("negative_audio_id"):
        source_audio_ids.append(str(negative_detail["negative_audio_id"]))
    metadata = base_metadata(
        example_type="mixed_contrast",
        args=args,
        source_audio_ids=source_audio_ids,
        source_partition=source_partition,
        source_mix={"speech": [detail], "snr_db": snr_db},
        negative_source=negative_detail,
    )
    metadata.update(
        {
            "actual_speech_segments": [{"start": start_s, "end": end_s}],
            "cut_point_segments": [],
        }
    )
    record = build_record(
        audio_id=audio_id,
        source="speech_boundary_ja_scorer_v6_native:mixed_contrast",
        duration_s=duration_s,
        text=str(detail.get("source_text") or ""),
        speech_segments=[TeacherSegment(start_s, end_s, score=1.0)],
        frame_hop_s=args.frame_hop_s,
        speech_label_dilation_s=args.speech_label_dilation_s,
        metadata=metadata,
    )
    return normalize_peak(mixed), record, {"speech": detail, "negative": negative_detail, "snr_db": snr_db}


def sample_long_chain_gap(
    *,
    example_index: int,
    boundary_index: int,
    rng: random.Random,
) -> tuple[str, float]:
    bucket_name, min_s, max_s = LONG_CHAIN_GAP_BUCKETS[
        (int(example_index) + int(boundary_index)) % len(LONG_CHAIN_GAP_BUCKETS)
    ]
    if max_s <= 0.0:
        return bucket_name, 0.0
    return bucket_name, rng.uniform(min_s, max_s)


def build_long_speech_chain_example(
    *,
    index: int,
    source_groups: Sequence[tuple[str, float, list[dict[str, Any]], Path]],
    negative_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, LabelRecord, dict[str, Any]]:
    utterance_target = rng.randint(args.long_chain_utterances_min, args.long_chain_utterances_max)
    target_duration_s = rng.uniform(args.long_chain_min_duration_s, args.long_chain_max_duration_s)
    parts: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    speech_segments: list[TeacherSegment] = []
    cut_points: list[dict[str, Any]] = []
    negative_details: list[dict[str, Any]] = []
    gap_bucket_counts: Counter[str] = Counter()
    cursor = 0
    source_partition: str | None = None
    previous_detail: dict[str, Any] | None = None

    while True:
        clip_index = len(details)
        row = choose_source_row(source_groups, rng, source_partition=source_partition)
        speech, detail = load_clip(
            row,
            rng=rng,
            min_speech_s=args.min_speech_s,
            max_speech_s=args.split_max_speech_s,
        )
        if source_partition is None:
            source_partition = str(detail["source_partition"])
        start_s = cursor / SAMPLE_RATE
        parts.append(speech)
        cursor += len(speech)
        end_s = cursor / SAMPLE_RATE
        speech_segments.append(TeacherSegment(start_s, end_s, score=1.0))
        details.append(detail)

        duration_s = cursor / SAMPLE_RATE
        minimum_ready = (
            len(details) >= args.long_chain_utterances_min
            and duration_s >= args.long_chain_min_duration_s
        )
        if minimum_ready and (len(details) >= utterance_target or duration_s >= target_duration_s):
            break
        if len(details) >= args.long_chain_utterances_max or duration_s >= args.long_chain_max_duration_s:
            break

        bucket_name, gap_s = sample_long_chain_gap(
            example_index=index,
            boundary_index=clip_index,
            rng=rng,
        )
        gap_start_s = cursor / SAMPLE_RATE
        gap_samples = int(round(gap_s * SAMPLE_RATE))
        gap_end_s = (cursor + gap_samples) / SAMPLE_RATE
        gap_bucket_counts[bucket_name] += 1
        cut_points.append(
            {
                "time_s": (gap_start_s + gap_end_s) / 2.0,
                "start": gap_start_s,
                "end": gap_end_s,
                "reason": "long_speech_chain_utterance_boundary",
                "gap_s": gap_s,
                "gap_bucket": bucket_name,
                "left_source_audio_id": detail.get("source_audio_id"),
                "right_source_pending": True,
                "source_switch": bool(
                    previous_detail
                    and previous_detail.get("source_audio_id") != detail.get("source_audio_id")
                ),
            }
        )
        if gap_samples > 0:
            gap_audio, negative_detail = load_or_synthesize_negative(
                negative_rows,
                samples=gap_samples,
                rng=rng,
                np_rng=np_rng,
                noise_rms=args.noise_rms,
                source_partition=source_partition,
            )
            parts.append(gap_audio)
            negative_details.append({**negative_detail, "gap_bucket": bucket_name})
            cursor += gap_samples
        previous_detail = detail

    for boundary_index, point in enumerate(cut_points):
        if boundary_index + 1 < len(details):
            point["right_source_audio_id"] = details[boundary_index + 1].get("source_audio_id")
            point["source_switch"] = (
                details[boundary_index].get("source_audio_id")
                != details[boundary_index + 1].get("source_audio_id")
            )
        point.pop("right_source_pending", None)

    audio = np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(1, dtype=np.float32)
    duration_s = len(audio) / SAMPLE_RATE
    audio_id = f"scorer-v6native-longchain-{index:06d}"
    source_partition = source_partition or partition_for_source_id(audio_id)
    source_audio_ids = [str(item.get("source_audio_id") or "") for item in details]
    source_audio_ids.extend(
        str(item.get("negative_audio_id") or "") for item in negative_details if item.get("negative_audio_id")
    )
    metadata = base_metadata(
        example_type="long_speech_chain",
        args=args,
        source_audio_ids=source_audio_ids,
        source_partition=source_partition,
        source_mix={
            "speech": details,
            "utterance_count": len(details),
            "target_duration_s": target_duration_s,
            "gap_bucket_counts": dict(sorted(gap_bucket_counts.items())),
        },
        negative_source={"gap_negative_sources": negative_details},
    )
    metadata.update(
        {
            "actual_speech_segments": [
                {"start": segment.start, "end": segment.end} for segment in speech_segments
            ],
            "utterance_boundaries": list(cut_points),
            "cut_point_segments": list(cut_points),
            "disable_implicit_gap_drop": True,
        }
    )
    record = build_record(
        audio_id=audio_id,
        source="speech_boundary_ja_scorer_v6_native:long_speech_chain",
        duration_s=duration_s,
        text=" ".join(str(item.get("source_text") or "") for item in details if item.get("source_text")),
        speech_segments=speech_segments,
        frame_hop_s=args.frame_hop_s,
        speech_label_dilation_s=args.speech_label_dilation_s,
        metadata=metadata,
    )
    return normalize_peak(audio), record, {"speech": details, "negative": negative_details}


def build_split_stress_example(
    *,
    index: int,
    source_groups: Sequence[tuple[str, float, list[dict[str, Any]], Path]],
    negative_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, LabelRecord, dict[str, Any]]:
    clip_count = rng.randint(args.split_clips_min, args.split_clips_max)
    parts: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    speech_segments: list[TeacherSegment] = []
    cut_points: list[dict[str, Any]] = []
    cursor = 0
    negative_details: list[dict[str, Any]] = []
    source_partition: str | None = None
    for clip_index in range(clip_count):
        row = choose_source_row(source_groups, rng, source_partition=source_partition)
        speech, detail = load_clip(
            row,
            rng=rng,
            min_speech_s=args.min_speech_s,
            max_speech_s=args.split_max_speech_s,
        )
        if source_partition is None:
            source_partition = str(detail["source_partition"])
        start_s = cursor / SAMPLE_RATE
        parts.append(speech)
        cursor += len(speech)
        end_s = cursor / SAMPLE_RATE
        speech_segments.append(TeacherSegment(start_s, end_s, score=1.0))
        details.append(detail)
        if clip_index >= clip_count - 1:
            continue
        gap_s = 0.0 if rng.random() < args.touch_gap_prob else rng.uniform(0.0, args.short_gap_max_s)
        cut_time_s = cursor / SAMPLE_RATE
        cut_points.append(
            {
                "time_s": cut_time_s,
                "reason": "source_switch_short_gap",
                "gap_s": gap_s,
                "left_source_audio_id": detail.get("source_audio_id"),
            }
        )
        gap_samples = int(round(gap_s * SAMPLE_RATE))
        if gap_samples > 0:
            gap_audio, negative_detail = load_or_synthesize_negative(
                negative_rows,
                samples=gap_samples,
                rng=rng,
                np_rng=np_rng,
                noise_rms=args.noise_rms,
                source_partition=source_partition,
            )
            parts.append(gap_audio)
            negative_details.append(negative_detail)
            cursor += gap_samples
    audio = np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(1, dtype=np.float32)
    duration_s = len(audio) / SAMPLE_RATE
    audio_id = f"scorer-v6native-split-{index:06d}"
    source_partition = source_partition or partition_for_source_id(audio_id)
    source_audio_ids = [str(item.get("source_audio_id") or "") for item in details]
    source_audio_ids.extend(
        str(item.get("negative_audio_id") or "") for item in negative_details if item.get("negative_audio_id")
    )
    metadata = base_metadata(
        example_type="split_stress",
        args=args,
        source_audio_ids=source_audio_ids,
        source_partition=source_partition,
        source_mix={"speech": details, "short_gap_negative_sources": negative_details},
        negative_source={"negative_sources": negative_details},
    )
    metadata.update(
        {
            "actual_speech_segments": [
                {"start": segment.start, "end": segment.end} for segment in speech_segments
            ],
            "utterance_boundaries": list(cut_points),
            "cut_point_segments": list(cut_points),
            "disable_implicit_gap_drop": True,
        }
    )
    record = build_record(
        audio_id=audio_id,
        source="speech_boundary_ja_scorer_v6_native:split_stress",
        duration_s=duration_s,
        text=" ".join(str(item.get("source_text") or "") for item in details if item.get("source_text")),
        speech_segments=speech_segments,
        frame_hop_s=args.frame_hop_s,
        speech_label_dilation_s=args.speech_label_dilation_s,
        metadata=metadata,
    )
    return normalize_peak(audio), record, {"speech": details, "negative": negative_details}


def build_scorer_v6_native_dataset(
    *,
    source_specs: Sequence[str],
    negative_manifests: Sequence[str],
    output_dir: Path,
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    source_groups, source_skipped = load_source_groups(
        source_specs,
        min_duration_s=args.min_speech_s,
        max_duration_s=args.source_max_duration_s,
    )
    negative_rows, negative_skipped = load_negative_rows(negative_manifests)
    mix_weights = scorer_mix_from_args(args)
    counts = allocate_counts(count, mix_weights)

    output_dir = output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[LabelRecord] = []
    manifest_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    split_rows: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    example_builders = (
        ("long_speech_chain", counts.get("long_speech_chain", 0)),
        ("split_stress", counts.get("split_stress", 0)),
        ("mixed_contrast", counts.get("mixed_contrast", 0)),
        ("pure_hard_negative", counts.get("pure_hard_negative", 0)),
        ("positive_speech_timeline", counts.get("positive_speech_timeline", 0)),
    )
    output_index = 0
    for example_type, type_count in example_builders:
        for _local_index in range(type_count):
            if example_type == "long_speech_chain":
                audio, record, detail = build_long_speech_chain_example(
                    index=output_index,
                    source_groups=source_groups,
                    negative_rows=negative_rows,
                    args=args,
                    rng=rng,
                    np_rng=np_rng,
                )
            elif example_type == "positive_speech_timeline":
                audio, record, detail = build_positive_example(
                    index=output_index,
                    source_groups=source_groups,
                    args=args,
                    rng=rng,
                )
            elif example_type == "pure_hard_negative":
                audio, record, detail = build_pure_negative_example(
                    index=output_index,
                    negative_rows=negative_rows,
                    args=args,
                    rng=rng,
                    np_rng=np_rng,
                )
            elif example_type == "mixed_contrast":
                audio, record, detail = build_mixed_contrast_example(
                    index=output_index,
                    source_groups=source_groups,
                    negative_rows=negative_rows,
                    args=args,
                    rng=rng,
                    np_rng=np_rng,
                )
            elif example_type == "split_stress":
                audio, record, detail = build_split_stress_example(
                    index=output_index,
                    source_groups=source_groups,
                    negative_rows=negative_rows,
                    args=args,
                    rng=rng,
                    np_rng=np_rng,
                )
            else:  # pragma: no cover
                raise ValueError(example_type)

            audio_path = audio_dir / f"{record.audio_id}.wav"
            sf.write(str(audio_path), audio, SAMPLE_RATE)
            partition = str((record.boundary_metadata or {}).get("source_partition") or "train")
            split_rows.setdefault(partition, []).append(len(records))
            records.append(record)
            manifest_rows.append(
                {
                    "audio_id": record.audio_id,
                    "audio": str(audio_path),
                    "duration_s": record.duration_s,
                    "sample_rate": SAMPLE_RATE,
                    "source": record.source,
                    "label_quality": record.label_quality,
                    "native_example_type": example_type,
                    "source_partition": partition,
                    "speech_frame_count": sum(int(value) for value in record.speech_frames),
                    "frame_count": len(record.speech_frames),
                    "text": record.text,
                }
            )
            detail_rows.append(
                {
                    "label_index": len(records) - 1,
                    "audio_id": record.audio_id,
                    "native_example_type": example_type,
                    "source_partition": partition,
                    "detail": detail,
                    "boundary_metadata": record.boundary_metadata,
                }
            )
            output_index += 1

    order = list(range(len(records)))
    rng.shuffle(order)
    records = [records[index] for index in order]
    manifest_rows = [manifest_rows[index] for index in order]
    detail_rows = [detail_rows[index] for index in order]
    split_rows = {"train": [], "val": [], "test": []}
    for index, record in enumerate(records):
        partition = str((record.boundary_metadata or {}).get("source_partition") or "train")
        split_rows.setdefault(partition, []).append(index)

    labels_path = output_dir / "scorer_v6_native_labels.jsonl"
    manifest_path = output_dir / "scorer_v6_native_manifest.json"
    training_manifest_path = output_dir / "scorer_v6_native_training_manifest.jsonl"
    details_path = output_dir / "scorer_v6_native_details.jsonl"
    splits_path = output_dir / "scorer_v6_native_splits.json"
    skipped_path = output_dir / "scorer_v6_native_skipped.json"
    summary_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    write_label_jsonl(labels_path, records)
    write_json(manifest_path, manifest_rows)
    write_jsonl(details_path, detail_rows)
    write_json(splits_path, split_rows)
    write_json(skipped_path, {"sources": source_skipped, "negative_sources": negative_skipped})
    examples = [
        TrainingExample(
            audio_id=record.audio_id,
            source=record.source,
            label_quality=record.label_quality,
            duration_s=record.duration_s,
            frame_hop_s=record.frame_hop_s,
            audio_path=str(manifest_row["audio"]),
            label_index=index,
            speech_frame_count=sum(int(value) for value in record.speech_frames),
            frame_count=len(record.speech_frames),
        )
        for index, (record, manifest_row) in enumerate(zip(records, manifest_rows, strict=True))
    ]
    write_training_manifest(path=training_manifest_path, examples=examples)

    total_frames = sum(len(record.speech_frames) for record in records)
    speech_frames = sum(sum(int(value) for value in record.speech_frames) for record in records)
    split_point_count = 0
    long_chain_gap_bucket_counts: Counter[str] = Counter()
    long_chain_utterance_counts: list[int] = []
    long_chain_durations: list[float] = []
    for record in records:
        metadata = dict(record.boundary_metadata or {})
        split_point_count += len(metadata.get("cut_point_segments") or [])
        if metadata.get("native_example_type") == "long_speech_chain":
            long_chain_durations.append(float(record.duration_s))
            actual_segments = list(metadata.get("actual_speech_segments") or [])
            long_chain_utterance_counts.append(len(actual_segments))
            for point in metadata.get("cut_point_segments") or []:
                if isinstance(point, Mapping):
                    bucket = str(point.get("gap_bucket") or "")
                    if bucket:
                        long_chain_gap_bucket_counts[bucket] += 1
    summary = {
        "schema": DATASET_SCHEMA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": repo_display_path(output_dir),
        "count": len(records),
        "seed": seed,
        "asr_repo_id": args.asr_repo_id,
        "feature_hash": args.feature_hash,
        "source_specs": list(source_specs),
        "negative_manifests": [repo_display_path(project_path(path)) for path in negative_manifests],
        "source_group_counts": dict(
            Counter(
                str((record.boundary_metadata or {}).get("source_mix", {}).get("speech", [{}])[0].get("source_group", ""))
                for record in records
                if (record.boundary_metadata or {}).get("source_mix", {}).get("speech")
            )
        ),
        "native_example_type_counts": dict(
            sorted(Counter(str((record.boundary_metadata or {}).get("native_example_type") or "") for record in records).items())
        ),
        "source_partition_counts": {name: len(rows) for name, rows in split_rows.items()},
        "frame_count": total_frames,
        "speech_frame_count": speech_frames,
        "speech_frame_ratio": speech_frames / total_frames if total_frames else 0.0,
        "split_point_count": split_point_count,
        "long_chain_gap_bucket_counts": dict(sorted(long_chain_gap_bucket_counts.items())),
        "long_chain_utterance_count_min": min(long_chain_utterance_counts) if long_chain_utterance_counts else 0,
        "long_chain_utterance_count_max": max(long_chain_utterance_counts) if long_chain_utterance_counts else 0,
        "long_chain_duration_s_min": min(long_chain_durations) if long_chain_durations else 0.0,
        "long_chain_duration_s_max": max(long_chain_durations) if long_chain_durations else 0.0,
        "training_examples": len(examples),
        "source_rows_skipped": len(source_skipped),
        "negative_rows": len(negative_rows),
        "negative_rows_skipped": len(negative_skipped),
        "labels": repo_display_path(labels_path),
        "manifest": repo_display_path(manifest_path),
        "training_manifest": repo_display_path(training_manifest_path),
        "details": repo_display_path(details_path),
        "splits": repo_display_path(splits_path),
        "skipped": repo_display_path(skipped_path),
        "config": {
            "mix": dict(mix_weights),
            "frame_hop_s": args.frame_hop_s,
            "speech_label_dilation_s": args.speech_label_dilation_s,
            "split_boundary_label_mode": args.split_boundary_label_mode,
            "split_boundary_radius_frames": args.split_boundary_radius_frames,
            "split_boundary_sigma_frames": args.split_boundary_sigma_frames,
            "min_speech_s": args.min_speech_s,
            "max_speech_s": args.max_speech_s,
            "source_max_duration_s": args.source_max_duration_s,
            "negative_min_s": args.negative_min_s,
            "negative_max_s": args.negative_max_s,
            "context_gap_min_s": args.context_gap_min_s,
            "context_gap_max_s": args.context_gap_max_s,
            "mixed_snr_db_min": args.mixed_snr_db_min,
            "mixed_snr_db_max": args.mixed_snr_db_max,
            "short_gap_max_s": args.short_gap_max_s,
            "touch_gap_prob": args.touch_gap_prob,
            "long_chain_gap_buckets": [
                {"name": name, "min_s": min_s, "max_s": max_s}
                for name, min_s, max_s in LONG_CHAIN_GAP_BUCKETS
            ],
            "long_chain_utterances_min": args.long_chain_utterances_min,
            "long_chain_utterances_max": args.long_chain_utterances_max,
            "long_chain_min_duration_s": args.long_chain_min_duration_s,
            "long_chain_max_duration_s": args.long_chain_max_duration_s,
        },
    }
    write_json(summary_path, summary)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SpeechBoundary-JA Scorer v6 Native Dataset",
        "",
        f"- Output: `{summary['output_dir']}`",
        f"- Schema: `{summary['schema']}`",
        f"- Records: `{summary['count']}`",
        f"- Mix: `{summary['native_example_type_counts']}`",
        f"- Source partitions: `{summary['source_partition_counts']}`",
        f"- Speech frame ratio: `{summary['speech_frame_ratio']}`",
        "",
        "## Outputs",
        "",
    ]
    for key in ("labels", "manifest", "training_manifest", "details", "splits", "skipped"):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SpeechBoundary-JA scorer v6-native dataset with speech and split-boundary heatmap labels."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="name=weight=manifest_path. Defaults keep anime NSFW:SFW at 1:1 plus galgame.",
    )
    parser.add_argument("--negative-manifest", action="append", default=[])
    parser.add_argument(
        "--type-mix",
        action="append",
        default=[],
        help=(
            "Override scorer example mix as name=weight. Valid names: "
            "long_speech_chain, positive_speech_timeline, pure_hard_negative, "
            "mixed_contrast, split_stress."
        ),
    )
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=260623)
    parser.add_argument("--asr-repo-id", default=QWEN_ASR_17B_REPO_ID)
    parser.add_argument("--feature-hash", default="")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--speech-label-dilation-s", type=float, default=0.06)
    parser.add_argument("--split-boundary-label-mode", choices=["hard", "gaussian"], default="gaussian")
    parser.add_argument("--split-boundary-radius-frames", type=int, default=1)
    parser.add_argument("--split-boundary-sigma-frames", type=float, default=1.0)
    parser.add_argument("--min-speech-s", type=float, default=0.05)
    parser.add_argument("--max-speech-s", type=float, default=6.0)
    parser.add_argument("--split-max-speech-s", type=float, default=3.0)
    parser.add_argument("--source-max-duration-s", type=float, default=12.0)
    parser.add_argument("--negative-min-s", type=float, default=1.2)
    parser.add_argument("--negative-max-s", type=float, default=8.0)
    parser.add_argument("--context-gap-min-s", type=float, default=0.2)
    parser.add_argument("--context-gap-max-s", type=float, default=1.2)
    parser.add_argument("--mixed-snr-db-min", type=float, default=-4.0)
    parser.add_argument("--mixed-snr-db-max", type=float, default=14.0)
    parser.add_argument("--short-gap-max-s", type=float, default=0.12)
    parser.add_argument("--touch-gap-prob", type=float, default=0.35)
    parser.add_argument("--split-clips-min", type=int, default=2)
    parser.add_argument("--split-clips-max", type=int, default=4)
    parser.add_argument("--long-chain-utterances-min", type=int, default=6)
    parser.add_argument("--long-chain-utterances-max", type=int, default=20)
    parser.add_argument("--long-chain-min-duration-s", type=float, default=10.0)
    parser.add_argument("--long-chain-max-duration-s", type=float, default=40.0)
    parser.add_argument("--noise-rms", type=float, default=0.015)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to agents/temp/YYYYMMDD_HHMMSS_scorer-v6-native-4096.",
    )
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.min_speech_s <= 0.0:
        parser.error("--min-speech-s must be positive")
    for name in (
        "speech_label_dilation_s",
        "max_speech_s",
        "split_max_speech_s",
        "source_max_duration_s",
        "negative_min_s",
        "negative_max_s",
        "context_gap_min_s",
        "context_gap_max_s",
        "short_gap_max_s",
        "touch_gap_prob",
        "long_chain_min_duration_s",
        "long_chain_max_duration_s",
        "noise_rms",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.split_boundary_radius_frames < 0:
        parser.error("--split-boundary-radius-frames must be non-negative")
    if args.split_boundary_sigma_frames <= 0.0:
        parser.error("--split-boundary-sigma-frames must be positive")
    if args.negative_max_s < args.negative_min_s:
        parser.error("--negative-max-s must be >= --negative-min-s")
    if args.context_gap_max_s < args.context_gap_min_s:
        parser.error("--context-gap-max-s must be >= --context-gap-min-s")
    if args.mixed_snr_db_max < args.mixed_snr_db_min:
        parser.error("--mixed-snr-db-max must be >= --mixed-snr-db-min")
    if not 0.0 <= args.touch_gap_prob <= 1.0:
        parser.error("--touch-gap-prob must be in [0, 1]")
    if args.split_clips_min < 2:
        parser.error("--split-clips-min must be >= 2")
    if args.split_clips_max < args.split_clips_min:
        parser.error("--split-clips-max must be >= --split-clips-min")
    if args.long_chain_utterances_min < 6:
        parser.error("--long-chain-utterances-min must be >= 6")
    if args.long_chain_utterances_max < args.long_chain_utterances_min:
        parser.error("--long-chain-utterances-max must be >= --long-chain-utterances-min")
    if args.long_chain_max_duration_s < args.long_chain_min_duration_s:
        parser.error("--long-chain-max-duration-s must be >= --long-chain-min-duration-s")
    try:
        scorer_mix_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "agents" / "temp" / f"{local_timestamp()}_scorer-v6-native-{args.count}"
    )
    summary = build_scorer_v6_native_dataset(
        source_specs=args.source or DEFAULT_SOURCE_SPECS,
        negative_manifests=args.negative_manifest,
        output_dir=output_dir,
        count=args.count,
        seed=args.seed,
        args=args,
    )
    print(f"output_dir={summary['output_dir']}")
    print(f"labels={summary['labels']}")
    print(f"manifest={summary['manifest']}")
    print(f"training_manifest={summary['training_manifest']}")
    print(
        "records={count} mix={mix}".format(
            count=summary["count"],
            mix=json.dumps(summary["native_example_type_counts"], ensure_ascii=False, sort_keys=True),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

