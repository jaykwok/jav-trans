#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from vad.base import SpeechSegment  # noqa: E402
from vad.fusionvad_ja import (  # noqa: E402
    frame_classification_counts,
    load_label_records,
    load_manifest_audio_map,
    metrics_from_frame_counts,
    segments_to_frame_labels,
)
from vad.fusionvad_ja.manifest import build_training_examples  # noqa: E402
from vad.fusionvad_ja.research_backends import get_research_vad_backend  # noqa: E402


def clean_vad_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace("/", "_")


def read_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audio_id = str(row.get("audio_id") or "")
            if audio_id:
                rows[audio_id] = row
    return rows


def padded_frames(frames: Iterable[int], *, pad_frames: int) -> list[int]:
    values = [1 if int(value) else 0 for value in frames]
    if pad_frames <= 0:
        return values
    out = [0] * len(values)
    for index, value in enumerate(values):
        if not value:
            continue
        start = max(0, index - pad_frames)
        end = min(len(out), index + pad_frames + 1)
        for offset in range(start, end):
            out[offset] = 1
    return out


def frames_to_segments(
    frames: Iterable[int],
    *,
    frame_hop_s: float,
    duration_s: float,
) -> list[SpeechSegment]:
    values = [1 if int(value) else 0 for value in frames]
    segments: list[SpeechSegment] = []
    start_index: int | None = None
    for index, value in enumerate(values + [0]):
        if value and start_index is None:
            start_index = index
        if not value and start_index is not None:
            start = max(0.0, min(float(start_index) * frame_hop_s, duration_s))
            end = max(0.0, min(float(index) * frame_hop_s, duration_s))
            if end > start:
                segments.append(SpeechSegment(start=start, end=end))
            start_index = None
    return segments


def normalize_segments(
    segments: Iterable[SpeechSegment | Mapping[str, Any]],
    *,
    duration_s: float,
    min_segment_s: float = 0.0,
) -> list[SpeechSegment]:
    out: list[SpeechSegment] = []
    for segment in segments:
        try:
            if isinstance(segment, SpeechSegment):
                start = float(segment.start)
                end = float(segment.end)
                score = segment.score
            else:
                start = float(segment.get("start", 0.0))
                end = float(segment.get("end", 0.0))
                raw_score = segment.get("score")
                score = None if raw_score is None else float(raw_score)
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(start, duration_s))
        end = max(0.0, min(end, duration_s))
        if end - start >= min_segment_s:
            out.append(SpeechSegment(start=start, end=end, score=score))
    return sorted(out, key=lambda item: (item.start, item.end))


def merge_segments(
    segments: Iterable[SpeechSegment],
    *,
    duration_s: float,
    merge_gap_s: float,
    min_segment_s: float = 0.0,
) -> list[SpeechSegment]:
    normalized = normalize_segments(
        segments,
        duration_s=duration_s,
        min_segment_s=min_segment_s,
    )
    merged: list[SpeechSegment] = []
    for segment in normalized:
        if not merged or segment.start - merged[-1].end > merge_gap_s:
            merged.append(SpeechSegment(segment.start, segment.end, score=segment.score))
        else:
            merged[-1].end = max(merged[-1].end, segment.end)
            if merged[-1].score is None:
                merged[-1].score = segment.score
            elif segment.score is not None:
                merged[-1].score = max(float(merged[-1].score), float(segment.score))
    return [segment for segment in merged if segment.end - segment.start >= min_segment_s]


def segment_overlap_s(left: SpeechSegment, right: SpeechSegment) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def total_overlap_s(segment: SpeechSegment, speech_segments: Iterable[SpeechSegment]) -> float:
    return sum(segment_overlap_s(segment, speech) for speech in speech_segments)


def write_wav_slice(
    *,
    source_audio: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
    output_path: Path,
) -> None:
    import soundfile as sf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_sample = max(0, min(len(source_audio), int(round(start_s * sample_rate))))
    end_sample = max(0, min(len(source_audio), int(round(end_s * sample_rate))))
    if end_sample <= start_sample:
        samples = np.zeros(max(1, int(round(0.02 * sample_rate))), dtype=np.float32)
    else:
        samples = np.ascontiguousarray(source_audio[start_sample:end_sample], dtype=np.float32)
    sf.write(str(output_path), samples, sample_rate)


def text_char_count(value: str) -> int:
    return len(str(value or "").strip())


def build_backend_segments(
    *,
    vad_name: str,
    examples,
    records,
    prediction_rows: Mapping[str, Mapping[str, Any]],
    fusionvad_pad_s: float,
    merge_gap_s: float,
    min_segment_s: float,
) -> tuple[dict[str, list[SpeechSegment]], list[dict[str, Any]]]:
    normalized_name = clean_vad_name(vad_name)
    by_audio_id: dict[str, list[SpeechSegment]] = {}
    errors: list[dict[str, Any]] = []

    if normalized_name == "fusionvad":
        for example in examples:
            record = records[example.label_index]
            row = prediction_rows.get(record.audio_id)
            if row is None:
                errors.append({"audio_id": record.audio_id, "vad": vad_name, "error": "missing_prediction"})
                by_audio_id[record.audio_id] = []
                continue
            raw_frames = row.get("speech_frames") or row.get("predictions") or []
            pad_frames = max(0, int(round(fusionvad_pad_s / record.frame_hop_s)))
            frames = padded_frames([int(value) for value in raw_frames], pad_frames=pad_frames)
            segments = frames_to_segments(
                frames,
                frame_hop_s=record.frame_hop_s,
                duration_s=record.duration_s,
            )
            by_audio_id[record.audio_id] = merge_segments(
                segments,
                duration_s=record.duration_s,
                merge_gap_s=merge_gap_s,
                min_segment_s=min_segment_s,
            )
        return by_audio_id, errors

    backend = get_research_vad_backend(vad_name)
    for index, example in enumerate(examples):
        record = records[example.label_index]
        try:
            result = backend.segment(example.audio_path, target_sr=16000)
            segments = merge_segments(
                result.segments,
                duration_s=record.duration_s,
                merge_gap_s=merge_gap_s,
                min_segment_s=min_segment_s,
            )
            by_audio_id[record.audio_id] = segments
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "audio_id": record.audio_id,
                    "audio_path": example.audio_path,
                    "vad": vad_name,
                    "error": str(exc),
                }
            )
            by_audio_id[record.audio_id] = []
        print(
            f"vad={vad_name} segmented={index + 1}/{len(examples)} errors={len(errors)}",
            flush=True,
        )
    return by_audio_id, errors


def summarize_text_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_count = len(rows)
    nonempty = [row for row in rows if text_char_count(row.get("text", "")) > 0]
    no_speech_overlap = [row for row in rows if float(row.get("manual_overlap_s") or 0.0) <= 0.0]
    low_speech_overlap = [row for row in rows if float(row.get("manual_overlap_ratio") or 0.0) < 0.1]
    negative_record_rows = [row for row in rows if row.get("label_quality") == "negative"]
    error_counts = Counter(
        str((row.get("asr_generation") or {}).get("error_kind") or "")
        for row in rows
    )
    return {
        "chunk_count": chunk_count,
        "nonempty_chunk_count": len(nonempty),
        "nonempty_chunk_ratio": len(nonempty) / chunk_count if chunk_count else 0.0,
        "total_text_chars": sum(text_char_count(row.get("text", "")) for row in rows),
        "raw_text_chars": sum(text_char_count(row.get("raw_text", "")) for row in rows),
        "no_speech_overlap_chunk_count": len(no_speech_overlap),
        "no_speech_overlap_nonempty_count": sum(
            1 for row in no_speech_overlap if text_char_count(row.get("text", "")) > 0
        ),
        "no_speech_overlap_text_chars": sum(
            text_char_count(row.get("text", "")) for row in no_speech_overlap
        ),
        "low_speech_overlap_chunk_count": len(low_speech_overlap),
        "low_speech_overlap_nonempty_count": sum(
            1 for row in low_speech_overlap if text_char_count(row.get("text", "")) > 0
        ),
        "low_speech_overlap_text_chars": sum(
            text_char_count(row.get("text", "")) for row in low_speech_overlap
        ),
        "negative_record_chunk_count": len(negative_record_rows),
        "negative_record_nonempty_count": sum(
            1 for row in negative_record_rows if text_char_count(row.get("text", "")) > 0
        ),
        "negative_record_text_chars": sum(
            text_char_count(row.get("text", "")) for row in negative_record_rows
        ),
        "asr_error_counts": dict(sorted(error_counts.items())),
    }


def evaluate_one_vad(
    *,
    vad_name: str,
    records,
    examples,
    segments_by_audio_id: Mapping[str, list[SpeechSegment]],
    output_dir: Path,
    asr_backend,
    asr_batch_size: int,
) -> dict[str, Any]:
    vad_key = clean_vad_name(vad_name)
    vad_dir = output_dir / vad_key
    chunk_dir = vad_dir / "chunks"
    vad_dir.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, Any]] = []
    aggregate = Counter()
    chunk_paths: list[str] = []
    chunk_meta: list[dict[str, Any]] = []
    total_predicted_audio_s = 0.0
    missed_speech_seconds = 0.0
    extra_audio_seconds = 0.0
    no_vad_records = 0

    for example in examples:
        record = records[example.label_index]
        segments = list(segments_by_audio_id.get(record.audio_id) or [])
        if not segments:
            no_vad_records += 1
        labels = record.speech_frames
        predictions = segments_to_frame_labels(
            segments,
            duration_s=record.duration_s,
            frame_hop_s=record.frame_hop_s,
        )
        counts = frame_classification_counts(labels=labels, predictions=predictions)
        aggregate.update(counts)
        missed_speech_seconds += int(counts.get("false_negative", 0)) * record.frame_hop_s
        extra_audio_seconds += int(counts.get("false_positive", 0)) * record.frame_hop_s
        speech_segments = frames_to_segments(
            record.speech_frames,
            frame_hop_s=record.frame_hop_s,
            duration_s=record.duration_s,
        )
        audio, sample_rate = load_audio_16k_mono(example.audio_path)
        for chunk_index, segment in enumerate(segments):
            duration_s = max(0.0, segment.end - segment.start)
            overlap_s = total_overlap_s(segment, speech_segments)
            chunk_path = chunk_dir / f"{record.audio_id}__{chunk_index:03d}.wav"
            write_wav_slice(
                source_audio=audio,
                sample_rate=sample_rate,
                start_s=segment.start,
                end_s=segment.end,
                output_path=chunk_path,
            )
            payload = {
                "vad": vad_name,
                "audio_id": record.audio_id,
                "source": record.source,
                "label_quality": record.label_quality,
                "chunk_index": chunk_index,
                "start": segment.start,
                "end": segment.end,
                "duration_s": duration_s,
                "score": segment.score,
                "chunk_path": str(chunk_path),
                "manual_overlap_s": overlap_s,
                "manual_overlap_ratio": overlap_s / duration_s if duration_s > 0.0 else 0.0,
            }
            segment_rows.append(payload)
            chunk_paths.append(str(chunk_path))
            chunk_meta.append(payload)
            total_predicted_audio_s += duration_s

    segments_path = vad_dir / "vad_segments.jsonl"
    with segments_path.open("w", encoding="utf-8") as handle:
        for row in segment_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    started = time.monotonic()
    asr_rows: list[dict[str, Any]] = []
    for offset in range(0, len(chunk_paths), asr_batch_size):
        batch_paths = chunk_paths[offset : offset + asr_batch_size]
        batch_meta = chunk_meta[offset : offset + asr_batch_size]
        if not batch_paths:
            continue
        print(
            f"vad={vad_name} anime_whisper_asr={offset + 1}-{offset + len(batch_paths)}/{len(chunk_paths)}",
            flush=True,
        )
        results = asr_backend.transcribe_texts(batch_paths)
        for meta, result in zip(batch_meta, results):
            asr_rows.append(
                {
                    **meta,
                    "text": result.get("text", ""),
                    "raw_text": result.get("raw_text", ""),
                    "language": result.get("language"),
                    "avg_logprob": result.get("avg_logprob"),
                    "no_speech_prob": result.get("no_speech_prob"),
                    "compression_ratio": result.get("compression_ratio"),
                    "asr_generation": result.get("asr_generation"),
                }
            )
    asr_elapsed_s = time.monotonic() - started

    asr_path = vad_dir / "anime_whisper_outputs.jsonl"
    with asr_path.open("w", encoding="utf-8") as handle:
        for row in asr_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metrics = metrics_from_frame_counts(counts=dict(aggregate), windows=len(examples))
    asr_summary = summarize_text_rows(asr_rows)
    summary = {
        "vad": vad_name,
        "examples": len(examples),
        "no_vad_records": no_vad_records,
        "segments": str(segments_path),
        "asr_outputs": str(asr_path),
        "frame_metrics": {
            **asdict(metrics),
            "missed_speech_seconds": missed_speech_seconds,
            "extra_audio_seconds": extra_audio_seconds,
            "extra_audio_ratio": (
                int(aggregate.get("predicted_positives", 0))
                / max(int(aggregate.get("positives", 0)), 1)
            ),
            "counts": dict(aggregate),
        },
        "asr_summary": {
            **asr_summary,
            "total_predicted_audio_s": total_predicted_audio_s,
            "asr_elapsed_s": asr_elapsed_s,
            "asr_realtime_factor_on_predicted_audio": (
                asr_elapsed_s / total_predicted_audio_s if total_predicted_audio_s > 0 else 0.0
            ),
        },
    }
    summary_path = vad_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run(args: argparse.Namespace) -> None:
    if args.whisper_max_new_tokens:
        os.environ["WHISPER_MAX_NEW_TOKENS"] = str(args.whisper_max_new_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_label_records(Path(args.labels))
    audio_map = load_manifest_audio_map(Path(args.manifest))
    examples, skipped = build_training_examples(
        records,
        manifest_audio_map=audio_map,
        trainable_only=not args.include_non_trainable,
    )
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("no examples to evaluate")

    prediction_rows = read_prediction_rows(Path(args.fusionvad_predictions))
    skipped_path = output_dir / "training_example_skipped.json"
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    from whisper.model_backend import create_whisper_model_backend

    asr_backend = create_whisper_model_backend(args.asr_backend, args.device)
    all_summaries: dict[str, Any] = {}
    vad_errors: dict[str, list[dict[str, Any]]] = {}
    try:
        for vad_name in args.vad:
            segments_by_audio_id, errors = build_backend_segments(
                vad_name=vad_name,
                examples=examples,
                records=records,
                prediction_rows=prediction_rows,
                fusionvad_pad_s=args.fusionvad_pad_s,
                merge_gap_s=args.merge_gap_s,
                min_segment_s=args.min_segment_s,
            )
            vad_errors[vad_name] = errors
            summary = evaluate_one_vad(
                vad_name=vad_name,
                records=records,
                examples=examples,
                segments_by_audio_id=segments_by_audio_id,
                output_dir=output_dir,
                asr_backend=asr_backend,
                asr_batch_size=args.asr_batch_size,
            )
            all_summaries[vad_name] = summary
            fm = summary["frame_metrics"]
            am = summary["asr_summary"]
            print(
                f"vad={vad_name} recall={fm['recall']:.4f} precision={fm['precision']:.4f} "
                f"extra_audio_ratio={fm['extra_audio_ratio']:.4f} "
                f"chunks={am['chunk_count']} negative_chars={am['negative_record_text_chars']} "
                f"no_speech_chars={am['no_speech_overlap_text_chars']}",
                flush=True,
            )
    finally:
        close = getattr(asr_backend, "close", None)
        if callable(close):
            close()

    summary = {
        "labels": args.labels,
        "manifest": args.manifest,
        "fusionvad_predictions": args.fusionvad_predictions,
        "asr_backend": args.asr_backend,
        "device": args.device,
        "examples": len(examples),
        "skipped": len(skipped),
        "settings": {
            "fusionvad_pad_s": args.fusionvad_pad_s,
            "merge_gap_s": args.merge_gap_s,
            "min_segment_s": args.min_segment_s,
            "asr_batch_size": args.asr_batch_size,
            "whisper_max_new_tokens": args.whisper_max_new_tokens,
        },
        "vad_errors": vad_errors,
        "results": all_summaries,
    }
    summary_path = output_dir / "downstream_asr_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary={summary_path}")
    print(f"skipped={skipped_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare VAD candidates through AnimeWhisper downstream ASR on FusionVAD-JA labels."
    )
    parser.add_argument("--labels", required=True, help="FusionVAD-JA label JSONL.")
    parser.add_argument("--manifest", required=True, help="Audio manifest matching labels.")
    parser.add_argument("--fusionvad-predictions", required=True, help="FusionVAD frame prediction JSONL.")
    parser.add_argument(
        "--vad",
        action="append",
        default=None,
        help="VAD name. Repeatable. Defaults to whisperseg-adaptive, fusion_lite, fusionvad.",
    )
    parser.add_argument("--asr-backend", default="anime-whisper")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fusionvad-pad-s", type=float, default=0.2)
    parser.add_argument("--merge-gap-s", type=float, default=0.0)
    parser.add_argument("--min-segment-s", type=float, default=0.05)
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--whisper-max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-non-trainable", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "vad-asr-downstream"),
    )
    args = parser.parse_args(argv)
    if args.vad is None:
        args.vad = ["whisperseg-adaptive", "fusion_lite", "fusionvad"]
    if args.fusionvad_pad_s < 0:
        parser.error("--fusionvad-pad-s must be non-negative")
    if args.merge_gap_s < 0:
        parser.error("--merge-gap-s must be non-negative")
    if args.min_segment_s < 0:
        parser.error("--min-segment-s must be non-negative")
    if args.asr_batch_size <= 0:
        parser.error("--asr-batch-size must be positive")
    if args.whisper_max_new_tokens is not None and args.whisper_max_new_tokens <= 0:
        parser.error("--whisper-max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
