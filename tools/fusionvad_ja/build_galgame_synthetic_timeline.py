#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402
from vad.fusionvad_ja import TeacherSegment, build_supervised_record, write_jsonl  # noqa: E402


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def valid_source_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    skipped = []
    for index, row in enumerate(rows):
        if row.get("error"):
            skipped.append({"index": index, "reason": "source_error", "error": str(row.get("error"))})
            continue
        audio = row.get("audio")
        if not audio:
            skipped.append({"index": index, "reason": "missing_audio"})
            continue
        audio_path = Path(str(audio))
        if not audio_path.exists():
            skipped.append({"index": index, "reason": "audio_not_found", "audio": str(audio_path)})
            continue
        valid.append(row)
    return valid, skipped


def load_valid_manifest_rows(paths: list[str] | None, *, role: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for manifest_path in paths or []:
        path = Path(manifest_path)
        rows, row_skipped = valid_source_rows(load_manifest_rows(path))
        for row in rows:
            updated = dict(row)
            updated["_manifest"] = str(path)
            valid.append(updated)
        for row in row_skipped:
            updated = dict(row)
            updated["manifest"] = str(path)
            updated["role"] = role
            skipped.append(updated)
    return valid, skipped


def crop_or_tile_audio(
    audio: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), 0
    if audio.size <= 0:
        raise ValueError("audio is empty")
    values = np.ascontiguousarray(audio, dtype=np.float32)
    if values.size >= samples:
        offset = int(rng.integers(0, values.size - samples + 1)) if values.size > samples else 0
        return np.ascontiguousarray(values[offset : offset + samples], dtype=np.float32), offset
    repeats = int(math.ceil(samples / values.size))
    tiled = np.tile(values, repeats)[:samples]
    return np.ascontiguousarray(tiled, dtype=np.float32), 0


def load_negative_audio(
    row: Mapping[str, Any],
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    audio_path = Path(str(row["audio"]))
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz audio after normalization, got {sample_rate}")
    clipped, offset = crop_or_tile_audio(audio, samples=samples, rng=rng)
    detail = {
        "audio_id": str(row.get("audio_id") or audio_path.stem),
        "audio": str(audio_path),
        "source": str(row.get("source") or "negative"),
        "manifest": str(row.get("_manifest") or ""),
        "source_offset_s": offset / sample_rate,
        "duration_s": clipped.size / sample_rate,
    }
    return clipped, detail


def clipped_speech_audio(
    row: Mapping[str, Any],
    *,
    rng: np.random.Generator,
    trim_head_s: float,
    trim_tail_s: float,
    max_speech_s: float,
    min_speech_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    audio_path = Path(str(row["audio"]))
    audio, sample_rate = load_audio_16k_mono(str(audio_path))
    if sample_rate != 16000:
        raise ValueError(f"expected 16kHz audio after normalization, got {sample_rate}")
    start_sample = max(0, int(round(max(0.0, trim_head_s) * sample_rate)))
    end_sample = max(start_sample, len(audio) - int(round(max(0.0, trim_tail_s) * sample_rate)))
    trimmed = np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)
    if trimmed.size < max(1, int(round(min_speech_s * sample_rate))):
        raise ValueError("speech clip is shorter than --min-speech-s after trim")
    if max_speech_s > 0.0:
        max_samples = max(1, int(round(max_speech_s * sample_rate)))
        if trimmed.size > max_samples:
            offset = int(rng.integers(0, trimmed.size - max_samples + 1))
            trimmed = np.ascontiguousarray(trimmed[offset : offset + max_samples], dtype=np.float32)
            start_sample += offset
            end_sample = start_sample + max_samples
    return trimmed, {
        "source_audio_id": str(row.get("audio_id") or audio_path.stem),
        "source_audio": str(audio_path),
        "source_input": str(row.get("input") or ""),
        "source_text": str(row.get("text") or ""),
        "source_start_s": start_sample / sample_rate,
        "source_end_s": end_sample / sample_rate,
        "duration_s": trimmed.size / sample_rate,
    }


def choose_real_negative_gap(
    *,
    rows: list[dict[str, Any]],
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not rows:
        raise ValueError("no negative rows available")
    first_index = int(rng.integers(0, len(rows)))
    errors: list[str] = []
    for attempt in range(min(len(rows), 8)):
        index = (first_index + attempt) % len(rows)
        row = rows[index]
        try:
            audio, detail = load_negative_audio(row, samples=samples, rng=rng)
            detail["row_index"] = index
            return audio, detail
        except Exception as exc:
            errors.append(f"{index}:{exc}")
    raise ValueError("; ".join(errors) or "failed to load real negative gap")


def synthesize_gap(
    *,
    samples: int,
    sample_rate: int,
    index: int,
    rng: np.random.Generator,
    noise_rms: float,
    hum_rms: float,
) -> tuple[np.ndarray, str]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), "empty"
    mode = index % 4
    if mode == 0:
        return np.zeros(samples, dtype=np.float32), "silence"
    if mode == 1:
        return rng.normal(0.0, max(0.0, noise_rms), samples).astype(np.float32), "white_noise"
    if mode == 2:
        time = np.arange(samples, dtype=np.float32) / sample_rate
        frequency = 80.0 + float(rng.integers(0, 80))
        return (max(0.0, hum_rms) * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32), "hum"
    base = rng.normal(0.0, max(0.0, noise_rms), samples).astype(np.float32)
    envelope = np.linspace(0.1, 1.0, samples, dtype=np.float32)
    return (base * envelope).astype(np.float32), "fade_noise"


def build_gap(
    *,
    samples: int,
    sample_rate: int,
    index: int,
    rng: np.random.Generator,
    noise_rms: float,
    hum_rms: float,
    negative_rows: list[dict[str, Any]],
    negative_gap_prob: float,
) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    if samples <= 0:
        return np.zeros(0, dtype=np.float32), "empty", None
    if negative_rows and negative_gap_prob > 0.0 and float(rng.random()) < negative_gap_prob:
        try:
            audio, detail = choose_real_negative_gap(rows=negative_rows, samples=samples, rng=rng)
            return audio, "real_negative", detail
        except Exception as exc:
            detail = {"error": str(exc), "fallback": "synthetic"}
    else:
        detail = None
    audio, mode = synthesize_gap(
        samples=samples,
        sample_rate=sample_rate,
        index=index,
        rng=rng,
        noise_rms=noise_rms,
        hum_rms=hum_rms,
    )
    return audio, mode, detail


def expand_segments(
    segments: list[TeacherSegment],
    *,
    duration_s: float,
    pad_s: float,
) -> list[TeacherSegment]:
    if pad_s <= 0.0:
        return list(segments)
    expanded = [
        TeacherSegment(
            start=max(0.0, segment.start - pad_s),
            end=min(duration_s, segment.end + pad_s),
            score=segment.score,
        )
        for segment in segments
    ]
    merged: list[TeacherSegment] = []
    for segment in sorted(expanded, key=lambda item: (item.start, item.end)):
        if segment.end <= segment.start:
            continue
        if not merged or segment.start > merged[-1].end:
            merged.append(segment)
        else:
            previous = merged[-1]
            merged[-1] = TeacherSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                score=previous.score if previous.score is not None else segment.score,
            )
    return merged


def mix_background_audio(
    audio: np.ndarray,
    *,
    background_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    snr_db_min: float,
    snr_db_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if audio.size <= 0:
        return audio, {"skipped": "empty_audio"}
    background_row_index = int(rng.integers(0, len(background_rows)))
    background, detail = load_negative_audio(
        background_rows[background_row_index],
        samples=audio.size,
        rng=rng,
    )
    snr_db = float(rng.uniform(snr_db_min, snr_db_max)) if snr_db_max > snr_db_min else float(snr_db_max)
    signal_rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64, copy=False)))))
    noise_rms = float(np.sqrt(np.mean(np.square(background.astype(np.float64, copy=False)))))
    if signal_rms <= 1e-8 or noise_rms <= 1e-8:
        return audio, {"skipped": "low_rms", "snr_db": snr_db, **detail}
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled = background * (target_noise_rms / max(noise_rms, 1e-12))
    mixed = np.clip(audio + scaled, -1.0, 1.0).astype(np.float32, copy=False)
    detail.update(
        {
            "row_index": background_row_index,
            "snr_db": snr_db,
            "signal_rms": signal_rms,
            "background_rms": noise_rms,
            "target_background_rms": target_noise_rms,
        }
    )
    return mixed, detail


def gap_samples(
    *,
    rng: np.random.Generator,
    sample_rate: int,
    min_s: float,
    max_s: float,
) -> int:
    if max_s <= 0.0:
        return 0
    lower = max(0.0, min_s)
    upper = max(lower, max_s)
    duration_s = float(rng.uniform(lower, upper)) if upper > lower else upper
    return max(0, int(round(duration_s * sample_rate)))


def build_synthetic_timeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    source_rows, skipped = valid_source_rows(load_manifest_rows(Path(args.manifest)))
    negative_rows, negative_skipped = load_valid_manifest_rows(args.negative_manifest, role="negative_gap")
    background_rows, background_skipped = load_valid_manifest_rows(args.background_manifest, role="background")
    skipped.extend(negative_skipped)
    skipped.extend(background_skipped)
    if args.shuffle:
        rng.shuffle(source_rows)
        rng.shuffle(negative_rows)
        rng.shuffle(background_rows)
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError("no valid Galgame source rows available")

    sample_rate = 16000
    records = []
    manifest_rows = []
    detail_rows = []
    gap_mode_counts: Counter[str] = Counter()
    background_mix_count = 0
    background_skip_count = 0
    source_cursor = 0
    synthetic_count = min(args.count, math.ceil(len(source_rows) / args.speech_clips_per_example))
    if args.reuse_sources:
        synthetic_count = args.count
    for output_index in range(synthetic_count):
        parts: list[np.ndarray] = []
        speech_segments: list[TeacherSegment] = []
        source_details: list[dict[str, Any]] = []
        current_sample = 0
        gap_index = output_index * (args.speech_clips_per_example + 2)

        leading_samples = gap_samples(
            rng=rng,
            sample_rate=sample_rate,
            min_s=args.leading_gap_min_s,
            max_s=args.leading_gap_max_s,
        )
        leading_gap, mode, gap_detail = build_gap(
            samples=leading_samples,
            sample_rate=sample_rate,
            index=gap_index,
            rng=rng,
            noise_rms=args.noise_rms,
            hum_rms=args.hum_rms,
            negative_rows=negative_rows,
            negative_gap_prob=args.negative_gap_prob,
        )
        if leading_gap.size:
            parts.append(leading_gap)
            current_sample += leading_gap.size
            gap_mode_counts[mode] += 1
            if gap_detail:
                source_details.append({"gap": "leading", "mode": mode, **gap_detail})

        for speech_index in range(args.speech_clips_per_example):
            if source_cursor >= len(source_rows):
                if not args.reuse_sources:
                    break
                source_cursor = 0
                if args.shuffle:
                    rng.shuffle(source_rows)
            row = source_rows[source_cursor]
            source_cursor += 1
            try:
                speech_audio, source_detail = clipped_speech_audio(
                    row,
                    rng=rng,
                    trim_head_s=args.trim_head_s,
                    trim_tail_s=args.trim_tail_s,
                    max_speech_s=args.max_speech_s,
                    min_speech_s=args.min_speech_s,
                )
            except Exception as exc:
                skipped.append(
                    {
                        "index": source_cursor - 1,
                        "audio_id": str(row.get("audio_id") or ""),
                        "reason": "speech_load_error",
                        "error": str(exc),
                    }
                )
                continue
            speech_start_sample = current_sample
            parts.append(speech_audio)
            current_sample += speech_audio.size
            speech_end_sample = current_sample
            speech_segments.append(
                TeacherSegment(
                    start=speech_start_sample / sample_rate,
                    end=speech_end_sample / sample_rate,
                    score=1.0,
                )
            )
            source_detail.update(
                {
                    "synthetic_start_s": speech_start_sample / sample_rate,
                    "synthetic_end_s": speech_end_sample / sample_rate,
                }
            )
            source_details.append(source_detail)

            if speech_index < args.speech_clips_per_example - 1:
                middle_samples = gap_samples(
                    rng=rng,
                    sample_rate=sample_rate,
                    min_s=args.gap_min_s,
                    max_s=args.gap_max_s,
                )
                middle_gap, mode, gap_detail = build_gap(
                    samples=middle_samples,
                    sample_rate=sample_rate,
                    index=gap_index + speech_index + 1,
                    rng=rng,
                    noise_rms=args.noise_rms,
                    hum_rms=args.hum_rms,
                    negative_rows=negative_rows,
                    negative_gap_prob=args.negative_gap_prob,
                )
                if middle_gap.size:
                    parts.append(middle_gap)
                    current_sample += middle_gap.size
                    gap_mode_counts[mode] += 1
                    if gap_detail:
                        source_details.append(
                            {"gap": f"middle-{speech_index}", "mode": mode, **gap_detail}
                        )

        trailing_samples = gap_samples(
            rng=rng,
            sample_rate=sample_rate,
            min_s=args.trailing_gap_min_s,
            max_s=args.trailing_gap_max_s,
        )
        trailing_gap, mode, gap_detail = build_gap(
            samples=trailing_samples,
            sample_rate=sample_rate,
            index=gap_index + args.speech_clips_per_example + 1,
            rng=rng,
            noise_rms=args.noise_rms,
            hum_rms=args.hum_rms,
            negative_rows=negative_rows,
            negative_gap_prob=args.negative_gap_prob,
        )
        if trailing_gap.size:
            parts.append(trailing_gap)
            current_sample += trailing_gap.size
            gap_mode_counts[mode] += 1
            if gap_detail:
                source_details.append({"gap": "trailing", "mode": mode, **gap_detail})

        if not speech_segments or not parts:
            continue
        audio = np.concatenate(parts).astype(np.float32, copy=False)
        background_detail: dict[str, Any] | None = None
        if background_rows and args.background_mix_prob > 0.0 and float(rng.random()) < args.background_mix_prob:
            try:
                audio, background_detail = mix_background_audio(
                    audio,
                    background_rows=background_rows,
                    rng=rng,
                    snr_db_min=args.background_snr_db_min,
                    snr_db_max=args.background_snr_db_max,
                )
                if background_detail.get("skipped"):
                    background_skip_count += 1
                else:
                    background_mix_count += 1
            except Exception as exc:
                background_skip_count += 1
                background_detail = {"error": str(exc)}
        audio = np.clip(audio, -1.0, 1.0)
        audio_id = f"{args.audio_id_prefix}-{output_index:06d}"
        audio_path = audio_dir / f"{audio_id}.wav"
        sf.write(str(audio_path), audio, sample_rate)
        duration_s = len(audio) / sample_rate
        label_segments = expand_segments(
            speech_segments,
            duration_s=duration_s,
            pad_s=args.speech_label_pad_s,
        )
        record = build_supervised_record(
            audio_id=audio_id,
            source=args.source,
            duration_s=duration_s,
            text=args.text_separator.join(
                str(item.get("source_text") or "") for item in source_details if item.get("source_text")
            ),
            speech_segments=label_segments,
            frame_hop_s=args.frame_hop_s,
        )
        records.append(record)
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio": str(audio_path),
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "source": args.source,
                "label_quality": record.label_quality,
                "input": ",".join(
                    str(item.get("source_input") or "") for item in source_details if item.get("source_input")
                ),
                "source_audio_ids": [
                    str(item.get("source_audio_id")) for item in source_details if item.get("source_audio_id")
                ],
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "speech_label_pad_s": args.speech_label_pad_s,
            }
        )
        detail_rows.append(
            {
                "audio_id": audio_id,
                "duration_s": duration_s,
                "speech_segments": [{"start": segment.start, "end": segment.end} for segment in label_segments],
                "actual_speech_segments": [{"start": segment.start, "end": segment.end} for segment in speech_segments],
                "sources": source_details,
                "background_mix": background_detail,
            }
        )

    labels_path = output_dir / args.output_jsonl
    manifest_path = output_dir / args.output_manifest
    details_path = output_dir / "synthetic_timeline_details.jsonl"
    skipped_path = output_dir / "synthetic_timeline_skipped.json"
    summary_path = output_dir / "synthetic_timeline_summary.json"
    write_jsonl(labels_path, records)
    manifest_path.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with details_path.open("w", encoding="utf-8") as handle:
        for row in detail_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    skipped_path.write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    total_frames = sum(len(record.speech_frames) for record in records)
    speech_frames = sum(sum(int(value) for value in record.speech_frames) for record in records)
    summary = {
        "manifest": str(Path(args.manifest)),
        "records": len(records),
        "source_rows": len(source_rows),
        "negative_rows": len(negative_rows),
        "background_rows": len(background_rows),
        "skipped": len(skipped),
        "duration_s_total": sum(record.duration_s for record in records),
        "speech_frame_ratio": (speech_frames / total_frames) if total_frames else 0.0,
        "label_quality_counts": dict(sorted(Counter(record.label_quality for record in records).items())),
        "gap_mode_counts": dict(sorted(gap_mode_counts.items())),
        "background_mix_count": background_mix_count,
        "background_skip_count": background_skip_count,
        "labels": str(labels_path),
        "output_manifest": str(manifest_path),
        "details": str(details_path),
        "skipped_report": str(skipped_path),
        "config": {
            "count": args.count,
            "speech_clips_per_example": args.speech_clips_per_example,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "reuse_sources": args.reuse_sources,
            "trim_head_s": args.trim_head_s,
            "trim_tail_s": args.trim_tail_s,
            "max_speech_s": args.max_speech_s,
            "min_speech_s": args.min_speech_s,
            "gap_min_s": args.gap_min_s,
            "gap_max_s": args.gap_max_s,
            "leading_gap_min_s": args.leading_gap_min_s,
            "leading_gap_max_s": args.leading_gap_max_s,
            "trailing_gap_min_s": args.trailing_gap_min_s,
            "trailing_gap_max_s": args.trailing_gap_max_s,
            "noise_rms": args.noise_rms,
            "hum_rms": args.hum_rms,
            "negative_manifest": args.negative_manifest,
            "negative_gap_prob": args.negative_gap_prob,
            "background_manifest": args.background_manifest,
            "background_mix_prob": args.background_mix_prob,
            "background_snr_db_min": args.background_snr_db_min,
            "background_snr_db_max": args.background_snr_db_max,
            "speech_label_pad_s": args.speech_label_pad_s,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"labels={labels_path}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact-timeline supervised VAD clips by concatenating Galgame speech islands and synthetic gaps."
    )
    parser.add_argument("--manifest", required=True, help="hf_audio_manifest.json from materialized Galgame audio.")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Optional source row limit before synthesis.")
    parser.add_argument("--speech-clips-per-example", type=int, default=2)
    parser.add_argument("--reuse-sources", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--source", default="galgame-synthetic-timeline")
    parser.add_argument("--audio-id-prefix", default="galgame-synth")
    parser.add_argument("--frame-hop-s", type=float, default=0.02)
    parser.add_argument("--trim-head-s", type=float, default=0.0)
    parser.add_argument("--trim-tail-s", type=float, default=0.0)
    parser.add_argument("--max-speech-s", type=float, default=4.0, help="Crop speech islands to this length; <=0 disables.")
    parser.add_argument("--min-speech-s", type=float, default=0.2)
    parser.add_argument("--gap-min-s", type=float, default=0.15)
    parser.add_argument("--gap-max-s", type=float, default=0.80)
    parser.add_argument("--leading-gap-min-s", type=float, default=0.10)
    parser.add_argument("--leading-gap-max-s", type=float, default=0.50)
    parser.add_argument("--trailing-gap-min-s", type=float, default=0.10)
    parser.add_argument("--trailing-gap-max-s", type=float, default=0.50)
    parser.add_argument("--noise-rms", type=float, default=0.01)
    parser.add_argument("--hum-rms", type=float, default=0.02)
    parser.add_argument(
        "--negative-manifest",
        action="append",
        help="Optional negative clip manifest JSON. Repeatable; used for real non-speech gaps.",
    )
    parser.add_argument("--negative-gap-prob", type=float, default=0.0)
    parser.add_argument(
        "--background-manifest",
        action="append",
        help="Optional negative clip manifest JSON. Repeatable; mixed under the final audio at SNR.",
    )
    parser.add_argument("--background-mix-prob", type=float, default=0.0)
    parser.add_argument("--background-snr-db-min", type=float, default=5.0)
    parser.add_argument("--background-snr-db-max", type=float, default=20.0)
    parser.add_argument(
        "--speech-label-pad-s",
        type=float,
        default=0.0,
        help="Expand speech labels on both sides without changing audio, useful for high-recall boundary training.",
    )
    parser.add_argument("--text-separator", default=" ")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "agents" / "temp" / "fusionvad-ja" / "galgame-synthetic-timeline"))
    parser.add_argument("--output-jsonl", default="labels.jsonl")
    parser.add_argument("--output-manifest", default="manifest.json")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.speech_clips_per_example <= 0:
        parser.error("--speech-clips-per-example must be positive")
    if args.frame_hop_s <= 0.0:
        parser.error("--frame-hop-s must be positive")
    if args.min_speech_s <= 0.0:
        parser.error("--min-speech-s must be positive")
    for name in (
        "trim_head_s",
        "trim_tail_s",
        "gap_min_s",
        "gap_max_s",
        "leading_gap_min_s",
        "leading_gap_max_s",
        "trailing_gap_min_s",
        "trailing_gap_max_s",
        "noise_rms",
        "hum_rms",
        "negative_gap_prob",
        "background_mix_prob",
        "speech_label_pad_s",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("negative_gap_prob", "background_mix_prob"):
        if getattr(args, name) > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be <= 1")
    if args.gap_max_s < args.gap_min_s:
        parser.error("--gap-max-s must be >= --gap-min-s")
    if args.leading_gap_max_s < args.leading_gap_min_s:
        parser.error("--leading-gap-max-s must be >= --leading-gap-min-s")
    if args.trailing_gap_max_s < args.trailing_gap_min_s:
        parser.error("--trailing-gap-max-s must be >= --trailing-gap-min-s")
    if args.background_snr_db_max < args.background_snr_db_min:
        parser.error("--background-snr-db-max must be >= --background-snr-db-min")
    return args


if __name__ == "__main__":
    build_synthetic_timeline(parse_args())
