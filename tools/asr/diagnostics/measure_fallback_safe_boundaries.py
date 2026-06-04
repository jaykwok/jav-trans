#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio.loading import load_audio_16k_mono  # noqa: E402


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def project_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p50": round(quantile(values, 0.5), 6),
        "p90": round(quantile(values, 0.9), 6),
        "p95": round(quantile(values, 0.95), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
    }


def normalize_segments(raw_segments: Any) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    for item in raw_segments or []:
        if not isinstance(item, dict):
            continue
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        if end <= start:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "duration_s": end - start,
                "score": row_float(item, "score", 0.0),
            }
        )
    return sorted(segments, key=lambda item: (item["start"], item["end"]))


def normalize_chunks(cache_payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, item in enumerate(cache_payload.get("processing_spans") or []):
        if not isinstance(item, dict):
            continue
        start = row_float(item, "start")
        end = row_float(item, "end", start)
        segments = normalize_segments(item.get("speech_segments") or [])
        core_start = row_float(item, "core_start", segments[0]["start"] if segments else start)
        core_end = row_float(item, "core_end", segments[-1]["end"] if segments else end)
        gaps = [
            max(0.0, current["start"] - previous["end"])
            for previous, current in zip(segments, segments[1:])
        ]
        chunks.append(
            {
                "chunk_index": index,
                "start": start,
                "end": end,
                "duration_s": max(0.0, end - start),
                "core_start": core_start,
                "core_end": core_end,
                "core_duration_s": max(0.0, core_end - core_start),
                "split_reason": str(item.get("split_reason") or ""),
                "speech_segments": segments,
                "speech_island_count": len(segments),
                "internal_gap_count": sum(1 for gap in gaps if gap > 0.0),
                "internal_gap_max_s": max(gaps or [0.0]),
                "internal_gap_total_s": sum(gaps),
                "left_padding_s": row_float(item, "left_padding_s", max(0.0, core_start - start)),
                "right_padding_s": row_float(item, "right_padding_s", max(0.0, end - core_end)),
            }
        )
    return chunks


def load_boundary_truth(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        audio_id = str(row.get("audio_id") or "")
        if audio_id:
            rows[audio_id] = row
    return rows


def cache_audio_id(cache_payload: dict[str, Any], diagnostics: list[dict[str, Any]], boundary_cache: Path) -> str:
    for row in diagnostics:
        video = str(row.get("video") or "")
        if video:
            return Path(video).stem
    audio_payload = dict(dict(cache_payload.get("signature") or {}).get("audio") or {})
    for key in ("path", "file", "name"):
        value = str(audio_payload.get(key) or "")
        if value:
            return Path(value).stem
    return boundary_cache.stem.split(".", 1)[0]


def parse_truth_segments(row: dict[str, Any]) -> list[dict[str, float]]:
    duration_s = row_float(row, "duration_s", 0.0)
    segments = []
    for item in row.get("actual_speech_segments") or row.get("speech_segments") or []:
        if not isinstance(item, dict):
            continue
        start = max(0.0, min(row_float(item, "start", row_float(item, "start_s", 0.0)), duration_s))
        end = max(0.0, min(row_float(item, "end", row_float(item, "end_s", 0.0)), duration_s))
        if end > start:
            segments.append({"start": start, "end": end})
    return sorted(segments, key=lambda item: (item["start"], item["end"]))


def best_truth_overlap(chunk: dict[str, Any], truth_segments: list[dict[str, float]]) -> tuple[dict[str, float] | None, float]:
    best_segment = None
    best_overlap = 0.0
    for segment in truth_segments:
        overlap = overlap_s(chunk["start"], chunk["end"], segment["start"], segment["end"])
        if overlap > best_overlap:
            best_segment = segment
            best_overlap = overlap
    return best_segment, best_overlap


def silence_stats(
    *,
    audio_cache: dict[str, tuple[np.ndarray, int]],
    source_audio_path: str,
    start_s: float,
    end_s: float,
    frame_s: float,
    threshold_dbfs: float,
) -> dict[str, float]:
    if not source_audio_path or end_s <= start_s:
        return {
            "total_silence_s": 0.0,
            "longest_silence_s": 0.0,
            "silence_ratio": 0.0,
            "measured_duration_s": 0.0,
        }
    path = project_path(source_audio_path)
    cache_key = str(path)
    if cache_key not in audio_cache:
        audio_cache[cache_key] = load_audio_16k_mono(str(path))
    audio, sample_rate = audio_cache[cache_key]
    start_sample = max(0, min(int(round(start_s * sample_rate)), int(audio.shape[0])))
    end_sample = max(start_sample, min(int(round(end_s * sample_rate)), int(audio.shape[0])))
    chunk = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    measured_duration_s = float(chunk.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    if chunk.size == 0 or frame_s <= 0.0:
        return {
            "total_silence_s": 0.0,
            "longest_silence_s": 0.0,
            "silence_ratio": 0.0,
            "measured_duration_s": measured_duration_s,
        }
    frame_samples = max(1, int(round(frame_s * sample_rate)))
    silent_frames: list[bool] = []
    frame_durations: list[float] = []
    for start in range(0, int(chunk.shape[0]), frame_samples):
        frame = chunk[start : min(int(chunk.shape[0]), start + frame_samples)]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float64, copy=False)))))
        dbfs = 20.0 * np.log10(max(rms, 1e-12))
        silent_frames.append(dbfs <= threshold_dbfs)
        frame_durations.append(float(frame.size / sample_rate))
    total_silence_s = sum(duration for silent, duration in zip(silent_frames, frame_durations) if silent)
    longest = 0.0
    current = 0.0
    for silent, duration in zip(silent_frames, frame_durations):
        if silent:
            current += duration
            longest = max(longest, current)
        else:
            current = 0.0
    return {
        "total_silence_s": total_silence_s,
        "longest_silence_s": longest,
        "silence_ratio": total_silence_s / measured_duration_s if measured_duration_s > 0.0 else 0.0,
        "measured_duration_s": measured_duration_s,
    }


def diagnostics_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            result[int(row.get("chunk_index"))] = row
        except (TypeError, ValueError):
            continue
    return result


def match_diagnostic(
    chunk: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
    *,
    min_overlap_ratio: float,
) -> dict[str, Any] | None:
    chunk_index = int(chunk["chunk_index"])
    if chunk_index in by_index:
        return by_index[chunk_index]

    c_start = row_float(chunk, "start")
    c_end = row_float(chunk, "end", c_start)
    c_duration = max(0.001, c_end - c_start)
    best: tuple[float, dict[str, Any] | None] = (0.0, None)
    for row in diagnostics:
        d_start = row_float(row, "start")
        d_end = row_float(row, "end", d_start)
        overlap = overlap_s(c_start, c_end, d_start, d_end)
        if overlap <= 0:
            continue
        d_duration = max(0.001, d_end - d_start)
        ratio = overlap / min(c_duration, d_duration)
        if ratio > best[0]:
            best = (ratio, row)
    return best[1] if best[0] >= min_overlap_ratio else None


def classify_row(
    chunk: dict[str, Any],
    diagnostic: dict[str, Any] | None,
    *,
    target_duration_s: float,
    truth_segments: list[dict[str, float]],
    silence: dict[str, float] | None = None,
    long_silence_s: float = 1.0,
) -> dict[str, Any]:
    fallback_type = str((diagnostic or {}).get("fallback_type") or "").strip()
    if diagnostic is not None and fallback_type == "":
        raise ValueError(
            f"diagnostics row is missing fallback_type for chunk {diagnostic.get('chunk_index')}"
        )
    fallback_subtype = str((diagnostic or {}).get("fallback_subtype") or "").strip()
    fallback_active = fallback_type not in {"", "none"}
    fallback_reason = (
        fallback_subtype
        if fallback_active and fallback_subtype not in {"", "none"}
        else fallback_type
        if fallback_active
        else ""
    )
    sentinel_fallback = fallback_active and bool((diagnostic or {}).get("sentinel_lines") or [])
    fallback_safe = not fallback_reason or chunk["duration_s"] <= target_duration_s
    truth_segment, truth_overlap = best_truth_overlap(chunk, truth_segments)
    truth_start_error_s = None
    truth_end_error_s = None
    if truth_segment is not None and truth_overlap > 0.0:
        truth_start_error_s = chunk["start"] - truth_segment["start"]
        truth_end_error_s = chunk["end"] - truth_segment["end"]
    silence = silence or {}
    risk_reasons: list[str] = []
    if fallback_active:
        risk_reasons.append("alignment_fallback")
    if sentinel_fallback:
        risk_reasons.append("sentinel_fallback")
    if chunk["duration_s"] > target_duration_s:
        risk_reasons.append("long_fallback_chunk" if fallback_active else "long_chunk")
    if chunk["speech_island_count"] > 1:
        risk_reasons.append("multi_speech_island")
    if chunk["internal_gap_max_s"] > 0.0:
        risk_reasons.append("internal_gap")
    if fallback_active and float(silence.get("longest_silence_s") or 0.0) >= long_silence_s:
        risk_reasons.append("fallback_crosses_long_silence")
    if not risk_reasons:
        risk_reasons.append("ok")

    return {
        "chunk_index": chunk["chunk_index"],
        "start": round(chunk["start"], 6),
        "end": round(chunk["end"], 6),
        "duration_s": round(chunk["duration_s"], 6),
        "core_start": round(chunk["core_start"], 6),
        "core_end": round(chunk["core_end"], 6),
        "core_duration_s": round(chunk["core_duration_s"], 6),
        "split_reason": chunk["split_reason"],
        "speech_island_count": chunk["speech_island_count"],
        "internal_gap_count": chunk["internal_gap_count"],
        "internal_gap_max_s": round(chunk["internal_gap_max_s"], 6),
        "internal_gap_total_s": round(chunk["internal_gap_total_s"], 6),
        "left_padding_s": round(chunk["left_padding_s"], 6),
        "right_padding_s": round(chunk["right_padding_s"], 6),
        "alignment_quality": str((diagnostic or {}).get("alignment_quality") or ""),
        "fallback_type": fallback_type,
        "fallback_subtype": fallback_subtype,
        "failure_bucket": str((diagnostic or {}).get("failure_bucket") or ""),
        "fallback_reason": fallback_reason,
        "sentinel_fallback": sentinel_fallback,
        "fallback_safe": fallback_safe,
        "risk_reasons": risk_reasons,
        "truth_start_error_s": None if truth_start_error_s is None else round(truth_start_error_s, 6),
        "truth_end_error_s": None if truth_end_error_s is None else round(truth_end_error_s, 6),
        "truth_overlap_s": round(truth_overlap, 6),
        "total_silence_s": round(float(silence.get("total_silence_s") or 0.0), 6),
        "longest_silence_s": round(float(silence.get("longest_silence_s") or 0.0), 6),
        "silence_ratio": round(float(silence.get("silence_ratio") or 0.0), 6),
        "display_text": str((diagnostic or {}).get("display_text") or (diagnostic or {}).get("text") or "")[:200],
    }


def summarize(rows: list[dict[str, Any]], *, target_duration_s: float) -> dict[str, Any]:
    fallback_rows = [row for row in rows if row["fallback_reason"]]
    unsafe_rows = [row for row in fallback_rows if not row["fallback_safe"]]
    sentinel_rows = [row for row in rows if row["sentinel_fallback"]]
    reason_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    alignment_quality_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row["risk_reasons"])
        split_counts.update([row["split_reason"] or ""])
        alignment_quality_counts.update([row["alignment_quality"] or ""])
        if row["fallback_reason"]:
            fallback_reason_counts.update([row["fallback_reason"]])
    truth_start_errors = [float(row["truth_start_error_s"]) for row in rows if row["truth_start_error_s"] is not None]
    truth_end_errors = [float(row["truth_end_error_s"]) for row in rows if row["truth_end_error_s"] is not None]
    fallback_silence_rows = [
        row
        for row in fallback_rows
        if float(row.get("longest_silence_s") or 0.0) > 0.0 or float(row.get("total_silence_s") or 0.0) > 0.0
    ]

    return {
        "chunk_count": len(rows),
        "target_duration_s": target_duration_s,
        "fallback_chunk_count": len(fallback_rows),
        "fallback_unsafe_count": len(unsafe_rows),
        "fallback_safe_ratio": round(
            (len(fallback_rows) - len(unsafe_rows)) / len(fallback_rows), 6
        )
        if fallback_rows
        else 1.0,
        "sentinel_fallback_count": len(sentinel_rows),
        "all_chunk_duration_s": stats([row["duration_s"] for row in rows]),
        "fallback_duration_s": stats([row["duration_s"] for row in fallback_rows]),
        "unsafe_fallback_duration_s": stats([row["duration_s"] for row in unsafe_rows]),
        "sentinel_fallback_duration_s": stats([row["duration_s"] for row in sentinel_rows]),
        "fallback_core_duration_s": stats([row["core_duration_s"] for row in fallback_rows]),
        "fallback_internal_gap_max_s": stats([row["internal_gap_max_s"] for row in fallback_rows]),
        "fallback_speech_island_count": stats([float(row["speech_island_count"]) for row in fallback_rows]),
        "truth_start_abs_error_s": stats([abs(value) for value in truth_start_errors]),
        "truth_end_abs_error_s": stats([abs(value) for value in truth_end_errors]),
        "truth_start_signed_error_s": stats(truth_start_errors),
        "truth_end_signed_error_s": stats(truth_end_errors),
        "fallback_longest_silence_s": stats([float(row.get("longest_silence_s") or 0.0) for row in fallback_rows]),
        "fallback_total_silence_s": stats([float(row.get("total_silence_s") or 0.0) for row in fallback_rows]),
        "fallback_silence_ratio": stats([float(row.get("silence_ratio") or 0.0) for row in fallback_rows]),
        "fallback_long_silence_count": sum(
            1 for row in fallback_rows if "fallback_crosses_long_silence" in row["risk_reasons"]
        ),
        "fallback_silence_measured_count": len(fallback_silence_rows),
        "risk_reason_counts": dict(reason_counts.most_common()),
        "split_reason_counts": dict(split_counts.most_common()),
        "fallback_reason_counts": dict(fallback_reason_counts.most_common()),
        "alignment_quality_counts": dict(alignment_quality_counts.most_common()),
    }


def build_markdown(summary: dict[str, Any], *, boundary_cache: Path, diagnostics: Path, top_rows: list[dict[str, Any]]) -> str:
    fallback = summary["fallback_duration_s"]
    unsafe = summary["unsafe_fallback_duration_s"]
    sentinel = summary["sentinel_fallback_duration_s"]
    truth_start = summary["truth_start_abs_error_s"]
    truth_end = summary["truth_end_abs_error_s"]
    silence = summary["fallback_longest_silence_s"]
    lines = [
        "# Fallback-Safe Boundary Metrics",
        "",
        f"- boundary cache: `{project_rel(boundary_cache)}`",
        f"- diagnostics: `{project_rel(diagnostics)}`",
        f"- target fallback duration: `{summary['target_duration_s']:.3f}s`",
        f"- chunks: `{summary['chunk_count']}`",
        f"- alignment fallback chunks: `{summary['fallback_chunk_count']}`",
        f"- unsafe fallback chunks: `{summary['fallback_unsafe_count']}`",
        f"- fallback safe ratio: `{summary['fallback_safe_ratio']:.3f}`",
        f"- sentinel fallback chunks: `{summary['sentinel_fallback_count']}`",
        f"- fallback crossing long silence: `{summary.get('fallback_long_silence_count', 0)}`",
        "",
        "## Duration",
        "",
        f"- fallback duration p50/p90/max: `{fallback['p50']:.3f}` / `{fallback['p90']:.3f}` / `{fallback['max']:.3f}`",
        f"- unsafe fallback duration p50/p90/max: `{unsafe['p50']:.3f}` / `{unsafe['p90']:.3f}` / `{unsafe['max']:.3f}`",
        f"- sentinel fallback duration p50/p90/max: `{sentinel['p50']:.3f}` / `{sentinel['p90']:.3f}` / `{sentinel['max']:.3f}`",
        f"- synthetic truth start abs error p50/p90/max: `{truth_start['p50']:.3f}` / `{truth_start['p90']:.3f}` / `{truth_start['max']:.3f}`",
        f"- synthetic truth end abs error p50/p90/max: `{truth_end['p50']:.3f}` / `{truth_end['p90']:.3f}` / `{truth_end['max']:.3f}`",
        f"- fallback longest silence p50/p90/max: `{silence['p50']:.3f}` / `{silence['p90']:.3f}` / `{silence['max']:.3f}`",
        "",
        "## Risk Reasons",
        "",
    ]
    for reason, count in summary.get("risk_reason_counts", {}).items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Longest Unsafe Fallback Chunks", ""])
    if not top_rows:
        lines.append("- none")
    for row in top_rows:
        text = row["display_text"].replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        lines.append(
            f"- chunk `{row['chunk_index']}` {row['start']:.2f}-{row['end']:.2f}s "
            f"duration `{row['duration_s']:.2f}s`, islands `{row['speech_island_count']}`, "
            f"gap_max `{row['internal_gap_max_s']:.2f}s`, reason `{row['fallback_reason']}`: {text}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether alignment fallback chunks are short enough to keep subtitle timing usable."
    )
    parser.add_argument("--boundary-cache", required=True, help="Boundary cache JSON path")
    parser.add_argument("--diagnostics", required=True, help="diagnostics.jsonl path or diagnostics directory")
    parser.add_argument("--output-dir", default="agents/temp/speech-boundary-ja/fallback-safe-boundary-metrics")
    parser.add_argument("--boundary-manifest", help="Optional synthetic boundary_manifest.jsonl with actual_speech_segments")
    parser.add_argument("--measure-audio-silence", action="store_true", help="Measure long silence spans inside fallback chunks")
    parser.add_argument(
        "--target-duration-s",
        type=float,
        default=8.0,
        help="Maximum acceptable duration for a chunk that may fall back to coarse timing.",
    )
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--silence-frame-s", type=float, default=0.10)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--long-silence-s", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)

    boundary_cache = project_path(args.boundary_cache)
    diagnostics_path = project_path(args.diagnostics)
    if diagnostics_path.is_dir():
        diagnostics_path = diagnostics_path / "diagnostics.jsonl"
    if args.target_duration_s <= 0:
        parser.error("--target-duration-s must be positive")

    cache_payload = read_json(boundary_cache)
    if not isinstance(cache_payload, dict):
        raise ValueError(f"Boundary cache must be a JSON object: {boundary_cache}")
    chunks = normalize_chunks(cache_payload)
    diagnostics = read_jsonl(diagnostics_path)
    by_index = diagnostics_by_index(diagnostics)
    boundary_manifest = project_path(args.boundary_manifest) if args.boundary_manifest else None
    truth_rows = load_boundary_truth(boundary_manifest)
    truth_audio_id = cache_audio_id(cache_payload, diagnostics, boundary_cache)
    truth_segments = parse_truth_segments(truth_rows.get(truth_audio_id) or {})
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    rows = []
    for chunk in chunks:
        diagnostic = match_diagnostic(
            chunk,
            diagnostics,
            by_index,
            min_overlap_ratio=args.min_overlap_ratio,
        )
        silence = None
        if args.measure_audio_silence and diagnostic:
            try:
                silence = silence_stats(
                    audio_cache=audio_cache,
                    source_audio_path=str(diagnostic.get("source_audio_path") or ""),
                    start_s=row_float(diagnostic, "start", row_float(chunk, "start")),
                    end_s=row_float(diagnostic, "end", row_float(chunk, "end")),
                    frame_s=float(args.silence_frame_s),
                    threshold_dbfs=float(args.silence_threshold_dbfs),
                )
            except Exception as exc:
                silence = {"error": str(exc)}
        rows.append(
            classify_row(
                chunk,
                diagnostic,
                target_duration_s=args.target_duration_s,
                truth_segments=truth_segments,
                silence=silence,
                long_silence_s=float(args.long_silence_s),
            )
        )
    summary = summarize(rows, target_duration_s=args.target_duration_s)
    summary.update(
        {
            "boundary_cache": project_rel(boundary_cache),
            "diagnostics": project_rel(diagnostics_path),
            "boundary_manifest": project_rel(boundary_manifest),
            "truth_audio_id": truth_audio_id,
            "truth_segment_count": len(truth_segments),
            "measure_audio_silence": bool(args.measure_audio_silence),
            "silence_frame_s": float(args.silence_frame_s),
            "silence_threshold_dbfs": float(args.silence_threshold_dbfs),
            "long_silence_s": float(args.long_silence_s),
            "min_overlap_ratio": args.min_overlap_ratio,
        }
    )
    unsafe_rows = [row for row in rows if row["fallback_reason"] and not row["fallback_safe"]]
    top_rows = sorted(unsafe_rows, key=lambda row: row["duration_s"], reverse=True)[: args.top_n]

    output_dir = project_path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "chunk_metrics.jsonl", rows)
    write_jsonl(output_dir / "unsafe_fallback_chunks.jsonl", top_rows)
    (output_dir / "summary.md").write_text(
        build_markdown(summary, boundary_cache=boundary_cache, diagnostics=diagnostics_path, top_rows=top_rows),
        encoding="utf-8",
    )
    print(f"summary={project_rel(output_dir / 'summary.json')}")
    print(f"unsafe={project_rel(output_dir / 'unsafe_fallback_chunks.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
