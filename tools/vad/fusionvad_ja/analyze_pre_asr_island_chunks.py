#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def compact_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
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
        if end < start:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "duration_s": max(0.0, end - start),
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
        vad_segments = normalize_segments(item.get("vad_segments") or [])
        chunks.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "duration_s": max(0.0, end - start),
                "core_start": row_float(item, "core_start", vad_segments[0]["start"] if vad_segments else start),
                "core_end": row_float(item, "core_end", vad_segments[-1]["end"] if vad_segments else end),
                "split_reason": str(item.get("split_reason") or ""),
                "vad_segments": vad_segments,
                "vad_seg_count": len(vad_segments),
            }
        )
    return chunks


def internal_gaps(segments: list[dict[str, float]]) -> list[float]:
    return [
        max(0.0, current["start"] - previous["end"])
        for previous, current in zip(segments, segments[1:])
    ]


def match_diagnostics(
    chunk: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    *,
    min_overlap_ratio: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    c_start = row_float(chunk, "start")
    c_end = row_float(chunk, "end", c_start)
    c_duration = max(0.001, c_end - c_start)
    for row in diagnostics:
        d_start = row_float(row, "start")
        d_end = row_float(row, "end", d_start)
        overlap = overlap_s(c_start, c_end, d_start, d_end)
        if overlap <= 0:
            continue
        d_duration = max(0.001, d_end - d_start)
        ratio = overlap / min(c_duration, d_duration)
        if ratio >= min_overlap_ratio:
            enriched = dict(row)
            enriched["_overlap_s"] = round(overlap, 6)
            enriched["_overlap_ratio"] = round(ratio, 6)
            matches.append(enriched)
    return matches


def classify_chunk(
    chunk: dict[str, Any],
    *,
    diagnostics: list[dict[str, Any]],
    long_chunk_s: float,
    long_gap_s: float,
    min_overlap_ratio: float,
) -> dict[str, Any]:
    segments = list(chunk["vad_segments"])
    gaps = internal_gaps(segments)
    matched = match_diagnostics(
        chunk,
        diagnostics,
        min_overlap_ratio=min_overlap_ratio,
    )
    fallback_subtypes = Counter(str(row.get("fallback_subtype") or "") for row in matched)
    failure_buckets = Counter(str(row.get("failure_bucket") or "") for row in matched)
    qualities = Counter(str(row.get("alignment_quality") or "") for row in matched)
    reasons: list[str] = []
    if len(segments) > 1 and max(gaps or [0.0]) >= long_gap_s:
        reasons.append("multi_island_long_gap")
    if len(segments) > 1 and max(gaps or [0.0]) < long_gap_s:
        reasons.append("multi_island_short_gap")
    if len(segments) <= 1 and chunk["duration_s"] >= long_chunk_s:
        reasons.append("continuous_speech_no_internal_gap")
    if chunk["duration_s"] >= long_chunk_s:
        reasons.append("long_chunk")
    if any(str(row.get("fallback_subtype") or "") == "vad_coarse_after_sentinel" for row in matched):
        reasons.append("vad_coarse_after_sentinel")
    if any("empty_text_for_chunk" in (row.get("failure_reasons") or []) for row in matched):
        reasons.append("asr_empty")
    if any(str(row.get("failure_bucket") or "") in {"repeat_repair_suggested", "long_low_information_text", "low_information_text"} for row in matched):
        reasons.append("low_information_or_repeat")
    if not reasons:
        reasons.append("normal_or_unmatched")

    return {
        "chunk_index": chunk["index"],
        "start": round(chunk["start"], 3),
        "end": round(chunk["end"], 3),
        "duration_s": round(chunk["duration_s"], 3),
        "core_start": round(chunk["core_start"], 3),
        "core_end": round(chunk["core_end"], 3),
        "core_duration_s": round(max(0.0, chunk["core_end"] - chunk["core_start"]), 3),
        "vad_seg_count": chunk["vad_seg_count"],
        "internal_gap_count": len([gap for gap in gaps if gap > 0.0]),
        "internal_gap_max_s": round(max(gaps or [0.0]), 3),
        "split_reason": chunk["split_reason"],
        "risk_reasons": reasons,
        "diagnostic_match_count": len(matched),
        "matched_chunk_indices": [row.get("chunk_index") for row in matched],
        "alignment_quality_counts": dict(qualities),
        "fallback_subtype_counts": dict(fallback_subtypes),
        "failure_bucket_counts": dict(failure_buckets),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [row_float(row, "duration_s") for row in rows]
    core_durations = [row_float(row, "core_duration_s") for row in rows]
    gap_max = [row_float(row, "internal_gap_max_s") for row in rows]
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row.get("risk_reasons") or [])
    return {
        "chunk_count": len(rows),
        "duration_s": compact_stats(durations),
        "core_duration_s": compact_stats(core_durations),
        "internal_gap_max_s": compact_stats(gap_max),
        "risk_reason_counts": dict(reason_counts.most_common()),
        "multi_island_count": sum(1 for row in rows if int(row.get("vad_seg_count") or 0) > 1),
        "matched_diagnostic_chunk_count": sum(1 for row in rows if row.get("diagnostic_match_count")),
    }


def build_markdown(summary: dict[str, Any], *, cache_path: Path, diagnostics_path: Path | None) -> str:
    lines = [
        "# Pre-ASR Speech-Island Chunk Analysis",
        "",
        f"- vad cache: `{project_rel(cache_path)}`",
        f"- diagnostics: `{project_rel(diagnostics_path) if diagnostics_path else ''}`",
        f"- chunks: {summary['chunk_count']}",
        f"- multi-island chunks: {summary['multi_island_count']}",
        f"- matched diagnostics chunks: {summary['matched_diagnostic_chunk_count']}",
        "",
        "## Duration",
        "",
    ]
    for key, stats in (
        ("duration_s", summary["duration_s"]),
        ("core_duration_s", summary["core_duration_s"]),
        ("internal_gap_max_s", summary["internal_gap_max_s"]),
    ):
        lines.append(
            f"- `{key}`: p50={stats['p50']:.3f}, p90={stats['p90']:.3f}, "
            f"p95={stats['p95']:.3f}, max={stats['max']:.3f}"
        )
    lines.extend(["", "## Risk Reasons", ""])
    for reason, count in summary.get("risk_reason_counts", {}).items():
        lines.append(f"- `{reason}`: {count}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze packed VAD chunks for R15/R16 pre-ASR speech-island splitting."
    )
    parser.add_argument("--vad-cache", required=True, help="VAD chunk cache JSON path")
    parser.add_argument("--diagnostics", help="diagnostics.jsonl or diagnostics directory")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/pre-asr-island-analysis",
    )
    parser.add_argument("--long-chunk-s", type=float, default=14.0)
    parser.add_argument("--long-gap-s", type=float, default=0.6)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    args = parser.parse_args(argv)

    cache_path = project_path(args.vad_cache)
    diagnostics_path: Path | None = None
    if args.diagnostics:
        diagnostics_path = project_path(args.diagnostics)
        if diagnostics_path.is_dir():
            diagnostics_path = diagnostics_path / "diagnostics.jsonl"

    cache_payload = read_json(cache_path)
    if not isinstance(cache_payload, dict):
        raise ValueError(f"VAD cache must be a JSON object: {cache_path}")
    chunks = normalize_chunks(cache_payload)
    diagnostics = read_jsonl(diagnostics_path) if diagnostics_path else []
    rows = [
        classify_chunk(
            chunk,
            diagnostics=diagnostics,
            long_chunk_s=args.long_chunk_s,
            long_gap_s=args.long_gap_s,
            min_overlap_ratio=args.min_overlap_ratio,
        )
        for chunk in chunks
    ]
    summary = summarize(rows)
    summary.update(
        {
            "vad_cache": project_rel(cache_path),
            "diagnostics": project_rel(diagnostics_path),
            "long_chunk_s": args.long_chunk_s,
            "long_gap_s": args.long_gap_s,
            "min_overlap_ratio": args.min_overlap_ratio,
        }
    )

    output_dir = project_path(args.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "chunk_analysis.jsonl", rows)
    (output_dir / "summary.md").write_text(
        build_markdown(summary, cache_path=cache_path, diagnostics_path=diagnostics_path),
        encoding="utf-8",
    )
    print(f"summary={project_rel(output_dir / 'summary.json')}")
    print(f"rows={project_rel(output_dir / 'chunk_analysis.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
