#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc import build_candidate, build_candidates


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def aligned_payload_to_candidates(payload: Mapping[str, Any], *, video_id: str = "") -> list[dict[str, Any]]:
    details = payload.get("asr_details") if isinstance(payload.get("asr_details"), Mapping) else {}
    transcript_chunks = [
        dict(row)
        for row in details.get("transcript_chunks") or []
        if isinstance(row, Mapping)
    ]
    if not transcript_chunks:
        return []
    chunks: list[dict[str, Any]] = []
    text_results: list[dict[str, Any]] = []
    for position, row in enumerate(transcript_chunks):
        chunk_index = int(row.get("index", position))
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start))
        audio_path = str(
            row.get("audio_path")
            or row.get("normalized_path")
            or payload.get("audio_path")
            or ""
        )
        chunks.append(
            {
                "index": chunk_index,
                "start": start,
                "end": end,
                "duration": float(row.get("duration", max(0.0, end - start)) or 0.0),
                "path": audio_path,
                "source_audio_path": str(payload.get("audio_path") or ""),
                **{
                    key: row[key]
                    for key in (
                        "speech_segment_count",
                        "boundary_split_reason",
                        "boundary_parent_chunk_id",
                        "speech_island_id",
                        "speech_island_count",
                        "speech_internal_gap_count",
                        "speech_internal_gap_max_s",
                        "boundary_score",
                        "boundary_reason",
                        "boundary_source",
                        "boundary_start_refine_delta_s",
                        "boundary_end_refine_delta_s",
                        "boundary_decision_source",
                    )
                    if key in row
                },
            }
        )
        text_results.append(
            {
                "text": str(row.get("text") or ""),
                "raw_text": str(row.get("raw_text") or row.get("text") or ""),
                "duration": float(row.get("duration", max(0.0, end - start)) or 0.0),
                "language": str(row.get("language") or "Japanese"),
                "normalized_path": audio_path,
                "avg_logprob": row.get("avg_logprob"),
                "no_speech_prob": row.get("no_speech_prob"),
                "compression_ratio": row.get("compression_ratio"),
                "alignment_fallback_start_s": row.get("alignment_fallback_start_s"),
                "alignment_fallback_end_s": row.get("alignment_fallback_end_s"),
                "alignment_fallback_source": row.get("alignment_fallback_source", ""),
            }
        )
    audio_id = Path(str(payload.get("audio_path") or video_id or "audio")).stem
    return build_candidates(
        chunks,
        text_results,
        audio_id=audio_id,
        video_id=video_id or audio_id,
    )


def diagnostics_to_candidates(rows: list[Mapping[str, Any]], *, video_id: str = "") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    text_results: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or start)
        chunk_index = int(row.get("chunk_index", row.get("index", position)))
        chunks.append(
            {
                "index": chunk_index,
                "start": start,
                "end": end,
                "duration": float(row.get("duration_s") or row.get("duration") or max(0.0, end - start)),
                "path": str(row.get("audio") or row.get("chunk_path") or ""),
                "source_audio_path": str(row.get("source_audio_path") or ""),
                **{
                    key: row[key]
                    for key in (
                        "speech_segment_count",
                        "boundary_split_reason",
                        "boundary_parent_chunk_id",
                        "speech_island_id",
                        "speech_island_count",
                        "speech_internal_gap_count",
                        "speech_internal_gap_max_s",
                        "boundary_score",
                        "boundary_reason",
                        "boundary_source",
                        "boundary_start_refine_delta_s",
                        "boundary_end_refine_delta_s",
                        "boundary_decision_source",
                    )
                    if key in row
                },
            }
        )
        text_results.append(
            {
                "text": str(row.get("text") or row.get("display_text") or ""),
                "raw_text": str(row.get("raw_text") or row.get("text") or ""),
                "duration": float(row.get("duration_s") or row.get("duration") or max(0.0, end - start)),
                "language": str(row.get("language") or "Japanese"),
                "normalized_path": str(row.get("audio") or row.get("chunk_path") or ""),
                "avg_logprob": row.get("avg_logprob"),
                "no_speech_prob": row.get("no_speech_prob"),
                "compression_ratio": row.get("compression_ratio"),
                "alignment_fallback_start_s": row.get("fallback_window_start"),
                "alignment_fallback_end_s": row.get("fallback_window_end"),
                "alignment_fallback_source": row.get("fallback_window_source", ""),
            }
        )
    for position, (chunk, result) in enumerate(zip(chunks, text_results)):
        candidates.append(
            build_candidate(
                chunk=chunk,
                text_result=result,
                position=position,
                chunks=chunks,
                text_results=text_results,
                audio_id=video_id or "diagnostics",
                video_id=video_id or str(rows[position].get("video") or "diagnostics"),
            )
        )
    return candidates


def run(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]]
    if args.aligned:
        payload = read_json(Path(args.aligned))
        if not isinstance(payload, Mapping):
            raise SystemExit("--aligned must point to a JSON object payload")
        rows = aligned_payload_to_candidates(payload, video_id=args.video_id)
    elif args.diagnostics:
        rows = diagnostics_to_candidates(read_jsonl(Path(args.diagnostics)), video_id=args.video_id)
    else:
        raise SystemExit("one of --aligned or --diagnostics is required")
    if args.max_items is not None:
        rows = rows[: args.max_items]
    count = write_jsonl(Path(args.output), rows)
    summary = {
        "schema": "cueqc_candidate_export_summary_v1",
        "output": args.output,
        "candidate_count": count,
        "source_aligned": args.aligned or "",
        "source_diagnostics": args.diagnostics or "",
        "video_id": args.video_id,
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"candidates={args.output}")
    print(f"count={count}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CueQC cluster-first candidate JSONL.")
    parser.add_argument("--aligned", help="aligned_segments.json payload from a full run")
    parser.add_argument("--diagnostics", help="diagnostics JSONL with chunk/text rows")
    parser.add_argument("--output", required=True, help="cueqc_candidates.jsonl")
    parser.add_argument("--summary", default="", help="optional summary JSON")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args(argv)
    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
