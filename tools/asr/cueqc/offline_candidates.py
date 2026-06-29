from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from asr.cueqc import build_candidates


def infer_transcript_path(aligned_path: Path) -> Path:
    name = aligned_path.name
    if name.endswith(".aligned_segments.json"):
        return aligned_path.with_name(name[: -len(".aligned_segments.json")] + ".transcript.json")
    if name == "aligned_segments.json":
        return aligned_path.with_name("transcript.json")
    if "aligned_segments" in name:
        return aligned_path.with_name(name.replace("aligned_segments", "transcript"))
    return aligned_path.with_suffix(".transcript.json")


def transcript_chunks_from_artifacts(
    aligned_payload: Mapping[str, Any],
    *,
    transcript_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    details = (
        aligned_payload.get("asr_details")
        if isinstance(aligned_payload.get("asr_details"), Mapping)
        else {}
    )
    embedded = [
        dict(row)
        for row in details.get("transcript_chunks") or []
        if isinstance(row, Mapping)
    ]
    if embedded:
        return embedded
    if not isinstance(transcript_payload, Mapping):
        return []
    return [
        dict(row)
        for row in transcript_payload.get("chunks") or []
        if isinstance(row, Mapping)
    ]


def aligned_payload_to_candidates(
    payload: Mapping[str, Any],
    *,
    video_id: str = "",
    transcript_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    transcript_chunks = transcript_chunks_from_artifacts(
        payload,
        transcript_payload=transcript_payload,
    )
    if not transcript_chunks:
        return []
    chunks: list[dict[str, Any]] = []
    text_results: list[dict[str, Any]] = []
    source_audio_path = str(payload.get("audio_path") or "")
    transcript_audio_path = (
        str(transcript_payload.get("audio_path") or "")
        if isinstance(transcript_payload, Mapping)
        else ""
    )
    for position, row in enumerate(transcript_chunks):
        chunk_index = int(row.get("index", position))
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start))
        audio_path = str(
            row.get("audio_path")
            or row.get("normalized_path")
            or transcript_audio_path
            or source_audio_path
            or ""
        )
        duration = float(row.get("duration", max(0.0, end - start)) or 0.0)
        chunks.append(
            {
                "index": chunk_index,
                "start": start,
                "end": end,
                "duration": duration,
                "path": audio_path,
                "source_audio_path": source_audio_path or transcript_audio_path,
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
                "duration": duration,
                "language": str(row.get("language") or "Japanese"),
                "normalized_path": audio_path,
                "avg_logprob": row.get("avg_logprob"),
                "no_speech_prob": row.get("no_speech_prob"),
                "compression_ratio": row.get("compression_ratio"),
                "alignment_window_start_s": row.get("alignment_window_start_s"),
                "alignment_window_end_s": row.get("alignment_window_end_s"),
                "alignment_window_source": row.get("alignment_window_source", ""),
            }
        )
    audio_id = Path(source_audio_path or transcript_audio_path or video_id or "audio").stem
    return build_candidates(
        chunks,
        text_results,
        audio_id=audio_id,
        video_id=video_id or audio_id,
    )


def compact_payload_requires_transcript(payload: Mapping[str, Any]) -> bool:
    details = (
        payload.get("asr_details")
        if isinstance(payload.get("asr_details"), Mapping)
        else {}
    )
    embedded = details.get("transcript_chunks")
    if isinstance(embedded, list) and embedded:
        return False
    try:
        return int(details.get("transcript_chunk_count") or 0) > 0
    except (TypeError, ValueError):
        return False
