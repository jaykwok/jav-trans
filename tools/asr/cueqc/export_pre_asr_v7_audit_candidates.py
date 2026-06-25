#!/usr/bin/env python3
"""Export current Pre-ASR CueQC v7 candidates for audio audit/labeling."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from asr.cueqc import text_features as cue_text_features  # noqa: E402


def project_path(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()


def repo_rel(value: str | Path | None) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        return raw.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _infer_video_id(path: Path, payload: Mapping[str, Any]) -> str:
    for key in ("video_id", "audio_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    name = path.name
    suffix = ".timings.json"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def _extract_rows(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    for nested_key in ("asr_details", "details"):
        details = payload.get(nested_key)
        if not isinstance(details, Mapping):
            continue
        inner = details.get(key)
        if isinstance(inner, list):
            return [dict(row) for row in inner if isinstance(row, Mapping)]
    return []


def _text_for_candidate(
    *,
    candidate: Mapping[str, Any],
    transcript_chunks: list[Mapping[str, Any]],
    aligned_segments: list[Mapping[str, Any]],
) -> tuple[str, str, str]:
    index = int(_safe_float(candidate.get("chunk_index", candidate.get("index")), 0.0))
    transcript_text = ""
    if 0 <= index < len(transcript_chunks):
        transcript = transcript_chunks[index]
        transcript_text = _clean_text(transcript.get("text") or transcript.get("raw_text"))
    start = _safe_float(candidate.get("start"))
    end = _safe_float(candidate.get("end"), start)
    overlaps: list[str] = []
    for segment in aligned_segments:
        seg_start = _safe_float(segment.get("start"))
        seg_end = _safe_float(segment.get("end"), seg_start)
        if min(end, seg_end) - max(start, seg_start) <= 0.01:
            continue
        text = _clean_text(segment.get("text") or segment.get("raw_text"))
        if text:
            overlaps.append(text)
    aligned_text = " / ".join(dict.fromkeys(overlaps))
    display = transcript_text or aligned_text
    return display, transcript_text, aligned_text


def _cluster_for_candidate(candidate: Mapping[str, Any]) -> str:
    features = candidate.get("features") if isinstance(candidate.get("features"), Mapping) else {}
    duration = _safe_float(candidate.get("duration_s"), _safe_float(candidate.get("end")) - _safe_float(candidate.get("start")))
    below_min = bool(candidate.get("below_subtitle_min_duration"))
    micro_action = str(candidate.get("micro_resolve_action") or "").strip()
    split_p90 = _safe_float(features.get("scorer_split_p90"))
    speech_mean = _safe_float(features.get("scorer_speech_mean"))
    if below_min:
        return "below_subtitle_min"
    if micro_action:
        return f"micro_{micro_action}"
    if duration >= 5.0:
        return "long_duration"
    if split_p90 >= 0.35:
        return "high_split_density"
    if speech_mean < 0.45:
        return "low_speech_mean"
    return "standard"


def export_candidates(*, timing_paths: list[str], output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_files: list[str] = []
    for raw_path in timing_paths:
        path = project_path(raw_path)
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"timings payload must be an object: {path}")
        video_id = _infer_video_id(path, payload)
        source_files.append(repo_rel(path))
        candidates = _extract_rows(payload, "pre_asr_candidates")
        transcript_chunks = _extract_rows(payload, "transcript_chunks")
        aligned_segments = _extract_rows(payload, "aligned_segments")
        for local_index, candidate in enumerate(candidates):
            item = dict(candidate)
            chunk_index = int(_safe_float(item.get("chunk_index", item.get("index", local_index)), float(local_index)))
            sample_id = str(item.get("sample_id") or item.get("candidate_id") or "").strip()
            if not sample_id:
                sample_id = f"preasr-{video_id}-chunk{chunk_index:05d}"
            start = _safe_float(item.get("start"))
            end = _safe_float(item.get("end"), start)
            duration = max(0.0, _safe_float(item.get("duration_s"), end - start))
            text, transcript_text, aligned_text = _text_for_candidate(
                candidate=item,
                transcript_chunks=transcript_chunks,
                aligned_segments=aligned_segments,
            )
            text_payload = cue_text_features(transcript_text, text, duration_s=duration)
            cluster_id = _cluster_for_candidate(item)
            rows.append(
                {
                    **item,
                    "schema": "pre_asr_cueqc_v7_audit_candidate",
                    "sample_id": sample_id,
                    "candidate_id": sample_id,
                    "video_id": video_id,
                    "audio_id": video_id,
                    "chunk_index": chunk_index,
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration_s": round(duration, 6),
                    "text": text,
                    "raw_text": transcript_text,
                    "text_preview": text[:160],
                    "compact_text": text_payload.get("compact_text", ""),
                    "text_features": {
                        key: value
                        for key, value in text_payload.items()
                        if key not in {"text", "raw_text", "compact_text", "context_compact_text"}
                    },
                    "audit_reference": {
                        "transcript_text": transcript_text,
                        "aligned_text": aligned_text,
                    },
                    "cluster_id": cluster_id,
                    "cluster_label": cluster_id,
                    "cluster_method": "pre_asr_v7_feature_bucket",
                    "cluster_backend": "rule_bucket_for_audit_only",
                    "cluster_confidence": 1.0,
                    "cluster_noise": False,
                    "audit_sampling_score": round(duration, 6),
                }
            )
    rows.sort(key=lambda row: (-_safe_float(row.get("duration_s")), str(row.get("video_id")), int(row.get("chunk_index") or 0)))
    for rank, row in enumerate(rows, start=1):
        row["duration_rank"] = rank
        row["duration_rank_key"] = f"{rank:06d}"

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("cluster_id") or "standard")].append(row)
    summaries: list[dict[str, Any]] = []
    for cluster_id, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        durations = [_safe_float(row.get("duration_s")) for row in members]
        video_counts = Counter(str(row.get("video_id") or "") for row in members)
        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_id,
                "cluster_method": "pre_asr_v7_feature_bucket",
                "cluster_noise": False,
                "count": len(members),
                "duration_min_s": round(min(durations), 6) if durations else 0.0,
                "duration_max_s": round(max(durations), 6) if durations else 0.0,
                "confidence_avg": 1.0,
                "video_counts": dict(sorted(video_counts.items())),
                "examples": [
                    {
                        "sample_id": row.get("sample_id"),
                        "start": row.get("start"),
                        "duration_s": row.get("duration_s"),
                        "text_preview": row.get("text_preview", ""),
                    }
                    for row in members[:5]
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = output_dir / "pre_asr_cueqc_v7_audit_candidates.jsonl"
    summaries_path = output_dir / "pre_asr_cueqc_v7_audit_summaries.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(clusters_path, rows)
    write_jsonl(summaries_path, summaries)
    summary = {
        "schema": "pre_asr_cueqc_v7_audit_candidate_export_summary",
        "candidate_count": len(rows),
        "cluster_count": len(summaries),
        "clusters": repo_rel(clusters_path),
        "cluster_summaries": repo_rel(summaries_path),
        "source_files": source_files,
        "cluster_counts": {summary["cluster_id"]: summary["count"] for summary in summaries},
        "duration_top10": [
            {
                "sample_id": row.get("sample_id"),
                "video_id": row.get("video_id"),
                "chunk_index": row.get("chunk_index"),
                "start": row.get("start"),
                "end": row.get("end"),
                "duration_s": row.get("duration_s"),
            }
            for row in rows[:10]
        ],
    }
    write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current Pre-ASR CueQC v7 candidates for audit.")
    parser.add_argument("--timings", action="append", required=True, help="Workflow .timings.json. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export_candidates(timing_paths=list(args.timings), output_dir=project_path(args.output_dir))
    print(
        json.dumps(
            {
                "ok": True,
                "candidate_count": summary["candidate_count"],
                "clusters": summary["clusters"],
                "cluster_summaries": summary["cluster_summaries"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
