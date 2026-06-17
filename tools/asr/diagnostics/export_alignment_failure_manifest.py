#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]


REVIEW_TYPE_BY_BUCKET = {
    "align_text_empty": "review_alignment_text",
    "empty_text_for_chunk": "review_empty_asr",
    "text_without_output_segment": "review_missing_output_segment",
    "repeat_repair_suggested": "review_repetition_repair",
    "partial_alignment": "review_partial_alignment",
    "vad_coarse_alignment": "review_coarse_timing",
    "proportional_alignment": "review_coarse_timing",
    "unknown_alignment_fallback": "review_unknown_fallback",
    "abnormal_char_density": "review_possible_hallucination",
    "diagnostic_warning": "review_diagnostic_warning",
}


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


def read_jsonl_rows(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_text in paths:
        path = project_path(path_text)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    row = dict(payload)
                    row["_candidate_jsonl"] = project_rel(path)
                    rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_type",
        "failure_bucket",
        "case_label",
        "sample_id",
        "source_audio_path",
        "start",
        "end",
        "duration_s",
        "alignment_quality",
        "fallback_type",
        "display_text",
        "align_text",
        "repetition_suggested_text",
        "text_density_level",
        "raw_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def sample_id_for_row(row: Mapping[str, Any]) -> str:
    case_label = str(row.get("case_label") or "case")
    video = str(row.get("video") or "video")
    chunk_index = int(row.get("chunk_index") or 0)
    bucket = str(row.get("failure_bucket") or "failure")
    return f"{case_label}__{video}__chunk{chunk_index:04d}__{bucket}"


def manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    bucket = str(row.get("failure_bucket") or "diagnostic_warning")
    start = row_float(row, "start")
    end = row_float(row, "end")
    duration = row_float(row, "duration_s") or max(0.0, end - start)
    source_audio_path = str(row.get("source_audio_path") or "")
    return {
        "sample_id": sample_id_for_row(row),
        "case_label": str(row.get("case_label") or ""),
        "video": str(row.get("video") or ""),
        "source_audio_path": source_audio_path,
        "aligned_path": str(row.get("aligned_path") or ""),
        "candidate_jsonl": str(row.get("_candidate_jsonl") or ""),
        "chunk_index": int(row.get("chunk_index") or 0),
        "position": int(row.get("position") or 0),
        "start": round(start, 3),
        "end": round(end, 3),
        "duration_s": round(duration, 3),
        "review_type": REVIEW_TYPE_BY_BUCKET.get(bucket, "review_diagnostic_warning"),
        "failure_bucket": bucket,
        "failure_reasons": list(row.get("failure_reasons") or []),
        "alignment_quality": str(row.get("alignment_quality") or ""),
        "fallback_type": str(row.get("fallback_type") or ""),
        "alignment_quality_reasons": list(row.get("alignment_quality_reasons") or []),
        "display_text": normalized_text(row.get("display_text")),
        "align_text": normalized_text(row.get("align_text")),
        "repetition_suggested_text": normalized_text(row.get("repetition_suggested_text")),
        "text_density_level": str(row.get("text_density_level") or ""),
        "text_density": dict(row.get("text_density") or {}),
        "repetition_repair": dict(row.get("repetition_repair") or {}),
        "analysis_text": normalized_text(row.get("analysis_text")),
        "text": normalized_text(row.get("text")),
        "raw_text": normalized_text(row.get("raw_text")),
        "review_original_text": normalized_text(row.get("review_original_text")),
        "manual_label": "",
        "manual_text": "",
        "notes": "",
    }


def export_manifest(rows: list[dict[str, Any]], *, output_dir: Path, max_rows: int | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = [manifest_row(row) for row in rows]
    manifest_rows.sort(
        key=lambda row: (
            str(row.get("review_type") or ""),
            str(row.get("failure_bucket") or ""),
            str(row.get("case_label") or ""),
            str(row.get("video") or ""),
            int(row.get("chunk_index") or 0),
        )
    )
    if max_rows is not None:
        manifest_rows = manifest_rows[:max_rows]

    manifest_path = output_dir / "alignment_failure_manifest.jsonl"
    csv_path = output_dir / "alignment_failure_manifest.csv"
    summary_path = output_dir / "alignment_failure_manifest_summary.json"
    write_jsonl(manifest_path, manifest_rows)
    write_csv(csv_path, manifest_rows)
    summary = {
        "input_rows": len(rows),
        "exported_rows": len(manifest_rows),
        "max_rows": max_rows,
        "review_type_counts": dict(sorted(Counter(str(row.get("review_type") or "") for row in manifest_rows).items())),
        "failure_bucket_counts": dict(sorted(Counter(str(row.get("failure_bucket") or "") for row in manifest_rows).items())),
        "alignment_quality_counts": dict(sorted(Counter(str(row.get("alignment_quality") or "") for row in manifest_rows).items())),
        "manifest_jsonl": project_rel(manifest_path),
        "manifest_csv": project_rel(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export alignment failure candidates into a review/closed-loop manifest."
    )
    parser.add_argument(
        "--failure-candidates",
        action="append",
        required=True,
        help="failure_candidates.jsonl from diagnose_asr_alignment.py. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="agents/temp/speech-boundary-ja/alignment-failure-manifest",
    )
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl_rows(args.failure_candidates)
    summary = export_manifest(rows, output_dir=project_path(args.output_dir), max_rows=args.max_rows)
    print(f"manifest={summary['manifest_jsonl']}")
    print(f"csv={summary['manifest_csv']}")
    print(f"rows={summary['exported_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
