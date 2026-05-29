#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INCLUDE_REVIEW_TYPES = [
    "review_repetition_repair",
    "review_low_information_text",
]
DEFAULT_SAMPLE_REVIEW_TYPES = [
    "review_coarse_timing:20",
]


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "audit_priority",
        "audit_reason",
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
        "low_information_level",
        "manual_label",
        "manual_text",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, float, str]:
    try:
        chunk_index = int(row.get("chunk_index") or 0)
    except (TypeError, ValueError):
        chunk_index = 0
    try:
        start = float(row.get("start") or 0.0)
    except (TypeError, ValueError):
        start = 0.0
    return (
        str(row.get("case_label") or ""),
        str(row.get("video") or ""),
        chunk_index,
        start,
        str(row.get("sample_id") or ""),
    )


def parse_sample_spec(value: str) -> tuple[str, int]:
    if ":" in value:
        review_type, count_text = value.split(":", 1)
    elif "=" in value:
        review_type, count_text = value.split("=", 1)
    else:
        raise ValueError(f"sample spec must be REVIEW_TYPE:COUNT: {value}")
    review_type = review_type.strip()
    if not review_type:
        raise ValueError(f"sample spec has empty review type: {value}")
    count = int(count_text)
    if count <= 0:
        raise ValueError(f"sample count must be positive: {value}")
    return review_type, count


def evenly_spaced_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    if count <= 1:
        return [rows[len(rows) // 2]]
    last = len(rows) - 1
    selected_indices = []
    seen: set[int] = set()
    for index in range(count):
        selected = round(index * last / (count - 1))
        while selected in seen and selected < last:
            selected += 1
        while selected in seen and selected > 0:
            selected -= 1
        seen.add(selected)
        selected_indices.append(selected)
    return [rows[index] for index in sorted(seen)]


def select_subset(
    rows: list[dict[str, Any]],
    *,
    include_review_types: list[str],
    sample_review_types: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for priority, review_type in enumerate(include_review_types, start=1):
        matches = sorted(
            [row for row in rows if str(row.get("review_type") or "") == review_type],
            key=row_sort_key,
        )
        for row in matches:
            sample_id = str(row.get("sample_id") or "")
            if sample_id in seen:
                continue
            selected.append(
                {
                    **row,
                    "audit_priority": priority,
                    "audit_reason": f"include:{review_type}",
                }
            )
            seen.add(sample_id)

    base_priority = len(include_review_types)
    for offset, (review_type, count) in enumerate(sample_review_types, start=1):
        matches = sorted(
            [
                row
                for row in rows
                if str(row.get("review_type") or "") == review_type
                and str(row.get("sample_id") or "") not in seen
            ],
            key=row_sort_key,
        )
        for row in evenly_spaced_rows(matches, count):
            sample_id = str(row.get("sample_id") or "")
            if sample_id in seen:
                continue
            selected.append(
                {
                    **row,
                    "audit_priority": base_priority + offset,
                    "audit_reason": f"sample_even:{review_type}",
                }
            )
            seen.add(sample_id)

    selected.sort(
        key=lambda row: (
            int(row.get("audit_priority") or 0),
            row_sort_key(row),
        )
    )
    return selected


def export_subset(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    include_review_types: list[str],
    sample_review_types: list[tuple[str, int]],
    source_manifest: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_subset(
        rows,
        include_review_types=include_review_types,
        sample_review_types=sample_review_types,
    )
    jsonl_path = output_dir / "alignment_failure_audit_subset.jsonl"
    csv_path = output_dir / "alignment_failure_audit_subset.csv"
    summary_path = output_dir / "alignment_failure_audit_subset_summary.json"
    write_jsonl(jsonl_path, selected)
    write_csv(csv_path, selected)
    summary = {
        "source_manifest": project_rel(source_manifest),
        "input_rows": len(rows),
        "exported_rows": len(selected),
        "include_review_types": include_review_types,
        "sample_review_types": {review_type: count for review_type, count in sample_review_types},
        "audit_reason_counts": dict(sorted(Counter(str(row.get("audit_reason") or "") for row in selected).items())),
        "review_type_counts": dict(sorted(Counter(str(row.get("review_type") or "") for row in selected).items())),
        "failure_bucket_counts": dict(sorted(Counter(str(row.get("failure_bucket") or "") for row in selected).items())),
        "manifest_jsonl": project_rel(jsonl_path),
        "manifest_csv": project_rel(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a prioritized manual audit subset from an alignment failure manifest."
    )
    parser.add_argument("--manifest", required=True, help="alignment_failure_manifest.jsonl")
    parser.add_argument(
        "--output-dir",
        default="agents/audits/fusionvad-ja/alignment-failure-audit-subset",
    )
    parser.add_argument(
        "--include-review-type",
        action="append",
        dest="include_review_types",
        help="Review type to include fully. Repeatable. Defaults to repetition and low-information reviews.",
    )
    parser.add_argument(
        "--sample-review-type",
        action="append",
        dest="sample_review_types",
        help="Evenly sample REVIEW_TYPE:COUNT rows. Repeatable. Defaults to review_coarse_timing:20.",
    )
    args = parser.parse_args(argv)
    args.include_review_types = args.include_review_types or list(DEFAULT_INCLUDE_REVIEW_TYPES)
    sample_values = args.sample_review_types or list(DEFAULT_SAMPLE_REVIEW_TYPES)
    try:
        args.sample_review_types = [parse_sample_spec(value) for value in sample_values]
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = project_path(args.manifest)
    rows = read_jsonl(manifest_path)
    summary = export_subset(
        rows,
        output_dir=project_path(args.output_dir),
        include_review_types=args.include_review_types,
        sample_review_types=args.sample_review_types,
        source_manifest=manifest_path,
    )
    print(f"manifest={summary['manifest_jsonl']}")
    print(f"csv={summary['manifest_csv']}")
    print(f"rows={summary['exported_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
