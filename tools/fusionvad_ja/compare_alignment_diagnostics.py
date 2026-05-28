#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def parse_labeled_path(value: str, index: int) -> tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in --diagnostics item: {value}")
        return label, project_path(path_text)
    path = project_path(value)
    return f"run{index + 1}", path


def ratio(part: int | float, total: int | float) -> float:
    total_value = float(total or 0)
    return float(part or 0) / total_value if total_value > 0 else 0.0


def compact_counts(counts: dict[str, Any], *, total: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in sorted(counts.items()):
        count = int(value or 0)
        out[str(key)] = {"count": count, "ratio": round(ratio(count, total), 6)}
    return out


def summarize_diagnostics(label: str, diagnostics_dir: Path) -> dict[str, Any]:
    summary_path = diagnostics_dir / "summary.json"
    diagnostics_path = diagnostics_dir / "diagnostics.jsonl"
    candidates_path = diagnostics_dir / "failure_candidates.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"summary.json must be an object: {summary_path}")

    diagnostics = read_jsonl(diagnostics_path)
    candidates = read_jsonl(candidates_path)
    chunk_count = int(summary.get("chunk_count") or len(diagnostics))
    output_segment_count = int(summary.get("output_segment_count") or 0)
    fallback_count = int(summary.get("fallback_chunk_count") or 0)
    candidate_count = int(summary.get("failure_candidate_count") or len(candidates))
    nonempty_count = int(summary.get("nonempty_chunk_count") or 0)

    quality_counts = {
        str(key): int(value or 0)
        for key, value in (summary.get("alignment_quality_counts") or {}).items()
    }
    fallback_type_counts = {
        str(key): int(value or 0)
        for key, value in (summary.get("fallback_type_counts") or {}).items()
    }
    failure_bucket_counts = {
        str(key): int(value or 0)
        for key, value in (summary.get("failure_bucket_counts") or {}).items()
    }
    if diagnostics and not quality_counts:
        quality_counts = dict(Counter(str(row.get("alignment_quality") or "") for row in diagnostics if row.get("alignment_quality")))
    if diagnostics and not fallback_type_counts:
        fallback_type_counts = dict(Counter(str(row.get("fallback_type") or "") for row in diagnostics if row.get("fallback_type")))
    if candidates and not failure_bucket_counts:
        failure_bucket_counts = dict(Counter(str(row.get("failure_bucket") or "") for row in candidates if row.get("failure_bucket")))

    forced_count = int(quality_counts.get("forced", 0))
    review_count = int(quality_counts.get("drop_or_review", 0))
    coarse_count = int(quality_counts.get("vad_coarse", 0)) + int(quality_counts.get("proportional", 0))
    partial_count = int(quality_counts.get("partial", 0))

    return {
        "label": label,
        "diagnostics_dir": project_rel(diagnostics_dir),
        "summary_json": project_rel(summary_path),
        "diagnostics_jsonl": project_rel(diagnostics_path) if diagnostics_path.exists() else "",
        "failure_candidates_jsonl": project_rel(candidates_path) if candidates_path.exists() else "",
        "case_count": int(summary.get("case_count") or 0),
        "chunk_count": chunk_count,
        "output_segment_count": output_segment_count,
        "nonempty_chunk_count": nonempty_count,
        "failure_candidate_count": candidate_count,
        "fallback_chunk_count": fallback_count,
        "forced_count": forced_count,
        "partial_count": partial_count,
        "coarse_alignment_count": coarse_count,
        "drop_or_review_count": review_count,
        "fallback_chunk_ratio": round(ratio(fallback_count, chunk_count), 6),
        "failure_candidate_ratio": round(ratio(candidate_count, chunk_count), 6),
        "forced_ratio": round(ratio(forced_count, chunk_count), 6),
        "drop_or_review_ratio": round(ratio(review_count, chunk_count), 6),
        "output_segments_per_chunk": round(ratio(output_segment_count, chunk_count), 6),
        "alignment_quality_counts": compact_counts(quality_counts, total=chunk_count),
        "fallback_type_counts": compact_counts(fallback_type_counts, total=chunk_count),
        "failure_bucket_counts": compact_counts(failure_bucket_counts, total=chunk_count),
        "cases": summary.get("cases") or [],
    }


def sorted_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row.get("label") or ""))


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ASR / Alignment Checkpoint Comparison",
        "",
        "| label | chunks | segments | forced | fallback | candidates | drop/review | partial | coarse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_runs(rows):
        lines.append(
            "| `{label}` | {chunks} | {segments} | {forced} ({forced_ratio:.1%}) | "
            "{fallback} ({fallback_ratio:.1%}) | {candidates} ({candidate_ratio:.1%}) | "
            "{review} ({review_ratio:.1%}) | {partial} | {coarse} |".format(
                label=row["label"],
                chunks=row["chunk_count"],
                segments=row["output_segment_count"],
                forced=row["forced_count"],
                forced_ratio=row["forced_ratio"],
                fallback=row["fallback_chunk_count"],
                fallback_ratio=row["fallback_chunk_ratio"],
                candidates=row["failure_candidate_count"],
                candidate_ratio=row["failure_candidate_ratio"],
                review=row["drop_or_review_count"],
                review_ratio=row["drop_or_review_ratio"],
                partial=row["partial_count"],
                coarse=row["coarse_alignment_count"],
            )
        )
    lines.extend(["", "## Failure Buckets", ""])
    for row in sorted_runs(rows):
        lines.append(f"### {row['label']}")
        buckets = row.get("failure_bucket_counts") or {}
        if not buckets:
            lines.append("- None")
            continue
        for bucket, payload in sorted(buckets.items(), key=lambda item: (-int(item[1].get("count") or 0), item[0])):
            lines.append(
                "- `{bucket}`: {count} ({ratio:.1%})".format(
                    bucket=bucket,
                    count=int(payload.get("count") or 0),
                    ratio=float(payload.get("ratio") or 0.0),
                )
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple ASR/alignment diagnostics runs by checkpoint or model label."
    )
    parser.add_argument(
        "--diagnostics",
        action="append",
        required=True,
        help="Diagnostics output directory, optionally label=path. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/asr-alignment-checkpoint-compare",
        help="Output directory for checkpoint_comparison.json/jsonl/md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [
        summarize_diagnostics(label, path)
        for index, value in enumerate(args.diagnostics)
        for label, path in [parse_labeled_path(value, index)]
    ]
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "checkpoint_comparison_rows.jsonl"
    summary_path = output_dir / "checkpoint_comparison.json"
    markdown_path = output_dir / "checkpoint_comparison.md"
    write_jsonl(rows_path, rows)
    write_json(
        summary_path,
        {
            "run_count": len(rows),
            "runs": sorted_runs(rows),
            "rows_jsonl": project_rel(rows_path),
            "summary_md": project_rel(markdown_path),
        },
    )
    markdown_path.write_text(build_markdown(rows), encoding="utf-8")
    print(f"runs={len(rows)} output={project_rel(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
