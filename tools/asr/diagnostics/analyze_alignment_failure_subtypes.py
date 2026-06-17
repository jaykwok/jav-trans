#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = "agents/temp/speech-boundary-ja/alignment-failure-subtype-analysis"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def resolve_diagnostics_path(value: str | Path) -> Path:
    path = project_path(value)
    if path.is_dir():
        path = path / "diagnostics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"diagnostics file not found: {path}")
    return path


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * quantile)
    return ordered[max(0, min(len(ordered) - 1, index))]


def numeric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "p50": round(statistics.median(values), 6),
        "p90": round(percentile(values, 0.9), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def text_preview(value: Any, limit: int = 80) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def is_failure_row(row: Mapping[str, Any]) -> bool:
    subtype = str(row.get("fallback_subtype") or "")
    bucket = str(row.get("failure_bucket") or "")
    return bool(row.get("failure_candidate")) or subtype not in {"", "none"} or bool(bucket)


def subtype_key(row: Mapping[str, Any]) -> str:
    subtype = str(row.get("fallback_subtype") or "").strip()
    if subtype and subtype != "none":
        return subtype
    bucket = str(row.get("failure_bucket") or "").strip()
    if bucket:
        return f"bucket:{bucket}"
    quality = str(row.get("alignment_quality") or "").strip()
    return quality or "unknown"


def recommended_route(subtype: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    durations = [as_float(row.get("duration_s")) for row in rows]
    p50_duration = statistics.median(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    nonempty_align = sum(1 for row in rows if str(row.get("align_text") or "").strip())
    has_sentinel_fallback = any(
        str(row.get("fallback_type") or "").strip() not in {"", "none"}
        and bool(row.get("sentinel_lines") or [])
        for row in rows
    )

    if has_sentinel_fallback:
        if p50_duration >= 6.0 or max_duration >= 10.0:
            return {
                "route": "aligner_robustness",
                "next_action": "Do not widen chunk packing further; inspect the sentinel fallback chunks and improve boundary planning, CTC/secondary alignment, or aligner-local splitting.",
            }
        return {
            "route": "sentinel_policy",
            "next_action": "Inspect sentinel chunks with short duration; likely an aligner quality rule or fallback-classification issue rather than chunk length.",
        }
    if subtype == "nonlexical_text":
        return {
            "route": "nonlexical_time_policy",
            "next_action": "Skip forced aligner for punctuation/nonlexical text, preserve display_text, and emit nonlexical/vad_coarse timing quality instead of counting it as an aligner failure.",
        }
    if subtype == "align_text_empty":
        return {
            "route": "prealign_policy",
            "next_action": "Review display_text->align_text cleaning; if text is symbol-only, bypass forced aligner with explicit drop_or_review/nonlexical quality rather than sending empty text.",
        }
    if subtype in {"asr_empty_text", "bucket:empty_text_for_chunk"}:
        return {
            "route": "asr_or_vad_empty",
            "next_action": "Treat as ASR/SpeechBoundary proposal issue; collect hard negatives and compare with ASR confidence/QC before changing aligner.",
        }
    if subtype.startswith("word_timing_") or nonempty_align:
        return {
            "route": "alignment_quality_threshold",
            "next_action": "Inspect word timing coverage/zero-duration thresholds before changing ASR or SpeechBoundary-JA.",
        }
    return {
        "route": "manual_triage",
        "next_action": "Inspect examples and assign this subtype to ASR, pre-align, fallback policy, or aligner robustness before new GPU runs.",
    }


def example_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_label": str(row.get("case_label") or ""),
        "video": str(row.get("video") or ""),
        "chunk_index": as_int(row.get("chunk_index")),
        "start": round(as_float(row.get("start")), 3),
        "end": round(as_float(row.get("end")), 3),
        "duration_s": round(as_float(row.get("duration_s")), 3),
        "alignment_quality": str(row.get("alignment_quality") or ""),
        "fallback_type": str(row.get("fallback_type") or ""),
        "fallback_subtype": str(row.get("fallback_subtype") or ""),
        "failure_bucket": str(row.get("failure_bucket") or ""),
        "compact_chars": as_int(row.get("compact_chars")),
        "prealign_align_len": as_int(row.get("prealign_align_len")),
        "aligned_segment_count": as_int(row.get("aligned_segment_count")),
        "text_preview": text_preview(row.get("analysis_text") or row.get("text")),
        "align_text_preview": text_preview(row.get("align_text")),
    }


def analyze_rows(rows: list[dict[str, Any]], *, examples_per_subtype: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failure_rows = [row for row in rows if is_failure_row(row)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        grouped[subtype_key(row)].append(row)

    subtype_summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for subtype, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        durations = [as_float(row.get("duration_s")) for row in members]
        compact_chars = [float(as_int(row.get("compact_chars"))) for row in members]
        align_lens = [float(as_int(row.get("prealign_align_len"))) for row in members]
        segment_counts = [float(as_int(row.get("aligned_segment_count"))) for row in members]
        route = recommended_route(subtype, members)
        sorted_examples = sorted(
            members,
            key=lambda row: (
                -as_float(row.get("duration_s")),
                -as_int(row.get("compact_chars")),
                str(row.get("video") or ""),
                as_int(row.get("chunk_index")),
            ),
        )[:examples_per_subtype]
        for row in sorted_examples:
            examples.append({"subtype_group": subtype, **example_row(row)})
        subtype_summaries.append(
            {
                "subtype_group": subtype,
                "count": len(members),
                "ratio_of_failures": round(len(members) / max(1, len(failure_rows)), 6),
                "duration_s": numeric_stats(durations),
                "compact_chars": numeric_stats(compact_chars),
                "prealign_align_len": numeric_stats(align_lens),
                "aligned_segment_count": numeric_stats(segment_counts),
                "alignment_quality_counts": dict(
                    Counter(str(row.get("alignment_quality") or "") for row in members).most_common()
                ),
                "fallback_type_counts": dict(
                    Counter(str(row.get("fallback_type") or "") for row in members).most_common()
                ),
                "failure_bucket_counts": dict(
                    Counter(str(row.get("failure_bucket") or "") for row in members).most_common()
                ),
                **route,
            }
        )

    summary = {
        "input_rows": len(rows),
        "failure_rows": len(failure_rows),
        "subtype_count": len(subtype_summaries),
        "alignment_quality_counts": dict(
            Counter(str(row.get("alignment_quality") or "") for row in rows if row.get("alignment_quality")).most_common()
        ),
        "fallback_subtype_counts": dict(
            Counter(subtype_key(row) for row in failure_rows).most_common()
        ),
        "failure_bucket_counts": dict(
            Counter(str(row.get("failure_bucket") or "") for row in failure_rows if row.get("failure_bucket")).most_common()
        ),
        "subtypes": subtype_summaries,
    }
    return summary, examples


def build_markdown(summary: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Alignment Failure Subtype Analysis",
        "",
        f"- input rows: `{summary['input_rows']}`",
        f"- failure rows: `{summary['failure_rows']}`",
        f"- subtype groups: `{summary['subtype_count']}`",
        "",
        "## Subtype Routes",
        "",
        "| subtype | count | duration p50/p90/max | route | next action |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in summary.get("subtypes") or []:
        duration = item.get("duration_s") or {}
        lines.append(
            "| `{subtype}` | {count} | {p50:.2f}/{p90:.2f}/{max_v:.2f}s | `{route}` | {action} |".format(
                subtype=item["subtype_group"],
                count=item["count"],
                p50=float(duration.get("p50") or 0.0),
                p90=float(duration.get("p90") or 0.0),
                max_v=float(duration.get("max") or 0.0),
                route=item.get("route") or "",
                action=str(item.get("next_action") or ""),
            )
        )
    lines.extend(["", "## Longest Examples", ""])
    for row in examples[:40]:
        lines.append(
            "- `{subtype}` {video} chunk={chunk} dur={dur:.2f}s chars={chars}: {text}".format(
                subtype=row["subtype_group"],
                video=row["video"],
                chunk=row["chunk_index"],
                dur=float(row["duration_s"]),
                chars=int(row["compact_chars"]),
                text=row["text_preview"] or "(empty)",
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate alignment diagnostics by fallback subtype and emit action-oriented next steps.",
    )
    parser.add_argument(
        "--diagnostics",
        required=True,
        help="diagnostics.jsonl or a diagnostics output directory containing diagnostics.jsonl.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples-per-subtype", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics_path = resolve_diagnostics_path(args.diagnostics)
    rows = read_jsonl(diagnostics_path)
    summary, examples = analyze_rows(rows, examples_per_subtype=max(0, args.examples_per_subtype))

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["diagnostics_jsonl"] = project_rel(diagnostics_path)
    summary["examples_jsonl"] = project_rel(output_dir / "subtype_examples.jsonl")
    summary["summary_md"] = project_rel(output_dir / "summary.md")
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "subtype_examples.jsonl", examples)
    (output_dir / "summary.md").write_text(build_markdown(summary, examples), encoding="utf-8")

    print(
        "subtypes={subtypes} failures={failures} output={output}".format(
            subtypes=summary["subtype_count"],
            failures=summary["failure_rows"],
            output=project_rel(output_dir),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
