#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from subtitles import writer as subtitle_writer  # noqa: E402


TEXT_KEEP_LABELS = {"keep_text", "left_keep_text", "right_keep_text"}
TIMING_KEEP_LABELS = {"timing_accurate", "coarse_timing_ok"}
KEEP_LABELS = TEXT_KEEP_LABELS | TIMING_KEEP_LABELS
ASR_QC_LABELS = {"bad_asr"}
SIDE_ASR_QC_LABELS = {"left_bad_asr", "right_bad_asr"}
HARD_DROP_LABELS = {"drop_non_speech"}
SIDE_HARD_DROP_LABELS = {"left_drop_non_speech", "right_drop_non_speech"}
LOW_INFO_LABELS = {"low_info_vocal"}
SIDE_LOW_INFO_LABELS = {"left_low_info_vocal", "right_low_info_vocal"}
MERGE_TIMING_LABELS = {"needs_realign", "needs_split"}
SIDE_LABEL_PREFIXES = ("left_", "right_")


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
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(dict(payload))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace(";", ",").split(",")
    return [str(item).strip() for item in raw if str(item).strip() and str(item).strip() != "-"]


def sample_key(row: Mapping[str, Any]) -> str:
    for key in ("sample_id", "audio", "position"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def manual_label(row: Mapping[str, Any]) -> str:
    for key in ("manual_label", "label", "decision", "status"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def manual_labels(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("manual_labels")
    if isinstance(raw, list):
        labels = [str(item).strip() for item in raw if str(item).strip()]
        if labels:
            return labels
    label = manual_label(row)
    return [label] if label else []


def problem_bucket(labels: list[str]) -> str:
    label_set = set(labels)
    if not label_set:
        return "unreviewed"
    side_problem = bool(label_set & (SIDE_ASR_QC_LABELS | SIDE_HARD_DROP_LABELS))
    side_keep = bool(label_set & ({"left_keep_text", "right_keep_text"} | SIDE_LOW_INFO_LABELS))
    if side_problem and side_keep:
        return "side_mixed"
    if label_set & SIDE_ASR_QC_LABELS:
        return "side_asr_qc"
    if label_set & SIDE_HARD_DROP_LABELS:
        return "side_asr_qc"
    if label_set & ASR_QC_LABELS:
        return "asr_qc"
    if label_set & HARD_DROP_LABELS and not (label_set & KEEP_LABELS or label_set & LOW_INFO_LABELS):
        return "asr_qc"
    if label_set & MERGE_TIMING_LABELS:
        return "merge_timing"
    if (label_set & (LOW_INFO_LABELS | SIDE_LOW_INFO_LABELS)) and label_set & KEEP_LABELS:
        return "low_info_keep"
    if label_set & (LOW_INFO_LABELS | SIDE_LOW_INFO_LABELS):
        return "low_info_review"
    if label_set & HARD_DROP_LABELS and label_set & KEEP_LABELS:
        return "low_info_keep"
    if label_set & KEEP_LABELS:
        return "keep"
    return "merge_timing"


def text_units(row: Mapping[str, Any]) -> float:
    text = str(row.get("merged_ja") or row.get("align_text") or row.get("text") or "")
    if not text:
        return 0.0
    return float(subtitle_writer._block_text_units({"ja_text": text}))


def cue_duration_s(row: Mapping[str, Any]) -> float:
    source_start = _as_float(row.get("source_start_s"))
    source_end = _as_float(row.get("source_end_s"))
    if source_end > source_start:
        return source_end - source_start
    chunk_start = _as_float(row.get("chunk_start_s"))
    chunk_end = _as_float(row.get("chunk_end_s"))
    if chunk_end > chunk_start:
        return chunk_end - chunk_start
    return _as_float(row.get("duration_s"))


def rate(part: int, total: int) -> float:
    return round(part / total, 6) if total else 0.0


def bucket_name(value: float, buckets: list[tuple[float, str]]) -> str:
    for upper, name in buckets:
        if value < upper:
            return name
    return buckets[-1][1]


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    side_label_counts: dict[str, Counter[str]] = {"left": Counter(), "right": Counter()}
    for row in rows:
        for raw_label in row.get("manual_labels", []):
            label = str(raw_label)
            if not label:
                continue
            label_counts.update([label])
            for prefix in SIDE_LABEL_PREFIXES:
                if label.startswith(prefix):
                    side_label_counts[prefix[:-1]].update([label.removeprefix(prefix)])
    bucket_counts = Counter(str(row["problem_bucket"]) for row in rows)
    total = len(rows)
    keep = bucket_counts.get("keep", 0) + bucket_counts.get("low_info_keep", 0)
    review_only = bucket_counts.get("low_info_review", 0)
    problem = total - keep - review_only
    return {
        "count": total,
        "keep": keep,
        "review_only": review_only,
        "problem": problem,
        "problem_rate": rate(problem, total),
        "label_counts": dict(label_counts.most_common()),
        "side_label_counts": {
            side: dict(counter.most_common())
            for side, counter in side_label_counts.items()
            if counter
        },
        "problem_bucket_counts": dict(bucket_counts.most_common()),
    }


def recommendation(summary: dict[str, Any]) -> dict[str, Any]:
    tag_stats = summary["risk_tag_stats"]
    strong_block: list[str] = []
    strong_penalty: list[str] = []
    review_only: list[str] = []
    for tag, stats in tag_stats.items():
        count = int(stats["count"])
        problem_rate = float(stats["problem_rate"])
        if count >= 2 and problem_rate >= 0.8:
            strong_block.append(tag)
        elif count >= 2 and problem_rate >= 0.6:
            strong_penalty.append(tag)
        else:
            review_only.append(tag)
    return {
        "baseline": "Keep th85 as the safer baseline until a second audit batch confirms th95-constrained.",
        "candidate": "th95-constrained",
        "strong_block_or_penalty_tags": sorted(strong_block),
        "soft_penalty_tags": sorted(strong_penalty),
        "review_only_tags": sorted(review_only),
        "suggested_planner_args": {
            "speaker_threshold": 0.95,
            "speaker_change_policy": "block",
            "fallback_risk_policy": "penalize",
            "max_gap_s": 0.5 if "loose_gap" in strong_block else 1.2,
            "speaker_score_penalty_threshold": 0.85 if "high_speaker_score" in strong_block else 0.0,
            "speaker_score_penalty": 0.12 if "high_speaker_score" in strong_block else 0.0,
            "max_reading_units_per_s": 0.0,
        },
        "notes": [
            "near_speaker_threshold has mixed labels, so it should not become a hard blocker from this batch alone.",
            "reading_density_high is useful for review/protection but is not reliable enough as a hard default gate.",
            "bad_asr and hard drop_non_speech should feed ASR QC or hard-negative pools; do not treat them as pure merge-policy failures.",
            "low_info_vocal is separate from hard drop_non_speech: moans, breaths, sighs, laughter, and short vocalizations may be transcript-worthy in the Galgame target domain.",
            "left_*/right_* labels identify side-specific failures inside a merged cue; side_mixed usually means one side is usable and the other should be split, fixed, or dropped.",
        ],
    }


def build_calibration(
    *,
    manual_labels_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manual_rows = read_jsonl(manual_labels_path)
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_key = {sample_key(row): row for row in manifest_rows if sample_key(row)}
    labeled_rows: list[dict[str, Any]] = []

    for row in manual_rows:
        key = sample_key(row)
        manifest = manifest_by_key.get(key, {})
        merged = {**manifest, **row}
        labels = manual_labels(merged)
        label = manual_label(merged) or (labels[0] if labels else "")
        tags = split_tags(merged.get("risk_tags") or merged.get("failure_bucket"))
        duration = cue_duration_s(merged)
        units = text_units(merged)
        labeled_rows.append(
            {
                **merged,
                "manual_label": label,
                "manual_labels": labels,
                "problem_bucket": problem_bucket(labels),
                "risk_tag_list": tags,
                "cue_duration_s": round(duration, 6),
                "text_units": round(units, 3),
                "reading_units_per_s": round(units / max(0.05, duration), 3),
                "speaker_change_score": _as_float(merged.get("speaker_change_score")),
                "planner_score": _as_float(merged.get("score")),
            }
        )

    reviewed = [row for row in labeled_rows if row["manual_labels"]]
    risk_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed:
        tags = row.get("risk_tag_list") or ["no_risk_tag"]
        for tag in tags:
            risk_rows[str(tag)].append(row)

    speaker_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    planner_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reading_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed:
        speaker_buckets[
            bucket_name(
                float(row["speaker_change_score"]),
                [
                    (0.75, "<0.75"),
                    (0.85, "0.75-0.85"),
                    (0.90, "0.85-0.90"),
                    (0.95, "0.90-0.95"),
                    (999.0, ">=0.95"),
                ],
            )
        ].append(row)
        planner_buckets[
            bucket_name(
                float(row["planner_score"]),
                [
                    (0.50, "<0.50"),
                    (0.60, "0.50-0.60"),
                    (0.70, "0.60-0.70"),
                    (0.80, "0.70-0.80"),
                    (999.0, ">=0.80"),
                ],
            )
        ].append(row)
        reading_buckets[
            bucket_name(
                float(row["reading_units_per_s"]),
                [
                    (8.0, "<8"),
                    (12.0, "8-12"),
                    (16.0, "12-16"),
                    (20.0, "16-20"),
                    (999.0, ">=20"),
                ],
            )
        ].append(row)

    summary = {
        "source_manual_labels": project_rel(manual_labels_path),
        "source_manifest": project_rel(manifest_path),
        "output_dir": project_rel(output_dir),
        "rows": len(labeled_rows),
        "reviewed_rows": len(reviewed),
        "overall": summarize_group(reviewed),
        "risk_tag_stats": {
            tag: summarize_group(rows)
            for tag, rows in sorted(risk_rows.items(), key=lambda item: (-len(item[1]), item[0]))
        },
        "speaker_score_buckets": {
            key: summarize_group(rows) for key, rows in sorted(speaker_buckets.items())
        },
        "planner_score_buckets": {
            key: summarize_group(rows) for key, rows in sorted(planner_buckets.items())
        },
        "reading_density_buckets": {
            key: summarize_group(rows) for key, rows in sorted(reading_buckets.items())
        },
    }
    summary["recommendation"] = recommendation(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = output_dir / "calibrated_manual_rows.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    write_jsonl(labeled_path, labeled_rows)
    summary["labeled_rows_jsonl"] = project_rel(labeled_path)
    summary["summary_json"] = project_rel(summary_path)
    summary["summary_md"] = project_rel(markdown_path)
    write_json(summary_path, summary)
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")
    return summary


def build_markdown(summary: dict[str, Any]) -> str:
    rec = summary["recommendation"]
    lines = [
        "# Cue Planner Manual Calibration",
        "",
        f"- manual_labels: `{summary['source_manual_labels']}`",
        f"- manifest: `{summary['source_manifest']}`",
        f"- reviewed: {summary['reviewed_rows']}/{summary['rows']}",
        "",
        "## Overall",
        "",
        f"- labels: `{summary['overall']['label_counts']}`",
        f"- side_labels: `{summary['overall']['side_label_counts']}`",
        f"- problem_rate: {summary['overall']['problem_rate']:.3f}",
        "",
        "## Risk Tags",
        "",
        "| tag | count | keep | review_only | problem | problem_rate | labels |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tag, stats in summary["risk_tag_stats"].items():
        lines.append(
            "| {tag} | {count} | {keep} | {review_only} | {problem} | {rate:.3f} | {labels} |".format(
                tag=tag,
                count=stats["count"],
                keep=stats["keep"],
                review_only=stats["review_only"],
                problem=stats["problem"],
                rate=stats["problem_rate"],
                labels=json.dumps(stats["label_counts"], ensure_ascii=False, sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- baseline: {rec['baseline']}",
            f"- candidate: {rec['candidate']}",
            f"- strong_block_or_penalty_tags: `{rec['strong_block_or_penalty_tags']}`",
            f"- soft_penalty_tags: `{rec['soft_penalty_tags']}`",
            f"- review_only_tags: `{rec['review_only_tags']}`",
            f"- suggested_planner_args: `{rec['suggested_planner_args']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate subtitle cue planner risk signals from manual audit labels."
    )
    parser.add_argument("--manual-labels", required=True, help="manual audit JSONL")
    parser.add_argument("--manifest", required=True, help="audit/source manifest JSONL")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/speech-boundary-ja/cue-planner-manual-calibration",
    )
    args = parser.parse_args(argv)
    summary = build_calibration(
        manual_labels_path=project_path(args.manual_labels),
        manifest_path=project_path(args.manifest),
        output_dir=project_path(args.output_dir),
    )
    print(
        "calibration={path} reviewed={reviewed}/{rows}".format(
            path=summary["summary_json"],
            reviewed=summary["reviewed_rows"],
            rows=summary["rows"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
