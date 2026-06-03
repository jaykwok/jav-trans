#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


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


def merge_key(row: Mapping[str, Any]) -> tuple[str, str]:
    left = row.get("left_cue_id")
    right = row.get("right_cue_id")
    if left not in (None, "") and right not in (None, ""):
        return (f"cue:{left}", f"cue:{right}")
    return (str(row.get("left_index") or ""), str(row.get("right_index") or ""))


def risk_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for tag in str(row.get("risk_tags") or "").split(","):
            tag = tag.strip()
            if tag:
                counts[tag] += 1
    return dict(counts.most_common())


def compare_reviews(
    *,
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    candidate_label: str,
) -> dict[str, Any]:
    baseline_rows = read_jsonl(baseline_path)
    candidate_rows = read_jsonl(candidate_path)
    baseline_keys = {merge_key(row) for row in baseline_rows}
    candidate_keys = {merge_key(row) for row in candidate_rows}
    extra = [row for row in candidate_rows if merge_key(row) not in baseline_keys]
    dropped = [row for row in baseline_rows if merge_key(row) not in candidate_keys]

    for rank, row in enumerate(extra, 1):
        row["extra_rank"] = rank
        row["comparison"] = f"{candidate_label}_extra_vs_baseline"
    for rank, row in enumerate(dropped, 1):
        row["dropped_rank"] = rank
        row["comparison"] = f"{candidate_label}_dropped_vs_baseline"

    output_dir.mkdir(parents=True, exist_ok=True)
    extra_path = output_dir / "candidate_extra_merge_review_items.jsonl"
    dropped_path = output_dir / "candidate_dropped_merge_review_items.jsonl"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    write_jsonl(extra_path, extra)
    write_jsonl(dropped_path, dropped)
    summary = {
        "source_baseline": project_rel(baseline_path),
        "source_candidate": project_rel(candidate_path),
        "candidate_label": candidate_label,
        "baseline_count": len(baseline_rows),
        "candidate_count": len(candidate_rows),
        "extra_count": len(extra),
        "dropped_count": len(dropped),
        "extra_risk_tag_counts": risk_counts(extra),
        "dropped_risk_tag_counts": risk_counts(dropped),
        "extra_jsonl": project_rel(extra_path),
        "dropped_jsonl": project_rel(dropped_path),
        "summary_md": project_rel(markdown_path),
    }
    write_json(summary_path, summary)
    markdown_path.write_text(build_markdown(summary, extra, dropped), encoding="utf-8")
    return summary


def build_markdown(
    summary: dict[str, Any],
    extra: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
) -> str:
    lines = [
        "# Cue Planner Merge Review Compare",
        "",
        f"- baseline: `{summary['source_baseline']}`",
        f"- candidate: `{summary['source_candidate']}`",
        f"- baseline_count: {summary['baseline_count']}",
        f"- candidate_count: {summary['candidate_count']}",
        f"- extra_count: {summary['extra_count']}",
        f"- dropped_count: {summary['dropped_count']}",
        "",
        "## Extra Risk Tags",
        "",
    ]
    if summary["extra_risk_tag_counts"]:
        lines.extend(f"- {key}: {value}" for key, value in summary["extra_risk_tag_counts"].items())
    else:
        lines.append("- none")
    lines.extend(["", "## Dropped Risk Tags", ""])
    if summary["dropped_risk_tag_counts"]:
        lines.extend(f"- {key}: {value}" for key, value in summary["dropped_risk_tag_counts"].items())
    else:
        lines.append("- none")
    lines.extend(["", "## Extra Top 10", "", "| rank | time | risk | left | right |", "|---:|---|---|---|---|"])
    for row in extra[:10]:
        lines.append(
            "| {rank} | {start}-{end} | {risk} | {left} | {right} |".format(
                rank=row.get("extra_rank"),
                start=row.get("timeline_start", ""),
                end=row.get("timeline_end", ""),
                risk=str(row.get("risk_tags") or "-").replace("|", " "),
                left=str(row.get("left_ja") or "").replace("|", " "),
                right=str(row.get("right_ja") or "").replace("|", " "),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two cue-planner merge review JSONL files.")
    parser.add_argument("--baseline", required=True, help="baseline merge_review_items.jsonl")
    parser.add_argument("--candidate", required=True, help="candidate merge_review_items.jsonl")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/cue-planner-merge-review-compare",
    )
    args = parser.parse_args(argv)
    summary = compare_reviews(
        baseline_path=project_path(args.baseline),
        candidate_path=project_path(args.candidate),
        output_dir=project_path(args.output_dir),
        candidate_label=args.candidate_label,
    )
    print(
        "extra={extra} dropped={dropped} out={out}".format(
            extra=summary["extra_count"],
            dropped=summary["dropped_count"],
            out=summary["summary_md"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
