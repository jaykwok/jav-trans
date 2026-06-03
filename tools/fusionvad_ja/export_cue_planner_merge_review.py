#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.fusionvad_ja.analyze_subtitle_cue_merge_candidates import (  # noqa: E402
    _reading_units_per_s,
    _text_units,
    merge_blocks,
)


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "priority_rank",
        "review_priority",
        "action_order",
        "timeline_start_s",
        "timeline_end_s",
        "left_index",
        "right_index",
        "score",
        "gap_s",
        "combined_duration_s",
        "combined_text_units",
        "combined_reading_units_per_s",
        "speaker_change_score",
        "speaker_threshold",
        "speaker_change",
        "crosses_chunk",
        "risky_chunks",
        "warn_chunks",
        "risk_tags",
        "left_ja",
        "right_ja",
        "merged_ja",
        "left_zh",
        "right_zh",
        "merged_zh",
        "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_text(value: Any, *, max_len: int = 140) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _block_text(block: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(block.get(key) or "").strip()
        if value:
            return value
    return ""


def _time_label(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _risk_tags(action: dict[str, Any], *, reading_warn_units_per_s: float) -> list[str]:
    tags: list[str] = []
    annotations = action.get("annotations") if isinstance(action.get("annotations"), dict) else {}
    speaker = annotations.get("speaker_pair") if isinstance(annotations.get("speaker_pair"), dict) else {}
    diagnostics = annotations.get("diagnostics") if isinstance(annotations.get("diagnostics"), dict) else {}

    score = _as_float(speaker.get("speaker_change_score"))
    threshold = _as_float(speaker.get("threshold"))
    if speaker.get("speaker_change"):
        tags.append("speaker_change")
    elif threshold > 0 and score >= threshold * 0.9:
        tags.append("near_speaker_threshold")
    elif score >= 0.8:
        tags.append("high_speaker_score")

    if diagnostics.get("risky_chunks"):
        tags.append("fallback_risk")
    if diagnostics.get("warn_chunks"):
        tags.append("fallback_warn")
    if diagnostics.get("crosses_chunk"):
        tags.append("crosses_chunk")

    if _as_float(action.get("combined_duration_s")) >= 6.0:
        tags.append("long_combined_duration")
    reading = _as_float(action.get("combined_reading_units_per_s"))
    if reading_warn_units_per_s > 0 and reading > reading_warn_units_per_s:
        tags.append("reading_density_high")
    if _as_float(action.get("gap_s")) >= 0.5:
        tags.append("loose_gap")

    return tags


def _review_priority(tags: list[str]) -> int:
    weights = {
        "speaker_change": 120,
        "fallback_risk": 90,
        "near_speaker_threshold": 55,
        "high_speaker_score": 40,
        "crosses_chunk": 35,
        "fallback_warn": 25,
        "reading_density_high": 20,
        "long_combined_duration": 15,
        "loose_gap": 10,
    }
    return sum(weights.get(tag, 0) for tag in tags)


def build_review_rows(
    *,
    before_blocks: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    reading_warn_units_per_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_order, action in enumerate(actions):
        left_index = int(action.get("left_index") or 0)
        right_index = int(action.get("right_index") or 0)
        if left_index < 0 or right_index < 0:
            continue
        if left_index >= len(before_blocks) or right_index >= len(before_blocks):
            continue
        left = before_blocks[left_index]
        right = before_blocks[right_index]
        merged = merge_blocks(left, right)
        start = min(_as_float(left.get("start")), _as_float(right.get("start")))
        end = max(_as_float(left.get("end")), _as_float(right.get("end")))
        combined_units = _as_float(action.get("combined_text_units"), _text_units(left) + _text_units(right))
        combined_duration = _as_float(action.get("combined_duration_s"), max(0.0, end - start))
        reading = _reading_units_per_s(combined_units, combined_duration)

        annotations = action.get("annotations") if isinstance(action.get("annotations"), dict) else {}
        speaker = annotations.get("speaker_pair") if isinstance(annotations.get("speaker_pair"), dict) else {}
        diagnostics = annotations.get("diagnostics") if isinstance(annotations.get("diagnostics"), dict) else {}

        enriched_action = {
            **action,
            "combined_reading_units_per_s": reading,
        }
        tags = _risk_tags(enriched_action, reading_warn_units_per_s=reading_warn_units_per_s)
        rows.append(
            {
                "action_order": action_order,
                "timeline_start_s": round(start, 6),
                "timeline_end_s": round(end, 6),
                "timeline_start": _time_label(start),
                "timeline_end": _time_label(end),
                "left_index": left_index,
                "right_index": right_index,
                "left_cue_id": left.get("cue_id"),
                "right_cue_id": right.get("cue_id"),
                "score": _as_float(action.get("score")),
                "gap_s": _as_float(action.get("gap_s")),
                "combined_duration_s": round(combined_duration, 6),
                "combined_text_units": round(combined_units, 3),
                "combined_reading_units_per_s": round(reading, 3),
                "speaker_change_score": speaker.get("speaker_change_score"),
                "speaker_threshold": speaker.get("threshold"),
                "speaker_change": bool(speaker.get("speaker_change")),
                "crosses_chunk": bool(diagnostics.get("crosses_chunk")),
                "risky_chunks": ",".join(str(v) for v in diagnostics.get("risky_chunks") or []),
                "warn_chunks": ",".join(str(v) for v in diagnostics.get("warn_chunks") or []),
                "risk_tags": ",".join(tags),
                "review_priority": _review_priority(tags),
                "left_ja": _compact_text(_block_text(left, "ja_text", "text")),
                "right_ja": _compact_text(_block_text(right, "ja_text", "text")),
                "merged_ja": _compact_text(_block_text(merged, "ja_text", "text"), max_len=220),
                "left_zh": _compact_text(_block_text(left, "zh_text", "zh")),
                "right_zh": _compact_text(_block_text(right, "zh_text", "zh")),
                "merged_zh": _compact_text(_block_text(merged, "zh_text", "zh"), max_len=220),
                "reasons": ",".join(str(reason) for reason in action.get("reasons") or []),
                "annotations": annotations,
            }
        )

    rows.sort(key=lambda row: (-int(row["review_priority"]), float(row["timeline_start_s"]), int(row["action_order"])))
    for rank, row in enumerate(rows, 1):
        row["priority_rank"] = rank
    return rows


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], *, top_n: int) -> str:
    lines = [
        "# Cue Planner Merge Review",
        "",
        f"- before_blocks: `{summary['source_before_blocks']}`",
        f"- planner_actions: `{summary['source_planner_actions']}`",
        f"- action_count: {summary['action_count']}",
        f"- reading_warn_units_per_s: {summary['reading_warn_units_per_s']}",
        "",
        "## Risk Tags",
        "",
    ]
    if summary["risk_tag_counts"]:
        lines.extend(f"- {key}: {value}" for key, value in summary["risk_tag_counts"].items())
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            f"## Top {min(top_n, len(rows))}",
            "",
            "| rank | time | priority | score | speaker | risk | left | right |",
            "|---:|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows[:top_n]:
        lines.append(
            "| {rank} | {start}-{end} | {priority} | {score:.3f} | {speaker} | {risk} | {left} | {right} |".format(
                rank=row["priority_rank"],
                start=row["timeline_start"],
                end=row["timeline_end"],
                priority=row["review_priority"],
                score=row["score"],
                speaker=row["speaker_change_score"] if row["speaker_change_score"] is not None else "",
                risk=row["risk_tags"] or "-",
                left=str(row["left_ja"]).replace("|", " "),
                right=str(row["right_ja"]).replace("|", " "),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_review(
    *,
    before_blocks_path: Path,
    planner_actions_path: Path,
    output_dir: Path,
    reading_warn_units_per_s: float,
    top_n: int,
) -> dict[str, Any]:
    before_blocks = read_json(before_blocks_path)
    actions = read_json(planner_actions_path)
    if not isinstance(before_blocks, list):
        raise ValueError(f"expected before blocks list: {before_blocks_path}")
    if not isinstance(actions, list):
        raise ValueError(f"expected planner actions list: {planner_actions_path}")
    rows = build_review_rows(
        before_blocks=[dict(block) for block in before_blocks if isinstance(block, dict)],
        actions=[dict(action) for action in actions if isinstance(action, dict)],
        reading_warn_units_per_s=reading_warn_units_per_s,
    )
    tag_counts: Counter[str] = Counter()
    for row in rows:
        for tag in str(row.get("risk_tags") or "").split(","):
            if tag:
                tag_counts.update([tag])
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "merge_review_items.jsonl"
    csv_path = output_dir / "merge_review_items.csv"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    summary = {
        "source_before_blocks": project_rel(before_blocks_path),
        "source_planner_actions": project_rel(planner_actions_path),
        "output_dir": project_rel(output_dir),
        "action_count": len(actions),
        "review_item_count": len(rows),
        "reading_warn_units_per_s": reading_warn_units_per_s,
        "risk_tag_counts": dict(tag_counts.most_common()),
        "manifest_jsonl": project_rel(jsonl_path),
        "manifest_csv": project_rel(csv_path),
        "summary_md": project_rel(markdown_path),
    }
    write_json(summary_path, summary)
    markdown_path.write_text(build_markdown(summary, rows, top_n=top_n), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a risk-prioritized review list from cue-planner merge actions."
    )
    parser.add_argument("--before-blocks", required=True, help="before_blocks.json")
    parser.add_argument("--planner-actions", required=True, help="planner_actions.json")
    parser.add_argument(
        "--output-dir",
        default="agents/temp/fusionvad-ja/cue-planner-merge-review",
    )
    parser.add_argument(
        "--reading-warn-units-per-s",
        type=float,
        default=16.0,
        help="Only used to tag high reading-density merges in the review list.",
    )
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)

    summary = build_review(
        before_blocks_path=project_path(args.before_blocks),
        planner_actions_path=project_path(args.planner_actions),
        output_dir=project_path(args.output_dir),
        reading_warn_units_per_s=float(args.reading_warn_units_per_s),
        top_n=max(0, int(args.top_n)),
    )
    print(
        "review={path} items={items}".format(
            path=summary["manifest_jsonl"],
            items=summary["review_item_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
