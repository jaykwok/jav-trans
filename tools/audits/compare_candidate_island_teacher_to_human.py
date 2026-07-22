#!/usr/bin/env python3
"""Compare candidate-island teacher preaudits with human frame truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FRAME_HOP_S = 0.02
SUMMARY_SCHEMA = "candidate_island_teacher_human_comparison_summary_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _runs(values: list[str], label: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values + ["__end__"]):
        if value == label and start is None:
            start = index
        elif value != label and start is not None:
            result.append((start, index))
            start = None
    return result


def _frame_labels(row: dict[str, Any], *, teacher: bool) -> list[str]:
    count = int(row.get("frame_count") or 0)
    labels = ["outside_candidate"] * count
    if teacher:
        for label, spans in (("inside_candidate", row.get("islands") or ()), ("unsure", row.get("unsure_spans") or ())):
            for span in spans:
                start = int(span["start_frame"]) if "start_frame" in span else round(float(span["start_s"]) / FRAME_HOP_S)
                end = int(span["end_frame"]) if "end_frame" in span else round(float(span["end_s"]) / FRAME_HOP_S)
                start = max(0, min(count, start))
                end = max(start, min(count, end))
                for index in range(start, end):
                    labels[index] = label
        return labels
    for span in row.get("spans") or ():
        start = max(0, min(count, int(span["start_frame"])))
        end = max(start, min(count, int(span["end_frame"])))
        for index in range(start, end):
            labels[index] = str(span["label"])
    return labels


def _metrics(human: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(human) != len(predicted):
        raise ValueError("human and teacher frame counts differ")
    human_inside = {"inside_candidate"}
    predicted_inside = {"inside_candidate"}
    human_outside = {"outside_candidate"}
    n_inside = sum(value in human_inside for value in human)
    n_outside = sum(value in human_outside for value in human)
    tp_inside = sum(h in human_inside and p in predicted_inside for h, p in zip(human, predicted))
    tp_outside = sum(h in human_outside and p in human_outside for h, p in zip(human, predicted))
    predicted_outside = sum(p in human_outside for p in predicted)
    human_runs = _runs(human, "inside_candidate")
    predicted_runs = _runs(predicted, "inside_candidate")
    fragmentation_counts: list[int] = []
    for hs, he in human_runs:
        fragmentation_counts.append(sum(ps < he and pe > hs for ps, pe in predicted_runs))
    return {
        "frame_count": len(human),
        "human_inside_frames": n_inside,
        "human_outside_frames": n_outside,
        "predicted_inside_frames": sum(p in predicted_inside for p in predicted),
        "predicted_unsure_frames": sum(p == "unsure" for p in predicted),
        "inside_candidate_recall": tp_inside / max(n_inside, 1),
        "safe_inside_recall_inside_or_unsure": sum(h in human_inside and p in {"inside_candidate", "unsure"} for h, p in zip(human, predicted)) / max(n_inside, 1),
        "outside_candidate_recall": tp_outside / max(n_outside, 1),
        "outside_candidate_precision": tp_outside / max(predicted_outside, 1),
        "extra_inside_rate_on_human_outside": sum(h in human_outside and p in predicted_inside for h, p in zip(human, predicted)) / max(n_outside, 1),
        "true_inside_deletion_rate": sum(h in human_inside and p == "outside_candidate" for h, p in zip(human, predicted)) / max(n_inside, 1),
        "human_inside_run_count": len(human_runs),
        "predicted_inside_run_count": len(predicted_runs),
        "human_runs_with_no_predicted_inside": sum(count == 0 for count in fragmentation_counts),
        "human_runs_fragmented_by_multiple_predicted_runs": sum(count > 1 for count in fragmentation_counts),
        "mean_predicted_runs_per_human_run": sum(fragmentation_counts) / max(len(fragmentation_counts), 1),
        "predicted_full_source_inside": bool(predicted_runs and predicted_runs[0][0] == 0 and predicted_runs[-1][1] == len(predicted) and len(predicted_runs) == 1),
    }


def compare(*, human_path: Path, teacher_specs: list[str], output_dir: Path) -> dict[str, Any]:
    human_rows = {str(row["source_id"]): row for row in _rows(human_path)}
    teachers: dict[str, dict[str, Any]] = {}
    for spec in teacher_specs:
        name, separator, path_text = spec.partition("=")
        if not separator or not name or not path_text:
            raise ValueError("teacher must use name=path")
        rows = {str(row["source_id"]): row for row in _rows(Path(path_text))}
        if set(rows) != set(human_rows):
            raise ValueError(f"teacher {name} source set differs from human")
        teachers[name] = rows
    per_source: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, dict[str, float]] = {name: {} for name in teachers}
    for source_id, human_row in sorted(human_rows.items()):
        human_labels = _frame_labels(human_row, teacher=False)
        per_source[source_id] = {"partition": human_row.get("partition"), "duration_s": human_row.get("duration_s"), "human_inside_runs": len(_runs(human_labels, "inside_candidate")), "teachers": {}}
        for name, teacher_rows in teachers.items():
            metrics = _metrics(human_labels, _frame_labels(teacher_rows[source_id], teacher=True))
            per_source[source_id]["teachers"][name] = metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    aggregate[name].setdefault(key, 0.0)
                    aggregate[name][key] += float(value)
    for name in aggregate:
        aggregate[name] = {key: value / max(len(human_rows), 1) for key, value in aggregate[name].items()}
        aggregate[name]["source_count"] = len(human_rows)
        aggregate[name]["sources_with_full_source_inside"] = sum(bool(per_source[s]["teachers"][name]["predicted_full_source_inside"]) for s in per_source)
        aggregate[name]["sources_with_any_true_inside_deletion"] = sum(per_source[s]["teachers"][name]["true_inside_deletion_rate"] > 0.0 for s in per_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "per_source.jsonl"
    detail_path.write_text("".join(json.dumps({"source_id": source_id, **payload}, ensure_ascii=False, sort_keys=True) + "\n" for source_id, payload in per_source.items()), encoding="utf-8")
    summary = {"schema": SUMMARY_SCHEMA, "human_verdicts": str(human_path), "teacher_specs": teacher_specs, "source_count": len(human_rows), "aggregate": aggregate, "per_source": str(detail_path), "training_manifest_allowed": False}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", required=True)
    parser.add_argument("--teacher", action="append", required=True, help="name=path")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(compare(human_path=Path(args.human), teacher_specs=args.teacher, output_dir=Path(args.output_dir)), ensure_ascii=False))
