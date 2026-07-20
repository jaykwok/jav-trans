#!/usr/bin/env python3
"""Select only candidate-vs-baseline extra Scorer v10 speech-drop spans."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audits.generate_scorer_v10_prediction_audit_html import (  # noqa: E402
    CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
    FRAME_HOP_S,
    truth_drop_spans,
)


SELECTION_SCHEMA = "scorer_v10_checkpoint_ab_extra_drop_selection_v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _drop_mask(row: dict[str, Any]) -> np.ndarray:
    values = np.zeros(int(row["frame_count"]), dtype=bool)
    for span in truth_drop_spans(row):
        values[int(span["start_frame"]): int(span["end_frame"])] = True
    return values


def _spans(values: np.ndarray) -> list[dict[str, Any]]:
    padded = np.r_[False, np.asarray(values, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    return [
        {
            "label": "truth_speech_model_background",
            "start_frame": int(start),
            "end_frame": int(end),
            "start_s": int(start) * FRAME_HOP_S,
            "end_s": int(end) * FRAME_HOP_S,
        }
        for start, end in changes
    ]


def build_selection(
    *, baseline: Path, candidate: Path, output: Path
) -> dict[str, Any]:
    baseline_rows = {row["source_id"]: row for row in _rows(baseline)}
    candidate_rows = {row["source_id"]: row for row in _rows(candidate)}
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Scorer v10 checkpoint A/B source identity mismatch")
    selected: list[dict[str, Any]] = []
    for source_id in sorted(candidate_rows):
        baseline_row = baseline_rows[source_id]
        candidate_row = candidate_rows[source_id]
        for key in ("frame_count", "audio", "truth_spans", "partition", "row_role"):
            if baseline_row.get(key) != candidate_row.get(key):
                raise ValueError(
                    f"Scorer v10 checkpoint A/B canonical field mismatch: {source_id} {key}"
                )
        baseline_drop = _drop_mask(baseline_row)
        candidate_drop = _drop_mask(candidate_row)
        extra = candidate_drop & ~baseline_drop
        if not np.any(extra):
            continue
        category = str(candidate_row.get("category") or "speech_edge_or_partial")
        if category not in {"speech_deletion", "speech_edge_or_partial"}:
            category = "speech_edge_or_partial"
        selected.append(
            {
                **candidate_row,
                "category": category,
                "audit_truth_drop_contract": CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
                "audit_truth_drop_spans": _spans(extra),
                "baseline_false_negative_frames": int(np.sum(baseline_drop)),
                "candidate_false_negative_frames": int(np.sum(candidate_drop)),
                "candidate_extra_false_negative_frames": int(np.sum(extra)),
            }
        )
    if not selected:
        raise ValueError("Scorer v10 checkpoint A/B has no extra speech-drop spans")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "schema": SELECTION_SCHEMA,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "output": str(output),
        "selection_contract": CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
        "source_count": len(selected),
        "extra_false_negative_frame_count": sum(
            int(row["candidate_extra_false_negative_frames"]) for row in selected
        ),
        "partition_counts": dict(Counter(str(row["partition"]) for row in selected)),
        "category_counts": dict(Counter(str(row["category"]) for row in selected)),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_selection(
                baseline=Path(args.baseline),
                candidate=Path(args.candidate),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
