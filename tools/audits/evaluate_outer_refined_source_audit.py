#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(*, audit_dir: Path, expected_labels: Path) -> dict[str, Any]:
    expected_ids = [str(row["sample_id"]) for row in _rows(expected_labels)]
    verdict_rows = _rows(audit_dir / "manual_verdicts.jsonl")
    verdict_by_id = {str(row["sample_id"]): row for row in verdict_rows}
    missing = [sample_id for sample_id in expected_ids if sample_id not in verdict_by_id]
    extra = sorted(set(verdict_by_id) - set(expected_ids))
    duplicate_count = len(verdict_rows) - len(verdict_by_id)
    edge_values = [
        str(verdict_by_id[sample_id][f"{edge}_model"])
        for sample_id in expected_ids
        if sample_id in verdict_by_id
        for edge in ("start", "end")
    ]
    counts = Counter(edge_values)
    complete = (
        not missing
        and not extra
        and duplicate_count == 0
        and len(edge_values) == len(expected_ids) * 2
        and not ({"", "unreviewed"} & set(counts))
    )
    clipped = int(counts.get("clipped_semantic", 0))
    too_wide = int(counts.get("too_wide_nonsemantic", 0))
    result = {
        "schema": "outer_refined_source_manual_gate_v1",
        "audit_dir": str(audit_dir),
        "expected_labels": str(expected_labels),
        "sample_count": len(expected_ids),
        "edge_count": len(expected_ids) * 2,
        "complete": complete,
        "missing_sample_ids": missing,
        "extra_sample_ids": extra,
        "duplicate_count": duplicate_count,
        "model_edge_counts": dict(sorted(counts.items())),
        "known_model_clipping_count": clipped,
        "model_too_wide_count": too_wide,
        "zero_clipping_pass": complete and clipped == 0,
        "edge_cleanup_pass": complete and clipped == 0 and too_wide == 0,
        "promotion_ready": complete and clipped == 0 and too_wide == 0,
    }
    (audit_dir / "gate_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Outer refined-source audit.")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--expected-labels", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                audit_dir=Path(args.audit_dir),
                expected_labels=Path(args.expected_labels),
            ),
            ensure_ascii=False,
        )
    )
