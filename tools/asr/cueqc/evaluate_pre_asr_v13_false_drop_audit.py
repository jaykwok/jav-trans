#!/usr/bin/env python3
"""Evaluate the complete manual audit of CueQC v13 false-drop predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_VERDICTS = {"safe_drop", "true_speech", "unsure", "unreviewed"}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    *, false_drop_manifest: Path, manual_verdicts: Path, output: Path
) -> dict[str, Any]:
    targets = _rows(false_drop_manifest)
    target_ids = [str(row.get("row_id") or "") for row in targets]
    if any(not item for item in target_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("false-drop manifest row_id values must be non-empty and unique")
    verdict_by_id: dict[str, dict[str, Any]] = {}
    for row in _rows(manual_verdicts):
        row_id = str(row.get("row_id") or "")
        verdict = str(row.get("verdict") or "unreviewed")
        if row_id not in set(target_ids):
            raise ValueError(f"manual verdict has no false-drop target: {row_id}")
        if row_id in verdict_by_id:
            raise ValueError(f"duplicate manual false-drop verdict: {row_id}")
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"unsupported manual false-drop verdict: {verdict}")
        verdict_by_id[row_id] = row
    resolved = [
        str(verdict_by_id.get(row_id, {}).get("verdict") or "unreviewed")
        for row_id in target_ids
    ]
    counts = Counter(resolved)
    complete = counts["unreviewed"] == 0 and len(verdict_by_id) == len(target_ids)
    true_speech = counts["true_speech"]
    unsure = counts["unsure"]
    summary = {
        "schema": "cueqc_v13_false_drop_manual_gate_summary_v1",
        "target_manifest_count": len(targets),
        "reviewed_target_count": len(targets) - counts["unreviewed"],
        "manual_verdict_counts": dict(sorted(counts.items())),
        "complete": complete,
        "true_semantic_keep_deletion_count": true_speech,
        "uncertain_count": unsure,
        "promote_allowed": bool(complete and true_speech == 0 and unsure == 0),
        "false_drop_manifest": str(false_drop_manifest),
        "manual_verdicts": str(manual_verdicts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-drop-manifest", required=True)
    parser.add_argument("--manual-verdicts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                false_drop_manifest=Path(args.false_drop_manifest),
                manual_verdicts=Path(args.manual_verdicts),
                output=Path(args.output),
            ),
            ensure_ascii=False,
        )
    )
