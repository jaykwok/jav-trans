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


def evaluate(*, audit_dir: Path) -> dict[str, Any]:
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _rows(audit_dir / "manual_verdicts.jsonl")
    expected_ids = [str(value) for value in summary["audio_ids"]]
    verdicts = {str(row["audio_id"]): row for row in rows}
    missing = [audio_id for audio_id in expected_ids if audio_id not in verdicts]
    extra = sorted(set(verdicts) - set(expected_ids))
    duplicate_count = len(rows) - len(verdicts)
    model_values = [
        str(verdicts[audio_id][f"{edge}_model"])
        for audio_id in expected_ids
        if audio_id in verdicts
        for edge in ("start", "end")
    ]
    target_values = [
        str(verdicts[audio_id][f"{edge}_target"])
        for audio_id in expected_ids
        if audio_id in verdicts
        for edge in ("start", "end")
    ]
    model_counts = Counter(model_values)
    target_counts = Counter(target_values)
    complete = (
        not missing
        and not extra
        and duplicate_count == 0
        and "unreviewed" not in model_counts
        and "unreviewed" not in target_counts
        and "" not in model_counts
        and "" not in target_counts
    )
    result = {
        "schema": "outer_v2_directional_tail_manual_gate_v1",
        "audit_dir": str(audit_dir),
        "sample_count": len(expected_ids),
        "edge_count": len(expected_ids) * 2,
        "complete": complete,
        "missing_audio_ids": missing,
        "extra_audio_ids": extra,
        "duplicate_count": duplicate_count,
        "model_edge_counts": dict(sorted(model_counts.items())),
        "training_target_edge_counts": dict(sorted(target_counts.items())),
        "known_model_clipping_count": int(model_counts.get("clipped_semantic", 0)),
        "model_too_wide_count": int(model_counts.get("too_wide_nonsemantic", 0)),
        "training_target_includes_nonsemantic_count": int(
            target_counts.get("includes_nonsemantic", 0)
        ),
        "training_target_clips_semantic_count": int(
            target_counts.get("clips_semantic", 0)
        ),
        "tail_clipping_gate_pass": complete
        and model_counts.get("clipped_semantic", 0) == 0,
        "promotion_ready": False,
        "next_gate": "actual_workflow_semantic_source_fixed5",
    }
    (audit_dir / "gate_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Outer v2 tail audit verdicts.")
    parser.add_argument("--audit-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(audit_dir=Path(parse_args().audit_dir)), ensure_ascii=False))
