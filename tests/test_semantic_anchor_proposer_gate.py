from __future__ import annotations

import json
from pathlib import Path

from tools.boundary.ja.evaluate_semantic_anchor_proposer_audit import (
    evaluate,
    safe_runs,
)


def _candidate(candidate_id: str, time_s: float, label: str) -> dict:
    return {"candidate_id": candidate_id, "time_s": time_s, "label": label}


def test_safe_runs_keep_separated_regions_distinct() -> None:
    candidates = [
        _candidate("c00", 0.1, "safe"),
        _candidate("c01", 0.2, "left_clipped"),
        _candidate("c02", 0.3, "safe"),
        _candidate("c03", 0.4, "safe"),
        _candidate("c04", 0.5, "right_clipped"),
    ]

    runs = safe_runs(candidates)

    assert [[row["candidate_id"] for row in run] for run in runs] == [
        ["c00"],
        ["c02", "c03"],
    ]


def test_gate_requires_one_contiguous_safe_run_per_event(tmp_path: Path) -> None:
    verdicts = tmp_path / "manual_verdicts.jsonl"
    rows = [
        {
            "event_key": "single",
            "complete": True,
            "candidates": [
                _candidate("c00", 0.1, "left_clipped"),
                _candidate("c01", 0.2, "safe"),
                _candidate("c02", 0.3, "safe"),
                _candidate("c03", 0.4, "right_clipped"),
            ],
        },
        {
            "event_key": "multiple",
            "complete": True,
            "candidates": [
                _candidate("c00", 0.1, "safe"),
                _candidate("c01", 0.2, "right_clipped"),
                _candidate("c02", 0.3, "safe"),
            ],
        },
    ]
    verdicts.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = evaluate(verdicts, output=tmp_path / "summary.json")

    assert summary["safe_coverage"] == 1.0
    assert summary["paired_edge_eligibility"] == 0.5
    assert summary["coverage_gate_passed"] is True
    assert summary["paired_edge_gate_passed"] is False
    assert summary["training_ready"] is False
    assert summary["events"][0]["route"] == "paired_inner_edges"
    assert summary["events"][1]["route"] == (
        "abstain_and_review_semantic_event_decomposition"
    )
