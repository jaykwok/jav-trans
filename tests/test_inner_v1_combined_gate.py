from __future__ import annotations

import json

from tools.boundary.ja.evaluate_inner_v1_combined_gate import evaluate


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _audit(tmp_path, name: str, verdicts: list[tuple[str, str]]):
    audit = tmp_path / name
    audit.mkdir()
    items = []
    manual = []
    for index, (start, end) in enumerate(verdicts):
        subisland_id = f"{name}-{index}"
        items.append(
            {
                "schema": "inner_subisland_edge_audit_item_v1",
                "subisland_id": subisland_id,
                "start_requires_inner": True,
                "end_requires_inner": True,
                "teacher_usage": "formal_inner_model_heldout_evaluation",
                "checkpoint_sha256": "checkpoint",
                "model_prediction": {
                    "start_action": "refined",
                    "end_action": "refined",
                },
            }
        )
        manual.append(
            {
                "schema": "inner_subisland_edge_manual_verdict_v1",
                "subisland_id": subisland_id,
                "start_verdict": start,
                "end_verdict": end,
            }
        )
    _write(audit / "inner_items.jsonl", items)
    _write(audit / "manual_verdicts.jsonl", manual)
    return audit


def test_combined_gate_accepts_95_percent_with_zero_clipping(tmp_path) -> None:
    first = _audit(
        tmp_path,
        "first",
        [("correct", "correct")] * 5,
    )
    second = _audit(
        tmp_path,
        "second",
        [("correct", "correct")] * 4 + [("correct", "too_wide")],
    )

    summary = evaluate(
        audit_dirs=[first, second], output=tmp_path / "gate.json"
    )

    assert summary["correct_rate"] == 0.95
    assert summary["zero_clipping_pass"] is True
    assert summary["promotion_ready"] is True


def test_combined_gate_rejects_any_clipped_edge(tmp_path) -> None:
    audit = _audit(
        tmp_path,
        "clipped",
        [("correct", "correct")] * 9 + [("correct", "clipped")],
    )

    summary = evaluate(audit_dirs=[audit], output=tmp_path / "gate.json")

    assert summary["correct_rate"] == 0.95
    assert summary["zero_clipping_pass"] is False
    assert summary["promotion_ready"] is False
