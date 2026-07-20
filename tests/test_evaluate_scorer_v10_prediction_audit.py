from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.evaluate_scorer_v10_prediction_audit import evaluate
from tools.audits.generate_scorer_v10_prediction_audit_html import (
    SUMMARY_SCHEMA,
    VERDICT_SCHEMA,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    manifest = tmp_path / "audit_manifest.jsonl"
    targets = [
        {
            "audit_id": "speech_deletion:delete",
            "source_id": "delete",
            "partition": "val",
            "row_role": "speech",
            "category": "speech_deletion",
        },
        {
            "audit_id": "speech_edge_or_partial:edge",
            "source_id": "edge",
            "partition": "test",
            "row_role": "speech",
            "category": "speech_edge_or_partial",
        },
        {
            "audit_id": "long_residual:long",
            "source_id": "long",
            "partition": "train",
            "row_role": "speech",
            "category": "long_residual",
        },
        {
            "audit_id": "background_false_keep:bg",
            "source_id": "bg",
            "partition": "test",
            "row_role": "all_background",
            "category": "background_false_keep",
        },
    ]
    _write_jsonl(manifest, targets)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "audit_manifest": str(manifest),
                "review_item_count": 4,
                "category_counts": {
                    "speech_deletion": 1,
                    "speech_edge_or_partial": 1,
                    "long_residual": 1,
                    "background_false_keep": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return summary, manifest, targets


def test_prediction_manual_gate_classifies_model_and_canonical_blockers(
    tmp_path: Path,
) -> None:
    summary, manifest, targets = _fixture(tmp_path)
    choices = [
        "true_speech_deleted",
        "canonical_should_be_background",
        "missed_background_or_gap",
        "canonical_contains_target_speech",
    ]
    verdicts = tmp_path / "manual_verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {**target, "schema": VERDICT_SCHEMA, "verdict": verdict}
            for target, verdict in zip(targets, choices, strict=True)
        ],
    )
    result = evaluate(
        audit_summary=summary,
        audit_manifest=manifest,
        manual_verdicts=verdicts,
        output=tmp_path / "gate.json",
    )
    assert result["manual_review_complete"] is True
    assert result["zero_clipping_violation_count"] == 1
    assert result["background_behavior_issue_count"] == 1
    assert result["canonical_repair_count"] == 2
    assert result["residual_gate_pass"] is False
    assert result["checkpoint_promotion_authorized"] is False


def test_prediction_manual_gate_can_pass_residuals_but_not_promote_alone(
    tmp_path: Path,
) -> None:
    summary, manifest, targets = _fixture(tmp_path)
    choices = [
        "canonical_should_be_background",
        "canonical_should_be_background",
        "acceptable_long_residual",
        "model_false_keep",
    ]
    verdicts = tmp_path / "manual_verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {**target, "schema": VERDICT_SCHEMA, "verdict": verdict}
            for target, verdict in zip(targets, choices, strict=True)
        ],
    )
    result = evaluate(
        audit_summary=summary,
        audit_manifest=manifest,
        manual_verdicts=verdicts,
        output=tmp_path / "gate.json",
    )
    assert result["zero_clipping_pass"] is True
    assert result["residual_gate_pass"] is False
    assert "canonical_repair_and_rescore_required" in result[
        "checkpoint_promotion_blockers"
    ]


def test_prediction_manual_gate_blocks_same_asr_unit_fragmentation(
    tmp_path: Path,
) -> None:
    summary, manifest, targets = _fixture(tmp_path)
    choices = [
        "canonical_should_be_background",
        "same_asr_unit_fragmented",
        "acceptable_long_residual",
        "canonical_contains_target_speech",
    ]
    verdicts = tmp_path / "manual_verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {**target, "schema": VERDICT_SCHEMA, "verdict": verdict}
            for target, verdict in zip(targets, choices, strict=True)
        ],
    )

    result = evaluate(
        audit_summary=summary,
        audit_manifest=manifest,
        manual_verdicts=verdicts,
        output=tmp_path / "gate.json",
    )

    assert result["zero_clipping_violation_count"] == 0
    assert result["workflow_continuity_issue_count"] == 1
    assert result["workflow_continuity_issue_ids"] == [
        "speech_edge_or_partial:edge"
    ]
    assert result["workflow_continuity_pass"] is False
    assert "manual_workflow_continuity_issue" in result[
        "checkpoint_promotion_blockers"
    ]


def test_prediction_manual_gate_rejects_category_mismatched_verdict(
    tmp_path: Path,
) -> None:
    summary, manifest, targets = _fixture(tmp_path)
    verdicts = tmp_path / "manual_verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {
                **targets[0],
                "schema": VERDICT_SCHEMA,
                "verdict": "acceptable_long_residual",
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid speech_deletion verdict"):
        evaluate(
            audit_summary=summary,
            audit_manifest=manifest,
            manual_verdicts=verdicts,
            output=tmp_path / "gate.json",
        )
