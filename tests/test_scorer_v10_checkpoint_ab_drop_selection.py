from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audits.build_scorer_v10_checkpoint_ab_extra_drop_selection import (
    SELECTION_SCHEMA,
    build_selection,
)
from tools.audits.generate_scorer_v10_prediction_audit_html import (
    CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
    CHECKPOINT_AB_REMAINING_DROP_CONTRACT,
    VERDICT_SCHEMA,
    _render_page,
    audit_truth_drop_spans,
)


def _row(prediction_spans: list[dict]) -> dict:
    return {
        "source_id": "sample",
        "audio": "audio.wav",
        "partition": "val",
        "row_role": "speech",
        "category": "speech_edge_or_partial",
        "frame_count": 5,
        "duration_s": 0.1,
        "false_negative_frames": 0,
        "false_positive_frames": 0,
        "max_predicted_speech_run_s": 0.1,
        "truth_spans": [
            {
                "label": "truth_speech",
                "start_frame": 0,
                "end_frame": 5,
                "start_s": 0.0,
                "end_s": 0.1,
            }
        ],
        "prediction_spans": prediction_spans,
    }


def test_checkpoint_ab_selection_contains_only_new_drop_difference(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "selection.jsonl"
    baseline.write_text(
        json.dumps(
            _row(
                [
                    {
                        "label": "model_speech",
                        "start_frame": 0,
                        "end_frame": 4,
                        "start_s": 0.0,
                        "end_s": 0.08,
                    }
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            _row(
                [
                    {
                        "label": "model_speech",
                        "start_frame": 0,
                        "end_frame": 2,
                        "start_s": 0.0,
                        "end_s": 0.04,
                    }
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_selection(baseline=baseline, candidate=candidate, output=output)
    selected = json.loads(output.read_text(encoding="utf-8"))

    assert summary["schema"] == SELECTION_SCHEMA
    assert summary["source_count"] == 1
    assert summary["extra_false_negative_frame_count"] == 2
    assert selected["audit_truth_drop_contract"] == CHECKPOINT_AB_EXTRA_DROP_CONTRACT
    assert selected["audit_truth_drop_spans"] == [
        {
            "label": "truth_speech_model_background",
            "start_frame": 2,
            "end_frame": 4,
            "start_s": 0.04,
            "end_s": 0.08,
        }
    ]
    assert audit_truth_drop_spans(selected) == selected["audit_truth_drop_spans"]
    page = _render_page([selected])
    assert "红色只显示新候选相对 baseline 新增" in page
    assert "candidate drop - baseline drop" in page


def test_checkpoint_ab_truth_drop_override_rejects_non_candidate_drop() -> None:
    row = _row(
        [
            {
                "label": "model_speech",
                "start_frame": 0,
                "end_frame": 4,
                "start_s": 0.0,
                "end_s": 0.08,
            }
        ]
    )
    row.update(
        audit_truth_drop_contract=CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
        audit_truth_drop_spans=[
            {
                "label": "truth_speech_model_background",
                "start_frame": 0,
                "end_frame": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="not a candidate drop"):
        audit_truth_drop_spans(row)


def test_checkpoint_ab_selection_carries_only_exact_audited_background_frames(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    prior_manifest = tmp_path / "prior_manifest.jsonl"
    prior_verdicts = tmp_path / "prior_verdicts.jsonl"
    output = tmp_path / "selection.jsonl"
    baseline_row = _row(
        [
            {
                "label": "model_speech",
                "start_frame": 0,
                "end_frame": 4,
                "start_s": 0.0,
                "end_s": 0.08,
            }
        ]
    )
    candidate_row = _row(
        [
            {
                "label": "model_speech",
                "start_frame": 0,
                "end_frame": 2,
                "start_s": 0.0,
                "end_s": 0.04,
            }
        ]
    )
    baseline.write_text(json.dumps(baseline_row) + "\n", encoding="utf-8")
    candidate.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")
    prior_row = {
        **candidate_row,
        "audit_id": "speech_edge_or_partial:sample",
        "audit_truth_drop_contract": CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
        "audit_truth_drop_spans": [
            {
                "label": "truth_speech_model_background",
                "start_frame": 2,
                "end_frame": 3,
                "start_s": 0.04,
                "end_s": 0.06,
            }
        ],
    }
    prior_manifest.write_text(json.dumps(prior_row) + "\n", encoding="utf-8")
    prior_verdicts.write_text(
        json.dumps(
            {
                "schema": VERDICT_SCHEMA,
                "audit_id": prior_row["audit_id"],
                "source_id": "sample",
                "partition": "val",
                "row_role": "speech",
                "category": "speech_edge_or_partial",
                "verdict": "canonical_should_be_background",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_selection(
        baseline=baseline,
        candidate=candidate,
        output=output,
        prior_audit_manifest=prior_manifest,
        prior_manual_verdicts=prior_verdicts,
    )
    selected = json.loads(output.read_text(encoding="utf-8"))

    assert summary["selection_contract"] == CHECKPOINT_AB_REMAINING_DROP_CONTRACT
    assert summary["total_extra_false_negative_frame_count_before_carryover"] == 2
    assert summary["carried_audited_background_frame_count"] == 1
    assert summary["remaining_false_negative_frame_count_requiring_review"] == 1
    assert selected["audit_truth_drop_contract"] == CHECKPOINT_AB_REMAINING_DROP_CONTRACT
    assert selected["candidate_extra_false_negative_frames"] == 2
    assert selected["candidate_extra_false_negative_frames_carried_as_background"] == 1
    assert selected["candidate_extra_false_negative_frames_requiring_review"] == 1
    assert selected["audit_truth_drop_spans"] == [
        {
            "label": "truth_speech_model_background",
            "start_frame": 3,
            "end_frame": 4,
            "start_s": 0.06,
            "end_s": 0.08,
        }
    ]
    assert audit_truth_drop_spans(selected) == selected["audit_truth_drop_spans"]
    page = _render_page([selected])
    assert "尚未被此前逐段人工确认为 background" in page
    assert "不继承整条 source 结论" in page
    assert "红段可为停顿，但左右属同一 ASR 单元，切开有害" in page


def test_checkpoint_ab_selection_does_not_carry_true_speech_verdict(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    prior_manifest = tmp_path / "prior_manifest.jsonl"
    prior_verdicts = tmp_path / "prior_verdicts.jsonl"
    output = tmp_path / "selection.jsonl"
    baseline_row = _row(
        [
            {
                "label": "model_speech",
                "start_frame": 0,
                "end_frame": 4,
                "start_s": 0.0,
                "end_s": 0.08,
            }
        ]
    )
    candidate_row = _row(
        [
            {
                "label": "model_speech",
                "start_frame": 0,
                "end_frame": 2,
                "start_s": 0.0,
                "end_s": 0.04,
            }
        ]
    )
    baseline.write_text(json.dumps(baseline_row) + "\n", encoding="utf-8")
    candidate.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")
    prior_row = {
        **candidate_row,
        "audit_id": "speech_edge_or_partial:sample",
        "audit_truth_drop_contract": CHECKPOINT_AB_EXTRA_DROP_CONTRACT,
        "audit_truth_drop_spans": [
            {
                "label": "truth_speech_model_background",
                "start_frame": 2,
                "end_frame": 3,
                "start_s": 0.04,
                "end_s": 0.06,
            }
        ],
    }
    prior_manifest.write_text(json.dumps(prior_row) + "\n", encoding="utf-8")
    prior_verdicts.write_text(
        json.dumps(
            {
                "schema": VERDICT_SCHEMA,
                "audit_id": prior_row["audit_id"],
                "source_id": "sample",
                "partition": "val",
                "row_role": "speech",
                "category": "speech_edge_or_partial",
                "verdict": "true_speech_clipped",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_selection(
        baseline=baseline,
        candidate=candidate,
        output=output,
        prior_audit_manifest=prior_manifest,
        prior_manual_verdicts=prior_verdicts,
    )
    selected = json.loads(output.read_text(encoding="utf-8"))

    assert summary["carried_audited_background_frame_count"] == 0
    assert summary["remaining_false_negative_frame_count_requiring_review"] == 2
    assert selected["audit_truth_drop_spans"] == [
        {
            "label": "truth_speech_model_background",
            "start_frame": 2,
            "end_frame": 4,
            "start_s": 0.04,
            "end_s": 0.08,
        }
    ]
