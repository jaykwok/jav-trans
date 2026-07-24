from __future__ import annotations

import numpy as np

from tools.audits.score_candidate_island_scorer_v11_checkpoint import (
    IGNORE_INDEX,
    evaluate_source_prediction,
    summarize_source_predictions,
)


def test_v11_source_audit_tracks_deletion_fragmentation_and_unsure() -> None:
    truth = np.asarray(
        [0, 1, 1, 1, 1, IGNORE_INDEX, 0, 1, 1, 0, 0, 0], dtype=np.int64
    )
    predicted = np.asarray([0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1], dtype=np.int64)

    result = evaluate_source_prediction(
        truth,
        predicted,
        tolerance_frames=1,
        long_residual_frames=3,
    )

    assert result["truth_inside_run_count"] == 2
    assert result["truth_outside_run_count"] == 3
    assert result["outside_run_mean_recall"] == 1 / 3
    assert result["true_inside_deletion_count"] == 1
    assert result["fragmented_truth_run_count"] == 1
    assert result["continuous_truth_run_count"] == 0
    assert result["prediction_inside_run_count"] == 3
    assert result["internal_drop_gap_count"] == 1
    assert result["internal_drop_gap_1_frame_count"] == 1
    assert result["internal_drop_gap_2_frame_count"] == 0
    assert result["internal_drop_gap_3_frame_count"] == 0
    assert result["internal_drop_gap_4plus_frame_count"] == 0
    assert result["prediction_drop_truth_keep_frame_count"] == 3
    assert result["prediction_keep_truth_drop_frame_count"] == 4
    assert result["long_residual_count"] == 1
    assert result["confusion_truth_by_prediction"] == [[1, 4], [3, 3]]


def test_v11_source_audit_perfect_prediction_passes_coverage() -> None:
    truth = np.asarray([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int64)

    result = evaluate_source_prediction(
        truth,
        truth.copy(),
        tolerance_frames=0,
        long_residual_frames=400,
    )

    assert result["inside_candidate_recall"] == 1.0
    assert result["outside_candidate_recall"] == 1.0
    assert result["outside_run_mean_recall"] == 1.0
    assert result["start_coverage"] == 1.0
    assert result["end_coverage"] == 1.0
    assert result["truth_run_continuity"] == 1.0
    assert result["true_inside_deletion_count"] == 0


def test_v11_source_audit_preserves_teacher_outside_negative_control() -> None:
    truth = np.zeros(12, dtype=np.int64)
    kept = np.ones(12, dtype=np.int64)
    result = evaluate_source_prediction(
        truth,
        kept,
        tolerance_frames=15,
        long_residual_frames=None,
    )

    assert result["inside_candidate_recall"] == 0.0
    assert result["outside_candidate_recall"] == 0.0
    assert result["outside_run_mean_recall"] == 0.0
    assert result["truth_run_continuity"] == 0.0
    assert result["all_outside_source"] is True
    assert result["outside_source_recall"] == 0.0
    assert result["all_outside_source_drop_success"] is False


def test_v11_source_audit_marks_all_outside_drop_as_success() -> None:
    truth = np.zeros(12, dtype=np.int64)
    result = evaluate_source_prediction(
        truth,
        truth.copy(),
        tolerance_frames=15,
        long_residual_frames=None,
    )

    assert result["all_outside_source"] is True
    assert result["outside_source_recall"] == 1.0
    assert result["all_outside_source_drop_success"] is True

    summary = summarize_source_predictions([result])
    assert summary["outside_source_macro_recall"] == 1.0
    assert summary["all_outside_source_count"] == 1
    assert summary["all_outside_source_drop_recall"] == 1.0
    assert summary["truth_run_continuity"] is None
    assert summary["truth_run_continuity_applicable"] is False
