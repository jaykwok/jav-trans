from __future__ import annotations

import json
from pathlib import Path

from tools.audits.build_scorer_v10_train_background_false_keep_selection import (
    SCORE_SUMMARY_SCHEMA,
    build_selection,
)


def test_train_background_selection_keeps_every_false_keep_without_duration_filter(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "source_id": "short",
            "partition": "train",
            "row_role": "all_background",
            "category": "background_false_keep",
            "truth_speech_frames": 0,
            "false_positive_frames": 1,
            "false_negative_frames": 0,
            "max_predicted_speech_run_s": 0.02,
        },
        {
            "source_id": "long",
            "partition": "train",
            "row_role": "all_background",
            "category": "background_false_keep",
            "truth_speech_frames": 0,
            "false_positive_frames": 40,
            "false_negative_frames": 0,
            "max_predicted_speech_run_s": 0.8,
        },
        {
            "source_id": "heldout",
            "partition": "val",
            "row_role": "all_background",
            "category": "background_false_keep",
            "truth_speech_frames": 0,
            "false_positive_frames": 100,
            "false_negative_frames": 0,
            "max_predicted_speech_run_s": 2.0,
        },
    ]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    score = tmp_path / "score.json"
    score.write_text(
        json.dumps(
            {
                "schema": SCORE_SUMMARY_SCHEMA,
                "checkpoint": "diagnostic.pt",
                "predictions": str(predictions),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selection"
    result = build_selection(
        score_summary_path=score,
        predictions_path=predictions,
        output_dir=output,
    )
    selected = [
        json.loads(line)
        for line in (output / "selection.jsonl").read_text().splitlines()
    ]
    assert [row["source_id"] for row in selected] == ["long", "short"]
    assert result["selection_count"] == 2
    assert result["duration_or_probability_filter_applied"] is False
    assert result["training_manifest_allowed"] is False
