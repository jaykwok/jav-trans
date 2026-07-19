from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.audits.generate_scorer_v10_prediction_audit_html import build_audit
from tools.audits.score_scorer_v10_checkpoint_audit import (
    _select_audit_rows,
    _span_rows,
)


def test_checkpoint_audit_span_rows_and_selection_include_required_residuals() -> None:
    spans = _span_rows(np.asarray([False, True, True, False]), label="truth_speech")
    assert spans == [
        {
            "label": "truth_speech",
            "start_frame": 1,
            "end_frame": 3,
            "start_s": 0.02,
            "end_s": 0.06,
        }
    ]
    rows = [
        {
            "source_id": "train-delete",
            "partition": "train",
            "category": "speech_deletion",
            "true_speech_deletions": 1,
            "max_predicted_speech_run_s": 1.0,
            "false_negative_frames": 10,
            "false_positive_frames": 0,
        },
        {
            "source_id": "val-bg",
            "partition": "val",
            "category": "background_false_keep",
            "true_speech_deletions": 0,
            "max_predicted_speech_run_s": 1.0,
            "false_negative_frames": 0,
            "false_positive_frames": 2,
        },
        {
            "source_id": "test-long",
            "partition": "test",
            "category": "normal",
            "true_speech_deletions": 0,
            "max_predicted_speech_run_s": 9.0,
            "false_negative_frames": 0,
            "false_positive_frames": 0,
        },
    ]
    assert {
        row["source_id"] for row in _select_audit_rows(rows, max_items=0)
    } == {"train-delete", "val-bg", "test-long"}


def test_prediction_audit_html_is_exact_span_and_saveable(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF-test")
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "source_id": "val-bg",
                "partition": "val",
                "row_role": "all_background",
                "category": "background_false_keep",
                "audio": str(audio),
                "duration_s": 1.0,
                "false_negative_frames": 0,
                "false_positive_frames": 1,
                "max_predicted_speech_run_s": 0.1,
                "truth_spans": [
                    {
                        "label": "truth_background",
                        "start_s": 0.0,
                        "end_s": 1.0,
                    }
                ],
                "prediction_spans": [
                    {"label": "model_speech", "start_s": 0.4, "end_s": 0.5}
                ],
                "true_speech_deletions": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = build_audit(selection=selection, output_dir=tmp_path / "audit")
    page = index.read_text(encoding="utf-8")
    assert "playExact" in page
    assert "speech_scorer_v10_prediction_manual_verdict_v1" in page
    assert (index.parent / "audio" / "item-000.wav").is_file()
