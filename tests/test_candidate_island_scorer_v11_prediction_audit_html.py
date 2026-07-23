from __future__ import annotations

from tools.audits.generate_candidate_island_scorer_v11_prediction_audit_html import (
    _page,
    build_items,
)


def test_v11_prediction_audit_includes_full_source_drop_and_long_residual() -> None:
    predictions = [
        {
            "source_id": "source-1",
            "partition": "test",
            "duration_s": 12.0,
            "frame_count": 600,
            "checkpoint_sha256": "a" * 64,
            "truth_spans": [],
            "prediction_spans": [],
            "prediction_drop_truth_keep_spans": [
                {"start_s": 1.0, "end_s": 1.2, "label": "drop"}
            ],
            "long_residual_spans": [
                {"start_s": 2.0, "end_s": 11.0, "label": "long"}
            ],
        }
    ]

    items = build_items(predictions)

    assert [item["category"] for item in items] == [
        "heldout_full_source",
        "prediction_drop_truth_keep",
        "long_residual_over_8s",
    ]


def test_v11_prediction_audit_escapes_jsonl_newlines_in_javascript() -> None:
    html = _page([])

    assert ".join('\\n')+'\\n'" in html
    assert ".join('\n')+'\n'" not in html
