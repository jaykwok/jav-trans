from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.datasets.analyze_split_cut_eager_operating_point import analyze


def _candidate(time_s: float, p_cut: float, p_continue: float, p_unsure: float) -> dict:
    return {
        "time_s": time_s,
        "p_cut": p_cut,
        "p_continue": p_continue,
        "p_unsure": p_unsure,
    }


def test_cut_eager_replay_rejects_same_sentence_pause_gate(tmp_path: Path) -> None:
    aligned_path = tmp_path / "aligned.json"
    labels_path = tmp_path / "labels.jsonl"
    aligned_path.write_text(
        json.dumps(
            {
                "boundary_signature": {
                    "boundary_pipeline": {
                        "semantic_split_model": {
                            "decision_config": {
                                "short_core_max_s": 6.0,
                                "short_core_cut_threshold": 0.9,
                                "normal_cut_threshold": 0.75,
                            }
                        }
                    }
                },
                "segments": [
                    {
                        "start": 0.0,
                        "end": 10.0,
                        "chunk_acoustic_start": 0.0,
                        "chunk_acoustic_end": 10.0,
                        "weak_cut_candidates": [
                            _candidate(3.0, 0.60, 0.35, 0.05),
                            _candidate(6.0, 0.45, 0.40, 0.15),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    labels_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"time_s": 3.0, "label": "cut", "flags": ["topic_shift"]},
                {"time_s": 6.0, "label": "continue", "flags": ["short_pause"]},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze(
        aligned_segments_path=aligned_path,
        omni_labels_path=labels_path,
    )

    assert summary["policies"]["current_threshold"]["accepted_additional_cut_count"] == 0
    assert summary["policies"]["gate_cut"]["accepted_additional_cut_count"] == 1
    assert summary["policies"]["probability_argmax"]["accepted_additional_cut_count"] == 2
    eager = summary["policies"]["probability_argmax"]
    assert eager["duration"]["p95_s"] == pytest.approx(4.0)
    assert eager["omni_truth"]["cut_precision"] == pytest.approx(0.5)
    assert eager["omni_truth"]["same_sentence_false_cut_rate"] == pytest.approx(1.0)
    assert summary["gate"]["v2_runtime_ab_allowed"] is False
