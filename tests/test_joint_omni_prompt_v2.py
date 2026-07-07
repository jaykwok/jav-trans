from __future__ import annotations

from tools.datasets.label_joint_boundary_preasr_with_omni import (
    PROMPT_VERSION,
    _build_prompt,
    _normalize_missed_boundaries,
)


def test_prompt_v2_requests_missed_boundaries() -> None:
    prompt = _build_prompt([], [], duration_s=75.0)
    assert PROMPT_VERSION == "joint_boundary_preasr_omni_v2"
    assert "missed_boundaries" in prompt
    assert "任务 A2" in prompt


def test_normalize_missed_boundaries_filters_and_sorts() -> None:
    rows = _normalize_missed_boundaries(
        [
            {"time_s": 10.5, "confidence": 0.9},
            {"time_s": 3.2, "confidence": 0.95, "flags": ["speaker_change"]},
            {"time_s": 3.2001, "confidence": 0.9},  # millisecond duplicate
            {"time_s": 90.0, "confidence": 0.99},  # outside window
            {"time_s": 5.0, "confidence": 0.3},  # low confidence
            {"time_s": "junk", "confidence": 0.9},
            "not-a-mapping",
        ],
        duration_s=75.0,
        confidence_threshold=0.8,
    )

    assert [row["time_s"] for row in rows] == [3.2, 10.5]
    assert rows[0]["flags"] == ["speaker_change"]


def test_normalize_missed_boundaries_handles_non_list() -> None:
    assert _normalize_missed_boundaries(
        None, duration_s=75.0, confidence_threshold=0.8
    ) == []
