from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.boundary.ja.label_multi_safe_event_repairs_with_omni import (
    build_prompt,
    select_multi_safe_run_representatives,
    validate_response,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(candidate_id: str, probability: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "time_s": float(candidate_id.removeprefix("c")),
        "proposer_probability": probability,
        "tick_audio": f"{candidate_id}.wav",
    }


def test_selects_highest_learned_candidate_from_every_separated_safe_run(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    timeline = tmp_path / "timeline.jsonl"
    candidates_a = [
        _candidate("c00", 0.10),
        _candidate("c01", 0.80),
        _candidate("c02", 0.20),
        _candidate("c03", 0.30),
        _candidate("c04", 0.90),
        _candidate("c05", 0.10),
        _candidate("c06", 0.60),
    ]
    candidates_b = [
        _candidate("c10", 0.20),
        _candidate("c11", 0.70),
        _candidate("c12", 0.10),
        _candidate("c13", 0.50),
    ]
    _write(
        timeline,
        [
            {
                "sample_id": "arbitrary-a",
                "reference_text": "甲。乙。",
                "text_units": [
                    {"unit_id": "u00", "text": "甲。"},
                    {"unit_id": "u01", "text": "乙。"},
                ],
            },
            {
                "sample_id": "arbitrary-b",
                "reference_text": "丙。丁。",
                "text_units": [
                    {"unit_id": "u00", "text": "丙。"},
                    {"unit_id": "u01", "text": "丁。"},
                ],
            },
        ],
    )
    _write(
        events,
        [
            {
                "event_key": "arbitrary-a__e00",
                "sample_id": "arbitrary-a",
                "reference_text": "甲。乙。",
                "left_unit_id": "u00",
                "right_unit_id": "u01",
                "left_text": "甲。",
                "right_text": "乙。",
                "candidates": candidates_a,
            },
            {
                "event_key": "arbitrary-b__e00",
                "sample_id": "arbitrary-b",
                "reference_text": "丙。丁。",
                "left_unit_id": "u00",
                "right_unit_id": "u01",
                "left_text": "丙。",
                "right_text": "丁。",
                "candidates": candidates_b,
            },
        ],
    )
    _write(
        verdicts,
        [
            {
                "event_key": "arbitrary-a__e00",
                "candidates": [
                    {"candidate_id": "c00", "label": "safe"},
                    {"candidate_id": "c01", "label": "safe"},
                    {"candidate_id": "c02", "label": "left_clipped"},
                    {"candidate_id": "c03", "label": "safe"},
                    {"candidate_id": "c04", "label": "safe"},
                    {"candidate_id": "c05", "label": "right_clipped"},
                    {"candidate_id": "c06", "label": "safe"},
                ],
            },
            {
                "event_key": "arbitrary-b__e00",
                "candidates": [
                    {"candidate_id": "c10", "label": "safe"},
                    {"candidate_id": "c11", "label": "safe"},
                    {"candidate_id": "c12", "label": "unsure"},
                    {"candidate_id": "c13", "label": "safe"},
                ],
            },
        ],
    )

    selected = select_multi_safe_run_representatives(
        events=events, verdicts=verdicts, timeline_labels=timeline
    )

    assert [row["candidate_id"] for row in selected] == [
        "c01",
        "c04",
        "c06",
        "c11",
        "c13",
    ]
    assert all(
        row["selection_contract"]
        == "highest_learned_proposer_probability_per_separated_safe_run_v1"
        for row in selected
    )
    assert json.loads(build_prompt(selected[0]))["reference_text"] == "甲。乙。"
    assert "manual" not in build_prompt(selected[0]).lower()


def test_semantic_response_requires_unique_exact_reference_boundary() -> None:
    selection = {
        "candidate_id": "c00",
        "reference_text": "同じ。別。同じ。別。",
    }
    with pytest.raises(ValueError, match="exactly once"):
        validate_response(
            {
                "candidate_id": "c00",
                "decision": "semantic_split",
                "left_text": "同じ。",
                "right_text": "別。",
                "confidence": 0.9,
            },
            selection,
        )


def test_selection_rejects_event_text_drift_from_independent_timeline(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    timeline = tmp_path / "timeline.jsonl"
    candidates = [_candidate("c00", 0.1), _candidate("c01", 0.2)]
    _write(
        timeline,
        [
            {
                "sample_id": "s",
                "reference_text": "甲。乙。",
                "text_units": [
                    {"unit_id": "u00", "text": "甲。"},
                    {"unit_id": "u01", "text": "乙。"},
                ],
            }
        ],
    )
    _write(
        events,
        [
            {
                "event_key": "s__e00",
                "sample_id": "s",
                "reference_text": "甲。乙。",
                "left_unit_id": "u00",
                "right_unit_id": "u01",
                "left_text": "硬编码特例",
                "right_text": "乙。",
                "candidates": candidates,
            }
        ],
    )
    _write(
        verdicts,
        [
            {
                "event_key": "s__e00",
                "candidates": [
                    {"candidate_id": "c00", "label": "safe"},
                    {"candidate_id": "c01", "label": "left_clipped"},
                    {"candidate_id": "c00", "label": "safe"},
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="does not match timeline"):
        select_multi_safe_run_representatives(
            events=events, verdicts=verdicts, timeline_labels=timeline
        )
