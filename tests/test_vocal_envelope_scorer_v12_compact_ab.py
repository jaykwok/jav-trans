from __future__ import annotations

import pytest

from tools.boundary.ja.label_vocal_envelope_scorer_v12_compact_ab import (
    COMPACT_SYSTEM_PROMPT,
    choose_adaptive_commit_frame,
    normalize_compact_response,
    parse_args,
)
from tools.boundary.ja.vocal_envelope_scorer_v12_teacher_contract import (
    SCORER_V12_TIME_GRID_CONTRACT_ID,
)


def test_partition_prompt_preserves_broad_vocal_and_complete_coverage() -> None:
    for text in (
        "吸气",
        "呼气",
        "亲吻/唾液声",
        "肉体撞击",
        "第一个区间必须从 0.00 开始",
        "15 秒",
        "0.01 秒",
        "0.02 秒帧",
    ):
        assert text in COMPACT_SYSTEM_PROMPT
    assert "严格 JSON 数组" in COMPACT_SYSTEM_PROMPT


def test_partition_response_compiles_complete_wire_partition_to_frames() -> None:
    normalized = normalize_compact_response(
        [
            {"s": 0.00, "e": 0.60, "t": "non_vocal"},
            {"s": 0.60, "e": 1.20, "t": "vocal"},
            {"s": 1.20, "e": 2.00, "t": "unsure"},
        ],
        duration_s=2.0,
        frame_count=100,
    )
    assert normalized["vocal_spans"] == [
        {"label": "vocal_candidate", "start_frame": 30, "end_frame": 60, "start_s": 0.6, "end_s": 1.2}
    ]
    assert normalized["unsure_spans"] == [
        {"label": "unsure", "start_frame": 60, "end_frame": 100, "start_s": 1.2, "end_s": 2.0}
    ]
    assert normalized["non_vocal_spans"][0]["start_frame"] == 0
    assert normalized["non_vocal_spans"][-1]["end_frame"] == 30
    assert normalized["time_grid_contract_id"] == SCORER_V12_TIME_GRID_CONTRACT_ID
    assert normalized["teacher_timestamp_step_s"] == 0.01
    assert normalized["scorer_frame_hop_s"] == 0.02
    assert normalized["quantized_boundary_frames"] == [0, 30, 60, 100]


def test_partition_uses_one_vocal_safe_boundary_on_half_frames() -> None:
    normalized = normalize_compact_response(
        [
            {"s": 0.00, "e": 0.61, "t": "non_vocal"},
            {"s": 0.61, "e": 1.21, "t": "vocal"},
            {"s": 1.21, "e": 2.00, "t": "non_vocal"},
        ],
        duration_s=2.0,
        frame_count=100,
    )
    # Vocal starts round down and vocal ends round up.  Both adjacent labels use
    # the same shared cut, so frame coverage remains contiguous and exact.
    assert normalized["quantized_boundary_frames"] == [0, 30, 61, 100]
    assert normalized["vocal_spans"] == [
        {
            "label": "vocal_candidate",
            "start_frame": 30,
            "end_frame": 61,
            "start_s": 0.6,
            "end_s": 1.22,
        }
    ]
    assert sum(
        span["end_frame"] - span["start_frame"]
        for key in ("vocal_spans", "non_vocal_spans", "unsure_spans")
        for span in normalized[key]
    ) == 100


def test_adaptive_commit_prefers_definite_nonvocal_seam_near_target() -> None:
    normalized = normalize_compact_response(
        [
            {"s": 0.00, "e": 14.00, "t": "vocal"},
            {"s": 14.00, "e": 16.00, "t": "non_vocal"},
            {"s": 16.00, "e": 20.00, "t": "vocal"},
        ],
        duration_s=20.0,
        frame_count=1000,
    )
    assert choose_adaptive_commit_frame(
        normalized, window_frame_count=1000
    ) == (750, "definite_nonvocal_seam")


def test_adaptive_commit_falls_back_without_safe_nonvocal_seam() -> None:
    normalized = normalize_compact_response(
        [{"s": 0.00, "e": 20.00, "t": "vocal"}],
        duration_s=20.0,
        frame_count=1000,
    )
    assert choose_adaptive_commit_frame(
        normalized, window_frame_count=1000
    ) == (750, "target_fallback")


def test_adaptive_teacher_defaults_to_serial_source_chains() -> None:
    args = parse_args(
        [
            "--manifest",
            "manifest.jsonl",
            "--selection-manifest",
            "selection.jsonl",
            "--output-dir",
            "out",
        ]
    )
    assert args.workers == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"spans": []}, "root must be a JSON array"),
        ([{"s": 0.001, "e": 2.00, "t": "vocal"}], "10ms time grid"),
        ([{"s": 0.00, "e": 0.30, "t": "vocal"}], "advertised duration"),
        ([{"s": 0.00, "e": 0.50, "t": "vocal"}, {"s": 0.60, "e": 2.00, "t": "non_vocal"}], "contiguous without gaps"),
        ([{"s": 0.00, "e": 2.00, "t": "vocal", "extra": 1}], "missing or extra"),
    ),
)
def test_compact_response_rejects_contract_drift(payload, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_compact_response(payload, duration_s=2.0, frame_count=100)


def test_partition_rejects_a_wire_span_that_collapses_on_the_frame_grid() -> None:
    with pytest.raises(ValueError, match="collapses below the 20ms"):
        normalize_compact_response(
            [
                {"s": 0.00, "e": 0.01, "t": "non_vocal"},
                {"s": 0.01, "e": 2.00, "t": "vocal"},
            ],
            duration_s=2.0,
            frame_count=100,
        )
