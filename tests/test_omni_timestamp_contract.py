from __future__ import annotations

import json

import pytest

from tools.boundary.ja import (
    label_acoustic_split_canonical_candidates_with_omni as split_teacher,
)
from tools.boundary.ja import (
    label_candidate_island_scorer_v11_dual_evidence_with_omni as dual_teacher,
)
from tools.boundary.ja import (
    label_candidate_island_scorer_v11_with_omni as scorer_teacher,
)
from tools.boundary.ja import (
    label_semantic_source_text_alignment_with_omni as source_alignment_teacher,
)
from tools.boundary.ja import (
    label_semantic_timeline_with_omni as semantic_timeline_teacher,
)
from tools.datasets import label_joint_boundary_preasr_with_omni as joint_teacher
from tools.datasets import label_timeline_with_omni as timeline_teacher
from tools.omni.timestamp_contract import (
    TIMESTAMP_CONTRACT_ID,
    TIMESTAMP_FORMAT,
    format_duration_timestamp,
    format_mmss_timestamp,
    parse_mmss_span,
    parse_mmss_timestamp,
    timestamp_request_contract,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.0, "00:00.000"),
        (59.5, "00:59.500"),
        (60.0, "01:00.000"),
        (65.153, "01:05.153"),
        (3605.153, "60:05.153"),
    ),
)
def test_format_mmss_timestamp(value: float, expected: str) -> None:
    assert format_mmss_timestamp(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("00:59.500", 59.5),
        ("01:00.000", 60.0),
        ("01:05.153", 65.153),
    ),
)
def test_parse_mmss_timestamp_crosses_minute_boundary(
    value: str,
    expected: float,
) -> None:
    assert parse_mmss_timestamp(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        105.153,
        "105.153",
        "1:05.153",
        "01:5.153",
        "01:60.000",
        "01:05.15",
        "01:05.1530",
        " 01:05.153",
    ),
)
def test_parse_mmss_timestamp_rejects_ambiguous_or_malformed_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="MM:SS.mmm"):
        parse_mmss_timestamp(value)


def test_timestamp_parser_rejects_source_overrun() -> None:
    assert parse_mmss_timestamp("01:05.153", duration_s=65.153) == 65.153
    with pytest.raises(ValueError, match="exceeds source duration"):
        parse_mmss_timestamp("01:05.154", duration_s=65.153)


def test_span_parser_requires_new_wire_keys_and_order() -> None:
    assert parse_mmss_span(
        {"start_ts": "00:59.500", "end_ts": "01:05.153"},
        field="span",
        duration_s=65.153,
    ) == (59.5, 65.153)
    with pytest.raises(ValueError, match="start_ts/end_ts"):
        parse_mmss_span(
            {"start_s": 59.5, "end_s": 65.153},
            field="span",
            duration_s=65.153,
        )
    with pytest.raises(ValueError, match="start < end"):
        parse_mmss_span(
            {"start_ts": "00:10.000", "end_ts": "00:09.999"},
            field="span",
            duration_s=20.0,
        )


def test_nullable_span_requires_both_coordinates_null() -> None:
    assert parse_mmss_span(
        {"start_ts": None, "end_ts": None},
        field="alignment",
        duration_s=1.0,
        allow_null=True,
    ) == (None, None)
    with pytest.raises(ValueError, match="two null"):
        parse_mmss_span(
            {"start_ts": None, "end_ts": "00:00.500"},
            field="alignment",
            duration_s=1.0,
            allow_null=True,
        )


def test_duration_contract_never_rounds_past_audio() -> None:
    assert format_duration_timestamp(65.1539) == "01:05.153"
    assert timestamp_request_contract(65.1539) == {
        "duration_ts": "01:05.153",
        "timestamp_format": TIMESTAMP_FORMAT,
        "timestamp_contract_id": TIMESTAMP_CONTRACT_ID,
        "coordinate_system": "0-based current uploaded-audio timeline",
    }


def test_all_current_interval_teacher_prompts_share_timestamp_contract() -> None:
    prompts = (
        scorer_teacher.SYSTEM_PROMPT,
        scorer_teacher.SAFE_OUTSIDE_SYSTEM_PROMPT,
        scorer_teacher.SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT,
        scorer_teacher.GREENLIGHT_SAFE_OUTSIDE_SYSTEM_PROMPT,
        scorer_teacher.FUNNEL_SAFE_OUTSIDE_SYSTEM_PROMPT,
        scorer_teacher.ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT,
        scorer_teacher.BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT,
        dual_teacher.PROTECT_SYSTEM_PROMPT,
        dual_teacher.REMOVE_SYSTEM_PROMPT,
        source_alignment_teacher.SYSTEM_PROMPT,
        semantic_timeline_teacher.SYSTEM_PROMPT,
        timeline_teacher.SYSTEM_PROMPT,
        split_teacher.SYSTEM_PROMPT,
    )
    for prompt in prompts:
        assert "MM:SS.mmm" in prompt
        assert '"start_s":' not in prompt
        assert '"end_s":' not in prompt


def test_current_teacher_requests_use_timestamp_strings_on_the_wire() -> None:
    requests = (
        scorer_teacher._prompt({"source_id": "s", "duration_s": 65.153}),
        dual_teacher._request_prompt(
            {"source_id": "s", "duration_s": 65.153},
            pass_name="protect",
        ),
        source_alignment_teacher.build_prompt(
            {
                "sample_id": "s",
                "duration_s": 65.153,
                "reference_text": "待って",
            }
        ),
        semantic_timeline_teacher.build_prompt(
            {
                "sample_id": "s",
                "duration_s": 65.153,
                "reference_text": "待って",
            }
        ),
        timeline_teacher.build_prompt(
            {
                "duration_s": 65.153,
                "text_units": [{"unit_id": "u0000", "text": "待って"}],
            }
        ),
        split_teacher.build_prompt(
            {"feature_index": 1, "time_s": 65.0},
            clip_start=60.0,
            clip_end=65.153,
        ),
    )
    for serialized in requests:
        payload = json.loads(serialized)
        assert payload["duration_ts"].count(":") == 1
        assert payload["timestamp_contract_id"] == TIMESTAMP_CONTRACT_ID
        assert "duration_s" not in payload

    split_prompt = joint_teacher._build_split_prompt(
        [{"time_s": 65.0, "label": "continue", "p_cut": 0.5}],
        duration_s=65.153,
    )
    cueqc_prompt = joint_teacher._build_pre_asr_prompt({"duration_s": 65.153})
    assert '"time_ts":"01:05.000"' in split_prompt
    assert '"duration_ts":"01:05.153"' in split_prompt
    assert '"time_s"' not in split_prompt
    assert '"duration_s"' not in split_prompt
    assert '"duration_ts":"01:05.153"' in cueqc_prompt
    assert '"duration_s"' not in cueqc_prompt
