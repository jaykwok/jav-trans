from __future__ import annotations

import pytest

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
