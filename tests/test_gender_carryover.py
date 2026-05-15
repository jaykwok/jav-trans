from __future__ import annotations

from audio.f0_gender import _apply_gender_carry_over


def _seg(start: float, end: float, gender=None, *, word_gender="keep"):
    words = [{"start": start, "end": end, "word": "x", "gender": word_gender}]
    return {"start": start, "end": end, "text": "x", "gender": gender, "words": words}


def _carry(segments, *, enabled=True, max_gap_s=10.0, max_segment_s=8.0):
    return _apply_gender_carry_over(
        segments,
        enabled=enabled,
        max_gap_s=max_gap_s,
        max_segment_s=max_segment_s,
    )


def test_none_segment_between_same_f_gender_inherits_f():
    result = _carry([
        _seg(0.0, 1.0, "F"),
        _seg(2.0, 3.0, None),
        _seg(4.0, 5.0, "F"),
    ])

    assert result[1]["gender"] == "F"
    assert result[1]["words"][0]["gender"] == "keep"


def test_none_segment_between_different_genders_stays_none():
    result = _carry([
        _seg(0.0, 1.0, "F"),
        _seg(2.0, 3.0, None),
        _seg(4.0, 5.0, "M"),
    ])

    assert result[1]["gender"] is None


def test_none_segment_over_max_gap_stays_none():
    result = _carry(
        [
            _seg(0.0, 1.0, "F"),
            _seg(20.0, 21.0, None),
            _seg(22.0, 23.0, "F"),
        ],
        max_gap_s=10.0,
    )

    assert result[1]["gender"] is None


def test_none_segment_over_max_segment_duration_stays_none():
    result = _carry(
        [
            _seg(0.0, 1.0, "F"),
            _seg(2.0, 12.0, None),
            _seg(13.0, 14.0, "F"),
        ],
        max_segment_s=8.0,
    )

    assert result[1]["gender"] is None


def test_disabled_carryover_stays_none():
    result = _carry(
        [
            _seg(0.0, 1.0, "F"),
            _seg(2.0, 3.0, None),
            _seg(4.0, 5.0, "F"),
        ],
        enabled=False,
    )

    assert result[1]["gender"] is None


def test_consecutive_none_segments_use_original_anchor_snapshot():
    result = _carry([
        _seg(0.0, 1.0, "F"),
        _seg(2.0, 3.0, None),
        _seg(3.2, 4.0, None),
        _seg(4.2, 5.0, None),
        _seg(6.0, 7.0, "F"),
    ])

    assert [segment["gender"] for segment in result] == ["F", "F", "F", "F", "F"]
