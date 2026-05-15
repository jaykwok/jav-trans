from __future__ import annotations

from audio.f0_gender import _apply_gender_carry_over
from pipeline.gender_split import split_segments_on_f0_gender_turns


def _word(index: int, gender: str | None) -> dict:
    return {
        "start": float(index),
        "end": float(index + 1),
        "word": f"w{index}",
        "gender": gender,
    }


def _segment(genders: list[str | None]) -> dict:
    words = [_word(index, gender) for index, gender in enumerate(genders)]
    known_genders = [gender for gender in genders if gender is not None]
    segment_gender = (
        max(("M", "F"), key=known_genders.count) if known_genders else None
    )
    return {
        "start": 0.0,
        "end": float(len(words)),
        "text": "".join(word["word"] for word in words),
        "gender": segment_gender,
        "words": words,
    }


def _split_genders(genders: list[str | None], monkeypatch) -> list[dict]:
    monkeypatch.setenv("F0_GENDER_NONE_TOLERANCE", "2")
    monkeypatch.setenv("SUBTITLE_MIN_DURATION_GENDER_TURN", "0.4")
    split, _count = split_segments_on_f0_gender_turns([_segment(genders)])
    return split


def _split_genders_tol3(genders: list[str | None], monkeypatch) -> list[dict]:
    monkeypatch.setenv("F0_GENDER_NONE_TOLERANCE", "3")
    monkeypatch.setenv("SUBTITLE_MIN_DURATION_GENDER_TURN", "0.4")
    split, _count = split_segments_on_f0_gender_turns([_segment(genders)])
    return split


def test_all_none_words_stay_one_group(monkeypatch):
    split = _split_genders([None, None, None, None, None], monkeypatch)

    assert len(split) == 1
    assert split[0]["gender"] is None


def test_single_none_between_same_gender_stays_in_gender_group(monkeypatch):
    split = _split_genders(["M", None, "M"], monkeypatch)

    assert len(split) == 1
    assert split[0]["gender"] == "M"
    assert [word["gender"] for word in split[0]["words"]] == ["M", None, "M"]


def test_three_nones_between_gender_turn_form_none_group(monkeypatch):
    split = _split_genders(["M", None, None, None, "F"], monkeypatch)

    assert [item["gender"] for item in split] == ["M", None, "F"]
    assert [[word["gender"] for word in item["words"]] for item in split] == [
        ["M"],
        [None, None, None],
        ["F"],
    ]


def test_two_nones_at_tolerance_form_none_group(monkeypatch):
    split = _split_genders(["M", None, None, "F"], monkeypatch)

    assert [item["gender"] for item in split] == ["M", None, "F"]
    assert [[word["gender"] for word in item["words"]] for item in split] == [
        ["M"],
        [None, None],
        ["F"],
    ]


def test_two_nones_below_tol3_do_not_split(monkeypatch):
    split = _split_genders_tol3(["M", None, None, "F"], monkeypatch)

    assert len(split) == 1
    assert [word["gender"] for word in split[0]["words"]] == ["M", None, None, "F"]


def test_two_nones_below_tol3_continued_f_stays_one_group(monkeypatch):
    """After absorbing 2 Nones at M→F, subsequent F words stay in the same group."""
    split = _split_genders_tol3(["M", None, None, "F", "F"], monkeypatch)

    assert len(split) == 1
    assert [word["gender"] for word in split[0]["words"]] == ["M", None, None, "F", "F"]


def test_three_nones_at_tol3_form_none_group(monkeypatch):
    split = _split_genders_tol3(["M", None, None, None, "F"], monkeypatch)

    assert [item["gender"] for item in split] == ["M", None, "F"]
    assert [[word["gender"] for word in item["words"]] for item in split] == [
        ["M"],
        [None, None, None],
        ["F"],
    ]


def test_post_split_carry_over_rescues_none_fragment():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "f1", "gender": "F"},
        {"start": 1.0, "end": 1.5, "text": "x", "gender": None},
        {"start": 1.5, "end": 2.5, "text": "f2", "gender": "F"},
    ]

    result = _apply_gender_carry_over(
        segments,
        enabled=True,
        max_gap_s=15.0,
        max_segment_s=12.0,
    )

    assert sum(1 for segment in result if segment.get("gender") is None) == 0
    assert [segment["gender"] for segment in result] == ["F", "F", "F"]


def test_long_gender_turn_with_single_none_splits_around_none(monkeypatch):
    split = _split_genders(["M"] * 5 + [None] + ["F"] * 5, monkeypatch)

    assert [item["gender"] for item in split] == ["M", None, "F"]
    assert [[word["gender"] for word in item["words"]] for item in split] == [
        ["M", "M", "M", "M", "M"],
        [None],
        ["F", "F", "F", "F", "F"],
    ]


def test_gender_turn_uses_dedicated_duration_floor(monkeypatch):
    monkeypatch.setenv("F0_GENDER_NONE_TOLERANCE", "2")
    monkeypatch.setenv("SUBTITLE_MIN_DURATION_GENDER_TURN", "0.4")
    split, count = split_segments_on_f0_gender_turns(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "mf",
                "gender": "M",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "m", "gender": "M"},
                    {"start": 0.4, "end": 0.8, "word": "f", "gender": "F"},
                ],
            }
        ]
    )

    assert count == 1
    assert [(item["start"], item["end"]) for item in split] == [(0.0, 0.4), (0.4, 0.8)]
