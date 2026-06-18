from subtitles import display_policy


def _cue(index: int, text: str, start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "ja_text": text,
        "zh_text": text,
        "source_segment_ids": [index],
    }


def _segment(text: str, start: float, end: float, **extra) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        **extra,
    }


def test_isolated_short_vocalization_is_kept():
    cues = [_cue(0, "あ", 0.0, 0.4)]
    segments = [_segment("あ", 0.0, 0.4, alignment_quality="boundary")]

    displayed, summary = display_policy.apply_display_policy(
        cues,
        source_segments=segments,
    )

    assert len(displayed) == 1
    assert displayed[0]["display_decision"] == "keep"
    assert summary["counts"]["keep"] == 1
    assert summary["decisions"][0]["display_decision"] == "keep"


def test_repeated_low_information_run_is_compacted():
    cues = [
        _cue(0, "あ", 0.0, 0.4),
        _cue(1, "あ", 0.5, 0.9),
        _cue(2, "あ", 1.0, 1.4),
    ]
    segments = [
        _segment("あ", 0.0, 0.4, fallback_subtype="nonlexical_text"),
        _segment("あ", 0.5, 0.9, fallback_subtype="nonlexical_text"),
        _segment("あ", 1.0, 1.4, fallback_subtype="nonlexical_text"),
    ]

    displayed, summary = display_policy.apply_display_policy(
        cues,
        source_segments=segments,
    )

    assert len(displayed) == 1
    assert displayed[0]["display_decision"] == "compact"
    assert displayed[0]["source_segment_ids"] == [0, 1, 2]
    assert displayed[0]["raw_texts"] == ["あ", "あ", "あ"]
    assert summary["cues_before"] == 3
    assert summary["cues_after"] == 1
    assert summary["counts"]["compact"] == 1
    assert summary["counts"]["drop"] == 2
    assert summary["decisions"][1]["compacted_into"] == 0


def test_stable_dialogue_is_kept_even_when_short_runs_are_nearby():
    cues = [
        _cue(0, "あ", 0.0, 0.4),
        _cue(1, "今日はいい天気ですね", 0.5, 2.4),
        _cue(2, "あ", 2.5, 2.9),
    ]
    segments = [
        _segment("あ", 0.0, 0.4, fallback_subtype="nonlexical_text"),
        _segment("今日はいい天気ですね", 0.5, 2.4, alignment_quality="boundary"),
        _segment("あ", 2.5, 2.9, fallback_subtype="nonlexical_text"),
    ]

    displayed, summary = display_policy.apply_display_policy(
        cues,
        source_segments=segments,
    )

    assert [item["display_decision"] for item in displayed] == [
        "review",
        "keep",
        "review",
    ]
    assert displayed[1]["ja_text"] == "今日はいい天気ですね"
    assert summary["counts"]["keep"] == 1
    assert summary["counts"]["review"] == 2
