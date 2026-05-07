import main


def _seg(text: str, gender):
    return {"start": 0.0, "end": 1.0, "text": text, "gender": gender}


def _word(start: float, end: float, text: str, gender):
    return {"start": start, "end": end, "word": text, "gender": gender}


def test_f0_filter_off_keeps_all_segments(monkeypatch):
    monkeypatch.delenv("F0_FILTER_NONE_SEGMENTS", raising=False)
    segments = [_seg("a", None), _seg("b", "F")]

    filtered, count = main._filter_f0_none_segments(segments)

    assert filtered == segments
    assert count == 0


def test_f0_filter_on_removes_none_gender_segments(monkeypatch):
    monkeypatch.setenv("F0_FILTER_NONE_SEGMENTS", "1")
    segments = [_seg("a", None), _seg("b", "F"), _seg("c", "M")]

    filtered, count = main._filter_f0_none_segments(segments)

    assert filtered == [_seg("b", "F"), _seg("c", "M")]
    assert count == 1


def test_f0_filter_on_all_none_returns_empty(monkeypatch):
    monkeypatch.setenv("F0_FILTER_NONE_SEGMENTS", "1")
    segments = [_seg("a", None), _seg("b", None)]

    filtered, count = main._filter_f0_none_segments(segments)

    assert filtered == []
    assert count == 2


def test_f0_filter_skips_when_f0_detection_failed(monkeypatch):
    monkeypatch.setenv("F0_FILTER_NONE_SEGMENTS", "1")
    segments = [_seg("a", None), _seg("b", None)]

    filtered, count = main._filter_f0_none_segments(segments, f0_failed=True)

    assert filtered == segments
    assert count == 0


def test_f0_gender_turn_split_aggressive_m_to_f():
    segment = {
        "start": 0.0,
        "end": 4.0,
        "text": "男声女声",
        "source_chunk_index": 7,
        "words": [
            _word(0.0, 1.0, "男", "M"),
            _word(1.0, 2.0, "声", "M"),
            _word(2.0, 3.0, "女", "F"),
            _word(3.0, 4.0, "声", "F"),
        ],
    }

    split, count = main._split_segments_on_f0_gender_turns([segment])

    assert count == 1
    assert [item["text"] for item in split] == ["男声", "女声"]
    assert [item["gender"] for item in split] == ["M", "F"]
    assert [(item["start"], item["end"]) for item in split] == [(0.0, 2.0), (2.0, 4.0)]
    assert all(item["source_chunk_index"] == 7 for item in split)


def test_f0_gender_turn_split_none_word_joins_previous_group():
    segment = {
        "start": 0.0,
        "end": 3.0,
        "text": "男无女",
        "words": [
            _word(0.0, 1.0, "男", "M"),
            _word(1.0, 2.0, "無", None),
            _word(2.0, 3.0, "女", "F"),
        ],
    }

    split, count = main._split_segments_on_f0_gender_turns([segment])
    filtered, filtered_count = main._filter_f0_none_segments(split, enabled=True)

    assert count == 1
    assert [item["text"] for item in split] == ["男無", "女"]
    assert [item["gender"] for item in split] == ["M", "F"]
    assert filtered == split
    assert filtered_count == 0


def test_f0_gender_turn_split_keeps_all_none_or_missing_words():
    all_none = {
        "start": 0.0,
        "end": 2.0,
        "text": "不明",
        "gender": None,
        "words": [
            _word(0.0, 1.0, "不", None),
            _word(1.0, 2.0, "明", None),
        ],
    }
    no_words = {"start": 2.0, "end": 3.0, "text": "なし", "gender": "M"}

    split, count = main._split_segments_on_f0_gender_turns([all_none, no_words])

    assert split == [all_none, no_words]
    assert count == 0


def test_f0_gender_turn_split_same_gender_stays_one_segment():
    segment = {
        "start": 0.0,
        "end": 2.0,
        "text": "同じ",
        "gender": "F",
        "words": [
            _word(0.0, 1.0, "同", "F"),
            _word(1.0, 2.0, "じ", "F"),
        ],
    }

    split, count = main._split_segments_on_f0_gender_turns([segment])

    assert split == [segment]
    assert count == 0

