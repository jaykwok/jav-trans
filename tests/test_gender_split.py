from whisper.pipeline import _should_split_on_gender


def test_same_gender_no_split():
    assert _should_split_on_gender({"gender": "M"}, {"gender": "M"}, 0.5) is False


def test_none_gender_no_split():
    assert _should_split_on_gender({"gender": None}, {"gender": "F"}, 0.5) is False
    assert _should_split_on_gender({"gender": "M"}, {"gender": None}, 0.5) is False


def test_gap_too_small_no_split():
    assert _should_split_on_gender({"gender": "M"}, {"gender": "F"}, 0.1) is False


def test_gender_switch_with_sufficient_gap_splits():
    assert _should_split_on_gender({"gender": "M"}, {"gender": "F"}, 0.2) is True
    assert _should_split_on_gender({"gender": "F"}, {"gender": "M"}, 0.2) is True
