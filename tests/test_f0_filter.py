import main
def _seg(text: str, gender):
    return {"start": 0.0, "end": 1.0, "text": text, "gender": gender}


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

