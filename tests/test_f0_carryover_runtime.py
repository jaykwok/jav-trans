import numpy as np

from audio.f0_gender import detect_gender_f0_word_level


def _fake_load(_path, sr, mono):
    del sr, mono
    return np.ones(16000 * 3, dtype=np.float32), 16000


def _pyin_fn(clip, fmin, fmax, sr, frame_length=None, hop_length=None):
    del fmin, fmax, sr, frame_length, hop_length
    frame_count = max(1, int(round(len(clip) / 1600)))
    f0 = np.full(frame_count, 220.0, dtype=np.float32)
    voiced = np.ones(frame_count, dtype=bool)
    return f0, voiced, np.ones(frame_count, dtype=np.float32)


def _segments():
    return [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.1, "end": 1.105, "text": "b"},
        {"start": 2.0, "end": 3.0, "text": "c"},
    ]


def test_detect_gender_carryover_disabled_runtime(monkeypatch):
    monkeypatch.setenv("F0_GENDER_CARRYOVER_ENABLED", "0")

    result = detect_gender_f0_word_level(
        "x.wav",
        _segments(),
        _load_fn=_fake_load,
        _pyin_fn=_pyin_fn,
        median_filter_frames=1,
    )

    assert [segment.get("gender") for segment in result] == ["F", None, "F"]


def test_detect_gender_carryover_enabled_runtime(monkeypatch):
    monkeypatch.setenv("F0_GENDER_CARRYOVER_ENABLED", "1")
    monkeypatch.setenv("F0_GENDER_CARRYOVER_MAX_GAP_S", "15.0")
    monkeypatch.setenv("F0_GENDER_CARRYOVER_MAX_SEGMENT_S", "12.0")

    result = detect_gender_f0_word_level(
        "x.wav",
        _segments(),
        _load_fn=_fake_load,
        _pyin_fn=_pyin_fn,
        median_filter_frames=1,
    )

    assert [segment.get("gender") for segment in result] == ["F", "F", "F"]
