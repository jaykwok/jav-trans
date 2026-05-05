import numpy as np

from audio.f0_gender import detect_gender_f0_word_level


def _fake_load(duration_s=4.0, sr=16000):
    y = np.ones(int(sr * duration_s), dtype=np.float32) * 0.1
    return lambda path, sr, mono: (y, 16000)


def test_word_level_two_words_split_gender():
    segment = {
        "start": 0.0,
        "end": 4.0,
        "text": "テストです",
        "words": [
            {"start": 0.0, "end": 2.0, "word": "テスト"},
            {"start": 2.0, "end": 4.0, "word": "です"},
        ],
    }

    def pyin_fn(clip, fmin, fmax, sr, frame_length=None, hop_length=None):
        f0 = np.concatenate(
            [
                np.full(20, 120.0, dtype=np.float32),
                np.full(20, 220.0, dtype=np.float32),
            ]
        )
        return f0, np.ones_like(f0, dtype=bool), np.ones_like(f0)

    result = detect_gender_f0_word_level(
        "x.wav",
        [segment],
        _load_fn=_fake_load(),
        _pyin_fn=pyin_fn,
        median_filter_frames=1,
    )

    assert result[0]["words"][0]["gender"] == "M"
    assert result[0]["words"][1]["gender"] == "F"
    assert result[0]["gender"] is not None


def test_segment_silence_returns_none_gender():
    segment = {
        "start": 0.0,
        "end": 4.0,
        "text": "テスト",
        "words": [{"start": 0.0, "end": 4.0, "word": "テスト"}],
    }

    def pyin_fn(clip, fmin, fmax, sr, frame_length=None, hop_length=None):
        f0 = np.full(40, 120.0, dtype=np.float32)
        return f0, np.zeros_like(f0, dtype=bool), np.ones_like(f0)

    result = detect_gender_f0_word_level(
        "x.wav",
        [segment],
        _load_fn=_fake_load(),
        _pyin_fn=pyin_fn,
    )

    assert result[0]["gender"] is None


def test_fallback_no_words_uses_text_interpolation():
    segment = {"start": 0.0, "end": 4.0, "text": "テスト"}

    def pyin_fn(clip, fmin, fmax, sr, frame_length=None, hop_length=None):
        f0 = np.full(40, 120.0, dtype=np.float32)
        return f0, np.ones_like(f0, dtype=bool), np.ones_like(f0)

    result = detect_gender_f0_word_level(
        "x.wav",
        [segment],
        _load_fn=_fake_load(),
        _pyin_fn=pyin_fn,
    )

    assert result[0]["gender"] == "M"
