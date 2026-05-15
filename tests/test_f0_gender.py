import importlib
import os

import numpy as np

from audio import f0_gender
from audio.f0_gender import detect_gender_f0


def _tone(frequency_hz, duration_s=5.0, sr=16000):
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    return (0.2 * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _silence(duration_s=5.0, sr=16000):
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _fake_load(y_data):
    return lambda path, sr, mono: (y_data, 16000)


def _fake_pyin(f0_val, voiced=True):
    def fn(clip, fmin, fmax, sr, **kwargs):
        n = 10
        f0 = np.full(n, f0_val)
        voiced_flag = np.ones(n, dtype=bool) if voiced else np.zeros(n, dtype=bool)
        return f0, voiced_flag, np.ones(n)

    return fn


def test_male_tone():
    result = detect_gender_f0(
        "x.wav",
        [{"start": 0.0, "end": 2.0, "text": "テスト", "words": [{"start": 0.0, "end": 2.0, "word": "テスト"}]}],
        _load_fn=_fake_load(_tone(120.0)),
        _pyin_fn=_fake_pyin(120.0),
    )
    assert result[0]["gender"] == "M"


def test_female_tone():
    result = detect_gender_f0(
        "x.wav",
        [{"start": 0.0, "end": 2.0, "text": "テスト", "words": [{"start": 0.0, "end": 2.0, "word": "テスト"}]}],
        _load_fn=_fake_load(_tone(220.0)),
        _pyin_fn=_fake_pyin(220.0),
    )
    assert result[0]["gender"] == "F"


def test_unvoiced_silence():
    result = detect_gender_f0(
        "x.wav",
        [{"start": 0.0, "end": 2.0, "text": "テスト", "words": [{"start": 0.0, "end": 2.0, "word": "テスト"}]}],
        _load_fn=_fake_load(_silence()),
        _pyin_fn=_fake_pyin(120.0, voiced=False),
    )
    assert result[0]["gender"] is None


def test_short_clip_returns_none():
    result = detect_gender_f0(
        "x.wav",
        [{"start": 0.0, "end": 0.005}],
        _load_fn=_fake_load(_tone(120.0)),
        _pyin_fn=_fake_pyin(120.0),
    )
    assert result[0]["gender"] is None


def test_clip_shorter_than_pyin_frame_is_skipped():
    def fail_pyin(*_args, **_kwargs):
        raise AssertionError("pyin should not be called for clips shorter than frame_length")

    result = detect_gender_f0(
        "x.wav",
        [{"start": 0.0, "end": 0.2, "text": "テスト"}],
        _load_fn=_fake_load(_tone(120.0, duration_s=0.2)),
        _pyin_fn=fail_pyin,
    )

    assert result[0]["gender"] is None


def test_empty_segments():
    result = detect_gender_f0("x.wav", [], _load_fn=_fake_load(_silence()))
    assert result == []


def test_env_threshold():
    previous_value = os.environ.get("F0_THRESHOLD_HZ")
    os.environ["F0_THRESHOLD_HZ"] = "200"
    module = importlib.reload(f0_gender)
    try:
        for f0_value, expected in [(120.0, "M"), (180.0, "M"), (220.0, "F")]:
            result = module.detect_gender_f0(
                "x.wav",
                [{"start": 0.0, "end": 2.0}],
                _load_fn=_fake_load(_tone(f0_value)),
                _pyin_fn=_fake_pyin(f0_value),
            )
            assert result[0]["gender"] == expected
    finally:
        if previous_value is None:
            os.environ.pop("F0_THRESHOLD_HZ", None)
        else:
            os.environ["F0_THRESHOLD_HZ"] = previous_value
        importlib.reload(f0_gender)


def test_audio_load_failure_returns_original():
    def bad_load(path, sr, mono):
        raise RuntimeError("no file")

    segs = [{"start": 0.0, "end": 2.0}]
    result = detect_gender_f0("x.wav", segs, _load_fn=bad_load)
    assert result == segs
    assert "gender" not in result[0]


def test_original_segments_not_mutated():
    segs = [{"start": 0.0, "end": 2.0}]
    detect_gender_f0(
        "x.wav",
        segs,
        _load_fn=_fake_load(_tone(120.0)),
        _pyin_fn=_fake_pyin(120.0),
    )
    assert "gender" not in segs[0]


def test_default_audio_loader_does_not_use_librosa_load(monkeypatch):
    def fail_if_librosa_is_used(*_args, **_kwargs):
        raise AssertionError("librosa.load should not be used")

    monkeypatch.setattr("librosa.load", fail_if_librosa_is_used, raising=False)
    monkeypatch.setattr(
        "soundfile.read",
        lambda *_args, **_kwargs: (_tone(120.0, duration_s=0.4), 16000),
    )

    result = detect_gender_f0(
        "sample.wav",
        [{"start": 0.0, "end": 0.4}],
        _pyin_fn=_fake_pyin(120.0),
    )

    assert result[0]["gender"] == "M"


# --- Bug fix: nan_ratio_threshold must be propagated through detect_gender_f0 ---

def _fake_pyin_mixed_nan(nan_ratio: float, f0_val: float = 120.0):
    """Return a pyin that produces the requested fraction of NaN frames."""
    def fn(clip, fmin, fmax, sr, **kwargs):
        n = 20
        n_nan = int(n * nan_ratio)
        f0 = np.full(n, f0_val, dtype=np.float32)
        f0[:n_nan] = np.nan
        voiced_flag = np.isfinite(f0)
        return f0, voiced_flag, np.ones(n)
    return fn


def test_nan_ratio_threshold_propagates_strict():
    """With threshold=0.50, nan_ratio=0.55 → None (stricter than default 0.6)."""
    segs = [{"start": 0.0, "end": 2.0}]
    result = detect_gender_f0(
        "x.wav",
        segs,
        nan_ratio_threshold=0.50,
        _load_fn=_fake_load(_tone(120.0)),
        _pyin_fn=_fake_pyin_mixed_nan(0.55),
    )
    assert result[0]["gender"] is None


def test_nan_ratio_threshold_propagates_lenient():
    """With threshold=0.75, nan_ratio=0.55 → classified (lenient, 0.55 < 0.75)."""
    segs = [{"start": 0.0, "end": 2.0}]
    result = detect_gender_f0(
        "x.wav",
        segs,
        nan_ratio_threshold=0.75,
        _load_fn=_fake_load(_tone(120.0)),
        _pyin_fn=_fake_pyin_mixed_nan(0.55),
    )
    # 45% of frames are valid and all at 120 Hz → should classify as "M"
    assert result[0]["gender"] == "M"
