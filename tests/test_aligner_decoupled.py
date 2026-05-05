from __future__ import annotations

from types import SimpleNamespace

from whisper.local_backend import (
    align_text_to_words,
    looks_like_word_timing_failure,
)


class FakeAligner:
    def align(self, audio, text, language):
        assert audio
        assert language == ["Japanese"]
        assert text == ["今日はいい天気ですね。"]
        return [
            SimpleNamespace(
                items=[
                    SimpleNamespace(text="今日は", start_time=0.0, end_time=0.4),
                    SimpleNamespace(text="いい天気", start_time=0.4, end_time=1.0),
                ]
            )
        ]


def test_align_arbitrary_text(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"")

    words, alignment_mode = align_text_to_words(
        str(audio_path),
        "今日はいい天気ですね。",
        "Japanese",
        aligner_handle=FakeAligner(),
    )

    assert alignment_mode == "forced_aligner"
    assert len(words) > 0
    assert words[-1]["word"].endswith("ですね。")


def test_sentinel_coverage():
    words = [{"word": "あ", "start": 0.0, "end": 0.2} for _ in range(5)]

    assert looks_like_word_timing_failure(words, scene_duration_sec=10.0) is True


