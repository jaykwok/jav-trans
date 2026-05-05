from dataclasses import dataclass

from whisper.local_backend import (
    clean_text_for_aligner,
    restore_timestamps_to_original,
)


@dataclass
class _Item:
    text: str
    start_time: float
    end_time: float


class _RecordingAligner:
    def __init__(self):
        self.texts = []

    def align(self, *, audio, text, language):
        del audio, language
        self.texts.extend(text)
        return [type("Result", (), {"items": [_Item(text[0], 0.0, 1.0)]})()]


def test_clean_text_for_aligner_removes_decorations_and_keeps_core():
    cleaned = clean_text_for_aligner("  あっっっ♡www 気持ちいい〜  ")

    assert cleaned == "あっっ 気持ちいい"
    assert clean_text_for_aligner("~~~♡ｗｗｗ!!!") == ""


def test_restore_timestamps_to_original_keeps_dirty_original_text():
    original = "あっっっ♡www気持ちいい~"
    cleaned = clean_text_for_aligner(original)
    restored = restore_timestamps_to_original(
        original,
        cleaned,
        [
            {"word": "あっ", "start": 0.0, "end": 0.4},
            {"word": "気持ちいい", "start": 0.5, "end": 1.2},
        ],
    )

    assert restored
    assert all(item["word"] for item in restored)
    assert restored[0]["start"] == 0.0
    assert restored[-1]["end"] == 1.2


def test_forced_align_words_sends_cleaned_text_and_restores(monkeypatch):
    aligner = _RecordingAligner()
    from whisper.local_backend import LocalAsrBackend

    backend = LocalAsrBackend("cpu")
    monkeypatch.setattr(backend, "_ensure_forced_aligner", lambda on_stage=None: aligner)

    words, mode = backend._forced_align_words(
        "missing.wav",
        "あっっっ♡www気持ちいい~",
        "Japanese",
    )

    assert mode == "forced_aligner"
    assert aligner.texts == ["あっっ気持ちいい"]
    assert "".join(item["word"] for item in words) == "あっっっ♡www気持ちいい~"


