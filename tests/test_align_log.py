from __future__ import annotations

from dataclasses import dataclass
import wave
from pathlib import Path

from asr import local_backend


@dataclass
class _Item:
    text: str
    start_time: float
    end_time: float


class _FakeAligner:
    def align(self, *, audio, text, language):
        del audio, text, language
        return [
            type(
                "Result",
                (),
                {
                    "items": [
                        _Item("東京", 0.0, 0.5),
                        _Item("です", 0.5, 1.0),
                    ]
                },
            )()
        ]


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def test_forced_align_words_prints_alignment_stats(monkeypatch, tmp_path, capsys):
    wav_path = tmp_path / "chunk.wav"
    _write_wav(wav_path)

    backend = local_backend.LocalAsrBackend("cpu")
    monkeypatch.setattr(
        backend,
        "_ensure_forced_aligner",
        lambda on_stage=None: _FakeAligner(),
    )
    monkeypatch.setattr(local_backend, "_clear_cuda_cache", lambda _device: None)

    word_dicts, mode = backend._forced_align_words(
        str(wav_path),
        "東京です",
        "Japanese",
    )

    captured = capsys.readouterr()
    assert mode == "forced_aligner"
    assert len(word_dicts) == 2
    assert "[align]" in captured.out
    assert "chunk_dur=2.0s" in captured.out
    assert "word_count=2" in captured.out
    assert "avg_word_dur=500ms" in captured.out
    assert "min_dur=500ms" in captured.out
