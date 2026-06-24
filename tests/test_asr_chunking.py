from __future__ import annotations

import importlib
import wave
from pathlib import Path


def _write_wav(path: Path, *, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * int(duration_s * sample_rate))


def test_chunk_export_keeps_any_nonzero_clamped_span(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_CHUNK_MIN_DURATION_S", "9.0")
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))

    from asr import chunking

    chunking = importlib.reload(chunking)
    audio = tmp_path / "source.wav"
    _write_wav(audio, duration_s=1.0)

    _chunk_dir, chunks = chunking._extract_wav_chunks(
        str(audio),
        [
            (0.00, 0.01),
            (0.10, 0.15),
            (0.20, 0.27),
            (0.30, 0.30),
            (0.40, 0.39),
            (0.95, 1.20),
            (1.10, 1.20),
        ],
    )

    assert [(round(chunk["start"], 2), round(chunk["end"], 2)) for chunk in chunks] == [
        (0.00, 0.01),
        (0.10, 0.15),
        (0.20, 0.27),
        (0.95, 1.00),
    ]
    assert [chunk["source_span_index"] for chunk in chunks] == [0, 1, 2, 5]
    assert all(Path(chunk["path"]).exists() for chunk in chunks)
