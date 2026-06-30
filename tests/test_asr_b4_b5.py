from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _reload_pipeline(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))
    from asr import pipeline as asr

    return importlib.reload(asr)


def _write_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def test_asr_text_transcribe_passes_only_static_context(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source.wav"
    _write_wav(source, 2.0)
    monkeypatch.setenv("ASR_CONTEXT", "actor-name")

    chunks = [
        {"index": 0, "start": 0.0, "end": 0.4, "path": str(tmp_path / "c0.wav"), "source_audio_path": str(source)},
        {"index": 1, "start": 0.45, "end": 0.9, "path": str(tmp_path / "c1.wav"), "source_audio_path": str(source)},
    ]

    class FakeBackend:
        is_subprocess = False
        request_batch_size = 1
        contexts: list[str] = []

        def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
            del audio_paths, on_stage
            self.contexts.extend(contexts or [])
            index = len(self.contexts) - 1
            text = f"text-{index}"
            return [
                {
                    "text": text,
                    "raw_text": text,
                    "duration": 0.4,
                    "language": "Japanese",
                    "normalized_path": str(tmp_path / f"c{index}.wav"),
                    "log": ["fake"],
                }
            ]

    backend = FakeBackend()
    results, _timings = asr._transcribe_asr_chunks_text_only(
        backend,
        chunks,
        "ASR 文本转写",
    )

    assert [result["text"] for result in results] == ["text-0", "text-1"]
    assert backend.contexts == ["actor-name", "actor-name"]
