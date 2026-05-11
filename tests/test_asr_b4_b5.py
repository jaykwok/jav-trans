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
    monkeypatch.setenv("ASR_SLIDING_CONTEXT_SEGS", "2")
    monkeypatch.setenv("VAD_MERGE_SHORT_MAX_S", "0.8")
    monkeypatch.setenv("VAD_MERGE_GAP_MAX_S", "0.3")
    from whisper import pipeline as asr

    return importlib.reload(asr)


def _write_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def test_sliding_initial_prompts_reset_on_gap_and_gender(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source.wav"
    _write_wav(source, 3.0)

    chunks = [
        {"index": 0, "start": 0.0, "end": 0.3, "path": str(tmp_path / "c0.wav"), "source_audio_path": str(source)},
        {"index": 1, "start": 0.35, "end": 0.6, "path": str(tmp_path / "c1.wav"), "source_audio_path": str(source)},
        {"index": 2, "start": 0.7, "end": 1.0, "path": str(tmp_path / "c2.wav"), "source_audio_path": str(source)},
        {"index": 3, "start": 1.7, "end": 2.0, "path": str(tmp_path / "c3.wav"), "source_audio_path": str(source)},
        {"index": 4, "start": 2.05, "end": 2.3, "path": str(tmp_path / "c4.wav"), "source_audio_path": str(source), "gender": "female"},
        {"index": 5, "start": 2.35, "end": 2.6, "path": str(tmp_path / "c5.wav"), "source_audio_path": str(source), "gender": "male"},
    ]

    class FakeBackend:
        is_subprocess = False
        request_batch_size = 1
        prompts: list[str | None] = []
        texts = ["固有名詞A", "続きB", "続きC", "リセットD", "女性E", "男性F"]

        def transcribe_texts(self, audio_paths, contexts=None, initial_prompts=None, on_stage=None):
            del audio_paths, contexts, on_stage
            self.prompts.extend(initial_prompts or [])
            index = len(self.prompts) - 1
            text = self.texts[index]
            return [
                {
                    "text": text,
                    "raw_text": text,
                    "duration": 0.3,
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

    assert [result["text"] for result in results] == FakeBackend.texts
    assert backend.prompts == [
        None,
        "固有名詞A",
        "固有名詞A\n続きB",
        None,
        "リセットD",
        None,
    ]


def test_short_vad_chunks_are_physically_merged_with_boundaries(monkeypatch, tmp_path):
    asr = _reload_pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source.wav"
    _write_wav(source, 2.0)

    chunks = [
        {"index": 0, "start": 0.0, "end": 0.35, "path": str(tmp_path / "c0.wav"), "source_audio_path": str(source)},
        {"index": 1, "start": 0.45, "end": 0.7, "path": str(tmp_path / "c1.wav"), "source_audio_path": str(source)},
        {"index": 2, "start": 1.3, "end": 1.6, "path": str(tmp_path / "c2.wav"), "source_audio_path": str(source)},
    ]

    merged = asr._merge_short_vad_chunks(tmp_path, chunks)

    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 0.7
    assert Path(merged[0]["path"]).exists()
    assert merged[0]["merged_from"] == [
        {"index": 0, "start": 0.0, "end": 0.35},
        {"index": 1, "start": 0.45, "end": 0.7},
    ]
    assert merged[1]["index"] == 2
