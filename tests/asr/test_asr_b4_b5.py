from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _pipeline(monkeypatch, tmp_path: Path):
    """Point the chunk root at a scratch directory and hand back the stage.

    This used to `importlib.reload` the module to make the new environment take
    effect. Every setting it reads is read inside a function, so the reload only
    forked the module's function objects away from the ones `asr.chunking` and
    `asr.transcribe` still held - patching one then had no effect on the other.
    """
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))
    from asr import pipeline as asr

    return asr


def _write_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


@pytest.mark.parametrize("keep_chunks", [False, True])
def test_asr_chunk_cleanup_can_be_deferred(monkeypatch, tmp_path, keep_chunks):
    asr = _pipeline(monkeypatch, tmp_path)
    source = tmp_path / "sample-a.wav"
    _write_wav(source, 0.1)
    monkeypatch.setenv("KEEP_ASR_CHUNKS", "1" if keep_chunks else "0")
    monkeypatch.setattr(asr, "_build_processing_spans", lambda *_a, **_kw: [])
    monkeypatch.setattr(asr, "_record_cuda_memory", lambda *_a, **_kw: None)
    monkeypatch.setattr(asr, "_release_stage_gpu_cache", lambda *_a, **_kw: None)

    cleanup = []
    segments, _, details = asr.transcribe_and_align(
        str(source), "cpu", include_details=True, defer_cleanup=cleanup.append,
    )

    chunk_root = tmp_path / "chunks"
    assert not segments and details["boundary_no_speech"]
    assert len(list(chunk_root.iterdir())) == 1
    if keep_chunks:
        assert not cleanup
    else:
        (remove_chunks,) = cleanup
        remove_chunks()
        assert not any(chunk_root.iterdir())


def test_asr_text_transcribe_passes_only_audio_paths(monkeypatch, tmp_path):
    asr = _pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source.wav"
    _write_wav(source, 2.0)

    chunks = [
        {"index": 0, "start": 0.0, "end": 0.4, "path": str(tmp_path / "c0.wav"), "source_audio_path": str(source)},
        {"index": 1, "start": 0.45, "end": 0.9, "path": str(tmp_path / "c1.wav"), "source_audio_path": str(source)},
    ]

    class FakeBackend:
        is_subprocess = False
        request_batch_size = 1

        def __init__(self) -> None:
            self.audio_paths: list[str] = []

        def transcribe_texts(self, audio_paths, on_stage=None):
            del on_stage
            self.audio_paths.extend(audio_paths)
            return [
                {
                    "text": f"text-{Path(path).stem}",
                    "raw_text": f"text-{Path(path).stem}",
                    "duration": 0.4,
                    "language": "Japanese",
                    "normalized_path": path,
                    "log": ["fake"],
                }
                for path in audio_paths
            ]

    backend = FakeBackend()
    stages = []
    results, _timings = asr._transcribe_asr_chunks_text_only(
        backend,
        chunks,
        "ASR 文本转写",
        on_stage=stages.append,
    )

    # Longest chunk (c1: 0.45s) runs before shorter chunk (c0: 0.40s), while
    # returned results are restored to chronological chunk index order.
    assert [result["text"] for result in results] == ["text-c0", "text-c1"]
    assert backend.audio_paths == [chunks[1]["path"], chunks[0]["path"]]
    assert stages[2] == "ASR 文本转写 1/2 batch=2 size=1 start"
    assert stages[3].startswith("ASR 文本转写 2/2 batch=2 size=1 done")


def test_asr_text_transcribe_rejects_short_batch_results(monkeypatch, tmp_path):
    asr = _pipeline(monkeypatch, tmp_path)
    source = tmp_path / "source.wav"
    _write_wav(source, 2.0)
    chunks = [
        {
            "index": index,
            "start": index * 0.5,
            "end": index * 0.5 + 0.4,
            "path": str(tmp_path / f"c{index}.wav"),
            "source_audio_path": str(source),
        }
        for index in range(2)
    ]

    class ShortBatchBackend:
        is_subprocess = False
        request_batch_size = 2

        def transcribe_texts(self, audio_paths, on_stage=None):
            del audio_paths, on_stage
            return [
                {
                    "text": "only-one-result",
                    "raw_text": "only-one-result",
                    "duration": 0.4,
                    "language": "Japanese",
                    "normalized_path": str(tmp_path / "c0.wav"),
                    "log": ["fake"],
                }
            ]

    with pytest.raises(RuntimeError, match="ASR batch result count mismatch"):
        asr._transcribe_asr_chunks_text_only(
            ShortBatchBackend(),
            chunks,
            "ASR 文本转写",
        )
