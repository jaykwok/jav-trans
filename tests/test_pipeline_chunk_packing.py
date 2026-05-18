from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path

from vad.base import SegmentationResult, SpeechSegment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_wav(path: Path, seconds: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * seconds))


def _segments() -> list[SpeechSegment]:
    return [
        SpeechSegment(start=index * 1.0, end=index * 1.0 + 0.8)
        for index in range(10)
    ]


class _StubVadBackend:
    name = "stub"

    def segment(
        self,
        audio_path: str,
        *,
        target_sr: int = 16000,
        threshold_override: float | None = None,
    ) -> SegmentationResult:
        del audio_path, target_sr, threshold_override
        segments = _segments()
        return SegmentationResult(
            segments=segments,
            groups=[[segment] for segment in segments],
            method=self.name,
            audio_duration_sec=10.0,
            parameters={"backend": self.name},
            processing_time_sec=0.0,
        )

    def signature(self) -> dict:
        return {"backend": self.name}


class _RecordingBackend:
    is_subprocess = False
    accepts_contexts = True
    timestamp_mode = "forced"
    request_batch_size = 1
    align_batch_size = 1

    def __init__(self) -> None:
        self.audio_paths: list[str] = []
        self.finalized_texts: list[str] = []

    def load(self, on_stage=None) -> None:
        return None

    def close(self) -> None:
        return None

    def unload_model(self, on_stage=None) -> None:
        return None

    def unload_forced_aligner(self, on_stage=None) -> None:
        return None

    def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
        del contexts, on_stage
        self.audio_paths.extend(audio_paths)
        results = []
        for index, path in enumerate(audio_paths):
            text = "怪しい怪しい怪しい怪しい怪しい" if index == 0 else "東京"
            results.append(
                {
                    "text": text,
                    "raw_text": text,
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.1,
                    "compression_ratio": 1.2,
                    "duration": 0.5,
                    "language": "Japanese",
                    "normalized_path": str(Path(path).resolve()),
                    "log": ["fake"],
                }
            )
        return results

    def finalize_text_results(self, text_results, on_stage=None):
        del on_stage
        self.finalized_texts.extend(result["text"] for result in text_results)
        return [
            (
                {
                    "words": [{"start": 0.0, "end": 0.5, "word": result["text"]}],
                    "text": result["text"],
                    "raw_text": result["raw_text"],
                    "alignment_mode": "fake",
                    "duration": result["duration"],
                    "language": result["language"],
                },
                ["Alignment 模式: fake"],
            )
            for result in text_results
        ]


class _LowLogprobBackend(_RecordingBackend):
    def transcribe_texts(self, audio_paths, contexts=None, on_stage=None):
        del contexts, on_stage
        self.audio_paths.extend(audio_paths)
        return [
            {
                "text": "低信頼テキスト",
                "raw_text": "低信頼テキスト",
                "avg_logprob": -1.5,
                "no_speech_prob": 0.1,
                "compression_ratio": 1.2,
                "duration": 0.5,
                "language": "Japanese",
                "normalized_path": str(Path(path).resolve()),
                "log": ["fake"],
            }
            for path in audio_paths
        ]


def _reload_pipeline(monkeypatch, tmp_path: Path, *, packing_enabled: str):
    monkeypatch.setenv("ASR_CHUNK_PACKING_ENABLED", packing_enabled)
    monkeypatch.setenv("ASR_CHUNK_PACK_MAX_S", "28.0")
    monkeypatch.setenv("ASR_CHUNK_PACK_GAP_MERGE_S", "1.5")
    monkeypatch.setenv("ASR_CHUNK_PACK_PADDING_S", "0.0")
    monkeypatch.setenv("ASR_CHUNK_ROOT", str(tmp_path / "chunks"))
    monkeypatch.setenv("VAD_MERGE_SHORT_MAX_S", "0")
    monkeypatch.setenv("VAD_MERGE_GAP_MAX_S", "0")
    monkeypatch.setenv("ASR_CHECKPOINT_ENABLED", "0")

    from whisper import pipeline as asr

    return importlib.reload(asr)


def _run_transcription(monkeypatch, tmp_path: Path, *, packing_enabled: str):
    asr = _reload_pipeline(monkeypatch, tmp_path, packing_enabled=packing_enabled)
    source = tmp_path / f"source_{packing_enabled}.wav"
    _write_wav(source, seconds=12.0)

    import vad

    monkeypatch.setattr(vad, "get_vad_backend", lambda: _StubVadBackend())
    backend = _RecordingBackend()
    monkeypatch.setattr(asr, "_resolve_asr_backend", lambda _device: backend)

    segments, log, details = asr._transcribe_and_align_local(str(source), "cpu")
    return backend, segments, log, details


def _run_transcription_with_backend(
    monkeypatch,
    tmp_path: Path,
    *,
    backend,
    packing_enabled: str = "1",
):
    asr = _reload_pipeline(monkeypatch, tmp_path, packing_enabled=packing_enabled)
    source = tmp_path / f"source_strict_{packing_enabled}.wav"
    _write_wav(source, seconds=12.0)

    import vad

    monkeypatch.setattr(vad, "get_vad_backend", lambda: _StubVadBackend())
    monkeypatch.setattr(asr, "_resolve_asr_backend", lambda _device: backend)

    segments, log, details = asr._transcribe_and_align_local(str(source), "cpu")
    return backend, segments, log, details


def test_chunk_packing_enabled_packs_vad_segments_before_transcribe(monkeypatch, tmp_path):
    backend, _segments, log, details = _run_transcription(
        monkeypatch,
        tmp_path,
        packing_enabled="1",
    )

    assert len(backend.audio_paths) < 10
    assert details["chunk_count"] == len(backend.audio_paths)
    assert any("[chunk] idx=0" in entry and "vad_seg_count=10" in entry for entry in log)


def test_chunk_packing_disabled_keeps_original_vad_chunk_count(monkeypatch, tmp_path):
    backend, _segments, log, details = _run_transcription(
        monkeypatch,
        tmp_path,
        packing_enabled="0",
    )

    assert len(backend.audio_paths) == 10
    assert details["chunk_count"] == 10
    assert not any(entry.startswith("[chunk]") for entry in log)


def test_strict_precision_drops_low_logprob_before_alignment(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_PRECISION_MODE", "strict")
    monkeypatch.setenv("ASR_QC_LOGPROB_THRESHOLD", "-1.0")
    backend, segments, log, details = _run_transcription_with_backend(
        monkeypatch,
        tmp_path,
        backend=_LowLogprobBackend(),
    )

    assert segments == []
    assert backend.finalized_texts == []
    assert details["asr_qc"]["dropped_uncertain_count"] == len(backend.audio_paths)
    assert details["stage_timings"]["asr_strict_dropped_chunks"] == len(backend.audio_paths)
    assert any(entry.startswith("ASR Strict Precision: dropped_uncertain=") for entry in log)
    assert all(chunk["text"] == "" for chunk in details["transcript_chunks"])
